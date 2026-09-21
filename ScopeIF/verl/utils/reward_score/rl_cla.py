import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import wandb

from .prompts.code_evaluation_prompts import code_evaluation_prompt
from .prompts.llm_evaluation_prompts import llm_evaluation_prompt
from .utils import (
    build_judge_client,
    check_judge_format,
    check_value_in_range,
    execute_function,
    extract_code_and_params,
    get_response_length_metrics,
    parse_judge,
    validate_critique_json,
)


def _get_parallel_workers():
    raw_value = os.getenv("SCOPEIF_PARALLEL_MAX_WORKERS")
    if raw_value is None:
        return 16
    try:
        return max(1, int(raw_value))
    except ValueError:
        return 16


def build_code_messages(d, j):
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": code_evaluation_prompt.format(
                prompt=d["prompt"],
                response=d["response"][:20000],
                checklist=d["checklist"][j],
                plan=d["targets"][j],
            ),
        },
    ]


def build_critique_messages(d, j):
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": llm_evaluation_prompt.format(
                prompt=d["prompt"],
                response=d["response"][:20000],
                checklist=d["checklist"][j],
                results=d["results"][j],
                plan=d["targets"][j],
            ),
        },
    ]


def generate_one_code(d, j, client, temperature):
    messages = build_code_messages(d, j)
    return client.get_response(messages, temperature=temperature)


def generate_one_critique(d, j, client, temperature):
    messages = build_critique_messages(d, j)
    return client.get_response(messages, temperature=temperature)


def _safe_generate_one_code(data, j, client, temperature):
    try:
        return generate_one_code(data, j, client, temperature)
    except Exception:
        return ""


def _safe_generate_one_critique(data, j, client, temperature):
    try:
        return generate_one_critique(data, j, client, temperature)
    except Exception:
        return ""


def _generate_code_with_retries(data, j, client, max_iters):
    for iter_num in range(max_iters):
        current_temperature = 0.0 if iter_num == 0 else 0.0
        parsed_text = _safe_generate_one_code(data, j, client, current_temperature)
        flag, intentional_none = _process_code_result(data, j, parsed_text)
        if flag or intentional_none:
            break


def _generate_critique_with_retries(data, j, client, max_iters):
    for iter_num in range(max_iters):
        current_temperature = 0.0 if iter_num == 0 else 0.0
        parsed_text = _safe_generate_one_critique(data, j, client, current_temperature)
        judge = parse_judge(parsed_text)
        is_valid, formatted_json = _extract_validated_critique(parsed_text, judge)

        data["critiques"][j] = parsed_text
        if is_valid:
            data["predict_labels"][j] = judge
            data["objective_results"][j] = formatted_json
            data["critique_valids"][j] = True
            return False

        data["predict_labels"][j] = None
        data["objective_results"][j] = None
        data["critique_valids"][j] = False

    return True


def _process_one_item(data, j, client, code_max_iters, critique_max_iters):
    _generate_code_with_retries(data, j, client, code_max_iters)
    return _generate_critique_with_retries(data, j, client, critique_max_iters)


def _run_parallel_items(data, item_indices, client, code_max_iters, critique_max_iters):
    if not item_indices:
        return []

    max_workers = min(_get_parallel_workers(), len(item_indices))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_process_one_item, data, j, client, code_max_iters, critique_max_iters)
            for j in item_indices
        ]
        unresolved = []
        for future in futures:
            try:
                unresolved.append(bool(future.result()))
            except Exception:
                unresolved.append(True)
        return unresolved


def _normalize_bound(x):
    if x in ["inf", float("inf")]:
        return float("inf")
    if x in ["-inf", float("-inf")]:
        return float("-inf")
    return float(x)


def _distance_to_range(actual, min_val, max_val):
    if actual < min_val:
        return min_val - actual
    if actual > max_val:
        return actual - max_val
    return 0.0


def _scale_for_range(min_val, max_val):
    min_val = max(min_val, 0)
    max_val = max(max_val, 0)
    is_min_inf = math.isinf(min_val)
    is_max_inf = math.isinf(max_val)
    if not is_min_inf and not is_max_inf:
        return max(1.0, (max_val + min_val) / 2.0)
    if not is_min_inf:
        return max(1.0, abs(min_val))
    if not is_max_inf:
        return max(1.0, abs(max_val))
    return 1.0


def calculate_fine_grained_reward(actual_val, required_range, alpha=3.0):
    min_val, max_val = required_range
    min_val = _normalize_bound(min_val)
    max_val = _normalize_bound(max_val)
    actual = float(actual_val)

    distance = _distance_to_range(actual, min_val, max_val)
    if distance == 0:
        return 1.0

    scale = _scale_for_range(min_val, max_val)
    return math.exp(-alpha * distance / scale)


def _is_intentional_none(parsed_text):
    if parsed_text.strip() == "无":
        return True

    block_match = re.search(r"\[辅助验证代码-开始\](.*?)\[辅助验证代码-结束\]", parsed_text, flags=re.DOTALL)
    if not block_match:
        return False

    content = block_match.group(1).strip()
    clean_content = re.sub(r"```(?:python)?", "", content)
    clean_content = clean_content.replace("```", "").strip()
    return clean_content == "无"


def _process_code_result(data, j, parsed_text):
    try:
        code, param = extract_code_and_params(parsed_text)
    except Exception:
        code, param = "", "[]"

    intentional_none = False
    if not code:
        intentional_none = _is_intentional_none(parsed_text)

    if not code and intentional_none:
        result, flag = "", True
    else:
        result, flag = execute_function(code, param, data["response"])

    if len(result) > 10000:
        result = ""
        flag = False

    data["codes"][j] = code
    data["params"][j] = param
    data["results"][j] = result
    data["flags"][j] = flag
    return flag, intentional_none


def _extract_validated_critique(parsed_text, _judge):
    if not check_judge_format(parsed_text):
        return False, None

    json_pattern = re.compile(r"```json\s*\n(.*?)\s*```", flags=re.DOTALL)
    json_matches = json_pattern.findall(parsed_text)
    if not json_matches:
        return False, None

    json_str = json_matches[-1].strip()
    valid, parsed_json = validate_critique_json(json_str)
    if not valid:
        return False, None

    if not parsed_json:
        return False, None

    expected_keys = {"计数目标", "要求数量", "实际数量"}
    for item in parsed_json:
        if set(item.keys()) != expected_keys:
            return False, None

        actual = float(item["实际数量"])
        if not actual.is_integer():
            return False, None
        item["实际数量"] = int(actual)


    # The critique verdict may disagree with the per-target quantities; the
    # normalized counting results are the source of truth for aggregation.
    return True, parsed_json


def _score_objective_results(obj_results, alpha):
    if not obj_results:
        return None, None, False

    try:
        all_satisfied = True
        item_rewards = []
        for target_item in obj_results:
            actual = target_item.get("实际数量")
            required = target_item.get("要求数量")
            if actual is None or required is None or not isinstance(required, list) or len(required) != 2:
                return None, None, False

            if not check_value_in_range(actual, required):
                all_satisfied = False

            item_reward = calculate_fine_grained_reward(actual, required, alpha=alpha)
            if isinstance(item_reward, (int, float, np.floating)) and math.isfinite(float(item_reward)):
                item_rewards.append(float(item_reward))
            else:
                return float(all_satisfied), None, True

        if not item_rewards:
            return float(all_satisfied), None, False

        safe_rewards = np.clip(item_rewards, 1e-5, 1.0)
        item_geom_mean = float(np.exp(np.mean(np.log(safe_rewards))))
        if not math.isfinite(item_geom_mean):
            return float(all_satisfied), None, True
        return float(all_satisfied), item_geom_mean, False
    except Exception:
        return None, None, True


def compute_score(*args, data_source=None, solution_str=None, ground_truth=None, extra_info=None, **kwargs):
    if args:
        if len(args) > 0 and solution_str is None:
            solution_str = args[0]
        if len(args) > 1 and ground_truth is None:
            ground_truth = args[1]
        if len(args) > 2 and extra_info is None:
            extra_info = args[2]

    if solution_str is None:
        raise ValueError("solution_str is required")
    if ground_truth is None:
        raise ValueError("ground_truth is required")

    extra_info = extra_info or {}
    response_metrics = get_response_length_metrics(solution_str)
    solution_text = solution_str if isinstance(solution_str, str) else str(solution_str)
    last_think_open = solution_text.rfind("<think>")
    last_think_close = solution_text.rfind("</think>")
    response_is_truncated = bool(extra_info.get("response_is_truncated", False))
    response_think_unclosed = last_think_open != -1 and (
        last_think_close == -1 or last_think_open > last_think_close
    )
    response_invalid = response_is_truncated or response_think_unclosed

    if response_invalid:
        return {
            "score": 0.0,
            "binary_reward": 0.0,
            "fine_reward": 0.0,
            "json_positive_judge_negative_rate": 0.0,
            "json_negative_judge_positive_rate": 0.0,
            "critique_unresolved": 0.0,
            "fine_reward_is_nan": False,
            "response_is_truncated": float(response_is_truncated),
            "response_think_unclosed": float(response_think_unclosed),
            "response_invalid": 1.0,
            "response_total_length": response_metrics["response_total_length"],
            "response_reasoning_length": response_metrics["response_reasoning_length"],
            "response_non_reasoning_length": response_metrics["response_non_reasoning_length"],
            "response_reasoning_fraction": response_metrics["response_reasoning_fraction"],
        }

    max_iters = 3
    code_max_iters = 2
    alpha = 3.0
    beta = 1.0


    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)

    checklist = ground_truth.get("checklist", [])
    targets = ground_truth.get("targets", [])
    data = {
        "prompt_id": ground_truth.get("prompt_id", 0),
        "prompt": ground_truth.get("prompt", ""),
        "response": response_metrics["response_text"],
        "checklist": checklist,
        "targets": targets,
        "codes": [None] * len(checklist),
        "params": [None] * len(checklist),
        "results": [None] * len(checklist),
        "flags": [None] * len(checklist),
        "critiques": [None] * len(checklist),
        "predict_labels": [None] * len(checklist),
        "objective_results": [None] * len(checklist),
        "critique_valids": [False] * len(checklist),
    }

    client = build_judge_client(max_retries=1)

    item_indices = list(range(len(data["checklist"])))
    unresolved_items = _run_parallel_items(data, item_indices, client, code_max_iters, max_iters)

    fine_reward_is_nan = False
    prompt_failed = any(unresolved_items)
    checklist_len = len(data["checklist"])

    if not data["checklist"]:
        final_reward = 0.0
        binary_reward = 0.0
        fine_reward = 0.0
        json_positive_judge_negative = 0.0
        json_negative_judge_positive = 0.0
    else:
        binary_reward = 0.0
        fine_check_rewards = []
        json_positive_judge_negative = 0.0
        json_negative_judge_positive = 0.0

        for j in range(checklist_len):
            llm_judge = data["predict_labels"][j]
            judge_value = float(bool(llm_judge)) if llm_judge is not None else 0.0
            obj_results = data["objective_results"][j]
            constraint_hard_reward = judge_value

            if obj_results and len(obj_results) > 0:
                objective_hard_reward, objective_soft_reward, objective_reward_is_nan = _score_objective_results(
                    obj_results, alpha
                )
                if objective_reward_is_nan:
                    fine_reward_is_nan = True
                if objective_hard_reward is not None:
                    constraint_hard_reward = objective_hard_reward
                    if llm_judge is not None:
                        if objective_hard_reward >= 1.0 and not bool(llm_judge):
                            json_positive_judge_negative += 1.0
                        elif objective_hard_reward < 1.0 and bool(llm_judge):
                            json_negative_judge_positive += 1.0
                if objective_soft_reward is not None:
                    fine_check_rewards.append(objective_soft_reward)
                else:
                    fine_check_rewards.append(judge_value)
            else:
                fine_check_rewards.append(judge_value)

            binary_reward += constraint_hard_reward / checklist_len

        if fine_check_rewards:
            fine_reward = float(np.mean(fine_check_rewards))
        else:
            fine_reward = float(binary_reward if checklist_len > 0 else 0.0)

        if not math.isfinite(fine_reward):
            fine_reward_is_nan = True
            fine_reward = float(binary_reward)

        prompt_failed = prompt_failed or fine_reward_is_nan
        final_reward = float(binary_reward) if prompt_failed else (
            beta * binary_reward + (1 - beta) * fine_reward
        )

    return {
        "score": final_reward,
        "binary_reward": binary_reward,
        "fine_reward": fine_reward,
        "json_positive_judge_negative_rate": float(json_positive_judge_negative / checklist_len)
        if checklist_len > 0
        else 0.0,
        "json_negative_judge_positive_rate": float(json_negative_judge_positive / checklist_len)
        if checklist_len > 0
        else 0.0,
        "critique_unresolved": float(prompt_failed),
        "fine_reward_is_nan": bool(fine_reward_is_nan),
        "response_is_truncated": float(response_is_truncated),
        "response_think_unclosed": float(response_think_unclosed),
        "response_invalid": 0.0,
        "response_total_length": response_metrics["response_total_length"],
        "response_reasoning_length": response_metrics["response_reasoning_length"],
        "response_non_reasoning_length": response_metrics["response_non_reasoning_length"],
        "response_reasoning_fraction": response_metrics["response_reasoning_fraction"],
    }
