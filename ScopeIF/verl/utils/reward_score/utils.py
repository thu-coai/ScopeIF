import math
import re
import json
import typing
import pysbd
import signal
import subprocess
import sys
from traceback import print_exc
import os
import httpx
from openai import OpenAI
from typing import Dict
seg = pysbd.Segmenter()


timeout = httpx.Timeout(
    timeout=float(os.getenv("OPENAI_TOTAL_TIMEOUT", "1800")),
    connect=float(os.getenv("OPENAI_CONNECT_TIMEOUT", "300")),
    read=float(os.getenv("OPENAI_READ_TIMEOUT", "1800")),
    write=float(os.getenv("OPENAI_WRITE_TIMEOUT", "300")),
    pool=float(os.getenv("OPENAI_POOL_TIMEOUT", "300")),
)


_CODE_EXECUTION_TIMEOUT_SECONDS = 3.0
_CODE_EXECUTOR_PATH = os.path.join(os.path.dirname(__file__), "code_executor_worker.py")


def execute_function(code, params_str, response):
    """Execute generated validation code in an isolated child process.

    This function is called from reward worker threads, where ``signal.signal``
    cannot be used. The child process preserves a hard timeout without splitting
    the existing per-constraint code-generation / critique pipeline.
    """
    if not code or not code.strip():
        return "", False

    process = None
    request = json.dumps(
        {
            "code": code,
            "params_str": params_str,
            "response": response,
        },
        ensure_ascii=False,
    )

    try:
        process = subprocess.Popen(
            [sys.executable, _CODE_EXECUTOR_PATH],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            start_new_session=True,
        )
        stdout, _stderr = process.communicate(
            input=request,
            timeout=_CODE_EXECUTION_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        # Kill the whole child process group so code spawned by generated code
        # cannot survive the three-second validation timeout.
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                process.kill()
            process.communicate()
        return "", False
    except Exception:
        if process is not None and process.poll() is None:
            process.kill()
            process.communicate()
        return "", False

    if process.returncode != 0:
        return "", False

    try:
        payload = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        return "", False

    result = payload.get("result")
    flag = payload.get("flag") is True
    if not flag or not isinstance(result, str):
        return "", False
    return result, True


def _execute_function_impl(code, params_str, response):
    """Run validation code inside the child process without timeout handling."""
    if not code or not code.strip():
        return "", False

    flag = True
    result = ""
    try:
        global_namespace = {
            "count_word": count_word,
            "load_json": load_json,
            "split_sentences": split_sentences,
            "re": re,
            "__builtins__": __builtins__
        }
        global_namespace.update(typing_names)

        exec(code, global_namespace)
        if "def " in code:
            function_name = code.split('def ')[1].split('(')[0].strip()
        else:
            function_name = code.split('(')[0].split()[-1]

        params = json.loads(params_str) if params_str else []
        input_dict = {}

        if isinstance(params, list):
            for item in params:
                if isinstance(item, dict) and "参数名" in item and "参数值" in item:
                    input_dict[item["参数名"]] = parser_extraction_results(item["参数值"], response)
        elif isinstance(params, dict):
            for k, v in params.items():
                input_dict[k] = parser_extraction_results(v, response)

        result = str(global_namespace[function_name](**input_dict))
    except Exception:
        flag = False

    if not flag:
        result = ""

    return result, flag


typing_names = {
    name: getattr(typing, name)
    for name in dir(typing)
    if (not name.startswith("_") and name != "re")
}


def check_judge_format(text):
    if "[[人工智能助手的回复满足了该要求]]" in text or "[[人工智能助手的回复没有满足该要求]]" in text:
        return True
    if "人工智能助手的回复满足了该要求" in text or "人工智能助手的回复没有满足该要求" in text:
        return True
    return False


def _parse_required_bound(value):
    """Normalize a range bound and reject NaN / bool-like pseudo numbers."""
    if isinstance(value, bool):
        return False, None

    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isnan(numeric):
            return False, None
        if math.isinf(numeric):
            return True, "inf" if numeric > 0 else "-inf"
        return True, numeric

    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return False, None
        if normalized in {"inf", "+inf", "infinity", "+infinity"}:
            return True, "inf"
        if normalized in {"-inf", "-infinity"}:
            return True, "-inf"
        try:
            numeric = float(normalized)
        except ValueError:
            return False, None
        if not math.isfinite(numeric):
            return False, None
        return True, numeric

    return False, None


def _parse_actual_quantity(value):
    """Normalize actual quantities and reject missing / non-finite values early."""
    if value is None or isinstance(value, bool):
        return False, None

    if isinstance(value, (int, float)):
        numeric = float(value)
    elif isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return False, None
        try:
            numeric = float(normalized)
        except ValueError:
            return False, None
    else:
        return False, None

    if not math.isfinite(numeric):
        return False, None

    return True, numeric


def validate_critique_json(json_str):
    try:
        json_str = json_str.replace('\xa0', ' ')
        json_str = re.sub(r'//.*', '', json_str)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        parsed_data = json.loads(json_str)

        if not isinstance(parsed_data, list):
            return False, None

        if len(parsed_data) == 0:
            return True, parsed_data

        for item in parsed_data:
            if not isinstance(item, dict):
                return False, None
                
            required_keys = {"计数目标", "要求数量", "实际数量"}
            if not required_keys.issubset(item.keys()):
                return False, None
                
            if not isinstance(item["计数目标"], str) or not item["计数目标"].strip():
                return False, None
                
            if not isinstance(item["要求数量"], list) or len(item["要求数量"]) != 2:
                return False, None

            normalized_required = []
            for x in item["要求数量"]:
                is_valid_bound, normalized_bound = _parse_required_bound(x)
                if not is_valid_bound:
                    return False, None
                normalized_required.append(normalized_bound)

            lower = float('-inf') if normalized_required[0] == "-inf" else float(normalized_required[0])
            upper = float('inf') if normalized_required[1] == "inf" else float(normalized_required[1])

            if lower > upper:
                return False, None

            is_valid_actual, normalized_actual = _parse_actual_quantity(item["实际数量"])
            if not is_valid_actual:
                return False, None

            item["计数目标"] = item["计数目标"].strip()
            item["要求数量"] = normalized_required
            item["实际数量"] = normalized_actual

        return True, parsed_data
    except json.JSONDecodeError:
        return False, None


def parse_judge(text):
    if "[[人工智能助手的回复满足了该要求]]" in text:
        return True
    if "[[人工智能助手的回复没有满足该要求]]" in text:
        return False
    if "人工智能助手的回复满足了该要求" in text:
        return True
    if "人工智能助手的回复没有满足该要求" in text:
        return False
    return False


def split_solution_parts(text):
    if text is None:
        text = ""
    elif not isinstance(text, str):
        text = str(text)

    if "</think>" not in text:
        return "", text

    reasoning_text, response_text = text.rsplit("</think>", 1)
    reasoning_text = re.sub(r"</?think>", "", reasoning_text).strip()
    return reasoning_text, response_text.strip()


def get_response_length_metrics(text):
    reasoning_text, response_text = split_solution_parts(text)
    stripped_text = text.strip() if isinstance(text, str) else str(text or "").strip()
    total_length = len(stripped_text)
    reasoning_length = len(reasoning_text)
    non_reasoning_length = len(response_text.strip())
    reasoning_fraction = reasoning_length / total_length if total_length > 0 else 0.0
    return {
        "reasoning_text": reasoning_text,
        "response_text": response_text,
        "response_total_length": float(total_length),
        "response_reasoning_length": float(reasoning_length),
        "response_non_reasoning_length": float(non_reasoning_length),
        "response_reasoning_fraction": float(reasoning_fraction),
    }


def parse_solution(text):
    return split_solution_parts(text)[1]


def parse_solution_oss(text):
    if "<|start|>assistant<|channel|>final<|message|>" not in text:
        return text.strip()
    return text.split("<|start|>assistant<|channel|>final<|message|>")[-1].strip()


class VLLMClient:
    def __init__(self, base_url, api_key="EMPTY", model_name="GPT-OSS-120B", max_retries=5):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
        self.model_name = model_name
        self.max_retries = max_retries

    def get_response(self, messages, max_tokens=32768, temperature=0.0):
        retry = self.max_retries
        while retry > 0:
            try:
                chat_response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    stream=True,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                segment = ""
                for chat in chat_response:
                    content = chat.choices[0].delta.content
                    if content:
                        segment += content
                
                if segment:
                    segment = parse_solution_oss(segment)
                    return segment
                    
            except Exception as e:
                print_exc()
            retry -= 1
        return ""


def build_judge_client(max_retries=1):
    """Judge model client. Point SCOPEIF_JUDGE_BASE_URL at an OpenAI-compatible endpoint."""
    return VLLMClient(
        base_url=os.getenv("SCOPEIF_JUDGE_BASE_URL", "http://localhost:8080/v1"),
        api_key=os.getenv("SCOPEIF_JUDGE_API_KEY", "EMPTY"),
        model_name=os.getenv("SCOPEIF_JUDGE_MODEL", "GPT-OSS-120B-Default"),
        max_retries=max_retries,
    )


def parser_extraction_results(result, response):
    if result == "ALL":
        return response
    elif result == "NONE":
        return ""
    return result


def check_contain(para, response):
    if para == "" or para == []:
        return True
    if isinstance(para, str) and para not in response:
        return False
    if isinstance(para, list):
        for p in para:
            if p != "" and p not in response:
                return False
    return True


def extract_plan(text):
    plan_pattern = r"\[计数目标-开始\]\n?(.*?)\n?\[计数目标-结束\]"
    plan_match = re.search(plan_pattern, text, re.S)
    extracted_plan = plan_match.group(1) if plan_match else ""
    return extracted_plan.strip()


def extract_code_and_params(code_context):
    code_pattern = re.compile(r'```python\s*\n(.*?)\s*```', flags=re.DOTALL)
    code_matches = code_pattern.findall(code_context)
    params_pattern = re.compile(r'```json\s*\n(.*?)\s*```', flags=re.DOTALL)
    params_matches = params_pattern.findall(code_context)
    code = code_matches[-1].strip() if code_matches else ""
    params = params_matches[-1].strip() if params_matches else "[]"
    return code, params


def extract_code(code_context):
    code_pattern = re.compile(r'```python\s*\n(.*?)\n```', flags=re.DOTALL)
    code_matches = code_pattern.findall(code_context)
    code = code_matches[-1].strip() if code_matches else None
    return code


def extract_extraction_result(code_context):
    result_pattern = re.compile(r'```json\s*\n(.*?)\n```', flags=re.DOTALL)
    result_matches = result_pattern.findall(code_context)
    extraction_result = result_matches[-1].strip() if result_matches else None
    return extraction_result


def count_word(s: str) -> int:
    count = 0
    found_english = False
    for i in range(len(s)):
        chr = s[i]
        is_chinese_char = '\u4e00' <= chr <= '\u9fff'
        is_english_char = re.match(r'^[a-zA-Z0-9]+$', chr) is not None
        is_english_punctuation = re.match(r'[.,;!?(){}\[\]<>:"\'`~\-+*/&^%$#@|\\]', chr) is not None
        if is_chinese_char:
            count += 1
            if found_english:
                count += 1
                found_english = False
        elif is_english_char or is_english_punctuation:
            if not found_english:
                found_english = True
        elif chr == ' ' or chr == '\n':
            if found_english:
                count += 1
                found_english = False
        else:
            count += 1
            if found_english:
                count += 1
                found_english = False
    if found_english:
        count += 1
    return count


def split_sentences(text):
    sentences = seg.segment(text)
    sentences = [s.strip() for s in sentences]
    sentences = [s for s in sentences if s != ""]
    return sentences


def load_json(s: str) -> Dict | None:
    s = s.strip()
    if not isinstance(s, str):
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    match = re.search(r'^```json\s*([\s\S]*?)\s*```$', s, re.DOTALL)
    if match:
        json_text = match.group(1)
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            pass
    match = re.search(r'^```JSON\s*([\s\S]*?)\s*```$', s, re.DOTALL)
    if match:
        json_text = match.group(1)
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            pass
    return None


def check_value_in_range(actual, required_range):
    min_val, max_val = required_range
    if min_val == "-inf" or min_val == float('-inf'):
        min_val = float('-inf')
    if max_val == "inf" or max_val == float('inf'):
        max_val = float('inf')
    min_val = float(min_val) if not isinstance(min_val, (int, float)) else min_val
    max_val = float(max_val) if not isinstance(max_val, (int, float)) else max_val
    actual = float(actual) if not isinstance(actual, (int, float)) else actual
    return min_val <= actual <= max_val
