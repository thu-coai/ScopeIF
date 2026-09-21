import ast
import contextlib
import io
import json
import math
import os
import re
import signal
import subprocess
import sys
import typing
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


try:
    import pysbd

    _SEGMENTER = pysbd.Segmenter()
except Exception:
    _SEGMENTER = None


typing_names = {
    name: getattr(typing, name)
    for name in dir(typing)
    if not name.startswith("_") and name != "re"
}


def parse_solution(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    if "<|start|>assistant<|channel|>final<|message|>" in text:
        text = text.split("<|start|>assistant<|channel|>final<|message|>")[-1]
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


def parse_judge(text: str) -> bool:
    text = parse_solution(text)
    if "[[人工智能助手的回复满足了该要求]]" in text:
        return True
    if "[[人工智能助手的回复没有满足该要求]]" in text:
        return False
    if "人工智能助手的回复满足了该要求" in text:
        return True
    if "人工智能助手的回复没有满足该要求" in text:
        return False
    return False


def has_single_exact_judge(text: str) -> bool:
    text = parse_solution(text)
    positive = text.count("[[人工智能助手的回复满足了该要求]]")
    negative = text.count("[[人工智能助手的回复没有满足该要求]]")
    return positive + negative == 1


def has_any_exact_judge(text: str) -> bool:
    text = parse_solution(text)
    return (
        "[[人工智能助手的回复满足了该要求]]" in text
        or "[[人工智能助手的回复没有满足该要求]]" in text
    )


def _last_fenced_block(text: str, language: str) -> Optional[str]:
    pattern = re.compile(rf"```{language}\s*\n(.*?)\s*```", flags=re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(text or "")
    return matches[-1].strip() if matches else None


def extract_code_and_params(text: str) -> Tuple[str, str]:
    text = parse_solution(text)
    code = _last_fenced_block(text, "python") or ""
    params = _last_fenced_block(text, "json") or "[]"
    return code.strip(), params.strip()


def extract_json_blocks(text: str) -> List[str]:
    text = parse_solution(text)
    return re.findall(r"```json\s*\n(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)


def load_json(s: str) -> Optional[Dict]:
    if not isinstance(s, str):
        return None
    s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    blocks = extract_json_blocks(s)
    if blocks:
        try:
            return json.loads(blocks[-1])
        except json.JSONDecodeError:
            return None
    return None


def validate_critique_json(json_str: str) -> Tuple[bool, Optional[List[Dict[str, Any]]]]:
    json_str = json_str.replace("\xa0", " ")
    try:
        data = json.loads(json_str)
    except Exception:
        try:
            without_comments = re.sub(r"//.*", "", json_str)
            without_comments = re.sub(r"/\*.*?\*/", "", without_comments, flags=re.DOTALL)
            data = json.loads(without_comments)
        except Exception:
            return False, None

    if not isinstance(data, list) or not data:
        return False, None
    for item in data:
        if not isinstance(item, dict):
            return False, None
        if not {"计数目标", "要求数量", "实际数量"}.issubset(item.keys()):
            return False, None
        if not isinstance(item["计数目标"], str) or not item["计数目标"].strip():
            return False, None
        if not isinstance(item["要求数量"], list) or len(item["要求数量"]) != 2:
            return False, None
        if isinstance(item["实际数量"], bool):
            return False, None
        try:
            low = normalize_bound(item["要求数量"][0])
            high = normalize_bound(item["要求数量"][1])
            actual = normalize_bound(item["实际数量"])
            if math.isnan(low) or math.isnan(high) or low > high or not math.isfinite(actual):
                return False, None
        except Exception:
            return False, None
    return True, data


def parse_objective_results(text: str) -> Optional[List[Dict[str, Any]]]:
    blocks = extract_json_blocks(text)
    if not blocks:
        return None
    merged = []
    for block in blocks:
        valid, parsed = validate_critique_json(block.strip())
        if not valid or not parsed:
            return None
        merged.extend(parsed)
    return merged or None


def parser_extraction_results(result: Any, response: str) -> Any:
    if result == "ALL":
        return response
    if result is None or result == "NONE":
        return ""
    return result


def count_word(s: str) -> int:
    count = 0
    found_english = False
    for ch in s:
        is_chinese_char = "\u4e00" <= ch <= "\u9fff"
        is_english_char = re.match(r"^[a-zA-Z0-9]+$", ch) is not None
        is_english_punctuation = re.match(r"""[.,;!?(){}\[\]<>:"'`~\-+*/&^%$#@|\\]""", ch) is not None
        if is_chinese_char:
            count += 1
            if found_english:
                count += 1
                found_english = False
        elif is_english_char or is_english_punctuation:
            if not found_english:
                found_english = True
        elif ch in {" ", "\n", "\t"}:
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


def split_sentences(text: str) -> List[str]:
    if _SEGMENTER is not None:
        sentences = _SEGMENTER.segment(text)
    else:
        sentences = re.split(r"(?<=[。！？!?\.])\s+|\n+", text)
    return [s.strip() for s in sentences if s and s.strip()]


def normalize_bound(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("Boolean is not a numeric bound")
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        raise ValueError("None is not a numeric bound")
    text = str(value).strip().lower().replace("+∞", "inf").replace("∞", "inf")
    compact = re.sub(r"\s+", "", text)
    if compact in {
        "inf",
        "+inf",
        "infinity",
        "+infinity",
        "float('inf')",
        'float("inf")',
        "float('+inf')",
        'float("+inf")',
    }:
        return float("inf")
    if compact in {
        "-inf",
        "-infinity",
        "float('-inf')",
        'float("-inf")',
    }:
        return float("-inf")
    return float(compact)


def check_value_in_range(actual: Any, required_range: Iterable[Any]) -> bool:
    low_raw, high_raw = list(required_range)
    if actual is None:
        return False
    low = normalize_bound(low_raw)
    high = normalize_bound(high_raw)
    value = normalize_bound(actual)
    return low <= value <= high


def objective_satisfied(data_list: Any, default_on_invalid: Optional[int] = 0) -> Optional[int]:
    if isinstance(data_list, str):
        try:
            data_list = ast.literal_eval(data_list)
        except Exception:
            return default_on_invalid
    if not isinstance(data_list, list) or not data_list:
        return default_on_invalid
    for item in data_list:
        try:
            required = item.get("要求数量", [0, 0])
            actual = item.get("实际数量", 0)
            if not check_value_in_range(actual, required):
                return 0
        except Exception:
            return default_on_invalid
    return 1


class ExecTimeout(Exception):
    pass


def _function_name(code: str) -> str:
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node.name
    raise ValueError("No function definition found")


def _validate_single_response_parameter(code: str) -> None:
    tree = ast.parse(code)
    function = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef)),
        None,
    )
    if function is None:
        raise ValueError("No function definition found")
    positional = [*function.args.posonlyargs, *function.args.args]
    if (
        len(positional) != 1
        or function.args.vararg is not None
        or function.args.kwarg is not None
        or function.args.kwonlyargs
    ):
        raise ValueError("Helper function must accept exactly one response parameter")


def _exec_namespace() -> Dict[str, Any]:
    namespace = {
        "count_word": count_word,
        "load_json": load_json,
        "split_sentences": split_sentences,
        "re": re,
        "math": math,
    }
    namespace.update(typing_names)
    return namespace


def _sandbox_worker_main() -> None:
    try:
        import resource

        payload = json.loads(sys.stdin.read())
        memory_limit_mb = int(payload.get("memory_limit_mb", 0))
        timeout_seconds = max(1, int(payload.get("timeout_seconds", 5)))
        if memory_limit_mb > 0:
            memory_limit_bytes = memory_limit_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (timeout_seconds, timeout_seconds + 1))

        mode = payload.get("mode")
        if mode == "probe":
            # Environment probe: executes no judge-generated code.
            output = {"ok": True, "result": {"has_pysbd": _SEGMENTER is not None}}
            sys.stdout.write(json.dumps(output, ensure_ascii=False))
            return

        namespace = _exec_namespace()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = str(payload["code"])
            exec(code, namespace)
            result = namespace[_function_name(code)](
                *payload.get("args", []),
                **payload.get("kwargs", {}),
            )
        if mode == "direct":
            result = int(bool(result))
        elif mode == "helper":
            result = str(result)
        else:
            raise ValueError(f"unknown execution mode: {mode!r}")
        output = {"ok": True, "result": result}
    except BaseException as exc:
        output = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    sys.stdout.write(json.dumps(output, ensure_ascii=False))


def _run_code(
    code: str,
    kwargs: Dict[str, Any],
    timeout_seconds: int = 5,
    memory_limit_mb: int = 1024,
    mode: str = "direct",
    args: Optional[List[Any]] = None,
) -> Any:
    payload = json.dumps(
        {
            "code": code,
            "args": args or [],
            "kwargs": kwargs,
            "timeout_seconds": timeout_seconds,
            "memory_limit_mb": memory_limit_mb,
            "mode": mode,
        },
        ensure_ascii=False,
    )
    process = subprocess.Popen(
        [sys.executable, "-I", str(Path(__file__).resolve()), "--exec-worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(input=payload, timeout=max(1, timeout_seconds) + 2)
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()
        raise ExecTimeout("exec timed out") from exc
    if process.returncode != 0:
        raise RuntimeError(f"isolated exec exited with code {process.returncode}: {stderr[-500:]}")
    try:
        output = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"isolated exec returned invalid output: {stdout[-500:]}") from exc
    if not output.get("ok"):
        raise RuntimeError(str(output.get("error", "isolated exec failed")))
    return output.get("result")


def execute_helper_code(
    code: str,
    params_str: str,
    response: str,
    timeout_seconds: int = 5,
    memory_limit_mb: int = 1024,
) -> Tuple[str, bool]:
    try:
        if not code:
            return "", False
        code = code.replace("-∞", "float('-inf')").replace("+∞", "float('inf')").replace("∞", "float('inf')")
        params = json.loads(params_str) if params_str else []
        kwargs: Dict[str, Any] = {}
        if isinstance(params, list):
            for item in params:
                if isinstance(item, dict) and "参数名" in item and "参数值" in item:
                    kwargs[item["参数名"]] = parser_extraction_results(item["参数值"], response)
        elif isinstance(params, dict):
            kwargs = {k: parser_extraction_results(v, response) for k, v in params.items()}
        result = _run_code(
            code,
            kwargs,
            timeout_seconds=timeout_seconds,
            memory_limit_mb=memory_limit_mb,
            mode="helper",
        )
        return str(result), True
    except Exception:
        return "", False


def is_intentional_no_code(text: str) -> bool:
    text = parse_solution(text)
    if text.strip() == "无":
        return True
    match = re.search(r"\[辅助验证代码-开始\](.*?)\[辅助验证代码-结束\]", text, flags=re.DOTALL)
    if not match:
        return False
    content = re.sub(r"```(?:python)?", "", match.group(1)).replace("```", "").strip()
    return content == "无"


def is_gpt_oss_model(model_path: str | Path) -> bool:
    return "gpt-oss" in str(model_path).lower().replace("_", "-")


def parse_response_with_reasoning(text: str) -> Tuple[str, str]:
    """Split a raw generation into (answer, reasoning trace).

    Handles both the ``<think>...</think>`` convention and the gpt-oss
    ``<|channel|>`` harmony format.
    """
    final_marker = "<|channel|>final<|message|>"
    if final_marker in text:
        prefix, response = text.rsplit(final_marker, 1)
        response = response.split("<|return|>", 1)[0].split("<|end|>", 1)[0]
        analysis_marker = "<|channel|>analysis<|message|>"
        reasoning = ""
        if analysis_marker in prefix:
            reasoning = prefix.rsplit(analysis_marker, 1)[1].split("<|end|>", 1)[0]
        return response.strip(), reasoning.strip()
    if "</think>" not in text:
        return text, ""
    reasoning, response = text.rsplit("</think>", 1)
    reasoning = reasoning.strip()
    if reasoning.startswith("<think>"):
        reasoning = reasoning[len("<think>") :].strip()
    return response.strip(), reasoning


if __name__ == "__main__" and "--exec-worker" in sys.argv:
    _sandbox_worker_main()
