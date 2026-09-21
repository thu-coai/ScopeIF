"""Child-process entry point for generated validation code execution."""

import contextlib
import json
import os
import sys

from utils import _execute_function_impl


def main():
    try:
        request = json.load(sys.stdin)
        code = request["code"]
        params_str = request.get("params_str", "[]")
        response = request.get("response", "")

        # Generated validation code may print arbitrary text. Suppress it so
        # stdout remains a clean JSON-only protocol for the parent process.
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                result, flag = _execute_function_impl(code, params_str, response)

        json.dump({"result": result, "flag": bool(flag)}, sys.stdout, ensure_ascii=False)
        return 0
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
