#!/usr/bin/env python3
"""Run one reward end-to-end against a live judge, without initializing torch.cuda.

    python3 smoke_test_reward.py --data rollouts.json --reward-module scopeif
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from multiprocessing import Process, Queue

import numpy as np


def _load_module_from_file(name, path):
    """Load a Python module directly from file path, bypassing package __init__."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


BASE = str(Path(__file__).resolve().parent / "verl/utils/reward_score")


def worker_process(
    worker_id,
    test_data,
    result_queue,
    base_url,
    model_name,
    api_key,
    max_retries,
    use_reward_utils_timeout,
    reward_module,
):
    """Run compute_score in a separate process, like a Ray RewardLoopWorker."""
    # Load modules directly from file, skipping verl.__init__ → torch.cuda
    utils = _load_module_from_file("reward_utils", f"{BASE}/utils.py")

    # Load prompt templates
    code_prompts = _load_module_from_file(
        "code_eval_prompts", f"{BASE}/prompts/code_evaluation_prompts.py")
    llm_prompts = _load_module_from_file(
        "llm_eval_prompts", f"{BASE}/prompts/llm_evaluation_prompts.py")

    # Prepare fused module's dependencies in sys.modules so relative imports resolve
    # Create a fake package structure
    import types
    pkg = types.ModuleType("_reward_score")
    pkg.utils = utils
    sys.modules["_reward_score"] = pkg
    sys.modules["_reward_score.utils"] = utils

    prompts_pkg = types.ModuleType("_reward_score.prompts")
    sys.modules["_reward_score.prompts"] = prompts_pkg
    sys.modules["_reward_score.prompts.code_evaluation_prompts"] = code_prompts
    sys.modules["_reward_score.prompts.llm_evaluation_prompts"] = llm_prompts

    # Now load the reward module with patched relative imports
    fused_path = f"{BASE}/{reward_module}.py"
    with open(fused_path) as f:
        source = f.read()

    # Rewrite relative imports to our fake package
    source = source.replace("from .prompts.", "from _reward_score.prompts.")
    source = source.replace("from .utils import", "from _reward_score.utils import")
    # Remove wandb import (not needed for smoke test)
    source = source.replace("import wandb", "wandb = None")

    fused = types.ModuleType("fused")
    exec(compile(source, fused_path, "exec"), fused.__dict__)

    # Monkey-patch VLLMClient to inject CLI params
    OriginalVLLMClient = utils.VLLMClient

    class PatchedVLLMClient(OriginalVLLMClient):
        def __init__(self, **kwargs):
            kwargs["base_url"] = base_url
            kwargs["model_name"] = model_name
            kwargs["api_key"] = api_key
            kwargs["max_retries"] = max_retries
            if use_reward_utils_timeout:
                super().__init__(**kwargs)
            else:
                self.client = utils.OpenAI(
                    api_key=kwargs["api_key"],
                    base_url=kwargs["base_url"],
                )
                self.model_name = kwargs["model_name"]
                self.max_retries = kwargs["max_retries"]

    fused.build_judge_client = lambda max_retries=1: PatchedVLLMClient()

    t0 = time.time()
    try:
        result = fused.compute_score(test_data["solution_str"], test_data["ground_truth"])
        elapsed = time.time() - t0
        result_queue.put({
            "worker_id": worker_id,
            "status": "ok",
            "elapsed": elapsed,
            **result,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        elapsed = time.time() - t0
        result_queue.put({
            "worker_id": worker_id,
            "status": "error",
            "elapsed": elapsed,
            "error": f"{type(e).__name__}: {e}",
        })


def main():
    parser = argparse.ArgumentParser(description="Smoke test for reward pipeline")
    parser.add_argument("--data", required=True, help="JSON file of {solution_str, ground_truth} records")
    parser.add_argument("--reward-module", default="scopeif",
                        choices=["scopeif", "scopeif_wo_hra", "rl_ila", "rl_cla"])
    parser.add_argument("--base-url", default="http://localhost:8080/v1")
    parser.add_argument("--model-name", default="GPT-OSS-120B-Default")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--max-threads", type=int, default=None,
                        help="Override IF_BINARY_PARALLEL_MAX_WORKERS")
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--connect-timeout", type=float, default=None,
                        help="Set OPENAI_CONNECT_TIMEOUT for this smoke test (seconds)")
    parser.add_argument("--use-reward-utils-timeout-defaults", action="store_true",
                        help="Set OPENAI_* timeout env vars to the defaults from verl/utils/reward_score/utils.py")
    parser.add_argument("--rounds", type=int, default=60)
    args = parser.parse_args()

    if args.max_threads is not None:
        os.environ["IF_BINARY_PARALLEL_MAX_WORKERS"] = str(args.max_threads)

    reward_utils_timeout_defaults = {
        "OPENAI_TOTAL_TIMEOUT": "1800",
        "OPENAI_CONNECT_TIMEOUT": "300",
        "OPENAI_READ_TIMEOUT": "1800",
        "OPENAI_WRITE_TIMEOUT": "300",
        "OPENAI_POOL_TIMEOUT": "300",
    }
    use_reward_utils_timeout = args.use_reward_utils_timeout_defaults or args.connect_timeout is not None
    if args.use_reward_utils_timeout_defaults:
        for key, value in reward_utils_timeout_defaults.items():
            os.environ.setdefault(key, value)
    if args.connect_timeout is not None:
        os.environ["OPENAI_CONNECT_TIMEOUT"] = str(args.connect_timeout)

    with open(args.data) as f:
        test_data = json.load(f)

    gt = json.loads(test_data["ground_truth"]) if isinstance(test_data["ground_truth"], str) else test_data["ground_truth"]
    max_threads_val = args.max_threads or int(os.getenv("IF_BINARY_PARALLEL_MAX_WORKERS", "16"))
    openai_timeout_keys = [
        "OPENAI_TOTAL_TIMEOUT",
        "OPENAI_CONNECT_TIMEOUT",
        "OPENAI_READ_TIMEOUT",
        "OPENAI_WRITE_TIMEOUT",
        "OPENAI_POOL_TIMEOUT",
    ]
    openai_timeout_env = {key: os.getenv(key) for key in openai_timeout_keys if os.getenv(key) is not None}

    print("=== Smoke Test Config ===")
    print(f"  API:              {args.base_url}")
    print(f"  Model:            {args.model_name}")
    print(f"  Num workers:      {args.num_workers}  (separate processes)")
    print(f"  Threads/worker:   {max_threads_val}")
    print(f"  Max retries:      {args.max_retries}")
    if use_reward_utils_timeout:
        if openai_timeout_env:
            missing_timeout_keys = [key for key in openai_timeout_keys if key not in openai_timeout_env]
            suffix = ""
            if missing_timeout_keys:
                suffix = "; remaining values use verl/utils/reward_score/utils.py defaults"
            print("  OPENAI timeouts:  reward-utils mode, " + ", ".join(f"{k}={v}s" for k, v in openai_timeout_env.items()) + suffix)
        else:
            print("  OPENAI timeouts:  reward-utils mode, using defaults from verl/utils/reward_score/utils.py")
    else:
        print("  OPENAI timeouts:  OpenAI SDK defaults (smoke test does not pass timeout=...)")
    print(f"  Rounds:           {args.rounds}")
    print(f"  Checklist items:  {len(gt.get('checklist', []))}")
    print(f"  Peak concurrency: {args.num_workers} x {max_threads_val} = {args.num_workers * max_threads_val}")
    print(f"  Response length:  {len(test_data['solution_str'])} chars")
    print()

    for rnd in range(1, args.rounds + 1):
        print(f"--- Round {rnd}/{args.rounds} ---")
        t_start = time.time()

        result_queue = Queue()
        procs = []
        for w in range(args.num_workers):
            p = Process(target=worker_process,
                        args=(w, test_data, result_queue, args.base_url,
                              args.model_name, args.api_key, args.max_retries,
                              use_reward_utils_timeout, args.reward_module))
            procs.append(p)
            p.start()

        results = []
        for _ in range(args.num_workers):
            r = result_queue.get()
            results.append(r)
            icon = "OK" if r["status"] == "ok" else "FAIL"
            detail = f"score={r['score']:.4f}" if r["status"] == "ok" else r.get("error", "")
            print(f"  Worker {r['worker_id']:>2d}: [{icon}] {r['elapsed']:.1f}s  {detail}")

        for p in procs:
            p.join()

        t_total = time.time() - t_start
        ok = sum(1 for r in results if r["status"] == "ok")
        fail = len(results) - ok
        times = [r["elapsed"] for r in results]
        scores = [r["score"] for r in results if r["status"] == "ok"]

        print(f"\n  Round {rnd} summary:")
        print(f"    Total time:   {t_total:.1f}s")
        print(f"    Success/Fail: {ok}/{fail}")
        print(f"    Worker time:  min={min(times):.1f}s  max={max(times):.1f}s  avg={np.mean(times):.1f}s")
        if scores:
            print(f"    Scores:       min={min(scores):.4f}  max={max(scores):.4f}  avg={np.mean(scores):.4f}")
        unres = [r.get("critique_unresolved", 0) for r in results if r["status"] == "ok"]
        if unres:
            print(f"    Unresolved:   {sum(1 for u in unres if u > 0)}/{len(unres)}")
        print()


if __name__ == "__main__":
    main()
