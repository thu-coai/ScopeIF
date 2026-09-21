#!/usr/bin/env python3
"""Convert a Hugging Face checkpoint into verl's FSDP1 model-only format.

Run with torchrun; the process count becomes the output world size.
"""

import argparse
import json
import os
import shutil
import warnings
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardedStateDictConfig,
    ShardingStrategy,
    StateDictType,
)
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from verl.utils.fsdp_utils import (
    fsdp_version,
    get_fsdp_state_ctx,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
)


_REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = _REPO_DIR / "checkpoints/hf/qwen3_8b_scopeif/checkpoint_420"
DEFAULT_TARGET = _REPO_DIR / "checkpoints/scopeif/qwen3_8b_scopeif/global_step_420"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-checkpoint", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target-global-step-dir", type=Path, default=DEFAULT_TARGET)
    return parser.parse_args()


def validate_paths(source: Path, target: Path, rank: int) -> None:
    error = None
    if rank == 0:
        if not source.is_dir():
            error = f"HF checkpoint does not exist: {source}"
        elif not (source / "config.json").is_file():
            error = f"Missing config.json in {source}"
        elif not ((source / "model.safetensors").is_file() or (source / "model.safetensors.index.json").is_file()):
            error = f"No safetensors model found in {source}"
        elif not target.is_dir():
            error = f"Target global-step directory does not exist: {target}"
        elif (target / "actor").exists():
            error = f"Refusing to overwrite existing actor directory: {target / 'actor'}"
        elif (target / ".actor_conversion_tmp").exists():
            error = f"Temporary conversion directory already exists: {target / '.actor_conversion_tmp'}"

    errors = [error]
    dist.broadcast_object_list(errors, src=0)
    if errors[0] is not None:
        raise RuntimeError(errors[0])


def expected_weight_keys(source: Path) -> set[str]:
    index_path = source / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open(encoding="utf-8") as file:
            return set(json.load(file)["weight_map"])

    from safetensors import safe_open

    with safe_open(source / "model.safetensors", framework="pt", device="cpu") as file:
        return set(file.keys())


def save_huggingface_metadata(model: FSDP, source: Path, destination: Path) -> None:
    unwrapped = model._fsdp_wrapped_module
    destination.mkdir(parents=True)
    unwrapped.config.save_pretrained(destination)
    AutoTokenizer.from_pretrained(source, trust_remote_code=True).save_pretrained(destination)
    try:
        GenerationConfig.from_pretrained(source).save_pretrained(destination)
    except OSError:
        pass


def main() -> None:
    args = parse_args()
    source = args.hf_checkpoint.resolve()
    target = args.target_global_step_dir.resolve()

    if "RANK" not in os.environ or "LOCAL_RANK" not in os.environ:
        raise RuntimeError("Use torchrun to launch this converter.")

    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    validate_paths(source, target, rank)

    staging = target / ".actor_conversion_tmp"
    if rank == 0:
        staging.mkdir()
    dist.barrier()

    device_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))
    config = AutoConfig.from_pretrained(source, trust_remote_code=True, attn_implementation="flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.pad_token_id = tokenizer.pad_token_id
    if hasattr(config, "num_nextn_predict_layers"):
        config.num_nextn_predict_layers = 0

    init_context = get_init_weight_context_manager(
        use_meta_tensor=not config.tie_word_embeddings,
        mesh=device_mesh,
    )
    with init_context(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = AutoModelForCausalLM.from_pretrained(
            source,
            config=config,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        model.to(torch.float32)

    dist.barrier()
    wrap_policy = get_fsdp_wrap_policy(model, {"min_num_params": 0}, is_lora=False)
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    )
    model = FSDP(
        model,
        cpu_offload=None,
        param_init_fn=init_fn,
        auto_wrap_policy=wrap_policy,
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        sync_module_states=True,
        device_mesh=device_mesh,
        use_orig_params=False,
        forward_prefetch=False,
    )

    state_config = ShardedStateDictConfig(offload_to_cpu=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with get_fsdp_state_ctx(model, StateDictType.SHARDED_STATE_DICT, state_config, None):
            state_dict = model.state_dict()

    expected_keys = expected_weight_keys(source)
    actual_keys = set(state_dict)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)[:10]
        unexpected = sorted(actual_keys - expected_keys)[:10]
        raise RuntimeError(f"State-dict key mismatch; missing={missing}, unexpected={unexpected}")

    shard_path = staging / f"model_world_size_{world_size}_rank_{rank}.pt"
    torch.save(state_dict, shard_path)
    del state_dict

    loaded = torch.load(shard_path, map_location="cpu", weights_only=False)
    if set(loaded) != expected_keys:
        raise RuntimeError(f"Saved shard is unreadable or has incorrect keys: {shard_path}")
    del loaded

    if rank == 0:
        save_huggingface_metadata(model, source, staging / "huggingface")
        with (staging / "fsdp_config.json").open("w", encoding="utf-8") as file:
            json.dump({"FSDP_version": fsdp_version(model), "world_size": world_size}, file, indent=4)

    dist.barrier()
    if rank == 0:
        shard_paths = sorted(staging.glob(f"model_world_size_{world_size}_rank_*.pt"))
        if len(shard_paths) != world_size or any(path.stat().st_size == 0 for path in shard_paths):
            raise RuntimeError(f"Expected {world_size} non-empty model shards, found {len(shard_paths)}")
        staging.rename(target / "actor")
        print(f"Converted {source} -> {target / 'actor'}")
        print(f"Preserved existing files in {target}, including data.pt")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
