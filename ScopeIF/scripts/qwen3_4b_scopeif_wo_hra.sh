#!/usr/bin/env bash
set -x

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}
DATA_DIR=${DATA_DIR:-$REPO_DIR/../ScopeInstruct}

cd "$REPO_DIR"
export PYTHONPATH=$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_V1=1

train_path=$DATA_DIR/train_scopeif_wo_hra.parquet
test_path=${VAL_PARQUET:-$train_path}

train_files="['$train_path']"
test_files="['$test_path']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.checkpoint.save_contents=['model'] \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward.num_workers=32 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.project_name='scopeif' \
    trainer.experiment_name='qwen3_4b_scopeif_wo_hra' \
    trainer.save_freq=30 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 2>&1 | tee qwen3_4b_scopeif_wo_hra.log
