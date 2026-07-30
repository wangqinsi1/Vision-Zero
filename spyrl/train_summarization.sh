#!/usr/bin/env bash
# SpyRL on text summarization (GovReport), parallel performing stage.
#
# Civilians see the full report, the spy sees the same report with a random contiguous
# 20% span masked out. All players summarize independently, then vote on who is the spy.
set -xeo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
spyrl_setup_experiment "${EXPERIMENT_NAME:-govreport_summarization}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.custom_cls.path="${SPYRL_DATASET_DIR}/govreport_spotdiff_parallel_dataset.py" \
    data.custom_cls.name=GovReportSpotDiffParallelDataset \
    data.train_files=unused \
    data.val_files=unused \
    data.train_batch_size=128 \
    data.train_max_samples=100000000 \
    data.max_prompt_length=25000 \
    data.max_response_length=3762 \
    data.filter_overlong_prompts=True \
    +data.num_players="${NUM_PLAYERS:-4}" \
    +data.num_rounds="${NUM_ROUNDS:-1}" \
    +data.report_max_tokens=16000 \
    +data.mask_fraction="${MASK_FRACTION:-0.2}" \
    actor_rollout_ref.rollout.agent.default_agent_loop=govreport_parallel \
    custom_reward_function.path="${SPYRL_REWARD}" \
    custom_reward_function.name=compute_score \
    +custom_reward_function.reward_kwargs.max_debug_prints=4 \
    actor_rollout_ref.model.path="${SPYRL_MODEL}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${SPYRL_NUM_GPUS}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.enforce_eager=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=NONE \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.use_inductor=False \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.max_num_batched_tokens=30000 \
    actor_rollout_ref.rollout.max_model_len=30000 \
    actor_rollout_ref.rollout.response_length=4096 \
    actor_rollout_ref.rollout.prompt_length=25000 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.ref.fsdp_config.use_torch_compile=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    +trainer.training_phase="${TRAINING_PHASE:-interactive}" \
    +trainer.interactive_cycle_length="${INTERACTIVE_CYCLE_LENGTH:-1}" \
    trainer.logger="${SPYRL_LOGGER}" \
    trainer.project_name='spyrl_govreport' \
    trainer.experiment_name="${SPYRL_EXPERIMENT}" \
    trainer.n_gpus_per_node="${SPYRL_NUM_GPUS}" \
    trainer.nnodes="${SPYRL_NNODES}" \
    trainer.save_freq="${SPYRL_SAVE_FREQ}" \
    trainer.default_local_dir="${SPYRL_CKPT_DIR}" \
    trainer.test_freq=-1 \
    trainer.total_epochs="${SPYRL_TOTAL_EPOCHS}" "$@"
