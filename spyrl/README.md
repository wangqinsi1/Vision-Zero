# SpyRL launch scripts

Every script here is a thin wrapper around `python -m verl.trainer.main_ppo`. They source
[`env.sh`](env.sh), which resolves the repo root from its own location, so they run correctly from any
working directory:

```bash
bash spyrl/train_summarization.sh
```

See the [root README](../README.md) for the method, results and the code map. This file documents the
scripts themselves.

## Scripts

| Script | Task | Environment | Agent loop | Reward |
|---|---|---|---|---|
| `train_summarization.sh` | GovReport summarization | `GovReportSpotDiffParallelDataset` | `govreport_parallel` | `spyrl_reward.py` |
| `train_creative_writing.sh` | WritingPrompts story writing | `WritingPromptsSpotDiffParallelDataset` | `writingprompts_parallel` | `spyrl_reward.py` |
| `train_math_reasoning.sh` | Nemotron-CC-Math reasoning | `NemotronCCMathSpotDiffParallelDataset` | `nemotron_cc_math_parallel` | `spyrl_reward.py` |
| `ablations/train_summarization_sequential.sh` | Summarization, players speak in turn | `GovReportSpotDiffTwoPlayerDataset` | `govreport_two_player` | `spyrl_reward.py` |
| `ablations/train_creative_writing_sequential.sh` | Writing, players speak in turn | `WritingPromptsSpotDiffTwoPlayerDataset` | `writingprompts_two_player` | `spyrl_reward.py` |
| `ablations/train_math_sequential.sh` | Math, players speak in turn, spy sees nothing | `NemotronCCMathSpotDiffTwoPlayerDataset` | `nemotron_cc_math_two_player` | `spyrl_reward.py` |
| `ablations/train_math_without_spy.sh` | "Without spy" (Table 4) | `NemotronCCMathNoSpyClueDataset` | `nemotron_cc_math_no_spy_clue` | `spyrl_no_spy_reward.py` |
| `ablations/train_math_dclm_corpus.sh` | Math game on a generic web corpus | `DCLMBaselineSpotDiffTwoPlayerDataset` | `nemotron_cc_math_two_player` | `spyrl_reward.py` |

`check_install.py` verifies that the runtime dependencies are present, that all seven agent loops
reached verl's registry, and that the files the scripts point at exist.

## Configuration

`env.sh` reads these from the environment, all with defaults:

| Variable | Default | Meaning |
|---|---|---|
| `SPYRL_MODEL` | `Qwen/Qwen3-4B-Instruct-2507` | Actor / reference model |
| `SPYRL_OUTPUT_DIR` | `<repo>/outputs` | Checkpoints + rollout transcripts |
| `SPYRL_CACHE_DIR` | `<output>/cache` | Inductor / Triton / vLLM JIT caches |
| `SPYRL_NUM_GPUS` | `8` | GPUs per node, and the vLLM tensor-parallel size |
| `SPYRL_NNODES` | `1` | Nodes |
| `SPYRL_TOTAL_EPOCHS` | `1` | `trainer.total_epochs` |
| `SPYRL_SAVE_FREQ` | `5` | `trainer.save_freq` |
| `SPYRL_LOGGER` | `["console"]` | `trainer.logger` |
| `TORCH_CUDA_ARCH_LIST` | `90` | `90` = H100/H200, `80` = A100 |

Per-script knobs: `EXPERIMENT_NAME`, `NUM_PLAYERS`, `NUM_ROUNDS`, `MASK_FRACTION`, `TRAINING_PHASE`,
`INTERACTIVE_CYCLE_LENGTH`.

Any extra argument is forwarded verbatim to Hydra:

```bash
NUM_PLAYERS=5 bash spyrl/train_math_reasoning.sh actor_rollout_ref.actor.optim.lr=5e-7
```

## Output layout

```
${SPYRL_OUTPUT_DIR}/${EXPERIMENT_NAME}/
├── global_step_*/     # verl checkpoints
└── rollouts.txt       # one full game per step: prompts, outputs, votes, rewards
```

`rollouts.txt` is written by rank 0 only, and is the fastest way to check that the game is behaving:
it contains each player's performing prompt and output, the detector prompt, every vote, whether the
vote was correct, and the resulting detection and performing rewards. Override its location with
`SPYRL_ROLLOUT_LOG`.

## On a cluster

Point the outputs and caches at node-local storage — the JIT caches in particular are slow on shared
filesystems:

```bash
SPYRL_OUTPUT_DIR=/mnt/localssd/${USER}/spyrl bash spyrl/train_summarization.sh
```

`env.sh` already suffixes the Inductor and Triton cache directories with `${SLURM_PROCID}` so ranks
do not contend on the same cache.
