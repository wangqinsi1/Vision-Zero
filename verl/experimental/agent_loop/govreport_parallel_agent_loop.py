# Copyright 2026 SpyRL Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SpyRL agent loop for GovReport summarization with a parallel performing stage.

One rollout plays one full game. In the performing stage every player writes a summary
concurrently from its own private view of the report; in the detection stage a detector reads
the full report plus all summaries and votes for the spy.

Which stage the rollout is *trained* on is chosen per step by ``trainer.training_phase`` (see
``RayPPOTrainer._get_interactive_training_phase``): the ``clue`` phase returns one performing
trajectory per player, the ``decision`` phase returns a detector trajectory. Performing-stage
generations are cached per ``(step, sample_idx)`` so that all ``rollout.n`` detector samples judge
the same game.

Set ``SPYRL_ROLLOUT_LOG`` to control where the first game of each step is dumped for inspection.
"""

import asyncio
import logging
import os
import random
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.dataset.govreport_spotdiff_parallel_dataset import GovReportSpotDiffParallelPromptBuilder
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_CLUE_CACHE = {}
_CLUE_CACHE_STEP = None
_OUTPUT_PATH = os.getenv("SPYRL_ROLLOUT_LOG", os.path.join(os.getcwd(), "outputs", "spyrl_rollout_log.txt"))
_OUTPUT_INITED = False
_FIRST_EXAMPLE_LOGGED = False


def _is_rank0() -> bool:
    return os.getenv("RANK", "0") == "0" and os.getenv("LOCAL_RANK", "0") == "0" and os.getenv("NODE_RANK", "0") == "0"


def _ensure_output_file() -> None:
    global _OUTPUT_INITED
    if not _is_rank0() or _OUTPUT_INITED:
        return
    os.makedirs(os.path.dirname(_OUTPUT_PATH), exist_ok=True)
    if not os.path.exists(_OUTPUT_PATH):
        with open(_OUTPUT_PATH, "w", encoding="utf-8") as handle:
            handle.write("OUTPUT LOG START\n")
    _OUTPUT_INITED = True


def _append_output(text: str) -> None:
    if not _is_rank0():
        return
    _ensure_output_file()
    with open(_OUTPUT_PATH, "a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def _log_first_example_to_output(
    *,
    step: int,
    sample_idx: Any,
    training_phase: str,
    game_id: str,
    num_players: int,
    num_rounds: int,
    clue_prompts: dict[tuple[int, int], str],
    clue_responses: dict[tuple[int, int], str],
    decision_prompt: str,
    decision_response: str,
    decision_responses: list[str],
) -> None:
    global _FIRST_EXAMPLE_LOGGED
    if _FIRST_EXAMPLE_LOGGED or not _is_rank0():
        return

    _append_output("=" * 80)
    _append_output(f"FIRST EXAMPLE | STEP {step} SAMPLE {sample_idx} PHASE {training_phase} GAME {game_id}")
    _append_output("CLUE/SUMMARY STAGE:")
    for round_num in range(1, num_rounds + 1):
        for player_id in range(1, num_players + 1):
            prompt = clue_prompts.get((round_num, player_id), "")
            resp = clue_responses.get((round_num, player_id), "")
            _append_output(f"[CLUE PROMPT] Round {round_num} Player {player_id}\n{prompt}")
            _append_output(f"[CLUE RESPONSE] Round {round_num} Player {player_id}\n{resp}")
    _append_output("[DECISION PROMPT]\n" + (decision_prompt or ""))
    if decision_responses:
        for idx, resp in enumerate(decision_responses, start=1):
            _append_output(f"[DECISION RESPONSE {idx}]\n{resp}")
    else:
        _append_output("[DECISION RESPONSE]\n" + (decision_response or ""))

    _FIRST_EXAMPLE_LOGGED = True


@register("govreport_parallel")
class GovReportParallelClueDecisionAgentLoop(AgentLoopBase):
    """Parallel clue -> decision agent loop for long-document summarization."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self._warned_truncate = False

    def _truncate_prompt_ids(self, prompt_ids: list[int]) -> list[int]:
        if hasattr(prompt_ids, "tolist"):
            prompt_ids = prompt_ids.tolist()
        if len(prompt_ids) <= self.prompt_length:
            return prompt_ids
        if not self._warned_truncate and _is_rank0():
            logger.warning(
                "Prompt exceeds max length (%s > %s); truncating from the left.",
                len(prompt_ids),
                self.prompt_length,
            )
            self._warned_truncate = True
        return prompt_ids[-self.prompt_length :]

    async def _generate_text(
        self, prompt_text: str, sampling_params: dict[str, Any]
    ) -> tuple[str, list[int], list[int]]:
        messages = [{"role": "user", "content": prompt_text}]
        prompt_ids = await self.apply_chat_template(messages)
        prompt_ids = self._truncate_prompt_ids(prompt_ids)
        sampling_params = dict(sampling_params)
        output = await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=None,
            video_data=None,
        )
        response_ids = output.token_ids[: self.response_length]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return response_text, prompt_ids, response_ids

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        extra_info = kwargs.get("extra_info", {}) or {}
        trajectory = kwargs.get("_trajectory", {}) or {}
        rollout_n = int(trajectory.get("rollout_n", 0))
        step = int(trajectory.get("step", 0))
        game_data = dict(extra_info.get("game_data", {}) or {})
        num_players = int(game_data.get("num_players", 2))
        num_rounds = int(game_data.get("num_rounds", 1))
        sample_idx = extra_info.get("sample_idx")

        global _CLUE_CACHE_STEP
        if _CLUE_CACHE_STEP != step:
            _CLUE_CACHE_STEP = step
            _CLUE_CACHE.clear()

        training_phase = str(kwargs.get("training_phase", "decision"))
        clue_prompts: dict[tuple[int, int], str] = {}
        clue_prompt_ids_map: dict[tuple[int, int], list[int]] = {}
        clue_texts: dict[tuple[int, int], str] = {}
        clue_responses: dict[tuple[int, int], str] = {}
        clue_response_ids_map: dict[tuple[int, int], list[int]] = {}
        clue_sequence: list[dict[str, Any]] = []
        metrics = {}
        cache_key = (step, sample_idx)
        decision_n = int(self.config.actor_rollout_ref.rollout.n)

        with simple_timer("generate_sequences", metrics):
            if rollout_n > 0 and cache_key in _CLUE_CACHE:
                cached = _CLUE_CACHE[cache_key]
                clue_prompts = cached["clue_prompts"]
                clue_prompt_ids_map = cached["clue_prompt_ids_map"]
                clue_texts = cached["clue_texts"]
                clue_responses = cached["clue_responses"]
                clue_response_ids_map = cached["clue_response_ids_map"]
                all_summaries = cached["all_summaries"]
                clue_sequence = cached.get("clue_sequence", [])
                decision_prompt = cached.get("decision_prompt", "")
                decision_responses = cached.get("decision_responses", [])
            else:
                govreport_data = game_data.get("govreport_data", {}) or {}
                full_report = str(govreport_data.get("report_text", ""))
                masked_report = str(govreport_data.get("masked_report_text", ""))
                mask_fraction = float(govreport_data.get("mask_fraction", 0.2))
                seed = int(self.dataset_config.get("seed", 0) or 0) + int(sample_idx or 0)
                rng = random.Random(seed)
                spy_player = rng.randint(1, num_players)
                civ_player_ids = [i for i in range(1, num_players + 1) if i != spy_player]
                rng.shuffle(civ_player_ids)

                player_document_content = [full_report] * num_players
                player_document_content[spy_player - 1] = masked_report
                game_data["spy_player"] = spy_player
                game_data["player_document_content"] = player_document_content

                for round_num in range(1, num_rounds + 1):
                    civ_prompt = GovReportSpotDiffParallelPromptBuilder.build_clue_prompt_parallel(
                        num_players=num_players,
                        round_num=round_num,
                        report_text=full_report,
                        is_spy=False,
                        mask_fraction=mask_fraction,
                    )
                    spy_prompt = GovReportSpotDiffParallelPromptBuilder.build_clue_prompt_parallel(
                        num_players=num_players,
                        round_num=round_num,
                        report_text=masked_report,
                        is_spy=True,
                        mask_fraction=mask_fraction,
                    )

                    civ_outputs = []
                    civ_tasks = [self._generate_text(civ_prompt, sampling_params) for _ in range(len(civ_player_ids))]
                    civ_results = await asyncio.gather(*civ_tasks)
                    for response, prompt_ids, response_ids in civ_results:
                        summary_text = GovReportSpotDiffParallelPromptBuilder.extract_summary(response)
                        civ_outputs.append(
                            {
                                "prompt": civ_prompt,
                                "prompt_ids": prompt_ids,
                                "response": response,
                                "response_ids": response_ids,
                                "summary_text": summary_text,
                            }
                        )

                    spy_response, spy_prompt_ids, spy_response_ids = await self._generate_text(
                        spy_prompt, sampling_params
                    )
                    spy_summary_text = GovReportSpotDiffParallelPromptBuilder.extract_summary(spy_response)

                    for idx, player_id in enumerate(civ_player_ids):
                        entry = civ_outputs[idx]
                        clue_prompts[(round_num, player_id)] = entry["prompt"]
                        clue_prompt_ids_map[(round_num, player_id)] = entry["prompt_ids"]
                        clue_responses[(round_num, player_id)] = entry["response"]
                        clue_response_ids_map[(round_num, player_id)] = entry["response_ids"]
                        clue_texts[(round_num, player_id)] = entry["summary_text"]
                        clue_sequence.append(
                            {"round": round_num, "player_id": player_id, "summary_text": entry["summary_text"]}
                        )

                    clue_prompts[(round_num, spy_player)] = spy_prompt
                    clue_prompt_ids_map[(round_num, spy_player)] = spy_prompt_ids
                    clue_responses[(round_num, spy_player)] = spy_response
                    clue_response_ids_map[(round_num, spy_player)] = spy_response_ids
                    clue_texts[(round_num, spy_player)] = spy_summary_text
                    clue_sequence.append(
                        {"round": round_num, "player_id": spy_player, "summary_text": spy_summary_text}
                    )

                all_summaries = GovReportSpotDiffParallelPromptBuilder.build_all_summaries_text_from_sequence(
                    clue_sequence, num_players, num_rounds
                )
                decision_prompt = GovReportSpotDiffParallelPromptBuilder.build_god_decision_prompt(
                    num_players=num_players,
                    report_text=full_report,
                    all_summaries=all_summaries,
                    mask_fraction=mask_fraction,
                )
                decision_responses = []
                if training_phase == "clue" and rollout_n == 0:
                    for _ in range(decision_n):
                        decision_response, _, _ = await self._generate_text(decision_prompt, sampling_params)
                        decision_responses.append(decision_response)

                _CLUE_CACHE[cache_key] = {
                    "clue_prompts": clue_prompts,
                    "clue_prompt_ids_map": clue_prompt_ids_map,
                    "clue_texts": clue_texts,
                    "clue_responses": clue_responses,
                    "clue_response_ids_map": clue_response_ids_map,
                    "all_summaries": all_summaries,
                    "clue_sequence": clue_sequence,
                    "decision_prompt": decision_prompt,
                    "decision_responses": decision_responses,
                }

            if training_phase == "decision":
                govreport_data = game_data.get("govreport_data", {}) or {}
                full_report = str(govreport_data.get("report_text", ""))
                mask_fraction = float(govreport_data.get("mask_fraction", 0.2))
                decision_prompt = GovReportSpotDiffParallelPromptBuilder.build_god_decision_prompt(
                    num_players=num_players,
                    report_text=full_report,
                    all_summaries=all_summaries,
                    mask_fraction=mask_fraction,
                )
                (
                    decision_response,
                    decision_prompt_ids,
                    decision_response_ids,
                ) = await self._generate_text(decision_prompt, sampling_params)
                if rollout_n == 0 and not _FIRST_EXAMPLE_LOGGED and decision_n > 1:
                    decision_responses = [decision_response]
                    for _ in range(decision_n - 1):
                        extra_response, _, _ = await self._generate_text(decision_prompt, sampling_params)
                        decision_responses.append(extra_response)
            else:
                decision_response = ""
                decision_prompt_ids = []
                decision_response_ids = []

            if rollout_n == 0:
                _log_first_example_to_output(
                    step=step,
                    sample_idx=sample_idx,
                    training_phase=training_phase,
                    game_id=str(game_data.get("game_id", "")),
                    num_players=num_players,
                    num_rounds=num_rounds,
                    clue_prompts=clue_prompts,
                    clue_responses=clue_responses,
                    decision_prompt=decision_prompt,
                    decision_response=decision_response,
                    decision_responses=decision_responses,
                )

        clue_rollout_total = max(1, num_players * num_rounds)
        clue_index = rollout_n % clue_rollout_total
        clue_round_num = clue_index // num_players + 1
        clue_player_id = clue_index % num_players + 1

        if training_phase == "clue":
            prompt_ids = clue_prompt_ids_map.get((clue_round_num, clue_player_id), [])
            response_ids = clue_response_ids_map.get((clue_round_num, clue_player_id), [])
            response_text = clue_responses.get((clue_round_num, clue_player_id), "")
            prompt_text = clue_prompts.get((clue_round_num, clue_player_id), "")
        else:
            prompt_ids = decision_prompt_ids
            response_ids = decision_response_ids
            response_text = decision_response
            prompt_text = decision_prompt

        response_mask = [1] * len(response_ids)

        updated_extra_info = dict(extra_info)
        updated_extra_info["prompt_text"] = prompt_text
        updated_extra_info["clue_texts"] = clue_texts
        updated_extra_info["clue_prompts"] = clue_prompts
        updated_extra_info["clue_responses"] = clue_responses
        updated_extra_info["all_clues"] = all_summaries
        updated_extra_info["training_phase"] = training_phase
        updated_extra_info["clue_player_id"] = clue_player_id
        updated_extra_info["clue_round_num"] = clue_round_num
        updated_extra_info["decision_n"] = decision_n
        updated_extra_info["decision_prompt"] = decision_prompt
        updated_extra_info["step"] = step
        if training_phase == "decision":
            updated_extra_info["decision_sample_index"] = rollout_n + 1
        updated_extra_info["decision_responses"] = (
            _CLUE_CACHE[cache_key].get("decision_responses", []) if training_phase == "clue" else [response_text]
        )

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=None,
            routed_experts=None,
            multi_modal_data={},
            num_turns=2,
            metrics=metrics,
            extra_fields={"agent_loop_extra_info": updated_extra_info},
        )
