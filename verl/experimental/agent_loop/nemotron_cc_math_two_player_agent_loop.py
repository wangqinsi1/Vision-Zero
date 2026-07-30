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
"""SpyRL agent loop for Nemotron-CC-Math reasoning with a sequential performing stage.

One rollout plays one full game. In the performing stage each player designs and solves a math
problem grounded in the document it received, speaking in turn and seeing the problems already
posed; the spy, which receives no document, speaks last. In the detection stage a detector
reads the reference document plus all problems and votes for the spy.

Which stage the rollout is *trained* on is chosen per step by ``trainer.training_phase``; the
performing-stage generations are cached per ``(step, sample_idx)`` and shared across the group.

Set ``VERL_DEBUG_AGENT_LOOP=1`` to print the first game of each step to stdout.
"""

import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.dataset.nemotron_cc_math_spotdiff_dataset import (
    NemotronCCMathSpotDiffPromptBuilder,
)
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


_CLUE_CACHE = {}
_CLUE_CACHE_STEP = None
_PRINT_COUNT = 0
_PRINTED_STEP = None


def _is_rank0() -> bool:
    return os.getenv("RANK", "0") == "0" and os.getenv("LOCAL_RANK", "0") == "0" and os.getenv("NODE_RANK", "0") == "0"


def _debug_enabled() -> bool:
    return os.getenv("VERL_DEBUG_AGENT_LOOP", "0") == "1"


@register("nemotron_cc_math_two_player")
class NemotronCCMathTwoPlayerClueDecisionAgentLoop(AgentLoopBase):
    """Two-player clue -> decision agent loop for Nemotron-CC-Math spot-the-difference."""

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
        global _PRINT_COUNT, _PRINTED_STEP
        extra_info = kwargs.get("extra_info", {}) or {}
        trajectory = kwargs.get("_trajectory", {}) or {}
        rollout_n = int(trajectory.get("rollout_n", 0))
        step = int(trajectory.get("step", 0))
        game_data = extra_info.get("game_data", {})
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
        max_debug_prints = int(os.getenv("VERL_MAX_DEBUG_PRINTS", "2"))
        should_print = _is_rank0() and rollout_n == 0 and _PRINT_COUNT < max_debug_prints
        decision_n = int(self.config.actor_rollout_ref.rollout.n)
        debug_log = _debug_enabled() and _is_rank0() and rollout_n == 0

        if debug_log:
            logger.warning(
                "[AGENT_LOOP] step=%s sample_idx=%s rollout_n=%s phase=%s num_players=%s num_rounds=%s",
                step,
                sample_idx,
                rollout_n,
                training_phase,
                num_players,
                num_rounds,
            )

        with simple_timer("generate_sequences", metrics):
            if rollout_n > 0 and cache_key in _CLUE_CACHE:
                cached = _CLUE_CACHE[cache_key]
                if debug_log:
                    logger.warning(
                        "[AGENT_LOOP] cache hit step=%s sample_idx=%s phase=%s",
                        step,
                        sample_idx,
                        training_phase,
                    )
                clue_prompts = cached["clue_prompts"]
                clue_prompt_ids_map = cached["clue_prompt_ids_map"]
                clue_texts = cached["clue_texts"]
                clue_responses = cached["clue_responses"]
                clue_response_ids_map = cached["clue_response_ids_map"]
                all_clues = cached["all_clues"]
                clue_sequence = cached.get("clue_sequence", [])
                decision_prompt = cached.get("decision_prompt", "")
                decision_responses = cached.get("decision_responses", [])
            else:
                spy_player = int(game_data.get("spy_player", 1))
                speaking_order = [i for i in range(1, num_players + 1) if i != spy_player]
                speaking_order.append(spy_player)
                for round_num in range(1, num_rounds + 1):
                    for player_id in speaking_order:
                        if debug_log:
                            logger.warning(
                                "[AGENT_LOOP] clue gen start step=%s sample_idx=%s round=%s player=%s",
                                step,
                                sample_idx,
                                round_num,
                                player_id,
                            )
                        previous_clues_text = (
                            NemotronCCMathSpotDiffPromptBuilder.build_previous_clues_text_from_sequence(clue_sequence)
                        )
                        clue_prompt = NemotronCCMathSpotDiffPromptBuilder.build_clue_prompt(
                            game_data=game_data,
                            player_id=player_id,
                            round_num=round_num,
                            previous_clues_text=previous_clues_text,
                        )
                        clue_prompts[(round_num, player_id)] = clue_prompt
                        clue_response, clue_prompt_ids, clue_response_ids = await self._generate_text(
                            clue_prompt, sampling_params
                        )
                        if debug_log:
                            logger.warning(
                                "[AGENT_LOOP] clue gen done step=%s sample_idx=%s round=%s player=%s "
                                "prompt_len=%s response_len=%s",
                                step,
                                sample_idx,
                                round_num,
                                player_id,
                                len(clue_prompt_ids),
                                len(clue_response_ids),
                            )
                        clue_prompt_ids_map[(round_num, player_id)] = clue_prompt_ids
                        clue_responses[(round_num, player_id)] = clue_response
                        clue_response_ids_map[(round_num, player_id)] = clue_response_ids
                        clue_text = NemotronCCMathSpotDiffPromptBuilder.extract_clue_answer(clue_response)
                        clue_texts[(round_num, player_id)] = clue_text
                        clue_sequence.append(
                            {
                                "round": round_num,
                                "player_id": player_id,
                                "clue_text": clue_text,
                            }
                        )

                all_clues = NemotronCCMathSpotDiffPromptBuilder.build_all_clues_text_from_sequence(
                    clue_sequence, num_players, num_rounds
                )
                decision_prompt = NemotronCCMathSpotDiffPromptBuilder.build_god_decision_prompt(
                    num_players=num_players,
                    original_document=game_data.get("nemotron_cc_math_data", {}).get("civilian_document_text", ""),
                    all_clues=all_clues,
                )
                # For clue training, generate decision samples once per game for reward calculation.
                decision_responses = []
                if training_phase == "clue" and rollout_n == 0:
                    for _ in range(decision_n):
                        if debug_log:
                            logger.warning(
                                "[AGENT_LOOP] decision gen start step=%s sample_idx=%s",
                                step,
                                sample_idx,
                            )
                        decision_response, _, _ = await self._generate_text(decision_prompt, sampling_params)
                        decision_responses.append(decision_response)
                        if debug_log:
                            logger.warning(
                                "[AGENT_LOOP] decision gen done step=%s sample_idx=%s",
                                step,
                                sample_idx,
                            )
                _CLUE_CACHE[cache_key] = {
                    "clue_prompts": clue_prompts,
                    "clue_prompt_ids_map": clue_prompt_ids_map,
                    "clue_texts": clue_texts,
                    "clue_responses": clue_responses,
                    "clue_response_ids_map": clue_response_ids_map,
                    "all_clues": all_clues,
                    "clue_sequence": clue_sequence,
                    "decision_prompt": decision_prompt,
                    "decision_responses": decision_responses,
                }
            if training_phase == "decision":
                decision_prompt = NemotronCCMathSpotDiffPromptBuilder.build_god_decision_prompt(
                    num_players=num_players,
                    original_document=game_data.get("nemotron_cc_math_data", {}).get("civilian_document_text", ""),
                    all_clues=all_clues,
                )
                (
                    decision_response,
                    decision_prompt_ids,
                    decision_response_ids,
                ) = await self._generate_text(decision_prompt, sampling_params)
            else:
                decision_response = ""
                decision_prompt_ids = []
                decision_response_ids = []
            if debug_log:
                clue_expected = num_players * num_rounds
                decision_expected = decision_n
                clue_count = len(clue_sequence)
                if training_phase == "decision":
                    decision_count = 1
                else:
                    decision_count = len(_CLUE_CACHE[cache_key].get("decision_responses", []))
                logger.warning(
                    "[AGENT_LOOP] sample counts step=%s sample_idx=%s phase=%s clues=%s/%s decisions=%s/%s",
                    step,
                    sample_idx,
                    training_phase,
                    clue_count,
                    clue_expected,
                    decision_count,
                    decision_expected,
                )
            if should_print and _PRINTED_STEP != step:
                _PRINTED_STEP = step
                _PRINT_COUNT += 1
                for round_num in range(1, num_rounds + 1):
                    for player_id in range(1, num_players + 1):
                        print(f"[CLUE PROMPT] Round {round_num} Player {player_id}")
                        print(clue_prompts.get((round_num, player_id), ""))
                        print(f"[CLUE RESPONSE] Round {round_num} Player {player_id}")
                        print(clue_responses.get((round_num, player_id), ""))
                print("[DECISION PROMPT]")
                print(
                    decision_prompt
                    if training_phase == "decision"
                    else _CLUE_CACHE[cache_key].get("decision_prompt", "")
                )
                print("[DECISION RESPONSE]")
                if training_phase == "decision":
                    print(decision_response)
                else:
                    print(_CLUE_CACHE[cache_key].get("decision_responses", []))

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
        updated_extra_info["all_clues"] = all_clues
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

        output = AgentLoopOutput(
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
        return output
