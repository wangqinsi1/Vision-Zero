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
# ruff: noqa: E501  -- this module is mostly verbatim prompt templates; wrapping them would change the prompts.
""" "Without spy" ablation environment (Table 4) on Nemotron-CC-Math.

Every player receives the same document, so there is no information asymmetry and no
environment-assigned identity to verify against. The detection stage degenerates into electing
the weakest output, and only the performing stage is trained. Use it together with
:mod:`verl.utils.spyrl_no_spy_reward` and the ``nemotron_cc_math_no_spy_clue`` agent loop.

Source: ``nvidia/Nemotron-CC-Math-v1``; set ``NEMOTRON_CC_MATH_CONFIG`` to choose the config
(default ``"3"``).
"""

import os
import random
import re
from typing import Any

import datasets
import torch
from torch.utils.data import Dataset


class NemotronCCMathNoSpyPromptBuilder:
    @staticmethod
    def truncate_to_first_n_whitespace_tokens(text: str, max_tokens: int) -> str:
        if not text:
            return text
        tokens = text.split()
        if len(tokens) <= max_tokens:
            return text
        return " ".join(tokens[:max_tokens])

    @staticmethod
    def build_clue_prompt(
        game_data: dict,
        player_id: int,
        round_num: int,
        previous_clues_text: str,
    ) -> str:
        num_players = game_data["num_players"]
        player_document = game_data["player_document_content"][player_id - 1]

        game_context = f"""You are PLAYER {player_id} of the game "Find the Weak Question". This is ROUND {round_num}. Your task is to write ONE document-grounded statement that looks like "condition + result" in a single sentence.

Role setup:
• There are {num_players} players, and all players see the SAME reference document.
• There is NO spy role in this game.
• Everyone should write one strong, verifiable clue so that weak clues can be identified later.

What you should do:
1. Read the reference document and previous clues.
2. Construct exactly ONE new scenario/condition from document knowledge points.
3. Compute or infer the final result and verify correctness.
4. Your statement must be specific and checkable, not vague, and not copied verbatim from the document.

Fixed template you MUST follow (choose exactly ONE):
• Template A (QA): "If <scenario with variables/conditions>, using <named rule/formula/definition>, then <final result/answer>."
• Template B (MCQ): "If <scenario with variables/conditions>, using <named rule/formula/definition>, what is <quantity/conclusion>? A) ... B) ... C) ... D) ... Answer: <letter>."

Required output format:
First, conduct your own thinking process and then put your one-sentence "Condition+Result" inside \\boxed{{}}.
Inside \\boxed{{...}}, write exactly ONE sentence following Template A or Template B.
Hard limit: The entire output must be <= 2000 tokens.

[Reference Document]
{player_document}
"""

        if previous_clues_text:
            game_context += f"\n\nPrevious clues:\n\n{previous_clues_text}"
        else:
            game_context += (
                "\n\nPrevious clues:\n"
                f"(You are PLAYER {player_id} and are the first to speak, so there is no previous output yet.)"
            )
        return game_context

    @staticmethod
    def build_bad_clue_decision_prompt(
        num_players: int,
        original_document: str,
        all_clues: str,
    ) -> str:
        return f"""You are a judge reviewing a multiplayer clue game. All players saw the SAME reference document, and each player provided one clue in the format "condition + result".

Your task:
Identify the single worst clue among all players. "Worst" means one or more of:
1. Wrong result under its stated condition.
2. Not grounded in the reference document.
3. Too generic, trivial, or hard to verify.
4. Poorly formed as a condition+result clue.

You must output exactly one player number, or \\boxed{{N/A}} if you truly cannot decide.

[Reference Document]
{original_document}

All Clues:
{all_clues}

Required output format:
Put your final answer in \\boxed{{PLAYER_NUMBER}} or \\boxed{{N/A}}."""

    @staticmethod
    def extract_clue_answer(clue_response: str) -> str:
        boxed_match = re.search(r"\\\\?boxed\{(.*?)\}", clue_response, re.DOTALL)
        if boxed_match:
            return boxed_match.group(1).strip()

        answer_match = re.search(r"<answer>(.*?)</answer>", clue_response, re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()

        answer_start_match = re.search(r"<answer>\s*(.*)", clue_response, re.DOTALL)
        if answer_start_match:
            answer_content = answer_start_match.group(1).strip()
            answer_content = re.split(r"</answer>|<think>|<answer>", answer_content)[0].strip()
            return answer_content
        return "No valid clue provided."

    @staticmethod
    def build_previous_clues_text_from_sequence(clue_sequence: list[dict[str, Any]]) -> str:
        if not clue_sequence:
            return ""
        lines = []
        rounds_seen: list[int] = []
        clues_by_round: dict[int, list[dict[str, Any]]] = {}
        for entry in clue_sequence:
            round_num = int(entry.get("round", 1))
            if round_num not in clues_by_round:
                clues_by_round[round_num] = []
                rounds_seen.append(round_num)
            clues_by_round[round_num].append(entry)
        for round_num in rounds_seen:
            lines.append(f"Round {round_num}:")
            for entry in clues_by_round[round_num]:
                player_id = entry.get("player_id")
                clue_text = entry.get("clue_text", "No clue provided")
                lines.append(f"Player {player_id}: {clue_text}")
            lines.append("")
        return "\n".join(lines).strip()

    @staticmethod
    def build_all_clues_text_from_sequence(
        clue_sequence: list[dict[str, Any]], num_players: int, num_rounds: int
    ) -> str:
        lookup: dict[tuple[int, int], str] = {}
        for entry in clue_sequence:
            round_num = int(entry.get("round", 1))
            player_id = int(entry.get("player_id", 1))
            clue_text = entry.get("clue_text", "No clue provided")
            lookup[(round_num, player_id)] = clue_text
        lines = []
        for round_num in range(1, num_rounds + 1):
            lines.append(f"Round {round_num}:")
            for player_id in range(1, num_players + 1):
                clue_text = lookup.get((round_num, player_id), "No clue provided")
                lines.append(f"Player {player_id}: {clue_text}")
            lines.append("")
        return "\n".join(lines).strip()


class NemotronCCMathNoSpyClueDataset(Dataset):
    """No-spy ablation dataset: all players receive the same document."""

    def __init__(
        self,
        data_files: str | list[str] | None,
        tokenizer: Any,
        config: Any,
        processor: Any | None = None,
        max_samples: int = -1,
    ):
        del data_files, tokenizer, processor
        self.config = config
        self.max_samples = max_samples

        self.num_players = int(config.get("num_players", 3))
        self.num_rounds = int(config.get("num_rounds", 1))
        self.seed = int(config.get("seed", 0) or 0)
        self.civilian_doc_max_tokens = int(config.get("civilian_doc_max_tokens", 1000))

        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        config_name = os.getenv("NEMOTRON_CC_MATH_CONFIG", "3")
        self.dataset = datasets.load_dataset(
            "nvidia/Nemotron-CC-Math-v1",
            config_name,
            split="train",
            token=token,
        )

        total = len(self.dataset)
        if self.max_samples and self.max_samples > 0 and self.max_samples < total:
            rng = random.Random(self.seed)
            indices = list(range(total))
            rng.shuffle(indices)
            self.indices = indices[: self.max_samples]
        else:
            self.indices = list(range(total))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        dataset_idx = self.indices[idx]
        sample = self.dataset[dataset_idx]
        document_text_full = sample.get("text", "")

        game_id = f"nemotron_cc_math_no_spy_sample_{dataset_idx}"
        document_text = NemotronCCMathNoSpyPromptBuilder.truncate_to_first_n_whitespace_tokens(
            document_text_full, self.civilian_doc_max_tokens
        )

        player_document_content = [document_text for _ in range(self.num_players)]

        game_data = {
            "game_id": game_id,
            "sample_idx": dataset_idx,
            "num_players": self.num_players,
            "num_rounds": self.num_rounds,
            "spy_player": 1,  # compatibility key used by existing reward interfaces
            "player_document_content": player_document_content,
            "nemotron_cc_math_data": {
                "document_text": document_text_full,
                "civilian_document_text": document_text,
                "civilian_document_max_whitespace_tokens": self.civilian_doc_max_tokens,
            },
        }

        messages = [{"role": "user", "content": ""}]
        return {
            "prompt": messages,
            "raw_prompt": messages,
            "data_source": "nemotron_cc_math_no_spy_clue",
            "reward_model": {"ground_truth": 1},
            "extra_info": {
                "prompt_text": "",
                "game_id": game_id,
                "sample_idx": dataset_idx,
                "game_data": game_data,
            },
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
            "index": idx,
        }
