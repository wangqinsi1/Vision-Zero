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
"""SpyRL mathematical-reasoning environment on Nemotron-CC-Math -- parallel performing stage.

Every player receives the same math-heavy web document, except the spy whose copy has a
random contiguous span (``mask_fraction``, 0.4 by default, matching Section 3.1 of the paper)
replaced by ``*``. Players design *and* solve a problem grounded in what they were given,
independently: nobody sees anyone else's problem before the detection stage.

This is the partial-masking counterpart of
:mod:`verl.utils.dataset.nemotron_cc_math_spotdiff_dataset`, where the spy instead
receives no document at all and players speak in turn.

Source: ``nvidia/Nemotron-CC-Math-v1``; set ``NEMOTRON_CC_MATH_CONFIG`` to choose the config
(default ``"3"``). Paired with the ``nemotron_cc_math_parallel`` agent loop.
"""

import os
import random
import re
from typing import Any

import datasets
import torch
from torch.utils.data import Dataset


class NemotronCCMathSpotDiffParallelPromptBuilder:
    @staticmethod
    def truncate_to_first_n_whitespace_tokens(text: str, max_tokens: int) -> str:
        if not text:
            return text
        tokens = text.split()
        if len(tokens) <= max_tokens:
            return text
        return " ".join(tokens[:max_tokens])

    @staticmethod
    def mask_random_contiguous_span(text: str, seed: int, fraction: float) -> dict[str, Any]:
        """
        Mask a random contiguous span with '*', deterministic given seed.

        Returns:
            dict with keys: masked_text, start, end, fraction
        """
        if not text:
            return {"masked_text": text, "start": 0, "end": 0, "fraction": 0.0}
        n = len(text)
        span = max(1, int(round(n * fraction)))
        rng = random.Random(seed)
        start = rng.randint(0, max(0, n - span))
        end = min(n, start + span)
        masked = text[:start] + ("*" * (end - start)) + text[end:]
        return {"masked_text": masked, "start": start, "end": end, "fraction": fraction}

    @staticmethod
    def extract_clue_answer(clue_response: str) -> str:
        """Extract the one-sentence statement a player produced in the performing stage."""
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
    def build_all_clues_text_from_sequence(
        clue_sequence: list[dict[str, Any]], num_players: int, num_rounds: int
    ) -> str:
        lookup: dict[tuple[int, int], str] = {}
        for entry in clue_sequence:
            round_num = int(entry.get("round", 1))
            player_id = int(entry.get("player_id", 1))
            lookup[(round_num, player_id)] = entry.get("clue_text", "No clue provided")
        lines: list[str] = []
        for round_num in range(1, num_rounds + 1):
            lines.append(f"Round {round_num}:")
            for player_id in range(1, num_players + 1):
                lines.append(f"Player {player_id}: {lookup.get((round_num, player_id), 'No clue provided')}")
            lines.append("")
        return "\n".join(lines).strip()

    @staticmethod
    def build_clue_prompt_parallel(
        *,
        num_players: int,
        round_num: int,
        document_text: str,
        is_spy: bool,
        mask_fraction: float,
    ) -> str:
        """
        Parallel clue stage: each player designs and solves ONE problem independently.
        - Civilians see the full document.
        - Spy sees the document with a random contiguous masked span.
        - Players do NOT see other players' statements in this stage.
        """
        percent = int(round(mask_fraction * 100))
        if is_spy:
            role_info = "Your role: SPY (you received a PARTIALLY MASKED document)"
            role_instruction = (
                "• You are the spy with a partially masked document. Other players see the full document."
            )
        else:
            role_info = "Your role: NOT SPY (you received the full document)"
            role_instruction = "• You are not the spy. Your document is the full document."

        game_context = f"""You are a player of the game "Who's the Odd One Out?". This is ROUND {round_num}. Your task is to write ONE document-grounded statement that looks like “condition + result” in a single sentence.

{role_info}

Game Instructions:
• There are {num_players} players in total. Only one player (the spy) received an incomplete view of the document, while all other players received the SAME full document.
• The player who is voted as the odd player will be severely punished, so you should convince others that you are not the odd player.
• Non-spy players see the full document. The spy sees a document where a random contiguous {percent}% span is masked with '*'.
• In this stage, players write their statements independently and cannot see other players' statements.
{role_instruction}

What you should do:
1. If you are NOT the spy:
    1) Identify key knowledge points in the text (e.g., a named formula/rule/defination).
    2) Construct ONE new scenario/condition based on those knowledge points. Do this quickly by using a fixed template (see below). Do NOT brainstorm or list alternatives.
    3) Compute or infer the final result/answer by thinking step by step and verify correctness. Ensure that the result you obtain is absolutely correct, otherwise you will be suspected of being a spy.
    4) Your final statement must NOT be too easy (that can be answer without reasoning), and NOT copy directly from the documents; it must paraphrase and create a novel setup through inference/assumptions.
    5) Assume other players cannot see the documents: do NOT say "this formula" or "the text says". Instead, explicitly name what you use (e.g., "the standard error formula sd/√n", "SUBTOTAL with function_num=103", "normal-sampling rule", "replication count N", etc.).
2. If you are the spy:
    1) Use only the visible part of your document and do not invent knowledge points that are not supported by it.
    2) Construct ONE plausible scenario using the same fixed template approach (no brainstorming).
    3) Compute/infer a correct, self-consistent answer.
    4) Your statement can be slightly more generic, but it should still appear to use document-like knowledge points and must not be obviously fabricated.

Fixed template you MUST follow (choose exactly ONE):
• Template A (QA): "If <scenario with variables/conditions>, using <named rule/formula/definition>, then <final result/answer>."
• Template B (MCQ): "If <scenario with variables/conditions>, using <named rule/formula/definition>, what is <quantity/conclusion>? A) ... B) ... C) ... D) ... Answer: <letter>."
Try to include multiple knowledge points when possible, but it is acceptable if only one is clearly used.

Required output format:
First, conduct your own thinking process and then put your one-sentence "Condition+Result" inside \\boxed{{}}.
Inside \\boxed{{...}}, write exactly ONE sentence following Template A or Template B.
Hard limit: The entire output must be <= 2000 tokens.
Example answer: \\boxed{{If an experiment repeats sampling N=500 times with population sd=1 and sample size n=20, using the standard error formula sd/\\sqrt{{n}}, then the sampling-distribution spread is about 0.22.}}.

[Your Knowledge Document]
{document_text}
"""
        return game_context

    @staticmethod
    def build_god_decision_prompt(
        *,
        num_players: int,
        document_text: str,
        all_clues: str,
        mask_fraction: float,
    ) -> str:
        """
        Decision stage: a judge reads the full document + all statements and identifies the spy.

        Output must be \\boxed{PLAYER_NUMBER} or \\boxed{N/A} so the existing reward parser can extract votes.
        """
        percent = int(round(mask_fraction * 100))
        decision_context = f"""You are analyzing a game of "Who's the Odd One Out?" where {num_players} players have just provided clues about a document they received. Your task is to identify which player is the spy (the one who saw an incomplete/masked document) based on their clues.

Game Setup:
• There are {num_players} players in total. Only one player (the spy) saw a document where a random contiguous {percent}% span was masked with '*'. All other players saw the SAME full document.
• Each player provided a clue independently without seeing other players' clues. These clues are subtle information obtained from the document.
• The spy tries to blend in by making a clue built on its partial view look as grounded as possible.
• Non-spy players try to give distinctive and correct clues that help identify the spy.

Your task:
Analyze all the clues and determine which player is most likely the spy. In this variant, each "clue" is a one-sentence **Condition + Result** created from the document. Your primary strategy is to **quickly verify** each clue against the reference document and judge who likely had full access to it.
Look for:
1. **Incorrect result**: If a player's stated result is wrong under its condition when checked against the reference document, they are highly likely to be the spy (they didn't see the knowledge points).
2. **Unsupported or unverifiable**: The clue uses facts/numbers/relationships that do not appear in the reference document, or the condition is too ambiguous to verify.
3. **Overly simple / generic**: If a player's condition and result are trivial, vague, or could fit many unrelated documents, they may be the spy trying to stay safe.
4. **Consistency with others**: Non-spy players tend to converge on the same document-specific themes/terms/quantities; a spy may drift, mismatch, or cover only part of the material.

Efficiency constraints (IMPORTANT):
• Do a fast check for each clue (supported? correct? too generic?) and pick the single most suspicious player.
• If you cannot immediately determine who the mole is, answer \\boxed{{N/A}} first and then continue thinking.
• Do NOT brainstorm. The entire output must be <= 2000 tokens.

The document below is what non-spy players see. Compare the clues against this reference:

[Reference Document]
{document_text}

Required output format:
First, conduct your private reasoning — may include suspicions, probabilities, evidence analysis, etc.
Then, put your final answer (PLAYER_NUMBER or N/A) inside \\boxed{{}}. If you are uncertain, you can answer N/A.
Example answer: \\boxed{{1}}; \\boxed{{2}}; \\boxed{{3}}; \\boxed{{N/A}}.
Hard limit: The entire output must be <= 2000 tokens.

All Clues from the Clue-giving Stage:
{all_clues}"""
        return decision_context


class NemotronCCMathSpotDiffParallelDataset(Dataset):
    """
    Parallel spot-the-difference dataset using `nvidia/Nemotron-CC-Math-v1`.

    Civilians see the sample's document in full.
    The spy sees the same document with a random contiguous span masked with '*'.
    """

    def __init__(
        self,
        data_files: str | list[str] | None,
        tokenizer: Any,
        config: Any,
        processor: Any | None = None,
        max_samples: int = -1,
    ):
        del data_files, processor
        self.tokenizer = tokenizer
        self.config = config
        self.max_samples = max_samples

        self.num_players = int(config.get("num_players", 2))
        self.num_rounds = int(config.get("num_rounds", 1))
        self.seed = int(config.get("seed", 0) or 0)
        # Section 3.1 of the paper masks 40% of the document for the math task.
        self.mask_fraction = float(config.get("mask_fraction", 0.4))
        self.civilian_doc_max_tokens = int(config.get("civilian_doc_max_tokens", 1000))

        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        config_name = os.getenv("NEMOTRON_CC_MATH_CONFIG", "3")
        print(f"Loading Nemotron-CC-Math-v1 (config {config_name}) from Hugging Face...")
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

        document_text_full = NemotronCCMathSpotDiffParallelPromptBuilder.truncate_to_first_n_whitespace_tokens(
            sample.get("text", ""), self.civilian_doc_max_tokens
        )

        rng = random.Random(self.seed + dataset_idx)
        spy_player = rng.randint(1, self.num_players)
        game_id = f"nemotron_cc_math_parallel_sample_{dataset_idx}"

        mask_info = NemotronCCMathSpotDiffParallelPromptBuilder.mask_random_contiguous_span(
            document_text_full, seed=self.seed + dataset_idx, fraction=self.mask_fraction
        )
        masked_document = mask_info["masked_text"]

        player_document_content = []
        for player_id in range(1, self.num_players + 1):
            if player_id == spy_player:
                player_document_content.append(masked_document)
            else:
                player_document_content.append(document_text_full)

        game_data = {
            "game_id": game_id,
            "sample_idx": dataset_idx,
            "num_players": self.num_players,
            "num_rounds": self.num_rounds,
            "spy_player": spy_player,
            "player_document_content": player_document_content,
            "nemotron_cc_math_data": {
                "document_text": document_text_full,
                "masked_document_text": masked_document,
                "mask_start": int(mask_info.get("start", 0)),
                "mask_end": int(mask_info.get("end", 0)),
                "mask_fraction": float(mask_info.get("fraction", self.mask_fraction)),
                "civilian_document_max_whitespace_tokens": self.civilian_doc_max_tokens,
                "source_dataset": "nvidia/Nemotron-CC-Math-v1",
            },
        }

        messages = [{"role": "user", "content": ""}]
        return {
            "prompt": messages,
            "raw_prompt": messages,
            "data_source": "nemotron_cc_math_spotdiff_parallel",
            "reward_model": {"ground_truth": spy_player},
            "extra_info": {
                "prompt_text": "",
                "game_id": game_id,
                "sample_idx": dataset_idx,
                "spy_player": spy_player,
                "game_data": game_data,
            },
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
            "index": idx,
        }
