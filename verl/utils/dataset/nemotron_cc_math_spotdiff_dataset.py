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
"""SpyRL mathematical-reasoning environment on Nemotron-CC-Math -- sequential performing stage.

Civilians receive a math-heavy web document (truncated to ``civilian_doc_max_tokens`` whitespace
tokens, 1000 by default) and must design *and* solve a problem grounded in it. The spy
receives an empty document and has to fabricate a plausible problem from its peers' outputs.
Players speak in turn; the spy speaks last, and player 1 -- who speaks first -- is never
the spy.

Source: ``nvidia/Nemotron-CC-Math-v1``; set ``NEMOTRON_CC_MATH_CONFIG`` to choose the config
(default ``"3"``). Paired with the ``nemotron_cc_math_two_player`` agent loop.
"""

import os
import random
import re
from typing import Any

import datasets
import torch
from torch.utils.data import Dataset


class NemotronCCMathSpotDiffPromptBuilder:
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
        player_thinking_history: str = "",
    ) -> str:
        num_players = game_data["num_players"]
        spy_player = game_data["spy_player"]
        player_document = game_data["player_document_content"][player_id - 1]

        if player_id == spy_player:
            role_info = "Your role: SPY (you received NO document)"
            role_instruction = "• You are the spy with no document. Other players see the same knowledge document."
            document_display = "[You see nothing - blank content]"
        else:
            role_info = "Your role: NOT SPY (you received the knowledge document)"
            role_instruction = "• You are not the spy. Your document is the same as most other players' document."
            document_display = f"[Your Knowledge Document]\n{player_document}"

        game_context = f"""You are PLAYER {player_id} of the game "Who's the Odd One Out?". This is ROUND {round_num}. Your task is to write ONE document-grounded statement that looks like “condition + result” in a single sentence.

{role_info}

Game Instructions:
• There are {num_players} players in total. Only one player (the spy) received NO documents, while all other players received the SAME set of documents.
• The player who is voted as the odd player will be severely punished, so you should convince others that you are not the odd player.
• The spy sees nothing (blank), while all other players see the same documents.
{role_instruction}

What you should do:
1. Compare your documents (or lack thereof) with all previous clues to understand what others are seeing.
2. If you are NOT the spy:
    1) Identify key knowledge points in the text (e.g., a named formula/rule/defination). 
    2) Construct ONE new scenario/condition based on those knowledge points. Do this quickly by using a fixed template (see below). Do NOT brainstorm or list alternatives.
    3) Compute or infer the final result/answer by thinking step by step and verify correctness. Ensure that the result you obtain is absolutely correct, otherwise you will be suspected of being a spy.
    4) Your final statement must NOT be too easy (that can be answer without reasoning), and NOT copy directly from the documents; it must paraphrase and create a novel setup through inference/assumptions.
    5) Assume other players cannot see the documents: do NOT say "this formula" or "the text says". Instead, explicitly name what you use (e.g., "the standard error formula sd/√n", "SUBTOTAL with function_num=103", "normal-sampling rule", "replication count N", etc.).
3. If you are the spy:
    1) Infer likely knowledge points by reading previous players’ statements.
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
Your clue must be differen with previous clues.
Hard limit: The entire output must be <= 2000 tokens.
Example answer: \\boxed{{If an experiment repeats sampling N=500 times with population sd=1 and sample size n=20, using the standard error formula sd/\\sqrt{{n}}, then the sampling-distribution spread is about 0.22.}}.

{document_display}
"""

        if previous_clues_text:
            game_context += f"\n\nPrevious questions:\n\n{previous_clues_text}"
        else:
            game_context += (
                "\n\nPrevious questions:\n"
                f"(You are PLAYER {player_id} and are the first to speak, so there is no previous output yet.)"
            )

        return game_context

    @staticmethod
    def build_god_decision_prompt(
        num_players: int,
        original_document: str,
        all_clues: str,
    ) -> str:
        decision_context = f"""You are analyzing a game of "Who's the Odd One Out?" where {num_players} players have just provided clues about documents they received. Your task is to identify which player is the spy (the one who received NO documents) based on their clues.

Game Setup:
• There are {num_players} players in total. Only one player (the spy) received NO documents, while all other players received the SAME set of documents.
• The spy sees nothing (blank content).
• Each player provided a clue trying to describe the documents while avoiding suspicion. These clues are subtle information obtained by bridging or comparing these documents.
• The spy tries to blend in by giving clues that seem consistent with what others see.
• Non-spy players try to give distinctive and correct clues that help identify the spy.

Your task:
Analyze all the clues and determine which player is most likely the spy. In this variant, each "clue" is a one-sentence **Condition + Result** created from the documents. Your primary strategy is to **quickly verify** each clue against the reference documents and judge who likely had access to the documents.
Look for:
1. **Incorrect result**: If a player's stated result is wrong under its condition when checked against the reference documents, they are highly likely to be the spy (they didn't see the knowledge points).
2. **Unsupported or unverifiable**: The clue uses facts/numbers/relationships that do not appear in the reference documents, or the condition is too ambiguous to verify.
3. **Overly simple / generic**: If a player's condition and result are trivial, vague, or could fit many unrelated documents, they may be the spy trying to stay safe.
4. **Consistency with others**: Non-spy players tend to converge on the same document-specific themes/terms/quantities; a spy may drift, mismatch, or imitate superficially.

Efficiency constraints (IMPORTANT):
• Do a fast check for each clue (supported? correct? too generic?) and pick the single most suspicious player.
• If you cannot immediately determine who the mole is, answer \\boxed{{N/A}} first and then continue thinking.  
• Do NOT brainstorm. The entire output must be <= 2000 tokens.

The documents below are what non-spy players see. Compare the clues against this reference:

[Reference Documents]
{original_document}

Required output format:
First, conduct your private reasoning — may include suspicions, probabilities, evidence analysis, etc.
Then, put your final answer (PLAYER_NUMBER or N/A) inside \\boxed{{}}. If you are uncertain, you can answer N/A.
Example answer: \\boxed{{1}}; \\boxed{{2}}; \\boxed{{3}}; \\boxed{{N/A}}. 
Hard limit: The entire output must be <= 2000 tokens.

All Clues from the Clue-giving Stage:
{all_clues}"""
        return decision_context

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
    def build_previous_clues_text(clue_texts: dict[int, str], num_rounds: int) -> str:
        if not clue_texts:
            return ""
        lines = []
        for round_num in range(1, num_rounds + 1):
            lines.append(f"Round {round_num}:")
            for player_id in sorted(clue_texts.keys()):
                lines.append(f"Player {player_id}: {clue_texts[player_id]}")
            lines.append("")
        return "\n".join(lines).strip()

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
    def build_all_clues_text(clue_texts: dict[int, str], num_players: int, num_rounds: int) -> str:
        lines = []
        for round_num in range(1, num_rounds + 1):
            lines.append(f"Round {round_num}:")
            for player_id in range(1, num_players + 1):
                clue_text = clue_texts.get(player_id, "No clue provided")
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


class NemotronCCMathSpotDiffTwoPlayerDataset(Dataset):
    """
    Two-player dataset for Nemotron-CC-Math Spot-the-Difference with sequential clues
    followed by a God decision phase in the agent loop.
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

        rng = random.Random(self.seed + dataset_idx)
        spy_player = rng.randint(2, self.num_players)
        game_id = f"nemotron_cc_math_two_player_sample_{dataset_idx}"

        document_text_civilian = NemotronCCMathSpotDiffPromptBuilder.truncate_to_first_n_whitespace_tokens(
            document_text_full, self.civilian_doc_max_tokens
        )

        player_document_content = []
        for player_id in range(1, self.num_players + 1):
            if player_id == spy_player:
                player_document_content.append("")
            else:
                player_document_content.append(document_text_civilian)

        game_data = {
            "game_id": game_id,
            "sample_idx": dataset_idx,
            "num_players": self.num_players,
            "num_rounds": self.num_rounds,
            "spy_player": spy_player,
            "player_document_content": player_document_content,
            "nemotron_cc_math_data": {
                "document_text": document_text_full,
                "civilian_document_text": document_text_civilian,
                "civilian_document_max_whitespace_tokens": self.civilian_doc_max_tokens,
            },
        }

        messages = [{"role": "user", "content": ""}]

        return {
            "prompt": messages,
            "raw_prompt": messages,
            "data_source": "nemotron_cc_math_spotdiff_two_player",
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
