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
"""SpyRL creative-writing environment on WritingPrompts -- parallel performing stage.

Civilians receive the full writing prompt; the spy receives the same prompt with a random
contiguous span (``mask_fraction``, 0.2 by default) replaced by ``*``. All players write their
stories independently before the detection stage.

Source: ``euclaise/writingprompts``, streamed; the first ``WRITINGPROMPTS_TAKE_N`` (default
100000) samples are materialized for random access. Paired with the ``writingprompts_parallel``
agent loop.
"""

import os
import random
import re
from typing import Any

import datasets
import torch
from torch.utils.data import Dataset


class WritingPromptsSpotDiffParallelPromptBuilder:
    @staticmethod
    def truncate_to_first_n_whitespace_tokens(text: str, max_tokens: int) -> str:
        if not text:
            return text
        tokens = text.split()
        if len(tokens) <= max_tokens:
            return text
        return " ".join(tokens[:max_tokens])

    @staticmethod
    def extract_story(story_response: str) -> str:
        """
        Extract the story text from a clue-stage model response.

        We ONLY accept content after an explicit "Answer:" marker (case-insensitive).
        This prevents chain-of-thought leakage into later players' visible context.
        """
        match = re.search(r"(?ims)^\s*answer\s*:\s*(.*)$", story_response)
        if match:
            extracted = match.group(1).strip()
            return extracted if extracted else "No story provided"
        return "No story provided"

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
    def build_all_stories_text_from_sequence(
        story_sequence: list[dict[str, Any]], num_players: int, num_rounds: int
    ) -> str:
        lookup: dict[tuple[int, int], str] = {}
        for entry in story_sequence:
            round_num = int(entry.get("round", 1))
            player_id = int(entry.get("player_id", 1))
            story_text = entry.get("story_text", "")
            lookup[(round_num, player_id)] = story_text
        lines: list[str] = []
        for round_num in range(1, num_rounds + 1):
            lines.append(f"Round {round_num}:")
            for player_id in range(1, num_players + 1):
                story_text = lookup.get((round_num, player_id), "")
                if not story_text:
                    story_text = "No story provided"
                lines.append(f"Player {player_id}: {story_text}")
            lines.append("")
        return "\n".join(lines).strip()

    @staticmethod
    def build_clue_prompt_parallel(
        *,
        num_players: int,
        round_num: int,
        writing_prompt: str,
        is_spy: bool,
        mask_fraction: float,
    ) -> str:
        """
        Parallel clue stage: each player writes a short creative story independently.
        - Civilians see the true writing prompt.
        - Spy sees nothing, and must write a plausible story without the prompt.
        - Players do NOT see other players' stories in this stage.
        """
        percent = int(round(mask_fraction * 100))
        if is_spy:
            role_info = "Your role: SPY (you received a PARTIALLY MASKED writing prompt)"
            role_instruction = (
                "• You are the spy with a partially masked writing prompt. Other players see the same full prompt."
            )
            prompt_display = f"[Your Writing Prompt (masked {percent}%)]\n{writing_prompt}"
        else:
            role_info = "Your role: NOT SPY (you received the writing prompt)"
            role_instruction = "• You are not the spy. Your writing prompt is the same as most other players' prompt."
            prompt_display = f"[Your Writing Prompt]\n{writing_prompt}"

        game_context = f"""You are a player of the game "Who's the Odd One Out?". This is ROUND {round_num}. Your task is to write ONE short creative story based on the writing prompt.

{role_info}

Game Instructions:
• There are {num_players} players in total. Only one player (the spy) received a PARTIALLY MASKED writing prompt, while all other players received the SAME full writing prompt.
• The player who is voted as the odd player will be severely punished, so you should convince others that you are not the odd player.
• The spy sees a writing prompt with a random contiguous {percent}% span masked with '*'.
• In this stage, players write stories independently and cannot see other players' stories.
{role_instruction}

What you should do:
1. If you are NOT the spy:
    1) Identify the core theme, mood, and implied constraints in the prompt.
    2) Write a short story that clearly matches the theme and is NOT generic.
    3) Make it deep: include subtext, a turning point, or a meaningful implication.
    4) Keep it coherent: clear setup → development → payoff, with smooth transitions.
    5) Be creative: fresh imagery, non-obvious choices, clever structure or constraints.
2. If you are the spy:
    1) Use only your partially masked prompt and do not invent missing details.
    2) Write a plausible story that fits what you can see, and keep it thematic, coherent, and creative.

Fixed template you MUST follow:
• Story template: Write ONE continuous short story in plain paragraphs (no bullet points), with 300-500 words (strict). Include a clear turning point and a payoff.

Required output format:
First, conduct your own thinking process.
Then, start a new line with exactly "Answer:" and write exactly ONE short story following the Story template after it (may span multiple paragraphs).

{prompt_display}
"""
        return game_context

    @staticmethod
    def build_god_decision_prompt(num_players: int, writing_prompt: str, all_stories: str) -> str:
        """
        Decision stage: a judge reads the writing prompt + all stories and identifies the spy.

        Output must be \\boxed{PLAYER_NUMBER} or \\boxed{N/A} so the existing reward parser can extract votes.
        """
        decision_context = f"""You are analyzing a game of "Who's the Odd One Out?" where {num_players} players have just written short stories based on a writing prompt. Your task is to identify which player is the spy (the one who received NO writing prompt) based on their story.

Game Setup:
• There are {num_players} players in total. Only one player (the spy) received a PARTIALLY MASKED writing prompt, while all other players received the SAME full writing prompt.
• The spy sees a writing prompt with a random contiguous 20% span masked with '*'.
• Each player wrote a short story independently without seeing other players' stories.
• The spy tries to blend in by using the partial prompt to write something that seems consistent.
• Non-spy players try to write stories that are thematically aligned, meaningful, coherent, and creative.

Your task:
Analyze the writing prompt and all the stories and determine which player is most likely the spy. Your primary strategy is to judge who likely had access to the prompt.
Look for:
1. **Off-theme / mismatch**: The story does not match the prompt's theme, constraints, or implied setting.
2. **Shallow / not meaningful**: Lacks subtext, turning point, or deeper implication.
3. **Weak narrative craft**: Straight-line narration, poor coherence, no payoff, abrupt transitions.
4. **Low creativity**: Generic style, cliché patterns, unimaginative choices.
5. **Consistency with others**: Non-spy players tend to converge on the same prompt-driven theme; a spy may drift or imitate superficially.

Efficiency constraints (IMPORTANT):
• Do a fast check for each story (on-theme? meaningful? coherent? creative?) and pick the single most suspicious player.
• If you cannot immediately determine who the mole is, answer \\boxed{{N/A}} first and then continue thinking.  
• Do NOT brainstorm. The entire output must be <= 2000 tokens.

The writing prompt below is what non-spy players see. Compare each story against this reference:

[Reference Writing Prompt]
{writing_prompt}

Required output format:
First, conduct your private reasoning — may include suspicions, probabilities, evidence analysis, etc.
Then, put your final answer (PLAYER_NUMBER or N/A) inside \\boxed{{}}. If you are uncertain, you can answer N/A.
Example answer: \\boxed{{1}}; \\boxed{{2}}; \\boxed{{3}}; \\boxed{{N/A}}. 
Hard limit: The entire output must be <= 2000 tokens.

All Stories from the Story-writing Stage:
{all_stories}"""
        return decision_context


class WritingPromptsSpotDiffParallelDataset(Dataset):
    """
    Parallel spot-the-difference dataset using `euclaise/writingprompts`.

    Civilians see the sample's `prompt`. The spy sees nothing.
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
        self.prompt_max_tokens = int(config.get("prompt_max_tokens", 128))
        self.mask_fraction = float(config.get("mask_fraction", 0.2))

        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        take_n_env = os.getenv("WRITINGPROMPTS_TAKE_N", "")
        take_n = int(take_n_env) if take_n_env.isdigit() else 100000

        print("Loading WritingPrompts dataset from Hugging Face (euclaise/writingprompts)...")
        ds = datasets.load_dataset(
            "euclaise/writingprompts",
            split="train",
            streaming=True,
            token=token,
        )
        print(f"Materializing first {take_n} samples for random access...")
        self.dataset_list = list(ds.take(take_n))
        print(f"Loaded {len(self.dataset_list)} WritingPrompts samples")

        total = len(self.dataset_list)
        if self.max_samples and self.max_samples > 0 and self.max_samples < total:
            rng = random.Random(self.seed)
            indices = list(range(total))
            rng.shuffle(indices)
            self.indices = indices[: self.max_samples]
        else:
            self.indices = list(range(total))

    def __len__(self) -> int:
        return len(self.indices)

    @staticmethod
    def _extract_prompt_field(sample: dict[str, Any]) -> str:
        prompt = sample.get("prompt", "")
        if isinstance(prompt, list) and prompt:
            prompt = prompt[0]
        if prompt is None:
            prompt = ""
        return str(prompt)

    def __getitem__(self, idx: int) -> dict:
        dataset_idx = self.indices[idx]
        sample = self.dataset_list[dataset_idx]

        prompt_text_full = self._extract_prompt_field(sample)
        prompt_text = WritingPromptsSpotDiffParallelPromptBuilder.truncate_to_first_n_whitespace_tokens(
            prompt_text_full, self.prompt_max_tokens
        )

        rng = random.Random(self.seed + dataset_idx)
        spy_player = rng.randint(1, self.num_players)
        game_id = f"writingprompts_parallel_sample_{dataset_idx}"

        mask_info = WritingPromptsSpotDiffParallelPromptBuilder.mask_random_contiguous_span(
            prompt_text, seed=self.seed + dataset_idx, fraction=self.mask_fraction
        )
        masked_prompt = mask_info["masked_text"]

        player_document_content = []
        for player_id in range(1, self.num_players + 1):
            if player_id == spy_player:
                player_document_content.append(masked_prompt)
            else:
                player_document_content.append(prompt_text)

        game_data = {
            "game_id": game_id,
            "sample_idx": dataset_idx,
            "num_players": self.num_players,
            "num_rounds": self.num_rounds,
            "spy_player": spy_player,
            "player_document_content": player_document_content,
            "writingprompts_data": {
                "prompt_text": prompt_text,
                "masked_prompt_text": masked_prompt,
                "mask_start": int(mask_info.get("start", 0)),
                "mask_end": int(mask_info.get("end", 0)),
                "mask_fraction": float(mask_info.get("fraction", self.mask_fraction)),
                "full_prompt_text": prompt_text_full,
                "prompt_max_whitespace_tokens": self.prompt_max_tokens,
                "source_dataset": "euclaise/writingprompts",
            },
        }

        messages = [{"role": "user", "content": ""}]
        return {
            "prompt": messages,
            "raw_prompt": messages,
            "data_source": "writingprompts_spotdiff_parallel",
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
