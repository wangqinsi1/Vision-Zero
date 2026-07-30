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
"""SpyRL summarization environment on GovReport -- sequential performing stage.

Variant of :mod:`verl.utils.dataset.govreport_spotdiff_parallel_dataset` in which players speak
in turn: each one sees the summaries written before it, and the spy always speaks last so
it can infer the masked content from its peers. The spy's report has a random contiguous
50% span replaced by ``*``; player 1 speaks first and is therefore never the spy.

Source: ``ccdv/govreport-summarization``, streamed; the first ``GOVREPORT_TAKE_N`` (default
20000) samples are materialized for random access. Paired with the ``govreport_two_player``
agent loop.
"""

import os
import random
import re
from typing import Any

import datasets
import torch
from torch.utils.data import Dataset


class GovReportSpotDiffPromptBuilder:
    @staticmethod
    def truncate_to_first_n_whitespace_tokens(text: str, max_tokens: int) -> str:
        if not text:
            return text
        tokens = text.split()
        if len(tokens) <= max_tokens:
            return text
        return " ".join(tokens[:max_tokens])

    @staticmethod
    def mask_random_contiguous_span(text: str, seed: int) -> dict[str, Any]:
        """
        Mask a random contiguous 50% span with '*', deterministic given seed.

        Returns:
            dict with keys: masked_text, start, end, fraction
        """
        if not text:
            return {"masked_text": text, "start": 0, "end": 0, "fraction": 0.0}
        n = len(text)
        span = max(1, n // 2)
        rng = random.Random(seed)
        start = rng.randint(0, max(0, n - span))
        end = min(n, start + span)
        masked = text[:start] + ("*" * (end - start)) + text[end:]
        return {"masked_text": masked, "start": start, "end": end, "fraction": 0.5}

    @staticmethod
    def extract_summary(clue_response: str) -> str:
        """
        Extract summary from clue-stage response.

        We ONLY accept content after an explicit "Answer:" marker (case-insensitive),
        otherwise return a safe placeholder to avoid leaking chain-of-thought.
        """
        match = re.search(r"(?ims)^\s*answer\s*:\s*(.*)$", clue_response)
        if match:
            extracted = match.group(1).strip()
            return extracted if extracted else "No valid rule provided"
        return "No valid rule provided"

    @staticmethod
    def build_previous_summaries_text_from_sequence(summary_sequence: list[dict[str, Any]]) -> str:
        if not summary_sequence:
            return ""
        lines: list[str] = []
        rounds_seen: list[int] = []
        summaries_by_round: dict[int, list[dict[str, Any]]] = {}
        for entry in summary_sequence:
            round_num = int(entry.get("round", 1))
            if round_num not in summaries_by_round:
                summaries_by_round[round_num] = []
                rounds_seen.append(round_num)
            summaries_by_round[round_num].append(entry)
        for round_num in rounds_seen:
            lines.append(f"Round {round_num}:")
            for entry in summaries_by_round[round_num]:
                player_id = entry.get("player_id")
                summary_text = entry.get("summary_text", "No valid rule provided")
                lines.append(f"Player {player_id}: {summary_text}")
            lines.append("")
        return "\n".join(lines).strip()

    @staticmethod
    def build_all_summaries_text_from_sequence(
        summary_sequence: list[dict[str, Any]], num_players: int, num_rounds: int
    ) -> str:
        lookup: dict[tuple[int, int], str] = {}
        for entry in summary_sequence:
            round_num = int(entry.get("round", 1))
            player_id = int(entry.get("player_id", 1))
            summary_text = entry.get("summary_text", "No valid rule provided")
            lookup[(round_num, player_id)] = summary_text
        lines: list[str] = []
        for round_num in range(1, num_rounds + 1):
            lines.append(f"Round {round_num}:")
            for player_id in range(1, num_players + 1):
                summary_text = lookup.get((round_num, player_id), "No valid rule provided")
                lines.append(f"Player {player_id}: {summary_text}")
            lines.append("")
        return "\n".join(lines).strip()

    @staticmethod
    def build_clue_prompt(
        game_data: dict[str, Any],
        player_id: int,
        round_num: int,
        previous_summaries_text: str,
    ) -> str:
        """
        Clue stage: each player writes ONE paragraph summary of the report.
        - Civilians see the full report.
        - Spy sees a report with a random contiguous 50% masked, and must infer missing info from others' summaries.
        """
        num_players = int(game_data["num_players"])
        spy_player = int(game_data["spy_player"])
        player_report = str(game_data["player_document_content"][player_id - 1] or "").strip()

        if player_id == spy_player:
            role_info = "Your role: SPY (you received a PARTIALLY MASKED report)"
            role_instruction = "• You are the spy with a partially masked report. Other players see the full report."
        else:
            role_info = "Your role: NOT SPY (you received the full report)"
            role_instruction = "• You are not the spy. Your report is the full report."

        game_context = f"""You are PLAYER {player_id} of the game "Who's the Odd One Out?". This is ROUND {round_num}. Your task is to write ONE paragraph summary of the report.

{role_info}

Game Instructions:
• There are {num_players} players in total. Only one player (the spy) received an incomplete view of the report, while all other players received the SAME full report.
• The player who is voted as the odd player will be severely punished, so you should convince others that you are not the odd player.
• Non-spy players see the full report. The spy sees a report where a random contiguous 50% span is masked with '*'.
{role_instruction}

What you should do:
1. Compare your report view with all previous summaries to understand what others are seeing.
2. If you are NOT the spy:
    1) Stay grounded in the report: summarize only what the text supports, and if something isn’t stated, leave it out rather than guessing.
    2) Aim for broad, balanced coverage: pull key information from across the whole report (including later sections), not just the opening.
    3) Keep it easy to scan: try to write with inline labels to clearly organize the content.
    4) Pack in information, not filler: be concrete and specific (important entities, actions, results, numbers when present), while avoiding fluff, repetition, and unnecessary detail.
3. If you are the spy:
    1) Infer missing report content by reading other players' summaries.
    2) Write a plausible, faithful-looking summary that matches the report as best as you can.
    3) Keep the same qualities: faithful, broad coverage, structured, high-density.

Fixed limit:
• Write ONE paragraph, try to no more than 500 words.
• Do not repeat previous players' summaries. Only by having a better summary than other players can you reduce the probability of being suspected of being the spy.

Required output format:
First, conduct your own thinking process.
Then, start a new line with exactly "Answer:" and write exactly ONE paragraph summary following the Summary template.

[Your Report]
{player_report}
"""

        if previous_summaries_text:
            game_context += f"\n\nPrevious summaries:\n\n{previous_summaries_text}"
        else:
            game_context += (
                "\n\nPrevious summaries:\n"
                f"(You are PLAYER {player_id} and are the first to write, so there is no previous output yet.)"
            )

        return game_context

    @staticmethod
    def build_god_decision_prompt(num_players: int, report_text: str, all_summaries: str) -> str:
        """
        Decision stage: a judge reads the full report + all summaries and identifies the spy.

        Output must be \\boxed{PLAYER_NUMBER} or \\boxed{N/A} so the existing reward parser can extract votes.
        """
        decision_context = f"""You are analyzing a game of "Who's the Odd One Out?" where {num_players} players have just written summaries of a report. Your task is to identify which player is the spy (the one who saw an incomplete/masked report) based on their summary.

Game Setup:
• There are {num_players} players in total. Only one player (the spy) saw a report where a random contiguous 50% span was masked with '*'. All other players saw the SAME full report.
• Each player wrote a summary trying to match the report while avoiding suspicion.
• The spy tries to blend in by inferring the missing content from others' summaries.
• Non-spy players should produce faithful, broad-coverage, structured, high-density summaries.

Your task:
Analyze the full report and all the summaries and determine which player is most likely the spy.
Look for:
1. **Not faithful / hallucination**: Includes facts, numbers, entities, or causal claims not supported by the report.
2. **Low coverage**: Misses key points that other summaries capture; focuses narrowly on one section.
3. **Unstructured**: No clear organization; hard to map to report.
4. **Repetition/ Too verbose / low density**: Many words with little information; repetition of previous players' summary; vague filler.
5. **Inconsistency with others**: Non-spy summaries tend to converge on the same report-grounded facts; a spy may drift or imitate superficially.

Efficiency constraints (IMPORTANT):
• Do a fast check for each summary and pick the single most suspicious player.
• If you cannot immediately determine who the mole is, answer \\boxed{{N/A}} first and then continue thinking.  
• Do NOT brainstorm. The entire output must be <= 2000 tokens.

The report below is what non-spy players see. Compare each summary against this reference:

[Reference Report]
{report_text}

Required output format:
First, conduct your private reasoning — may include suspicions, probabilities, evidence analysis, etc.
Then, put your final answer (PLAYER_NUMBER or N/A) inside \\boxed{{}}. If you are uncertain, you can answer N/A.
Example answer: \\boxed{{1}}; \\boxed{{2}}; \\boxed{{3}}; \\boxed{{N/A}}. 
Hard limit: The entire output must be <= 2000 tokens.

All Summaries from the Summary-writing Stage:
{all_summaries}"""
        return decision_context


class GovReportSpotDiffTwoPlayerDataset(Dataset):
    """
    Spot-the-difference dataset using `ccdv/govreport-summarization`.

    Civilians see the sample's `report` in full.
    The spy sees the same report with a random contiguous 50% span masked with '*'.
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
        # Optional whitespace truncation (set high by default; can be overridden).
        self.report_max_tokens = int(config.get("report_max_tokens", 1000000))

        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        take_n_env = os.getenv("GOVREPORT_TAKE_N", "")
        take_n = int(take_n_env) if take_n_env.isdigit() else 20000

        print("Loading GovReport summarization dataset from Hugging Face (ccdv/govreport-summarization)...")
        ds = datasets.load_dataset(
            "ccdv/govreport-summarization",
            split="train",
            streaming=True,
            token=token,
        )
        print(f"Materializing first {take_n} samples for random access...")
        self.dataset_list = list(ds.take(take_n))
        print(f"Loaded {len(self.dataset_list)} GovReport samples")

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
    def _extract_report_field(sample: dict[str, Any]) -> str:
        report = sample.get("report", "")
        if report is None:
            report = ""
        return str(report)

    def __getitem__(self, idx: int) -> dict:
        dataset_idx = self.indices[idx]
        sample = self.dataset_list[dataset_idx]

        report_text_full = self._extract_report_field(sample)
        report_text_full = GovReportSpotDiffPromptBuilder.truncate_to_first_n_whitespace_tokens(
            report_text_full, self.report_max_tokens
        )

        rng = random.Random(self.seed + dataset_idx)
        spy_player = rng.randint(2, self.num_players)
        game_id = f"govreport_two_player_sample_{dataset_idx}"

        mask_info = GovReportSpotDiffPromptBuilder.mask_random_contiguous_span(
            report_text_full, seed=self.seed + dataset_idx
        )
        masked_report = mask_info["masked_text"]

        player_document_content = []
        for player_id in range(1, self.num_players + 1):
            if player_id == spy_player:
                player_document_content.append(masked_report)
            else:
                player_document_content.append(report_text_full)

        game_data = {
            "game_id": game_id,
            "sample_idx": dataset_idx,
            "num_players": self.num_players,
            "num_rounds": self.num_rounds,
            "spy_player": spy_player,
            "player_document_content": player_document_content,
            "govreport_data": {
                "report_text": report_text_full,
                "masked_report_text": masked_report,
                "mask_start": int(mask_info.get("start", 0)),
                "mask_end": int(mask_info.get("end", 0)),
                "mask_fraction": float(mask_info.get("fraction", 0.5)),
                "report_max_whitespace_tokens": self.report_max_tokens,
                "source_dataset": "ccdv/govreport-summarization",
            },
        }

        messages = [{"role": "user", "content": ""}]
        return {
            "prompt": messages,
            "raw_prompt": messages,
            "data_source": "govreport_spotdiff_two_player",
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
