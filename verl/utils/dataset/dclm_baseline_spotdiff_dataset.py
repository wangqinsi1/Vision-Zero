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
"""Training-corpus ablation: the math game played on generic web text instead of math text.

Identical game rules, prompts and rewards to
:mod:`verl.utils.dataset.nemotron_cc_math_spotdiff_dataset` -- only the source documents change,
from ``nvidia/Nemotron-CC-Math-v1`` to ``mlfoundations/dclm-baseline-1.0``. It measures how much
of the reasoning gain depends on the training corpus being math-heavy. This ablation is not
reported in the paper. Paired with the ``nemotron_cc_math_two_player`` agent loop.
"""

import os
import random
from typing import Any

import datasets
import torch
from torch.utils.data import Dataset

from verl.utils.dataset.nemotron_cc_math_spotdiff_dataset import (
    NemotronCCMathSpotDiffPromptBuilder,
)


class DCLMBaselineSpotDiffTwoPlayerDataset(Dataset):
    """
    Two-player dataset for Spot-the-Difference using DCLM-Baseline documents.

    This mirrors NemotronCCMathSpotDiffTwoPlayerDataset but swaps the source
    documents to mlfoundations/dclm-baseline-1.0. Game rules, prompts, rewards,
    and loss logic remain unchanged.
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
        # Default truncation length for DCLM baseline documents.
        self.civilian_doc_max_tokens = int(config.get("civilian_doc_max_tokens", 3762))

        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        take_n_env = os.getenv("DCLM_TAKE_N", "")
        take_n = int(take_n_env) if take_n_env.isdigit() else 100000

        print("Loading DCLM-Baseline dataset from Hugging Face (mlfoundations/dclm-baseline-1.0)...")
        ds = datasets.load_dataset(
            "mlfoundations/dclm-baseline-1.0",
            split="train",
            streaming=True,
            token=token,
        )
        # Materialize a prefix of the stream so the sampler can index it randomly.
        print(f"Materializing first {take_n} samples for random access...")
        self.dataset_list = list(ds.take(take_n))
        print(f"Loaded {len(self.dataset_list)} DCLM-Baseline samples")

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

    def __getitem__(self, idx: int) -> dict:
        dataset_idx = self.indices[idx]
        sample = self.dataset_list[dataset_idx]
        document_text_full = sample.get("text", "")

        rng = random.Random(self.seed + dataset_idx)
        spy_player = rng.randint(2, self.num_players)
        game_id = f"dclm_baseline_two_player_sample_{dataset_idx}"

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
            # Keep nemotron_cc_math_data for compatibility with the existing agent loop.
            "nemotron_cc_math_data": {
                "document_text": document_text_civilian,
                "full_document_text": document_text_full,
                "civilian_document_text": document_text_civilian,
                "civilian_document_max_whitespace_tokens": self.civilian_doc_max_tokens,
                "source_dataset": "mlfoundations/dclm-baseline-1.0",
            },
            # Also include dclm_baseline_data to mirror VLM-R1 structure.
            "dclm_baseline_data": {
                "document_text": document_text_civilian,
                "full_document_text": document_text_full,
                "civilian_document_max_whitespace_tokens": self.civilian_doc_max_tokens,
            },
        }

        messages = [{"role": "user", "content": ""}]

        return {
            "prompt": messages,
            "raw_prompt": messages,
            "data_source": "dclm_baseline_spotdiff_two_player",
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
