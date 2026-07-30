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
"""Reward function for the "Without spy" ablation of SpyRL (Table 4 in the paper).

Information asymmetry is removed: every player sees the same input, so there is no
environment-assigned identity to verify against. The detectors still vote, but their votes only
elect a *pseudo* spy (the most-suspected player) instead of being checked against a
ground-truth identity, so only the performing stage is trained and the detection reward is
disabled. This ablation isolates how much of SpyRL's gain comes from the adversarial spy
mechanism rather than from peer comparison alone.

Wire it up with ``custom_reward_function.path=<this file>`` and
``custom_reward_function.name=compute_score``.
"""

import random
import re
from typing import Any

import torch

_CLUE_REWARD_CACHE: dict[tuple[str, int | None], dict[str, Any]] = {}
_CLUE_REWARD_CACHE_STEP: int | None = None


def _extract_answer_content(text: str) -> str:
    boxed_match = re.search(r"\\\\?boxed\{(.*?)\}", text, re.DOTALL)
    if boxed_match:
        return boxed_match.group(1).strip()

    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    return ""


def _extract_vote(text: str) -> dict[str, Any] | None:
    answer_content = _extract_answer_content(text)
    if not answer_content:
        return None
    answer_upper = answer_content.strip().upper()
    if answer_upper in {"N/A", "NA"}:
        return {"voted_spy": "N/A", "reasoning": "Uncertain vote"}
    numbers = re.findall(r"\b([1-9])\b", answer_content)
    if numbers:
        return {"voted_spy": int(numbers[0]), "reasoning": "Direct number vote"}
    return None


class _RoleBaselines:
    def __init__(self, alpha: float = 0.9):
        self.b_spy = 0.0
        self.b_civ = 0.0
        self.alpha = alpha

    def _reduce_mean(self, value: float) -> float:
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return value
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor = torch.tensor([value], device=device, dtype=torch.float32)
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        world_size = torch.distributed.get_world_size()
        if world_size > 0:
            tensor /= float(world_size)
        return float(tensor.item())

    def _sync(self) -> None:
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor = torch.tensor([self.b_spy, self.b_civ], device=device, dtype=torch.float32)
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        world_size = torch.distributed.get_world_size()
        if world_size > 0:
            tensor /= float(world_size)
        self.b_spy, self.b_civ = tensor.tolist()

    def update_role_baselines(self, spy_reward: float, civilian_avg_reward: float) -> None:
        spy_reward = self._reduce_mean(spy_reward)
        civilian_avg_reward = self._reduce_mean(civilian_avg_reward)
        self.b_spy = self.alpha * self.b_spy + (1 - self.alpha) * spy_reward
        self.b_civ = self.alpha * self.b_civ + (1 - self.alpha) * civilian_avg_reward
        self._sync()

    def apply_unified_role_advantage_adjustment(self, rewards: list[float], spy_player: int) -> list[float]:
        spy_reward = rewards[spy_player - 1]
        civilian_rewards = [rewards[i] for i in range(len(rewards)) if i != spy_player - 1]
        civilian_avg = sum(civilian_rewards) / len(civilian_rewards) if civilian_rewards else 0.0
        self.update_role_baselines(spy_reward, civilian_avg)
        adjusted_rewards = []
        for i, reward in enumerate(rewards):
            player_id = i + 1
            if player_id == spy_player:
                adjusted_reward = reward - self.b_spy
            else:
                adjusted_reward = reward - self.b_civ
            adjusted_rewards.append(adjusted_reward)
        return adjusted_rewards


_ROLE_BASELINES = _RoleBaselines(alpha=0.9)


def _simulate_votes(game_data: dict[str, Any], num_players: int) -> list[dict[str, Any]]:
    game_seed = hash(game_data.get("game_id", "")) % 1000000
    random.seed(game_seed)
    simulated_votes = []
    for _ in range(1, num_players + 1):
        voted_player = random.randint(1, num_players)
        simulated_votes.append({"voted_spy": voted_player, "reasoning": "Simulated vote for no-spy clue training"})
    return simulated_votes


def _pick_bad_player_from_votes(votes: list[dict[str, Any]], num_players: int) -> int:
    counts = {player_id: 0 for player_id in range(1, num_players + 1)}
    for vote_info in votes:
        voted = vote_info.get("voted_spy")
        if isinstance(voted, int) and 1 <= voted <= num_players:
            counts[voted] += 1
    max_votes = max(counts.values()) if counts else 0
    candidates = [pid for pid, c in counts.items() if c == max_votes]
    return min(candidates) if candidates else 1


def _calculate_strategic_clue_rewards(
    game_data: dict[str, Any],
    all_votes: list[dict[str, Any]],
    num_players: int,
    beta: float = 0.1,
    lambda_param: float = 0.1,
    apply_role_advantage: bool = True,
) -> dict[str, Any]:
    spy_player = game_data["spy_player"]
    vote_counts = {player_id: 0 for player_id in range(1, num_players + 1)}
    total_god_voters = 0
    na_votes = 0
    invalid_votes = 0

    for vote_info in all_votes:
        if vote_info and "voted_spy" in vote_info:
            voted_spy = vote_info["voted_spy"]
            if voted_spy == "N/A" or voted_spy == "NA":
                na_votes += 1
            elif isinstance(voted_spy, int) and 1 <= voted_spy <= num_players:
                vote_counts[voted_spy] += 1
                total_god_voters += 1
            else:
                invalid_votes += 1
        else:
            invalid_votes += 1

    v_U = vote_counts[spy_player]
    civilian_votes = [vote_counts[i] for i in range(1, num_players + 1) if i != spy_player]
    v_C_bar = sum(civilian_votes) / len(civilian_votes) if civilian_votes else 0.0
    delta_psi = v_U - v_C_bar

    rewards = [0.0] * num_players
    for player_id in range(1, num_players + 1):
        player_idx = player_id - 1
        if player_id == spy_player:
            rewards[player_idx] = -beta * delta_psi
        else:
            num_civilians = num_players - 1
            shared_reward = beta * delta_psi / num_civilians
            v_C_j = vote_counts[player_id]
            individual_suspicion = -lambda_param * (v_C_j - v_C_bar)
            rewards[player_idx] = shared_reward + individual_suspicion

    metrics = {
        "pseudo_bad_player": spy_player,
        "spy_raw_reward": rewards[spy_player - 1],
        "civilian_raw_reward_mean": (
            sum([rewards[i] for i in range(num_players) if i != spy_player - 1]) / (num_players - 1)
            if num_players > 1
            else 0.0
        ),
        "suspicion_potential_psi": delta_psi,
        "bad_votes_received": v_U,
        "other_votes_avg": v_C_bar,
        "total_valid_votes": total_god_voters,
        "na_votes": na_votes,
        "invalid_votes": invalid_votes,
    }

    if apply_role_advantage:
        rewards = _ROLE_BASELINES.apply_unified_role_advantage_adjustment(rewards, spy_player)
        metrics["spy_adjusted_reward"] = rewards[spy_player - 1]

    return {"rewards": rewards, "metrics": metrics}


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    max_debug_prints: int = 2,
) -> dict[str, Any]:
    del data_source, solution_str, ground_truth, max_debug_prints
    extra_info = extra_info or {}
    training_phase = str(extra_info.get("training_phase", "clue"))
    if training_phase != "clue":
        return {"score": 0.0, "phase": "decision_disabled"}

    step = extra_info.get("step")
    game_data = dict(extra_info.get("game_data", {}) or {})
    game_id = str(game_data.get("game_id", ""))
    sample_idx = extra_info.get("sample_idx")

    global _CLUE_REWARD_CACHE_STEP
    if _CLUE_REWARD_CACHE_STEP != step:
        _CLUE_REWARD_CACHE_STEP = step
        _CLUE_REWARD_CACHE.clear()

    num_players = int(game_data.get("num_players", 3))
    clue_player_id = int(extra_info.get("clue_player_id", 1))
    cache_key = (game_id or f"step_{step}_sample_{sample_idx}", step)
    cached = _CLUE_REWARD_CACHE.get(cache_key)

    if cached is None:
        decision_responses = extra_info.get("decision_responses", [])
        votes = []
        for response in decision_responses:
            vote = _extract_vote(response)
            if vote is not None:
                votes.append(vote)
        if not votes:
            votes = _simulate_votes(game_data, num_players)

        bad_player = _pick_bad_player_from_votes(votes, num_players)
        game_data["spy_player"] = bad_player
        clue_result = _calculate_strategic_clue_rewards(
            game_data=game_data,
            all_votes=votes,
            num_players=num_players,
            beta=0.1,
            lambda_param=0.1,
            apply_role_advantage=True,
        )
        cached = {
            "rewards": clue_result["rewards"],
            "metrics": clue_result.get("metrics", {}),
        }
        _CLUE_REWARD_CACHE[cache_key] = cached

    player_rewards = cached["rewards"]
    reward = player_rewards[clue_player_id - 1]
    return {
        "score": reward,
        "phase": "clue",
        "clue_player_id": clue_player_id,
        "clue_reward_metrics": cached.get("metrics", {}),
    }
