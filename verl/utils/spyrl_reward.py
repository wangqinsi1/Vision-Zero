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
"""Reward function for SpyRL, the reference implementation of RLSVR (Reinforcement Learning with
Self-Verifiable Rewards).

This module implements the two coupled reward signals of the SpyRL game and is shared by
every domain (summarization, creative writing, mathematical reasoning):

* ``training_phase == "decision"`` -- the *detection* stage reward. Each detector votes for
  the player it believes is the spy; the vote is checked against the environment
  assigned identity, so the reward is fully verifiable.
* ``training_phase == "clue"`` -- the *performing* stage reward. Players are rewarded by how
  few suspicion votes they attracted relative to their peers, following the zero-sum
  formulation of Eq. (5) in the paper. Role-Advantage Estimation (RAE) removes the reward
  bias induced by the information asymmetry between the spy and the civilians.

Wire it up with ``custom_reward_function.path=<this file>`` and
``custom_reward_function.name=compute_score``.
"""

import os
import random
import re
from typing import Any

import torch

_PRINT_COUNT = 0
_CLUE_REWARD_CACHE: dict[tuple[str, int | None], dict[str, Any]] = {}
_CLUE_REWARD_CACHE_STEP: int | None = None
_DECISION_REWARD_CACHE: dict[tuple[str, int | None], dict[str, Any]] = {}
_DECISION_REWARD_CACHE_STEP: int | None = None
_FIRST_SAMPLE_BY_STEP: dict[int, Any] = {}
_LOGGED_STEPS: set[int] = set()
_OUTPUT_PATH = os.getenv("SPYRL_ROLLOUT_LOG", os.path.join(os.getcwd(), "outputs", "spyrl_rollout_log.txt"))
_OUTPUT_INITED = False


def _is_rank0() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def _extract_thinking(text: str) -> str:
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        return think_match.group(1).strip()
    return ""


def _extract_answer_content(text: str) -> str:
    boxed_match = re.search(r"\\\\?boxed\{(.*?)\}", text, re.DOTALL)
    if boxed_match:
        return boxed_match.group(1).strip()

    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()

    answer_start_match = re.search(r"<answer>\s*(.*)", text, re.DOTALL)
    if answer_start_match:
        answer_content = answer_start_match.group(1).strip()
        answer_content = re.split(r"</answer>|<think>|<answer>", answer_content)[0].strip()
        return answer_content

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


def _append_output(text: str) -> None:
    if not _is_rank0():
        return
    _ensure_output_file()
    with open(_OUTPUT_PATH, "a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def _ensure_output_file() -> None:
    global _OUTPUT_INITED
    if not _is_rank0() or _OUTPUT_INITED:
        return
    os.makedirs(os.path.dirname(_OUTPUT_PATH), exist_ok=True)
    if not os.path.exists(_OUTPUT_PATH):
        with open(_OUTPUT_PATH, "w", encoding="utf-8") as handle:
            handle.write("OUTPUT LOG START\n")
    _OUTPUT_INITED = True


def _compute_decision_details(solution_str: str, spy_player: int, decision_sample_index: int) -> dict[str, Any]:
    thinking_content = _extract_thinking(solution_str)
    answer_content = _extract_answer_content(solution_str)
    fmt_ok = bool(thinking_content and answer_content and len(thinking_content) > 10)

    vote_info = _extract_vote(solution_str)
    if vote_info and "voted_spy" in vote_info:
        voted_spy = vote_info["voted_spy"]
        if decision_sample_index == spy_player:
            if voted_spy == spy_player:
                accuracy_component = -1.0
            elif voted_spy == "N/A":
                accuracy_component = 0.2
            else:
                accuracy_component = 0.5
        else:
            if voted_spy == spy_player:
                accuracy_component = 1.0
            elif voted_spy == "N/A":
                accuracy_component = -0.5
            else:
                accuracy_component = -1.0
    else:
        voted_spy = None
        accuracy_component = -1.0

    lambda_fmt = 0.3
    beta_acc = 1.2
    gamma = 0.99

    r_fmt = lambda_fmt * (2 * int(fmt_ok) - 1)
    r_acc = accuracy_component * beta_acc
    shaped_term = gamma * (1.0 if accuracy_component > 0 else 0.0)
    reward = r_fmt + r_acc + shaped_term

    return {
        "reward": reward,
        "fmt_ok": fmt_ok,
        "voted_spy": voted_spy,
        "accuracy_component": accuracy_component,
    }


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
        # Match VLM-R1: use global mean of current-game rewards across processes.
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

    def apply_role_advantage_without_update(self, rewards: list[float], spy_player: int) -> list[float]:
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
    spy_player = game_data["spy_player"]
    simulated_votes = []
    for player_id in range(1, num_players + 1):
        if player_id == spy_player:
            possible_targets = [p for p in range(1, num_players + 1) if p != spy_player]
            voted_spy = random.choice(possible_targets)
        else:
            if random.random() < 0.6:
                voted_spy = spy_player
            else:
                possible_targets = [p for p in range(1, num_players + 1) if p != player_id]
                voted_spy = random.choice(possible_targets)
        simulated_votes.append({"voted_spy": voted_spy, "reasoning": "Simulated vote for clue-only training"})
    return simulated_votes


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
        "spy_raw_reward": rewards[spy_player - 1],
        "civilian_raw_reward_mean": (
            sum([rewards[i] for i in range(num_players) if i != spy_player - 1]) / (num_players - 1)
            if num_players > 1
            else 0.0
        ),
        "suspicion_potential_psi": delta_psi,
        "spy_votes_received": v_U,
        "civilian_votes_avg": v_C_bar,
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
    del data_source
    global _PRINT_COUNT

    extra_info = extra_info or {}
    training_phase = str(extra_info.get("training_phase", "decision"))
    step = extra_info.get("step")
    game_data = extra_info.get("game_data", {})
    game_id = str(game_data.get("game_id", ""))
    sample_idx = extra_info.get("sample_idx")

    if _is_rank0() and _PRINT_COUNT < max_debug_prints:
        _PRINT_COUNT += 1
        print("[GPU0 PROMPT]")
        print(extra_info.get("prompt_text", ""))
        print("[GPU0 RESPONSE]")
        print(solution_str)
        print(f"[TRAINING PHASE] {training_phase}")

    if _is_rank0() and step is not None:
        if step not in _FIRST_SAMPLE_BY_STEP:
            _FIRST_SAMPLE_BY_STEP[step] = sample_idx if sample_idx is not None else "__unknown__"
    should_log_sample = (
        _is_rank0()
        and step is not None
        and (
            sample_idx is None
            or _FIRST_SAMPLE_BY_STEP.get(step) == sample_idx
            or _FIRST_SAMPLE_BY_STEP.get(step) == "__unknown__"
        )
        and step not in _LOGGED_STEPS
    )
    _ensure_output_file()

    if training_phase == "clue":
        global _CLUE_REWARD_CACHE_STEP
        if _CLUE_REWARD_CACHE_STEP != step:
            _CLUE_REWARD_CACHE_STEP = step
            _CLUE_REWARD_CACHE.clear()
        num_players = int(game_data.get("num_players", 2))
        num_rounds = int(game_data.get("num_rounds", 1))
        clue_player_id = int(extra_info.get("clue_player_id", 1))
        clue_round_num = int(extra_info.get("clue_round_num", 1))
        decision_n = int(extra_info.get("decision_n", 0))
        decision_responses = extra_info.get("decision_responses", [])

        cache_key = (game_id or f"step_{step}_sample_{sample_idx}", step)
        cached = _CLUE_REWARD_CACHE.get(cache_key)
        if cached is None:
            votes = []
            for response in decision_responses:
                votes.append(_extract_vote(response))
            if not votes:
                votes = _simulate_votes(game_data, num_players)

            clue_result = _calculate_strategic_clue_rewards(
                game_data=game_data,
                all_votes=votes,
                num_players=num_players,
                beta=0.1,
                lambda_param=0.1,
                apply_role_advantage=True,
            )
            player_rewards = clue_result["rewards"]
            cached = {
                "rewards": player_rewards,
                "metrics": clue_result.get("metrics", {}),
                "num_players": num_players,
                "num_rounds": num_rounds,
                "decision_count": len(decision_responses),
                "decision_n": decision_n,
            }
            _CLUE_REWARD_CACHE[cache_key] = cached

            if _is_rank0() and clue_player_id == 1 and clue_round_num == 1:
                print(
                    f"[REWARD DEBUG] game={game_id} clues={num_players * num_rounds} "
                    f"decisions={len(decision_responses)}/{decision_n}"
                )
                print(
                    "[REWARD DEBUG] clue player rewards: "
                    + ", ".join([f"P{i + 1}={r:.3f}" for i, r in enumerate(player_rewards)])
                )
            if should_log_sample and clue_player_id == 1 and clue_round_num == 1:
                spy_player = int(game_data.get("spy_player", 1))
                decision_details = []
                for idx, response in enumerate(decision_responses, start=1):
                    details = _compute_decision_details(response, spy_player, idx)
                    decision_details.append(details)
                votes = [d["voted_spy"] for d in decision_details]
                correct_flags = [v == spy_player for v in votes]
                decision_rewards = [d["reward"] for d in decision_details]

                clue_prompts = extra_info.get("clue_prompts", {})
                clue_responses = extra_info.get("clue_responses", {})
                decision_prompt = extra_info.get("decision_prompt", "")

                _append_output("=" * 80)
                _append_output(f"STEP {step} SAMPLE {sample_idx} PHASE clue GAME {game_id}")
                _append_output("CLUES:")
                for round_num in range(1, num_rounds + 1):
                    for player_id in range(1, num_players + 1):
                        clue_prompt = clue_prompts.get((round_num, player_id), "")
                        clue_response = clue_responses.get((round_num, player_id), "")
                        _append_output(f"[CLUE PROMPT] Round {round_num} Player {player_id}\n{clue_prompt}")
                        _append_output(f"[CLUE RESPONSE] Round {round_num} Player {player_id}\n{clue_response}")
                _append_output("[DECISION PROMPT]\n" + decision_prompt)
                for idx, response in enumerate(decision_responses, start=1):
                    _append_output(f"[DECISION RESPONSE {idx}]\n{response}")
                _append_output("[DECISION VOTES] " + ", ".join([f"G{i + 1}={v}" for i, v in enumerate(votes)]))
                _append_output(
                    "[DECISION CORRECT] "
                    + ", ".join([f"G{i + 1}={'Y' if c else 'N'}" for i, c in enumerate(correct_flags)])
                )
                _append_output(
                    "[DECISION REWARDS] " + ", ".join([f"G{i + 1}={r:.3f}" for i, r in enumerate(decision_rewards)])
                )
                _append_output(
                    "[CLUE REWARDS] " + ", ".join([f"P{i + 1}={r:.3f}" for i, r in enumerate(player_rewards)])
                )
                _LOGGED_STEPS.add(step)

        player_rewards = cached["rewards"]
        reward = player_rewards[clue_player_id - 1]

        return {
            "score": reward,
            "phase": "clue",
            "num_rounds": num_rounds,
            "clue_player_id": clue_player_id,
            "clue_reward_metrics": cached.get("metrics", {}),
        }

    spy_player = int(game_data.get("spy_player", ground_truth or 1))
    decision_sample_index = int(extra_info.get("decision_sample_index", 1))
    decision_details = _compute_decision_details(solution_str, spy_player, decision_sample_index)
    reward = decision_details["reward"]
    fmt_ok = decision_details["fmt_ok"]
    voted_spy = decision_details["voted_spy"]
    accuracy_component = decision_details["accuracy_component"]

    global _DECISION_REWARD_CACHE_STEP
    if _DECISION_REWARD_CACHE_STEP != step:
        _DECISION_REWARD_CACHE_STEP = step
        _DECISION_REWARD_CACHE.clear()
    decision_n = int(extra_info.get("decision_n", 0))
    if decision_n > 0:
        cache_key = (game_id or f"step_{step}_sample_{sample_idx}", step)
        cached = _DECISION_REWARD_CACHE.get(cache_key)
        if cached is None:
            cached = {
                "rewards": [None] * decision_n,
                "votes": [None] * decision_n,
                "correct": [None] * decision_n,
                "responses": [None] * decision_n,
                "printed": False,
                "logged_header": False,
                "logged_summary": False,
                "logged_indices": [False] * decision_n,
                "decision_n": decision_n,
            }
            _DECISION_REWARD_CACHE[cache_key] = cached
        if 1 <= decision_sample_index <= decision_n:
            cached["rewards"][decision_sample_index - 1] = reward
            cached["votes"][decision_sample_index - 1] = voted_spy
            cached["correct"][decision_sample_index - 1] = voted_spy == spy_player
            cached["responses"][decision_sample_index - 1] = solution_str
        if _is_rank0() and not cached["printed"] and all(r is not None for r in cached["rewards"]):
            cached["printed"] = True
            print(
                "[REWARD DEBUG] decision rewards: "
                + ", ".join([f"G{i + 1}={r:.3f}" for i, r in enumerate(cached["rewards"])])
            )
        if should_log_sample and step is not None:
            num_players = int(game_data.get("num_players", 2))
            num_rounds = int(game_data.get("num_rounds", 1))
            clue_prompts = extra_info.get("clue_prompts", {})
            clue_responses = extra_info.get("clue_responses", {})
            decision_prompt = extra_info.get("decision_prompt", "")

            if not cached["logged_header"]:
                _append_output("=" * 80)
                _append_output(f"STEP {step} SAMPLE {sample_idx} PHASE decision GAME {game_id}")
                _append_output("CLUES:")
                for round_num in range(1, num_rounds + 1):
                    for player_id in range(1, num_players + 1):
                        clue_prompt = clue_prompts.get((round_num, player_id), "")
                        clue_response = clue_responses.get((round_num, player_id), "")
                        _append_output(f"[CLUE PROMPT] Round {round_num} Player {player_id}\n{clue_prompt}")
                        _append_output(f"[CLUE RESPONSE] Round {round_num} Player {player_id}\n{clue_response}")
                _append_output("[DECISION PROMPT]\n" + decision_prompt)
                cached["logged_header"] = True

            if 1 <= decision_sample_index <= decision_n:
                decision_idx = decision_sample_index - 1
                if not cached["logged_indices"][decision_idx]:
                    cached["logged_indices"][decision_idx] = True
                    _append_output(f"[DECISION RESPONSE {decision_sample_index}]\n{solution_str}")
                    _append_output(f"[DECISION VOTE {decision_sample_index}] {voted_spy}")
                    _append_output(
                        f"[DECISION CORRECT {decision_sample_index}] {'Y' if voted_spy == spy_player else 'N'}"
                    )
                    _append_output(f"[DECISION REWARD {decision_sample_index}] {reward:.3f}")

            if not cached["logged_summary"] and all(r is not None for r in cached["rewards"]):
                cached["logged_summary"] = True
                votes = cached["votes"]
                correct_flags = cached["correct"]
                decision_rewards = cached["rewards"]
                vote_objs = []
                for response in cached["responses"]:
                    vote_objs.append(_extract_vote(response or ""))
                clue_result = _calculate_strategic_clue_rewards(
                    game_data=game_data,
                    all_votes=vote_objs,
                    num_players=num_players,
                    beta=0.1,
                    lambda_param=0.1,
                    apply_role_advantage=False,
                )
                clue_rewards = _ROLE_BASELINES.apply_role_advantage_without_update(clue_result["rewards"], spy_player)
                _append_output("[DECISION VOTES] " + ", ".join([f"G{i + 1}={v}" for i, v in enumerate(votes)]))
                _append_output(
                    "[DECISION CORRECT] "
                    + ", ".join([f"G{i + 1}={'Y' if c else 'N'}" for i, c in enumerate(correct_flags)])
                )
                _append_output(
                    "[DECISION REWARDS] " + ", ".join([f"G{i + 1}={r:.3f}" for i, r in enumerate(decision_rewards)])
                )
                _append_output("[CLUE REWARDS] " + ", ".join([f"P{i + 1}={r:.3f}" for i, r in enumerate(clue_rewards)]))
                _LOGGED_STEPS.add(step)

    return {
        "score": reward,
        "phase": "decision",
        "fmt_ok": fmt_ok,
        "voted_spy": voted_spy,
        "accuracy_component": accuracy_component,
    }
