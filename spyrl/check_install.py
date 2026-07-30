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
"""Smoke-check that an SpyRL install is complete.

Verifies the runtime deps are importable, that every SpyRL agent loop reached verl's registry,
and that the environment/reward modules the launch scripts point at exist. Run it after
``setup.sh`` (or any time training fails with a confusing import error):

    python spyrl/check_install.py
"""

import importlib
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

AGENT_LOOPS = [
    "govreport_parallel",
    "govreport_two_player",
    "writingprompts_parallel",
    "writingprompts_two_player",
    "nemotron_cc_math_parallel",
    "nemotron_cc_math_two_player",
    "nemotron_cc_math_no_spy_clue",
]

MODULE_FILES = [
    "verl/utils/dataset/govreport_spotdiff_parallel_dataset.py",
    "verl/utils/dataset/govreport_spotdiff_dataset.py",
    "verl/utils/dataset/writingprompts_spotdiff_parallel_dataset.py",
    "verl/utils/dataset/writingprompts_spotdiff_dataset.py",
    "verl/utils/dataset/nemotron_cc_math_spotdiff_parallel_dataset.py",
    "verl/utils/dataset/nemotron_cc_math_spotdiff_dataset.py",
    "verl/utils/dataset/nemotron_cc_math_no_spy_clue_dataset.py",
    "verl/utils/dataset/dclm_baseline_spotdiff_dataset.py",
    "verl/utils/spyrl_reward.py",
    "verl/utils/spyrl_no_spy_reward.py",
]

REQUIRED_PACKAGES = ["torch", "transformers", "datasets", "ray", "hydra", "omegaconf", "tensordict"]
OPTIONAL_PACKAGES = ["vllm", "flash_attn"]


def main() -> int:
    failures: list[str] = []

    print("== python ==")
    print(f"   {sys.version.split()[0]} at {sys.executable}")

    print("== required packages ==")
    for name in REQUIRED_PACKAGES:
        if importlib.util.find_spec(name) is None:
            print(f"   MISSING  {name}")
            failures.append(f"missing required package: {name}")
        else:
            print(f"   ok       {name}")

    print("== optional packages ==")
    for name in OPTIONAL_PACKAGES:
        found = importlib.util.find_spec(name) is not None
        print(f"   {'ok      ' if found else 'absent  '} {name}")
    if importlib.util.find_spec("vllm") is None:
        failures.append("vllm is not installed -- the rollout engine used by every recipe script")

    print("== spyrl modules on disk ==")
    for rel in MODULE_FILES:
        path = REPO_ROOT / rel
        print(f"   {'ok      ' if path.exists() else 'MISSING '} {rel}")
        if not path.exists():
            failures.append(f"missing file: {rel}")

    print("== agent loop registry ==")
    try:
        import verl.experimental.agent_loop  # noqa: F401  -- importing registers the loops
        from verl.experimental.agent_loop.agent_loop import _agent_loop_registry

        for name in AGENT_LOOPS:
            registered = name in _agent_loop_registry
            print(f"   {'ok      ' if registered else 'MISSING '} {name}")
            if not registered:
                failures.append(f"agent loop not registered: {name}")
    except Exception as exc:  # noqa: BLE001 -- report any import failure verbatim
        print(f"   FAILED to import verl.experimental.agent_loop: {exc}")
        failures.append(f"cannot import verl.experimental.agent_loop: {exc}")

    print()
    if failures:
        print(f"FAILED ({len(failures)} problem(s)):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("All good. Launch a run with e.g. bash spyrl/train_summarization.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
