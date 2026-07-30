# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from .agent_loop import AgentLoopBase, AgentLoopManager, AgentLoopWorker, AsyncLLMServerManager
from .govreport_parallel_agent_loop import GovReportParallelClueDecisionAgentLoop
from .govreport_two_player_agent_loop import GovReportTwoPlayerClueDecisionAgentLoop
from .nemotron_cc_math_no_spy_clue_agent_loop import NemotronCCMathNoSpyClueAgentLoop
from .nemotron_cc_math_parallel_agent_loop import NemotronCCMathParallelClueDecisionAgentLoop
from .nemotron_cc_math_two_player_agent_loop import NemotronCCMathTwoPlayerClueDecisionAgentLoop
from .single_turn_agent_loop import SingleTurnAgentLoop
from .tool_agent_loop import ToolAgentLoop
from .writingprompts_parallel_agent_loop import WritingPromptsParallelClueDecisionAgentLoop
from .writingprompts_two_player_agent_loop import WritingPromptsTwoPlayerClueDecisionAgentLoop

_ = [
    SingleTurnAgentLoop,
    ToolAgentLoop,
    NemotronCCMathParallelClueDecisionAgentLoop,
    NemotronCCMathTwoPlayerClueDecisionAgentLoop,
    NemotronCCMathNoSpyClueAgentLoop,
    WritingPromptsTwoPlayerClueDecisionAgentLoop,
    WritingPromptsParallelClueDecisionAgentLoop,
    GovReportTwoPlayerClueDecisionAgentLoop,
    GovReportParallelClueDecisionAgentLoop,
]

__all__ = ["AgentLoopBase", "AgentLoopManager", "AsyncLLMServerManager", "AgentLoopWorker"]
