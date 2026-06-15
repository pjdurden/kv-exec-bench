# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Prajjwal Chittori

"""kv-exec-bench: executable-correctness evaluation for generation under KV-cache compression."""

from .datasets import build_code_exec_df, build_tool_call_df
from .scorers import score_code_exec, score_tool_call

__all__ = ["build_code_exec_df", "build_tool_call_df", "score_code_exec", "score_tool_call"]
__version__ = "0.1.0"
