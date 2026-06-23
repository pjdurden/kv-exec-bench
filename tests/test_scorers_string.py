# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Prajjwal Chittori
import json

import pandas as pd

from kv_exec_bench.scorers import score_code_string, score_tool_call


def test_edit_sim_is_one_for_exact_match():
    df = pd.DataFrame(
        [
            {
                "context": "def f():\n",
                "predicted_answer": "    return 1\n",
                "answer": json.dumps({"canonical_solution": "    return 1\n", "test": "", "entry_point": "f"}),
                "task": "humaneval",
            }
        ]
    )
    out = score_code_string(df)
    assert out["humaneval"]["edit_sim"] == 1.0


def test_edit_sim_drops_for_divergent_output():
    df = pd.DataFrame(
        [
            {
                "context": "def f():\n",
                "predicted_answer": "    raise ValueError('xyz')\n",
                "answer": json.dumps({"canonical_solution": "    return 1\n", "test": "", "entry_point": "f"}),
                "task": "humaneval",
            }
        ]
    )
    out = score_code_string(df)
    assert out["humaneval"]["edit_sim"] < 0.5


def test_name_substring_hits_without_valid_json():
    df = pd.DataFrame(
        [
            {
                "context": "",
                "predicted_answer": "I will use get_weather but emit broken json {oops",
                "answer": json.dumps(
                    {"name": "get_weather", "arguments": {"city": "Paris"}, "schema": {"type": "object"}}
                ),
                "task": "tool_call",
            }
        ]
    )
    out = score_tool_call(df)
    assert out["tool_call"]["name_substring"] == 1.0
    assert out["tool_call"]["json_valid"] == 0.0
