# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Prajjwal Chittori

"""Scorer unit tests — no model or GPU needed; predicted_answer is supplied synthetically."""

import json

import pandas as pd
import pytest

from kv_exec_bench import scorers
from kv_exec_bench.datasets import build_tool_call_df


def _tool_row(predicted, name="get_weather", arguments=None, schema=None):
    arguments = arguments if arguments is not None else {"city": "Paris"}
    schema = schema or {
        "type": "object",
        "properties": {"city": {"type": "string"}, "units": {"type": "string", "enum": ["c", "f"]}},
        "required": ["city"],
        "additionalProperties": False,
    }
    return {
        "context": "catalog",
        "question": "q",
        "answer_prefix": "",
        "answer": json.dumps({"name": name, "arguments": arguments, "schema": schema}),
        "task": "tool_call",
        "predicted_answer": predicted,
    }


def test_tool_call_perfect():
    df = pd.DataFrame([_tool_row('{"name": "get_weather", "arguments": {"city": "Paris"}}')])
    s = scorers.score_tool_call(df)["tool_call"]
    assert s == {"json_valid": 1.0, "name_match": 1.0, "schema_valid": 1.0, "args_exact": 1.0}


def test_tool_call_trailing_prose_still_parses():
    df = pd.DataFrame([_tool_row('Sure! {"name": "get_weather", "arguments": {"city": "Paris"}} hope that helps')])
    assert scorers.score_tool_call(df)["tool_call"]["args_exact"] == 1.0


def test_tool_call_malformed_json_scores_zero():
    df = pd.DataFrame([_tool_row('{"name": "get_weather", "arguments": {city: Paris')])
    assert scorers.score_tool_call(df)["tool_call"]["json_valid"] == 0.0


def test_tool_call_wrong_name():
    df = pd.DataFrame([_tool_row('{"name": "get_stock_price", "arguments": {"city": "Paris"}}')])
    s = scorers.score_tool_call(df)["tool_call"]
    assert s["json_valid"] == 1.0 and s["name_match"] == 0.0


def test_tool_call_schema_violation_invalid_enum():
    df = pd.DataFrame([_tool_row('{"name": "get_weather", "arguments": {"city": "Paris", "units": "kelvin"}}')])
    s = scorers.score_tool_call(df)["tool_call"]
    assert s["json_valid"] == 1.0 and s["schema_valid"] == 0.0 and s["args_exact"] == 0.0


def test_tool_call_schema_valid_but_not_exact():
    df = pd.DataFrame([_tool_row('{"name": "get_weather", "arguments": {"city": "London"}}')])
    s = scorers.score_tool_call(df)["tool_call"]
    assert s["schema_valid"] == 1.0 and s["args_exact"] == 0.0


def test_extract_json_object_none_when_absent():
    assert scorers.extract_json_object("no json here") is None


def test_build_tool_call_df_contract():
    df = build_tool_call_df()
    assert set(df.columns) == {"context", "question", "answer_prefix", "answer", "task", "max_new_tokens"}
    assert len(df) > 0
    # gold answers validate against their own schema
    import jsonschema

    for _, row in df.iterrows():
        spec = json.loads(row["answer"])
        jsonschema.validate(spec["arguments"], spec["schema"])


# --- code_exec ------------------------------------------------------------------------------------

_HUMANEVAL_LIKE = {
    "context": 'def add(a, b):\n    """Return a+b."""\n',
    "question": "",
    "answer_prefix": "",
    "answer": json.dumps(
        {"test": "def check(f):\n    assert f(1, 2) == 3\n    assert f(-1, 1) == 0\n", "entry_point": "add"}
    ),
    "task": "humaneval",
}


def _code_row(completion):
    return {**_HUMANEVAL_LIKE, "predicted_answer": completion}


def test_code_exec_requires_flag(monkeypatch):
    monkeypatch.delenv(scorers.EXEC_ENV_FLAG, raising=False)
    df = pd.DataFrame([_code_row("    return a + b\n")])
    with pytest.raises(RuntimeError):
        scorers.score_code_exec(df)


def test_code_exec_pass_and_fail(monkeypatch):
    monkeypatch.setenv(scorers.EXEC_ENV_FLAG, "1")
    df = pd.DataFrame([_code_row("    return a + b\n"), _code_row("    return a - b\n")])
    s = scorers.score_code_exec(df)["humaneval"]
    assert s == {"pass@1": 0.5, "passed": 1, "total": 2}


def test_run_program_timeout_is_caught():
    # an infinite loop must be caught by the timeout/rlimit as a fail, not hang the suite
    assert scorers.run_program("while True:\n    pass\n", timeout=2) is False
    assert scorers.run_program("import sys\nsys.exit(0)\n", timeout=2) is True
