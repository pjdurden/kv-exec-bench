# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Prajjwal Chittori
import json

import pandas as pd

from kv_exec_bench.sweep import run_sweep


def fake_run(df, model, press_name, compression_ratio, **kw):
    """Stand in for runner.run: gold call when uncompressed, garbage when a press is applied."""
    df = df.copy()

    def pred(row):
        if press_name == "none":
            spec = json.loads(row["answer"])
            return json.dumps({"name": spec["name"], "arguments": spec["arguments"]})
        return "{broken"

    df["predicted_answer"] = [pred(r) for _, r in df.iterrows()]
    return df


def test_sweep_emits_tidy_rows_with_baseline():
    rows = run_sweep(models=["m1"], presses=["SnapKV"], ratios=[0.5], tasks=["tool_call"], run_fn=fake_run, limit=2)
    df = pd.DataFrame(rows)
    assert set(df.columns) == {"model", "press", "compression_ratio", "task", "metric", "value"}
    assert (df["press"] == "none").any()  # baseline always present
    assert (df["press"] == "SnapKV").any()
    base = df[(df.press == "none") & (df.metric == "name_match")].value.iloc[0]
    comp = df[(df.press == "SnapKV") & (df.metric == "name_match")].value.iloc[0]
    assert base > comp
