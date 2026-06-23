# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Prajjwal Chittori
"""Grid-sweep driver: run models x presses x ratios x tasks, emit tidy metric rows.

Each emitted row is one (model, press, compression_ratio, task, metric) measurement, ready to drop
into a DataFrame / CSV for plotting. `run_fn` is injectable so tests can stand in for generation.
The uncompressed `none` baseline is always run once per (model, task) regardless of `ratios`.
"""

from .datasets import build_code_exec_df, build_tool_call_df
from .runner import run
from .scorers import score_code_exec, score_code_string, score_tool_call


def _build(task, limit):
    # CPU note uses the network-free inline tool catalog; BFCL is deferred follow-up.
    if task == "tool_call":
        df = build_tool_call_df()
        return df.head(limit) if limit else df
    if task == "code_exec":
        return build_code_exec_df(limit=limit)
    raise ValueError(f"unknown task {task!r}")


def _score(task, df):
    if task == "tool_call":
        return score_tool_call(df)
    # code_exec: structured pass@1 + string anchor (edit_sim), merged per task key.
    merged = {}
    for d in (score_code_exec(df), score_code_string(df)):
        for k, v in d.items():
            merged.setdefault(k, {}).update(v)
    return merged


def _emit(rows, model, press, ratio, scored):
    for task, metrics in scored.items():
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                rows.append(
                    {
                        "model": model,
                        "press": press,
                        "compression_ratio": ratio,
                        "task": task,
                        "metric": metric,
                        "value": float(value),
                    }
                )


def run_sweep(models, presses, ratios, tasks, run_fn=run, limit=None, device=None):
    """Run the full grid and return a list of tidy metric-row dicts."""
    rows = []
    for model in models:
        for task in tasks:
            base_df = _build(task, limit)
            done = run_fn(base_df, model=model, press_name="none", compression_ratio=0.0, device=device)
            _emit(rows, model, "none", 0.0, _score(task, done))
            for press in presses:
                for ratio in ratios:
                    out = run_fn(base_df, model=model, press_name=press, compression_ratio=ratio, device=device)
                    _emit(rows, model, press, ratio, _score(task, out))
    return rows
