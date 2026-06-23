# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Prajjwal Chittori
"""Render the contrast figure(s) and divergence table from a sweep results CSV.

The results CSV has columns: model, press, compression_ratio, task, metric, value.

The contrast figure is the paper's core claim: a 'standard' string metric stays high across
compression ratios while an executable/structured metric collapses. At each ratio we average over
presses; the ratio-0 point is the uncompressed baseline.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def _mean_by_ratio(df, task, metric):
    """Return a (ratio -> mean value) Series for one metric/task, averaged over presses."""
    sub = df[(df.task == task) & (df.metric == metric)]
    return sub.groupby("compression_ratio")["value"].mean().sort_index()


def make_contrast_figure(df, task, standard_metric, exec_metric, out_path):
    """Plot standard vs exec metric against compression ratio, one panel per model. Save a PNG."""
    models = sorted(df[df.task == task]["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), squeeze=False)
    for ax, model in zip(axes[0], models):
        md = df[df.model == model]
        std = _mean_by_ratio(md, task, standard_metric)
        exe = _mean_by_ratio(md, task, exec_metric)
        ax.plot(std.index, std.values, marker="o", label=f"{standard_metric} (string)")
        ax.plot(exe.index, exe.values, marker="s", label=f"{exec_metric} (exec/struct)")
        ax.set_title(model)
        ax.set_xlabel("compression ratio")
        ax.set_ylabel("score")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def divergence_table(df, standard_metric, exec_metric):
    """Per model: retention (value at max ratio / value at ratio 0) for the two metrics."""
    rows = []
    for model in sorted(df["model"].unique()):
        md = df[df.model == model]

        def retention(metric):
            s = md[md.metric == metric].groupby("compression_ratio")["value"].mean().sort_index()
            if len(s) == 0 or s.iloc[0] == 0:
                return float("nan")
            return round(s.iloc[-1] / s.iloc[0], 4)

        rows.append(
            {"model": model, "standard_retention": retention(standard_metric), "exec_retention": retention(exec_metric)}
        )
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description="render contrast figures + divergence table")
    p.add_argument("csv", help="results CSV from `kv-exec-bench sweep`")
    p.add_argument("--out", default="results/figures", help="output directory for PNGs")
    args = p.parse_args(argv)

    df = pd.read_csv(args.csv)
    # tool_call: json_valid (shallow well-formedness anchor) vs schema_valid (real correctness).
    if (df.task == "tool_call").any():
        make_contrast_figure(
            df, "tool_call", "json_valid", "schema_valid", os.path.join(args.out, "contrast_tool_call.png")
        )
        print("\ntool_call divergence (retention at max ratio):")
        print(divergence_table(df[df.task == "tool_call"], "json_valid", "schema_valid").to_string(index=False))
    # code_exec: edit_sim (string anchor) vs pass@1 (executable).
    code_tasks = df[df.task != "tool_call"]["task"].unique()
    for t in code_tasks:
        make_contrast_figure(df, t, "edit_sim", "pass@1", os.path.join(args.out, f"contrast_{t}.png"))
        print(f"\n{t} divergence (retention at max ratio):")
        print(divergence_table(df[df.task == t], "edit_sim", "pass@1").to_string(index=False))


if __name__ == "__main__":
    main()
