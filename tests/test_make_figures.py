# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Prajjwal Chittori
import os

import matplotlib

matplotlib.use("Agg")  # headless

import pandas as pd  # noqa: E402

from scripts.make_figures import divergence_table, make_contrast_figure  # noqa: E402

DF = pd.DataFrame(
    [
        {
            "model": "m1",
            "press": "none",
            "compression_ratio": 0.0,
            "task": "tool_call",
            "metric": "name_substring",
            "value": 1.0,
        },
        {
            "model": "m1",
            "press": "none",
            "compression_ratio": 0.0,
            "task": "tool_call",
            "metric": "schema_valid",
            "value": 1.0,
        },
        {
            "model": "m1",
            "press": "SnapKV",
            "compression_ratio": 0.5,
            "task": "tool_call",
            "metric": "name_substring",
            "value": 0.95,
        },
        {
            "model": "m1",
            "press": "SnapKV",
            "compression_ratio": 0.5,
            "task": "tool_call",
            "metric": "schema_valid",
            "value": 0.30,
        },
    ]
)


def test_contrast_figure_written(tmp_path):
    out = tmp_path / "fig.png"
    make_contrast_figure(DF, "tool_call", "name_substring", "schema_valid", str(out))
    assert os.path.getsize(out) > 0


def test_divergence_table_shows_gap():
    tbl = divergence_table(DF, "name_substring", "schema_valid")
    row = tbl[tbl.model == "m1"].iloc[0]
    assert row.standard_retention > row.exec_retention
