# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Prajjwal Chittori

"""Generation runner — consumes `kvpress` as a dependency, does not patch it.

This drives KVPress's `kv-press-text-generation` transformers pipeline with a chosen press and
compression ratio, fills in `predicted_answer` for each row, and hands the DataFrame to the scorers.
Because the catalog/prompt is shared across rows, rows are grouped by `context` so a prefill press
compresses the context once and answers every question from the compressed cache — the regime where
structured-output degradation actually shows up.
"""

from typing import Optional

import pandas as pd


def build_press(press_name: str, compression_ratio: float):
    """Instantiate a KVPress press by name, or return None for `none`/empty (no compression)."""
    if not press_name or press_name.lower() == "none":
        return None
    import kvpress

    cls_name = press_name if press_name.endswith("Press") else f"{press_name}Press"
    try:
        press_cls = getattr(kvpress, cls_name)
    except AttributeError as e:  # pragma: no cover - user input
        available = sorted(n for n in dir(kvpress) if n.endswith("Press"))
        raise ValueError(f"Unknown press {press_name!r}. Available: {', '.join(available)}") from e
    return press_cls(compression_ratio=compression_ratio)


def run(
    df: pd.DataFrame,
    model: str,
    press_name: str = "none",
    compression_ratio: float = 0.0,
    device: Optional[str] = None,
    max_context_length: Optional[int] = None,
) -> pd.DataFrame:
    """Generate `predicted_answer` for every row of `df` and return the filled DataFrame.

    `df` must carry the column contract from `datasets.py`. Importing `kvpress` registers both the
    presses and the `kv-press-text-generation` pipeline.
    """
    import kvpress  # noqa: F401  (registers the pipeline + presses as a side effect)
    from transformers import pipeline

    pipe = pipeline("kv-press-text-generation", model=model, device=device)
    press = build_press(press_name, compression_ratio)

    df = df.copy()
    df["predicted_answer"] = None
    for context, group in df.groupby("context"):
        answer_prefix = group["answer_prefix"].iloc[0]
        max_new_tokens = int(group["max_new_tokens"].iloc[0])
        questions = group["question"].tolist()
        output = pipe(
            context,
            questions=questions,
            answer_prefix=answer_prefix,
            press=press,
            max_new_tokens=max_new_tokens,
            max_context_length=max_context_length,
        )
        df.loc[group.index, "predicted_answer"] = output["answers"]
    return df
