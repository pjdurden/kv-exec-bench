# kv-exec-bench Preprint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take the already-built `~/kv-exec-bench` harness, run one real GPU grid sweep, produce the contrast figure, write a 4-6 page preprint, post it to arXiv, push the repo, and close the project for good.

**Architecture:** Add the missing measurement pieces (a string-overlap "standard" anchor metric, a BFCL tool dataset, a grid-sweep driver, a figure generator) on top of the existing scorers/runner/datasets. Run the sweep on Modal GPUs. Aggregate to one tidy CSV, render the contrast figures, drop them into a LaTeX paper, publish, archive.

**Tech Stack:** Python 3.12, pandas, jsonschema, kvpress (+ transformers<5.3, torch), HuggingFace datasets, Modal (serverless GPU), matplotlib, LaTeX (arXiv).

## Global Constraints

- License: Apache-2.0; SPDX header `# SPDX-License-Identifier: Apache-2.0` + `# Copyright (c) 2026 Prajjwal Chittori` on every new source file (match existing files).
- Solo author. **Never name the employer (ether.fi / EtherFi) anywhere** in code, repo, paper, or arXiv metadata.
- kvpress pins `transformers<5.3`; the `[run]` extra and all generation live in that env. Scorer/dataset/figure code stays light (`pandas`, `jsonschema`, `matplotlib`) and must import without torch.
- `code_exec` executes model output: only ever behind `KV_EXEC_BENCH_ALLOW_CODE_EXECUTION=1`, on throwaway/CI/GPU hardware.
- Dataset column contract is fixed: `context, question, answer_prefix, answer, task, max_new_tokens`. Every dataset builder produces exactly these columns.
- Compute: CPU-only on the local box (AMD Ryzen 5 5500U, 6c/12t, AVX2, 14 GiB RAM, no GPU). No GPU rental, $0 spend. The exact hardware string is disclosed verbatim in the paper.
- Git: `~/kv-exec-bench` is not yet a git repo. Local commits are fine; **never push unsigned commits** — leave the push for the user to sign and push manually.
- No em dashes in any prose intended to be posted/pasted externally (paper abstract blurbs, arXiv comments, README marketing lines).

---

## File Structure

Existing (do not rewrite, only extend where noted):
- `kv_exec_bench/scorers.py` — add `score_code_string`, extend `score_tool_call` with a `name_substring` anchor.
- `kv_exec_bench/datasets.py` — add `build_bfcl_df`, extend `build_code_exec_df` to carry `canonical_solution`.
- `kv_exec_bench/runner.py` — unchanged (reused as-is).
- `kv_exec_bench/cli.py` — add a `sweep` subcommand.

New:
- `kv_exec_bench/sweep.py` — grid driver: iterate models x presses x ratios x datasets, score, emit tidy rows.
- `scripts/make_figures.py` — read results CSV, render contrast figures + a divergence table.
- `modal_app.py` — Modal GPU app that installs `.[run]`, runs the sweep, writes `results.csv` to a Modal volume.
- `data/bfcl/` — downloaded BFCL `simple` + `multiple` JSON (user-fetched; gitignored).
- `tests/test_scorers_string.py`, `tests/test_datasets_bfcl.py`, `tests/test_sweep.py`, `tests/test_make_figures.py`.
- `paper/main.tex`, `paper/refs.bib`, `paper/figures/` — the preprint.
- `results/results.csv`, `results/figures/` — sweep outputs (committed so the paper is reproducible-on-paper).

---

## Task 1: String-overlap "standard" anchor scorers

The paper's claim is "standard string metrics stay flat while exec-correctness collapses." Today the harness only has exec/structured metrics. This task adds the cheap string anchors that form the *other line* in the contrast figure: edit-similarity for code, and a name-substring hit for tool calls (lenient, does not require valid JSON).

**Files:**
- Modify: `kv_exec_bench/scorers.py`
- Test: `tests/test_scorers_string.py`

**Interfaces:**
- Consumes: a scored DataFrame with columns `context, predicted_answer, answer, task`. For code rows, `answer` JSON now also carries `canonical_solution` (added in Task 2; for this task's tests, supply it inline).
- Produces:
  - `score_code_string(df: pd.DataFrame) -> dict` — per task: `{"edit_sim": float}`, mean over rows of `difflib.SequenceMatcher(None, predicted, canonical).ratio()`.
  - `score_tool_call(df)` gains a fifth key per task: `name_substring` (fraction of rows where the gold tool name appears as a substring of `predicted_answer`, regardless of JSON validity).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scorers_string.py
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Prajjwal Chittori
import json
import pandas as pd
from kv_exec_bench.scorers import score_code_string, score_tool_call


def test_edit_sim_is_one_for_exact_match():
    df = pd.DataFrame([{
        "context": "def f():\n",
        "predicted_answer": "    return 1\n",
        "answer": json.dumps({"canonical_solution": "    return 1\n",
                              "test": "", "entry_point": "f"}),
        "task": "humaneval",
    }])
    out = score_code_string(df)
    assert out["humaneval"]["edit_sim"] == 1.0


def test_edit_sim_drops_for_divergent_output():
    df = pd.DataFrame([{
        "context": "def f():\n",
        "predicted_answer": "    raise ValueError('xyz')\n",
        "answer": json.dumps({"canonical_solution": "    return 1\n",
                              "test": "", "entry_point": "f"}),
        "task": "humaneval",
    }])
    out = score_code_string(df)
    assert out["humaneval"]["edit_sim"] < 0.5


def test_name_substring_hits_without_valid_json():
    df = pd.DataFrame([{
        "context": "",
        "predicted_answer": "I will use get_weather but emit broken json {oops",
        "answer": json.dumps({"name": "get_weather", "arguments": {"city": "Paris"},
                              "schema": {"type": "object"}}),
        "task": "tool_call",
    }])
    out = score_tool_call(df)
    assert out["tool_call"]["name_substring"] == 1.0
    assert out["tool_call"]["json_valid"] == 0.0  # the JSON really is broken
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/kv-exec-bench && pip install -e ".[dev]" && pytest tests/test_scorers_string.py -v`
Expected: FAIL — `score_code_string` not defined; `name_substring` key missing.

- [ ] **Step 3: Implement**

In `kv_exec_bench/scorers.py`, add `import difflib` at the top, then add:

```python
def score_code_string(df: pd.DataFrame) -> dict:
    """String-overlap anchor for code: edit-similarity of completion vs canonical solution.

    This is the 'standard metric' baseline for the contrast figure. It runs no code and needs no
    GPU; it reads `canonical_solution` from the row's `answer` JSON.
    """
    scores = {}
    for task, df_task in df.groupby("task"):
        sims = []
        for _, row in df_task.iterrows():
            canonical = json.loads(row["answer"]).get("canonical_solution", "")
            pred = row["predicted_answer"] or ""
            sims.append(difflib.SequenceMatcher(None, pred, canonical).ratio())
        scores[str(task)] = {"edit_sim": round(sum(sims) / len(sims), 4)}
    return scores
```

In `score_tool_call`, initialize the aggregate dict with the extra key and increment it before the JSON-extraction guard:

```python
        agg = {"json_valid": 0, "name_match": 0, "schema_valid": 0,
               "args_exact": 0, "name_substring": 0}
        n = len(df_task)
        for _, row in df_task.iterrows():
            spec = json.loads(row["answer"])
            pred_text = row["predicted_answer"] or ""
            agg["name_substring"] += int(spec["name"] in pred_text)
            call = extract_json_object(pred_text)
            if not isinstance(call, dict):
                continue
            # ... existing json_valid / name_match / schema_valid / args_exact logic unchanged ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_scorers_string.py tests/test_scorers.py -v`
Expected: PASS (new tests pass; existing scorer tests still pass).

- [ ] **Step 5: Commit**

```bash
git add kv_exec_bench/scorers.py tests/test_scorers_string.py
git commit -m "feat: add string-overlap anchor metrics (edit_sim, name_substring)"
```

---

## Task 2: canonical_solution for code (BFCL deferred)

> **REVISED (CPU-only):** The BFCL builder is **deferred** to post-publication follow-up — the CPU
> note uses the network-free inline tool catalog. The ONLY required part of this task now is the
> one-line `build_code_exec_df` extension to carry `canonical_solution` (needed by Task 1's
> `edit_sim` anchor on the real code sweep). Skip the BFCL steps; do Step 3's `canonical_solution`
> edit and a quick test. The full BFCL spec below is kept for the follow-up.

Carry HumanEval's `canonical_solution` so Task 1's anchor works on real data; (deferred) replace the
inline catalog with BFCL for a future headline.

**Files:**
- Modify: `kv_exec_bench/datasets.py`
- Test: `tests/test_datasets_bfcl.py`

**Interfaces:**
- Consumes: BFCL `simple`/`multiple` records (official gorilla JSON) as a list of dicts, each with `question`, `function` (list of `{name, description, parameters}`), and a paired ground-truth `{func_name: {arg: [acceptable_values...]}}`. The user downloads these to `data/bfcl/`.
- Produces:
  - `bfcl_records_to_df(records, ground_truths) -> pd.DataFrame` — pure converter (no IO), 6-column contract. `context` = rendered function catalog (reuse `render_catalog`-style formatting). `answer` JSON = `{name, arguments, schema}` where `arguments` takes the *first* acceptable value per arg and `schema` is the matching function's `parameters`.
  - `build_bfcl_df(data_dir="data/bfcl", categories=("simple","multiple"), limit=None) -> pd.DataFrame` — thin loader that reads the JSON files and calls the converter.
  - `build_code_exec_df` extended: `answer` JSON now also includes `"canonical_solution": r["canonical_solution"]`.

- [ ] **Step 1: Write the failing test** (pure converter, no network)

```python
# tests/test_datasets_bfcl.py
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Prajjwal Chittori
import json
from kv_exec_bench.datasets import bfcl_records_to_df, COLUMNS

RECORD = {
    "id": "simple_0",
    "question": "What's the weather in Paris?",
    "function": [{
        "name": "get_weather",
        "description": "Get weather for a city.",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"]},
    }],
}
GT = {"get_weather": {"city": ["Paris"]}}


def test_converter_produces_column_contract():
    df = bfcl_records_to_df([RECORD], [GT])
    assert list(df.columns) == COLUMNS
    assert len(df) == 1


def test_converter_extracts_gold_name_args_schema():
    df = bfcl_records_to_df([RECORD], [GT])
    ans = json.loads(df.iloc[0]["answer"])
    assert ans["name"] == "get_weather"
    assert ans["arguments"] == {"city": "Paris"}        # first acceptable value
    assert ans["schema"]["required"] == ["city"]         # matching function's schema
    assert df.iloc[0]["task"] == "tool_call"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_datasets_bfcl.py -v`
Expected: FAIL — `bfcl_records_to_df` not defined.

- [ ] **Step 3: Implement**

In `kv_exec_bench/datasets.py` add:

```python
import os

BFCL_CATALOG_HEADER = (
    "You are a function-calling assistant with access to the following tools. For the user's "
    'request, respond with ONLY a single JSON object {"name": <tool name>, "arguments": {...}} and '
    "nothing else.\n\nTools:\n"
)


def _render_functions(functions) -> str:
    lines = [BFCL_CATALOG_HEADER]
    for i, fn in enumerate(functions, 1):
        lines.append(f"{i}. {fn['name']} — {fn.get('description','')}\n"
                     f"   parameters (JSON Schema): {json.dumps(fn['parameters'])}\n")
    return "".join(lines)


def bfcl_records_to_df(records, ground_truths) -> pd.DataFrame:
    """Convert BFCL (record, ground_truth) pairs to the 6-column contract. Pure, no IO."""
    rows = []
    for rec, gt in zip(records, ground_truths):
        functions = rec["function"]
        gold_name = next(iter(gt))
        gold_args = {arg: vals[0] for arg, vals in gt[gold_name].items()}
        schema = next(f["parameters"] for f in functions if f["name"] == gold_name)
        rows.append({
            "context": _render_functions(functions),
            "question": "\nUser request: " + rec["question"] + "\n",
            "answer_prefix": "",
            "answer": json.dumps({"name": gold_name, "arguments": gold_args, "schema": schema}),
            "task": "tool_call",
            "max_new_tokens": TOOL_CALL_MAX_NEW_TOKENS,
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def build_bfcl_df(data_dir="data/bfcl", categories=("simple", "multiple"), limit=None) -> pd.DataFrame:
    """Load BFCL question + ground-truth JSON from data_dir and build the tool_call DataFrame.

    Expects, per category, gorilla's `BFCL_v3_<cat>.json` and `possible_answer/BFCL_v3_<cat>.json`
    (one JSON object per line). Download from the gorilla repo's berkeley-function-call-leaderboard
    data directory.
    """
    records, gts = [], []
    for cat in categories:
        q_path = os.path.join(data_dir, f"BFCL_v3_{cat}.json")
        a_path = os.path.join(data_dir, "possible_answer", f"BFCL_v3_{cat}.json")
        with open(q_path) as qf, open(a_path) as af:
            for ql, al in zip(qf, af):
                rec = json.loads(ql)
                # BFCL "question" is a list of chat turns; take the first user turn's content.
                q = rec["question"]
                if isinstance(q, list):
                    q = q[0][0]["content"] if isinstance(q[0], list) else q[0]["content"]
                records.append({"function": rec["function"], "question": q})
                gts.append(json.loads(al)["ground_truth"])
    if limit is not None:
        records, gts = records[:limit], gts[:limit]
    return bfcl_records_to_df(records, gts)
```

Extend `build_code_exec_df`'s row `answer` to include the canonical solution:

```python
                "answer": json.dumps({"test": r["test"], "entry_point": r["entry_point"],
                                      "canonical_solution": r["canonical_solution"]}),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_datasets_bfcl.py -v`
Expected: PASS.

> **Note on BFCL format drift / fallback:** the official BFCL schema shifts between versions. If
> `build_bfcl_df`'s file parsing does not match the version you downloaded, fix the field access in
> the loader only (the pure converter and its test are the stable contract). If BFCL integration
> exceeds ~half a day, fall back to a scaled inline catalog (expand `TOOLS` to ~60 tools) and note
> BFCL as the headline follow-up in the paper. Do not let this task block the sweep.

- [ ] **Step 5: Commit**

```bash
git add kv_exec_bench/datasets.py tests/test_datasets_bfcl.py
git commit -m "feat: BFCL tool dataset builder + carry HumanEval canonical_solution"
```

---

## Task 3: Grid-sweep driver

One function that runs the whole experiment matrix and returns tidy rows ready for a CSV. This is what Modal calls.

**Files:**
- Create: `kv_exec_bench/sweep.py`
- Modify: `kv_exec_bench/cli.py` (add `sweep` subcommand)
- Test: `tests/test_sweep.py`

**Interfaces:**
- Consumes: `run` from `runner.py`; `score_tool_call`, `score_code_exec`, `score_code_string` from `scorers.py`; dataset builders from `datasets.py`.
- Produces: `run_sweep(models, presses, ratios, tasks, run_fn=run, limit=None) -> list[dict]`. Each dict is one (model, press, ratio, task, metric) row: `{"model","press","compression_ratio","task","metric","value"}`. `run_fn` is injectable so the test mocks generation. `press="none"` is always run once at ratio 0.0 regardless of `ratios`.

- [ ] **Step 1: Write the failing test** (mock `run_fn` so no model loads)

```python
# tests/test_sweep.py
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Prajjwal Chittori
import json
import pandas as pd
from kv_exec_bench.sweep import run_sweep


def fake_run(df, model, press_name, compression_ratio, **kw):
    df = df.copy()
    # Emit the gold call verbatim when uncompressed, garbage when compressed — so metrics differ.
    def pred(row):
        if press_name == "none":
            spec = json.loads(row["answer"])
            return json.dumps({"name": spec["name"], "arguments": spec["arguments"]})
        return "{broken"
    df["predicted_answer"] = [pred(r) for _, r in df.iterrows()]
    return df


def test_sweep_emits_tidy_rows_with_baseline():
    rows = run_sweep(models=["m1"], presses=["SnapKV"], ratios=[0.5],
                     tasks=["tool_call"], run_fn=fake_run, limit=2)
    df = pd.DataFrame(rows)
    assert set(df.columns) == {"model", "press", "compression_ratio", "task", "metric", "value"}
    assert (df["press"] == "none").any()        # baseline always present
    assert (df["press"] == "SnapKV").any()
    # uncompressed schema_valid should beat compressed
    base = df[(df.press == "none") & (df.metric == "name_match")].value.iloc[0]
    comp = df[(df.press == "SnapKV") & (df.metric == "name_match")].value.iloc[0]
    assert base > comp
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sweep.py -v`
Expected: FAIL — module `kv_exec_bench.sweep` missing.

- [ ] **Step 3: Implement**

```python
# kv_exec_bench/sweep.py
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Prajjwal Chittori
"""Grid-sweep driver: run models x presses x ratios x tasks, emit tidy metric rows."""

from .datasets import build_code_exec_df, build_tool_call_df
from .runner import run
from .scorers import score_code_exec, score_code_string, score_tool_call


def _build(task, limit):
    # CPU note uses the network-free inline tool catalog; BFCL (Task 2) is deferred follow-up.
    if task == "tool_call":
        df = build_tool_call_df()
        return df.head(limit) if limit else df
    if task == "code_exec":
        return build_code_exec_df(limit=limit)
    raise ValueError(f"unknown task {task!r}")


def _score(task, df):
    if task == "tool_call":
        return score_tool_call(df)
    # code_exec: structured (pass@1) + string anchor (edit_sim), merged per task key
    merged = {}
    for d in (score_code_exec(df), score_code_string(df)):
        for k, v in d.items():
            merged.setdefault(k, {}).update(v)
    return merged


def _emit(rows, model, press, ratio, scored):
    for task, metrics in scored.items():
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                rows.append({"model": model, "press": press, "compression_ratio": ratio,
                             "task": task, "metric": metric, "value": float(value)})


def run_sweep(models, presses, ratios, tasks, run_fn=run, limit=None, device=None):
    rows = []
    for model in models:
        for task in tasks:
            base_df = _build(task, limit)
            # baseline once
            done = run_fn(base_df, model=model, press_name="none",
                          compression_ratio=0.0, device=device)
            _emit(rows, model, "none", 0.0, _score(task, done))
            for press in presses:
                for ratio in ratios:
                    out = run_fn(base_df, model=model, press_name=press,
                                 compression_ratio=ratio, device=device)
                    _emit(rows, model, press, ratio, _score(task, out))
    return rows
```

Add to `kv_exec_bench/cli.py` a `sweep` subcommand that calls `run_sweep` with CLI-provided lists and writes a CSV via `pandas.DataFrame(rows).to_csv(out, index=False)`. (Follow the existing argparse pattern in the file.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sweep.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add kv_exec_bench/sweep.py kv_exec_bench/cli.py tests/test_sweep.py
git commit -m "feat: grid-sweep driver + sweep CLI subcommand"
```

---

## Task 4: Figure + divergence-table generator

Turn `results.csv` into the paper's money figure and a divergence table. Pure pandas/matplotlib, runs locally on the CSV (no GPU).

**Files:**
- Create: `scripts/make_figures.py`
- Test: `tests/test_make_figures.py`

**Interfaces:**
- Consumes: a results CSV with columns `model, press, compression_ratio, task, metric, value`.
- Produces: `make_contrast_figure(df, task, standard_metric, exec_metric, out_path)` writes a PNG and returns the matplotlib Figure; `divergence_table(df, standard_metric, exec_metric) -> pd.DataFrame` (per model: retention of standard vs exec metric at the highest ratio).

- [ ] **Step 1: Write the failing test** (synthetic CSV, headless backend)

```python
# tests/test_make_figures.py
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Prajjwal Chittori
import matplotlib
matplotlib.use("Agg")
import os
import pandas as pd
from scripts.make_figures import make_contrast_figure, divergence_table

DF = pd.DataFrame([
    {"model": "m1", "press": "none", "compression_ratio": 0.0, "task": "tool_call", "metric": "name_substring", "value": 1.0},
    {"model": "m1", "press": "none", "compression_ratio": 0.0, "task": "tool_call", "metric": "schema_valid", "value": 1.0},
    {"model": "m1", "press": "SnapKV", "compression_ratio": 0.5, "task": "tool_call", "metric": "name_substring", "value": 0.95},
    {"model": "m1", "press": "SnapKV", "compression_ratio": 0.5, "task": "tool_call", "metric": "schema_valid", "value": 0.30},
])


def test_contrast_figure_written(tmp_path):
    out = tmp_path / "fig.png"
    make_contrast_figure(DF, "tool_call", "name_substring", "schema_valid", str(out))
    assert os.path.getsize(out) > 0


def test_divergence_table_shows_gap():
    tbl = divergence_table(DF, "name_substring", "schema_valid")
    row = tbl[tbl.model == "m1"].iloc[0]
    assert row.standard_retention > row.exec_retention  # string holds, structure drops
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pip install matplotlib && pytest tests/test_make_figures.py -v`
Expected: FAIL — `scripts.make_figures` missing. (Add empty `scripts/__init__.py` so it imports.)

- [ ] **Step 3: Implement** `scripts/make_figures.py` with both functions: group by `compression_ratio`, plot standard vs exec metric as two lines per model, save PNG; `divergence_table` computes `standard_retention = value_at_max_ratio / value_at_ratio_0` for each metric per model and returns a DataFrame with columns `model, standard_retention, exec_retention`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_make_figures.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/make_figures.py scripts/__init__.py tests/test_make_figures.py
git commit -m "feat: contrast-figure + divergence-table generator"
```

---

## Task 5: Install the run stack + CPU smoke

Reinstall the generation stack (torch CPU + kvpress + transformers<5.3) and confirm one cell runs end to end on the CPU box.

**Files:** none (environment + verification).

- [ ] **Step 1: Install the run extra**

> **Gotcha (hit on 2026-06-23):** the `run` extra lists `torch` unconstrained, so plain pip pulls the
> multi-GB **CUDA** build and fills the disk on a CPU box. Install CPU torch FIRST from the CPU index,
> then the extra (torch already satisfied), and use `--no-cache-dir` to avoid a 4GB pip cache:
> ```
> .venv/bin/pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
> .venv/bin/pip install --no-cache-dir -e ".[run]"
> ```
Expected: torch (CPU), transformers<5.3, kvpress install. Then verify:
`.venv/bin/python -c "import kvpress, torch; print('ok', torch.__version__)"` -> `ok ...`.

- [ ] **Step 2: Discover available presses** (kvpress version may not ship all of them)

Run: `.venv/bin/python -c "import kvpress; print(sorted(n for n in dir(kvpress) if n.endswith('Press')))"`
Record the list; the sweep uses only presses that exist (drop any missing from the grid below).

- [ ] **Step 3: Smoke run** (smallest model, one press, tiny limit)

Run: `.venv/bin/kv-exec-bench tool_call --model Qwen/Qwen2.5-0.5B-Instruct --press SnapKV --compression-ratio 0.5`
Expected: prints a metrics dict; `schema_valid` below the uncompressed baseline. Proves the CPU path.

---

## Task 6: Run the CPU sweep

Run the experiment that the box can actually do, in the background, and capture `results/results.csv`. No GPU, $0.

> Run as a background job (it is slow on CPU). Models: Qwen2.5-0.5B-Instruct, plus Qwen2.5-1.5B-Instruct only if Step 1 confirms RAM headroom. Presses: the subset of {Random, StreamingLLM, SnapKV, Knorm, ExpectedAttention, PyramidKV} that Task 5 Step 2 reported. Ratios: 0.25, 0.5, 0.75.

- [ ] **Step 1: RAM check for 1.5B** (optional second model)

Run: `.venv/bin/kv-exec-bench tool_call --model Qwen/Qwen2.5-1.5B-Instruct` and watch `free -h`.
If it OOMs or swaps hard, drop to 0.5B-only and note single-model in the paper.

- [ ] **Step 2: tool_call sweep** (fast; full inline catalog)

Run (background): `.venv/bin/kv-exec-bench sweep --models "<models>" --presses "<available>" --ratios "0.25,0.5,0.75" --tasks tool_call --out results/results_tool.csv`
Expected: CSV with `none` baseline + every (model, press, ratio) row for tool metrics including `name_substring`.

- [ ] **Step 3: code_exec sweep** (slow; small slice, isolation on)

Run (background): `KV_EXEC_BENCH_ALLOW_CODE_EXECUTION=1 .venv/bin/kv-exec-bench sweep --models "<models>" --presses "<available>" --ratios "0.25,0.5,0.75" --tasks code_exec --limit 20 --out results/results_code.csv`
Expected: CSV with `pass@1` and `edit_sim` per condition.

- [ ] **Step 4: Merge + sanity-check**

Concatenate the two CSVs into `results/results.csv`. Sanity: `none` baselines highest; structured/exec metrics fall faster than the string anchors (`name_substring`, `edit_sim`) as ratio rises.

- [ ] **Step 5: Figures + table**

Run: `.venv/bin/python scripts/make_figures.py results/results.csv --out results/figures/`
Expected: contrast PNGs + printed divergence table. Copy headline figure(s) to `paper/figures/`.

- [ ] **Step 6: Commit**

```bash
git add results/results.csv results/figures/
git commit -m "data: CPU sweep results (Qwen2.5-0.5B/1.5B) + figures"
```

---

## Task 7: Paper scaffold

**Files:**
- Create: `paper/main.tex`, `paper/refs.bib`, `paper/figures/` (figures copied from Task 6)

- [ ] **Step 1:** Write `paper/main.tex` using a standard arXiv-friendly class (`article` or the NeurIPS/ICML workshop style if targeting one). Section skeleton with real headings: Abstract, 1 Introduction, 2 Related Work, 3 Benchmark Design, 4 Experimental Setup, 5 Results, 6 Discussion & Limitations, 7 Conclusion. Add `\input` for the figures.
- [ ] **Step 2:** Populate `refs.bib` with the must-cite work: KV-compression methods (SnapKV, StreamingLLM, H2O, PyramidKV, Expected Attention / kvpress), eval suites (RULER, LongBench, InfiniteBench, NIAH), code/tool eval (HumanEval, MBPP, BFCL), and kvpress itself.
- [ ] **Step 3:** Build to confirm it compiles:
```bash
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
Expected: `main.pdf` builds with placeholder prose but real structure, figures, and references.
- [ ] **Step 4: Commit**

```bash
git add paper/
git commit -m "docs: paper scaffold (structure, refs, figures)"
```

---

## Task 8: Write the paper body with real numbers

- [ ] **Step 1:** Write Introduction: state the blind spot, the one-sentence finding, contributions C1/C2/C3 (verbatim from spec). Keep prose human-voiced; lead with the result.
- [ ] **Step 2:** Related Work: contrast existing KV-compression eval suites (string/retrieval/extracted-answer only) with this benchmark; position relative to BFCL and HumanEval.
- [ ] **Step 3:** Benchmark Design: the two tasks, the four tool metrics + pass@1 + the string anchors, the shared-context prefill regime (catalog compressed once, every request answered from the compressed cache), kvpress-as-dependency, execution isolation.
- [ ] **Step 4:** Experimental Setup: the exact grid (models, presses, ratios, inline tool catalog + HumanEval slice). **Disclose the hardware verbatim and prominently:** "All experiments were run on a single CPU-only machine: AMD Ryzen 5 5500U (6 cores / 12 threads, AVX2, no AVX-512), 14 GiB RAM, Ubuntu 22.04 (Linux 6.8), PyTorch CPU build, no GPU." State plainly that this bounds scale to <=1.5B models and a small HumanEval slice, and that the contribution is a benchmark release + small-scale demonstration, not a large empirical study.
- [ ] **Step 5:** Results: insert the contrast figure(s) and the divergence table from `results/`. State numbers from `results.csv` exactly. Call out which press/ratio/size is worst, and whether the gap widens with size or ratio.
- [ ] **Step 6:** Discussion & Limitations: small-model caveat, prefill-only regime, BFCL category coverage, exec-isolation is eval-grade not adversarial. If the result is null at low ratios, frame as degradation-onset/threshold finding.
- [ ] **Step 7:** Abstract + Conclusion last, once numbers are locked.
- [ ] **Step 8:** Rebuild PDF, commit:

```bash
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
git add paper/
git commit -m "docs: full paper body with measured results"
```

---

## Task 9: Internal accuracy review

- [ ] **Step 1:** Cross-check every number in the paper text/figures against `results/results.csv`. Fix any mismatch. (Optionally dispatch this as a fresh subagent: "verify each numeric claim in paper/main.tex against results/results.csv".)
- [ ] **Step 2:** Overclaim scan: every claim must be supported by a number in the table; soften anything that isn't. Confirm no employer mention anywhere; confirm author/affiliation correct.
- [ ] **Step 3:** Commit any fixes:

```bash
git add paper/
git commit -m "docs: accuracy + overclaim review pass"
```

---

## Task 10: Finalize repo

- [ ] **Step 1:** Update `README.md`: replace the smoke-run table with the real headline result, add a "Reproduce" section pointing at `modal_app.py` and `scripts/make_figures.py`, and a "Paper" line linking the arXiv ID (filled after Task 11).
- [ ] **Step 2:** Run full test + lint:
```bash
pytest && flake8 kv_exec_bench scripts tests && black --check . && isort --check .
```
Expected: all pass.
- [ ] **Step 3:** Initialize git if not already, stage everything, make the final local commit. **Do not push** — leave it for the user to sign and push.
```bash
cd ~/kv-exec-bench && git init 2>/dev/null; git add -A
git commit -m "feat: kv-exec-bench v0.1 + preprint artifacts"
```
> **USER-IN-THE-LOOP:** sign and push the branch yourself, then create the public repo.

---

## Task 11: arXiv submission

> **USER-IN-THE-LOOP:** arXiv account + first-time **endorsement** in `cs.CL`/`cs.LG` may be required.

- [ ] **Step 1:** Build the arXiv tarball: `main.tex`, `refs.bib` (or the generated `.bbl`), and `paper/figures/`. Strip absolute paths; verify it compiles from a clean dir.
- [ ] **Step 2:** Write the arXiv metadata: title, abstract (no em dashes), authors, categories (`cs.CL` primary, `cs.LG` cross-list), comments line (code URL).
- [ ] **Step 3 (USER):** Upload, request endorsement if prompted, submit. Record the arXiv ID.
- [ ] **Step 4:** Backfill the arXiv ID into `README.md` and commit (local; user pushes).

---

## Task 12: Close the project

- [ ] **Step 1:** Update the memory file `project_kv_exec_bench.md`: mark CONCLUDED, record arXiv ID + repo URL, note the result one-liner. Add a `[[project_ai_infra_career_pivot]]` link (resume bullet).
- [ ] **Step 2:** Add a one-line resume bullet candidate to the project memory (for the next `~/resume` sync): "kv-exec-bench: first executable-correctness benchmark for KV-cache compression (arXiv:XXXX), code pass@1 + tool-schema validity; showed standard metrics mask structural degradation."
- [ ] **Step 3:** Confirm no open threads: tests pass, paper posted, repo pushed (by user), memory updated. Declare done. Do not reopen for scope creep (14B+, MBPP, decode-time) — those are explicitly out of scope.

---

## Self-Review (plan vs spec)

- **Spec coverage:** C1/C2/C3 -> Tasks 7-8; money figure -> Tasks 4, 6, 8; grid (models/presses/ratios/datasets) -> Tasks 2,3,6; BFCL -> Task 2; Modal compute -> Tasks 5-6; contrast anchors -> Tasks 1,4; paper structure -> Tasks 7-8; venue/arXiv -> Task 11; risks (null result, BFCL effort, focus collision, AI-slop) -> noted in Tasks 2,8,9. Closure (the user's explicit ask) -> Tasks 10-12. Covered.
- **Placeholder scan:** code steps carry real code; LaTeX prose steps are content-writing tasks (acceptable — the deliverable is the writing). No TBDs.
- **Type consistency:** `run_sweep` row schema `{model,press,compression_ratio,task,metric,value}` is used identically in Tasks 3, 4, 6. `answer` JSON gains `canonical_solution` in Task 2, consumed by `score_code_string` in Task 1. `bfcl_records_to_df`/`build_bfcl_df` names consistent across Tasks 2, 3, 5, 6.

## Sequencing note

Tasks 1-4 are pure local code (TDD, no GPU, no spend) and can be done now regardless of the nanoserve focus window. Tasks 5-6 are the GPU spend and the only hard external dependency. Tasks 7-12 are writing + publishing. If nanoserve stays the priority, do 1-4 now and gate 5-12; the harness work is wasted-proof either way.
