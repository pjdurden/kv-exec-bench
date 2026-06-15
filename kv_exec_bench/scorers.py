# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Prajjwal Chittori

"""Executable-correctness scorers for generation under KV-cache compression.

The KV-compression eval suites in common use (RULER, LongBench, Loogle, InfiniteBench, NIAH, and the
math sets) only ever score token/string overlap, retrieval hits, or an extracted answer. None of them
check whether the generated output actually *runs* or whether a generated tool call is *well-formed*.
That is a real blind spot for KV compression: substring and boxed-answer metrics can sit flat while a
compressed cache quietly breaks structured generation — one dropped token in a function body, or a
malformed `arguments` object, fails hard but barely moves string_match.

This module is the scoring half of that benchmark. Both scorers take a DataFrame whose rows carry the
prompt the model completed (`context`), the model output (`predicted_answer`), and a JSON reference
(`answer`); each returns a JSON-serializable dict keyed by task.

  * score_code_exec  — unit-test pass@1; *executes* the model output (opt-in, isolated subprocess).
  * score_tool_call  — JSON parse + JSON-Schema validity + exactness; runs NO code.
"""

import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import jsonschema
import pandas as pd

try:  # rlimits are POSIX-only; on other platforms we fall back to the wall-clock timeout alone.
    import resource
except ImportError:  # pragma: no cover - non-POSIX
    resource = None


# --------------------------------------------------------------------------------------------------
# code_exec: unit-test pass@1 (executes model output)
# --------------------------------------------------------------------------------------------------

EXEC_ENV_FLAG = "KV_EXEC_BENCH_ALLOW_CODE_EXECUTION"
TIMEOUT_SECONDS = 15
# Address-space cap for the child (bytes); guards against runaway allocations in generated code.
MEM_LIMIT_BYTES = 4 * 1024 * 1024 * 1024


def _set_child_limits():  # pragma: no cover - runs only in the forked child
    """Apply CPU-time and address-space rlimits in the subprocess before exec."""
    resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT_SECONDS, TIMEOUT_SECONDS))
    resource.setrlimit(resource.RLIMIT_AS, (MEM_LIMIT_BYTES, MEM_LIMIT_BYTES))


def run_program(program: str, timeout: int = TIMEOUT_SECONDS) -> bool:
    """Run `program` in an isolated subprocess. Return True iff it exits 0 within the limits.

    Isolation is best-effort eval-grade, not an adversarial sandbox: a fresh temp cwd, a stripped
    environment, a wall-clock timeout, and (on POSIX) CPU-time and address-space rlimits. Run it on
    throwaway/CI hardware, not a workstation with secrets in the env.
    """
    posix = resource is not None and os.name == "posix"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "candidate.py")
        with open(path, "w") as f:
            f.write(program)
        try:
            proc = subprocess.run(
                [sys.executable, path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=d,
                env={"PATH": "", "PYTHONPATH": "", "PYTHONDONTWRITEBYTECODE": "1"},
                preexec_fn=_set_child_limits if posix else None,
            )
            return proc.returncode == 0
        except (subprocess.TimeoutExpired, MemoryError):
            return False


def _build_program(row: pd.Series) -> str:
    """Assemble a runnable program from the prompt, the completion, and the reference tests."""
    spec = json.loads(row["answer"])
    completion = row["predicted_answer"] or ""
    return f"{row['context']}{completion}\n\n{spec['test']}\n\ncheck({spec['entry_point']})\n"


def score_code_exec(df: pd.DataFrame) -> dict:
    """Score code generation by unit-test pass@1, per task.

    Refuses to run unless ``KV_EXEC_BENCH_ALLOW_CODE_EXECUTION=1`` is set, so importing this module or
    scoring tool_call never executes untrusted code.
    """
    if os.environ.get(EXEC_ENV_FLAG) != "1":
        raise RuntimeError(
            f"score_code_exec executes model-generated code and is disabled by default. "
            f"Set {EXEC_ENV_FLAG}=1 to enable it, and run on isolated/CI hardware."
        )

    programs = [_build_program(row) for _, row in df.iterrows()]
    max_workers = min(8, (os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        passed_flags = list(pool.map(run_program, programs))

    scores = {}
    for task, idx in df.groupby("task").groups.items():
        flags = [passed_flags[df.index.get_loc(i)] for i in idx]
        n = len(flags)
        passed = int(sum(flags))
        scores[str(task)] = {"pass@1": round(passed / n, 4), "passed": passed, "total": n}
    return scores


# --------------------------------------------------------------------------------------------------
# tool_call: JSON parse + JSON-Schema validity (no execution)
# --------------------------------------------------------------------------------------------------


def extract_json_object(text: str):
    """Return the first balanced top-level JSON object in `text`, or None.

    Scans brace depth (ignoring braces inside strings) so trailing prose after the call doesn't break
    parsing, then json.loads the candidate span.
    """
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break  # malformed; try the next opening brace
        start = text.find("{", start + 1)
    return None


def score_tool_call(df: pd.DataFrame) -> dict:
    """Score tool/function calling, per task, with four nested metrics from lenient to strict.

    * json_valid   - a JSON object could be extracted from the output
    * name_match   - the selected tool name is correct
    * schema_valid - the arguments validate against the tool's JSON Schema (shape/types/required)
    * args_exact   - the arguments exactly equal the gold arguments (right values too)
    """
    scores = {}
    for task, df_task in df.groupby("task"):
        agg = {"json_valid": 0, "name_match": 0, "schema_valid": 0, "args_exact": 0}
        n = len(df_task)
        for _, row in df_task.iterrows():
            spec = json.loads(row["answer"])  # {"name", "arguments", "schema"}
            call = extract_json_object(row["predicted_answer"] or "")
            if not isinstance(call, dict):
                continue
            agg["json_valid"] += 1
            agg["name_match"] += int(call.get("name") == spec["name"])
            args = call.get("arguments")
            try:
                jsonschema.validate(args, spec["schema"])
                agg["schema_valid"] += 1
            except jsonschema.ValidationError:
                pass
            agg["args_exact"] += int(args == spec["arguments"])
        scores[str(task)] = {k: round(v / n, 4) for k, v in agg.items()}
    return scores
