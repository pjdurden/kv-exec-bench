# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Prajjwal Chittori

"""Command-line entry point: build a benchmark, generate under a press, score, print JSON.

Examples
--------
    # tool_call (no code execution), uncompressed baseline:
    kv-exec-bench tool_call --model Qwen/Qwen2.5-0.5B-Instruct

    # tool_call under SnapKV at compression ratio 0.5:
    kv-exec-bench tool_call --model Qwen/Qwen2.5-0.5B-Instruct --press SnapKV --compression-ratio 0.5

    # code_exec (executes model output — opt in and run on throwaway hardware):
    KV_EXEC_BENCH_ALLOW_CODE_EXECUTION=1 \\
      kv-exec-bench code_exec --model Qwen/Qwen2.5-0.5B-Instruct --limit 20
"""

import argparse
import json

import pandas as pd

from . import datasets, runner, scorers
from .sweep import run_sweep


def _csv_list(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def main(argv=None):
    p = argparse.ArgumentParser(prog="kv-exec-bench", description=__doc__)
    p.add_argument("benchmark", choices=["code_exec", "tool_call", "sweep"])
    p.add_argument("--model", help="HF model id or local path (single-run modes)")
    p.add_argument("--press", default="none", help="KVPress press name (e.g. SnapKV, StreamingLLM) or 'none'")
    p.add_argument("--compression-ratio", type=float, default=0.0)
    p.add_argument("--device", default=None, help="torch device, e.g. cuda:0 or cpu")
    p.add_argument("--max-context-length", type=int, default=None)
    p.add_argument("--limit", type=int, default=None, help="cap number of problems/rows per task")
    # sweep-only args (comma-separated lists)
    p.add_argument("--models", type=_csv_list, help="sweep: comma-separated model ids")
    p.add_argument("--presses", type=_csv_list, help="sweep: comma-separated press names")
    p.add_argument("--ratios", type=_csv_list, help="sweep: comma-separated compression ratios")
    p.add_argument("--tasks", type=_csv_list, help="sweep: comma-separated tasks (tool_call,code_exec)")
    p.add_argument("--out", default=None, help="sweep: path to write the results CSV")
    args = p.parse_args(argv)

    if args.benchmark == "sweep":
        rows = run_sweep(
            models=args.models,
            presses=args.presses,
            ratios=[float(r) for r in args.ratios],
            tasks=args.tasks,
            limit=args.limit,
            device=args.device,
        )
        df_out = pd.DataFrame(rows)
        if args.out:
            df_out.to_csv(args.out, index=False)
            print(f"wrote {len(df_out)} rows to {args.out}")
        else:
            print(df_out.to_csv(index=False))
        return

    if not args.model:
        p.error("--model is required for code_exec/tool_call")

    if args.benchmark == "code_exec":
        df = datasets.build_code_exec_df(limit=args.limit)
        scorer = scorers.score_code_exec
    else:
        df = datasets.build_tool_call_df()
        scorer = scorers.score_tool_call

    df = runner.run(
        df,
        model=args.model,
        press_name=args.press,
        compression_ratio=args.compression_ratio,
        device=args.device,
        max_context_length=args.max_context_length,
    )
    result = {
        "benchmark": args.benchmark,
        "model": args.model,
        "press": args.press,
        "compression_ratio": args.compression_ratio,
        "scores": scorer(df),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
