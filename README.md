# kv-exec-bench

executable-correctness eval for text generation under KV-cache compression.

the eval suites people use to validate KV compression (RULER, LongBench, Loogle, InfiniteBench, NIAH,
the math sets) only ever score token/string overlap, retrieval hits, or an extracted answer. nothing
checks whether the output actually runs, or whether a tool call is well-formed.

that's a blind spot for KV compression specifically. substring and boxed-answer metrics can sit flat
while a compressed cache quietly breaks structured generation. one dropped token in a function body,
or a malformed `arguments` object, fails hard but barely moves `string_match`. aggregate accuracy
holds, functional correctness degrades.

this measures the two things those suites miss. it runs on top of
[NVIDIA/kvpress](https://github.com/NVIDIA/kvpress) as a dependency (it doesn't patch it), so any
press in kvpress works.

## the two benchmarks

`tool_call` picks one tool from a long shared catalog (that catalog is the context the press
compresses) and emits a single JSON call. the call is parsed and validated against the tool's JSON
schema. no code runs. four metrics, lenient to strict: `json_valid`, `name_match`, `schema_valid`,
`args_exact`.

`code_exec` completes a HumanEval function and scores unit-test pass@1. this one executes model
output, so it's off unless you set `KV_EXEC_BENCH_ALLOW_CODE_EXECUTION=1`. execution is in a subprocess
with a fresh temp cwd, a stripped env, a wall-clock timeout, and (on POSIX) CPU + address-space
rlimits. eval-grade isolation, not an adversarial sandbox. run it on throwaway/CI hardware, not a box
with secrets in the env.

## what it catches

the tool catalog is shared across requests, so a prefill press compresses it once and answers every
request from the compressed cache. that's where a dropped or garbled tool definition turns into an
invalid call. quick CPU run, Qwen2.5-0.5B-Instruct, the inline `tool_call` set:

| condition       | json_valid | name_match | schema_valid | args_exact |
|-----------------|-----------:|-----------:|-------------:|-----------:|
| no press        | 1.00       | 1.00       | 1.00         | 0.50       |
| SnapKV, cr 0.5  | 1.00       | 0.50       | 0.25         | 0.00       |

valid JSON survives the compression, tool selection and schema-valid args don't. a string-overlap
metric never sees that. it's a tiny smoke run, not a leaderboard. reproduce with the commands below
and scale up the model and the dataset.

## install

scorers and dataset builders are light (`pandas`, `jsonschema`). the runner pulls the model stack, and
kvpress pins `transformers<5.3`, so keep that in its own env:

```bash
pip install -e .            # scorers + tool_call dataset, enough for the tests
pip install -e ".[run]"     # adds kvpress + torch + transformers<5.3 for generation
```

## run

```bash
# tool_call, uncompressed baseline
kv-exec-bench tool_call --model Qwen/Qwen2.5-0.5B-Instruct

# tool_call under SnapKV at compression ratio 0.5
kv-exec-bench tool_call --model Qwen/Qwen2.5-0.5B-Instruct --press SnapKV --compression-ratio 0.5

# code_exec (executes model output; opt in, throwaway hardware)
KV_EXEC_BENCH_ALLOW_CODE_EXECUTION=1 \
  kv-exec-bench code_exec --model Qwen/Qwen2.5-0.5B-Instruct --limit 20
```

`--press` takes any kvpress press name (`SnapKV`, `StreamingLLM`, `Knorm`, `Random`, ...) or `none`.
`--device cuda:0` on GPU.

## from python

```python
from kv_exec_bench import build_tool_call_df, score_tool_call
from kv_exec_bench.runner import run

df = build_tool_call_df()
df = run(df, model="Qwen/Qwen2.5-0.5B-Instruct", press_name="SnapKV", compression_ratio=0.5)
print(score_tool_call(df))
```

the dataset is six columns (`context, question, answer_prefix, answer, task, max_new_tokens`); each
scorer takes the filled frame and returns a JSON-serializable dict keyed by task. swapping the inline
tools for BFCL/xLAM/ToolACE, or adding MBPP next to HumanEval, is a drop-in as long as you produce
those columns.

## tests

```bash
pip install -e ".[dev]"
pytest
```

scorer tests need no model or GPU; `predicted_answer` is supplied synthetically.

## license

Apache-2.0.
