# kv-exec-bench preprint — design

Date: 2026-06-23
Status: design approved, pending spec review
Author: solo (do not name employer anywhere in the paper or repo)

## Goal

A credibility preprint, on-thesis for the inference-infra job push (Jan/Feb 2027), built on the
already-implemented `~/kv-exec-bench` repo. Scope is deliberately "minimum credible": a 4-6 page
workshop-style empirical note whose deliverable is one decisive contrast figure. Scoped to finish in
weeks, not months, to stay inside the known 3-4 month motivation window.

## Claim and contributions

Working title: *Perplexity Holds, Programs Break: Executable Correctness as the Blind Spot of
KV-Cache Compression.*

- **C1 (gap).** Every long-context KV-compression eval suite in common use (RULER, LongBench,
  InfiniteBench, Loogle, NIAH, GSM-style math) scores only string/token overlap, retrieval hits, or
  an extracted answer. None checks whether generated output actually runs, or whether a generated
  tool call is well-formed.
- **C2 (benchmark).** `kv-exec-bench` measures the two missing things: code unit-test pass@1 and
  tool-call JSON-Schema validity, under KV compression. Built on top of NVIDIA/kvpress as a
  dependency (does not patch it), so any press works drop-in. Includes eval-grade execution isolation
  for the code task.
- **C3 (result).** Across a model-size ladder x presses x compression ratios, standard string/
  perplexity metrics stay approximately flat while executable-correctness metrics collapse. Quantify
  the divergence so the implicit "compression is near-lossless" claim is shown false for structured
  generation.

## The money figure (the paper stands or falls on this)

Side-by-side line plot. x-axis: compression ratio {0, 0.25, 0.5, 0.75}.

- One line: a **standard** metric (extracted-answer / string-overlap, plus gold-completion
  perplexity under the compressed cache) — stays high.
- Other line: an **exec** metric (code pass@1; tool `schema_valid` / `args_exact`) — drops.

Repeated per model size. The argument is unassailable if the gap (a) widens with ratio and (b) is
consistent across presses. Secondary figure: divergence quantified (e.g. correlation between
standard-metric retention and exec-metric retention is weak/near-zero).

## Experimental grid

- **Models (instruct, tool-capable):** Qwen2.5-0.5B-Instruct, Qwen2.5-3B-Instruct,
  Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct.
- **Presses:** `none` (baseline), `Random` (floor), StreamingLLM, SnapKV, Knorm, ExpectedAttention
  (or PyramidKV). ~5 presses + baseline.
- **Compression ratios:** 0.25, 0.5, 0.75.
- **Datasets:**
  - `code_exec`: full HumanEval (164). MBPP deferred to future work unless time allows.
  - `tool_call`: **BFCL** (Berkeley Function-Calling Leaderboard) — simple + multiple categories.
    Replaces the 11 inline requests for the headline numbers. The inline TOOLS catalog stays as a
    fast, network-free smoke set.
- **Contrast anchors:**
  - code: CodeBLEU / edit-similarity to canonical solution (string) vs pass@1 (exec).
  - tool: `name_match` / substring presence (lenient string) vs `schema_valid` + `args_exact`
    (structured).
  - global: perplexity of the gold completion under the compressed cache as the "looks fine" signal.

## Compute plan (REVISED 2026-06-23: CPU-only, no rental)

Decision: skip GPU rental entirely. Publish off the local CPU box. The paper discloses the hardware
explicitly and scopes every claim to that scale.

- **Hardware (disclosed verbatim in the paper):** AMD Ryzen 5 5500U, 6 cores / 12 threads, AVX2 (no
  AVX-512), 14 GiB RAM (~8 GiB usable for the run), Ubuntu 22.04 (Linux 6.8), PyTorch CPU build, no
  GPU. Single laptop-class CPU.
- **Models:** Qwen2.5-0.5B-Instruct (primary); Qwen2.5-1.5B-Instruct if RAM allows, for a two-point
  size trend. 3B+ excluded (does not fit in ~8 GiB alongside the OS).
- **Tasks:** `tool_call` on the inline catalog (BFCL dropped for the CPU note — keeps it
  network-free and fast; noted as the headline follow-up). `code_exec` on a small HumanEval slice
  (limit ~20) since 512-token CPU generation is slow.
- **Cost: $0.** Wall-clock is the only budget; sweep runs in the background.
- This makes the contribution explicitly a *small-scale demonstration + benchmark release*, not a
  large empirical study. That is stated as a limitation, not hidden.

## Paper structure (~6 pages)

1. Intro — the blind spot, stated crisply.
2. Related work — KV-compression methods and the eval suites they use; code/tool-call evaluation
   (HumanEval, BFCL).
3. Benchmark design — the two tasks, the four tool metrics + pass@1, the shared-context prefill
   regime (catalog compressed once, every request answered from the compressed cache), kvpress
   integration, execution isolation.
4. Experiments — the grid above.
5. Results — contrast figures, divergence quantification, which press/ratio/size is worst.
6. Discussion, limitations, release (Apache-2.0 repo + reproduction commands).

## Venue

arXiv `cs.CL` / `cs.LG`, then a workshop: **ENLSP** (NeurIPS Efficient NLP) or **ES-FoMo** (ICML
Efficient Systems for Foundation Models). Both take exactly this empirical-systems note. Confirm the
2026/2027 deadlines when drafting.

## Risks and mitigations

- **Null result** (large models hold at ratio 0.5): reframe as a degradation-onset / threshold
  finding — "exec correctness breaks at compression ratios where standard metrics are still nominal."
  Still publishable.
- **"This is obvious":** counter with the quantified divergence and the fact that no compression
  paper reports it. Obvious-in-hindsight + nobody-measured-it = a legitimate benchmark contribution.
- **BFCL integration effort:** datasets.py already exposes a 6-column drop-in contract
  (`context, question, answer_prefix, answer, task, max_new_tokens`); if BFCL eats time, ship with
  HumanEval + a scaled inline/xLAM tool set and note BFCL as the headline follow-up.
- **Focus collision:** conflicts with the committed nanoserve 100-day sole-focus window
  (2026-06-16 -> ~2026-09-24, all else parked except light vLLM/llguidance PRs). Decision pending:
  run now (nanoserve slips) or queue this for after. Scope kept small precisely so it can slot in
  without blowing the motivation budget.
- **AI-slop perception** (prior lesson: noir PR closed as AI-slop): this is code-with-a-correct-
  answer plus measured results, not a UX/opinion contribution. Keep prose human-voiced; lead with
  numbers.

## Out of scope (YAGNI)

- 14B+ models, additional presses beyond the six, MBPP, per-layer / per-head drop analysis. All
  deferred to a potential full benchmark paper later.
- Any non-prefill (decode-time) compression regime.
- Live serving / latency benchmarking — this paper is about correctness, not throughput.

## Repo state note

`~/kv-exec-bench` is built and verified but not yet a git repo and not pushed. Initializing/pushing is
a separate step. Per the hard git-signing rule, any push is left for the user to sign and push
manually; do not push unsigned commits.
