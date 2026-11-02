# Bootstrap FP8 engine (.NET + Metal)

Independent implementation for LLM Bootstrap experiment
`2026-10-29-bootstrap-dogfood-001`. The contract was frozen at
`5f7f7099be4a7c261140d6fbc4f800d7065266e1` before engine source was written.
The empty starting workspace is `5fe496f89f7bd68eeb6ffc2f2f2efef375771c5e`.

The host is C#/.NET 10. The native bridge calls only Foundation, Metal and libc.
Original Metal kernels perform all model projections, normalization, convolution,
gated-delta state updates, full attention, top-8 MoE/shared experts and argmax.
There is no MLX, Transformers, llama.cpp, Infer.NET or external inference service
in this candidate's execution path. Baseline source was inspected for model
architecture, but no backend code was copied.

The model is official `Qwen/Qwen3.6-35B-A3B-FP8`, pinned at
`95a723d08a9490559dae23d0cff1d9466213d989`. Packed E4M3 weights and their block
scales are read directly; source BF16 tensors retain their values. Default computation,
KV and recurrent state are FP32. The opt-in `--fast-prefill` preset uses FP16
projection and attention operands with FP32 accumulation, softmax, KV and recurrent
state; see the fast-prefill measurements below. `--tuned-prefill` now selects FP16
KV storage for both target and draft; `--kv-cache-precision fp32` retains the
previous storage for comparison. The default served window is the checkpoint's
native 262144 tokens. `--context-window` can select a smaller window (1–262144).
KV storage starts at at most 4096 tokens and grows as positions are consumed;
requesting a large output allowance does not reserve that whole allowance in memory.
The general prefill workspace reserves 128 MiB for attention scores; tuned online
attention keeps its scores inside the threadgroup, with a device-workspace
fallback for fewer than 32 queries.
Vision and the target MTP head are excluded. Optional DFlash2 speculation is described below. Greedy text generation has thinking
disabled. No OS memory/power limits are modified. All model files are external
to Git and are checked against the manifest.

## Build and verify

```sh
dotnet build src/Bootstrap.Engine -c Release
dotnet run --project src/Bootstrap.Engine -c Release --no-build -- kernel-self-test
dotnet run --project src/Bootstrap.Engine -c Release --no-build -- tokenizer-self-test --model-dir MODEL_DIR --fixtures ABSOLUTE_FIXTURE_PATH
dotnet run --project src/Bootstrap.Engine -c Release --no-build -- generate --model-dir MODEL_DIR --prompt "Say hello in one short sentence." --max-tokens 24
```

The build invokes the original native bridge build with the installed Apple clang.
Metal shaders compile at runtime. The source-pinned `model-manifest.json` and
`kernels.metal` are copied to the executable directory. Tests include all FP8 byte
patterns and projections plus byte-BPE/chat-template fixtures from an independent
reference tokenizer. These implementation checks are separate from the repository
owned acceptance evaluations.

## Serve and reproduce

Obtain the complete official checkpoint snapshot at the revision above from
[Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8/tree/95a723d08a9490559dae23d0cff1d9466213d989).
Keep model files outside this repository. Startup verifies the weight, tokenizer
and chat-template hashes before serving. The SDK is pinned in `global.json`.

```sh
dotnet src/Bootstrap.Engine/bin/Release/net10.0/Bootstrap.Engine.dll serve --model-dir /path/to/snapshot --port 17863
```

`serve` listens on all IPv4/IPv6 interfaces and accepts any HTTP Host name.
Browser CORS permits every origin, method and request header, including OPTIONS
preflight for streaming requests. The endpoint has no authentication requirement.

The original Kestrel server provides the selected experiment behavior through
`/v1/models`, `/v1/chat/completions`, `/v1/responses` and `/v1/messages`, including
streaming and generic model-generated tool calls. It serializes inference and
reuses matching prompt prefixes with bounded hybrid-state checkpoints. This is an experimental local endpoint, not a
claim of implementing every optional feature of those APIs.

Default prefill works in blocks of at most 512 tokens, grouping experts on GPU
and using FP32 SIMD-group matrix operations with vectorized packed-FP8 staging.
Larger dense tiles reuse decoded weights across 32 prompt tokens. Decode uses
vectorized GEMV and deterministic SIMD top-eight routing. The default path uses no FP16/BF16 activation
substitution; `--fast-prefill` explicitly enables reduced-precision operands. A diagnostic `--sequential-prefill` option retains the
original token-by-token prefill path.

```sh
dotnet src/Bootstrap.Engine/bin/Release/net10.0/Bootstrap.Engine.dll compare-prefill --model-dir /path/to/snapshot
```

This checks identical generated tokens between default and sequential prefill on
a 626-token fixture crossing the 512-token boundary. Kernel checks cover all FP8
byte patterns, independent CPU dot products including partial matrix tiles, and
routing tie/probability cases. Initial real-model generation produced `Hello!`
with IDs `[9419,0,248046]`. These checks do not establish numerical equivalence
to another model implementation or general output quality.

The separate LLM Bootstrap evaluation repository is pinned in the experiment
contract. Run its complete `engine-api-conformance` before its `engine-throughput`.
The experiment selects `prefill.in2k.out1,decode.in128.out256`, one warm-up and
three formal passes, `ignore_eos=true`, no prefix reuse (`--no-prefix-cache` on
current builds). Exact runtime hashes and
all favorable/unfavorable measurements belong to the separate results report.
Do not interpret build/unit checks as throughput or semantic-quality evidence.

For development only, `BOOTSTRAP_PROFILE=1` serializes each dispatch into its own
command submission and prints aggregated GPU diagnostics on shutdown. This
substantially changes scheduling/residency and must be unset during formal
conformance/throughput. It is never a substitute for the repository evaluator.

The baseline uses independently pinned affine8/BF16 MLX-LM and a generic protocol
adapter; it is not precision-matched to this official FP8/FP32 engine. No controlled
speedup, SOTA, HumanEval, autonomy or self-hosting claim is made. Model weights,
binaries and raw private evidence are not in Git.

The earlier supplemental Pi HumanEval campaign used `--context-window 32768`
and 4096 output tokens. Those historical measurements remain tied to that build.
For interactive use, omit `--context-window` to use the native window. There is
no separate 4096 output cap: a positive `max_tokens`/`max_output_tokens` is an
upper bound, clamped to the space remaining after the formatted prompt. Omitting
it on Chat/Responses permits the entire remaining window; Messages still requires
`max_tokens`. Generation stops on EOS, stop strings, cancellation, or that bound.
Input that leaves no output space is rejected. API and engine use the same rules.
A 2D attention score grid avoids overflow of flattened thread counts.

Native context is a positional limit, not a guarantee that every request fits
physical memory or retains quality at that length. KV still uses FP32; allocations
can fail under memory pressure. No OS memory limit or weight precision is changed.
Set a lower `--context-window` when a bounded local memory footprint is needed.

Tool-enabled requests stream during generation on Chat Completions, Responses,
and Messages. An incremental parser hides the model's XML control markup, emits
ordinary text immediately, and converts function parameters into JSON argument
deltas. String and JSON object/array values stream as they arrive; numeric and
boolean scalars wait for their parameter delimiter so they can be validated and
normalized. IDs and tool names precede argument fragments. Final arguments match
the concatenated fragments exactly. Incomplete, undefined or malformed calls do
not receive a successful stream completion. Clients execute tools only after the
completed call/turn, rather than on a partial argument fragment.

Run `streaming-self-test` for deterministic parser/writer checks, including every
two-way split of the mixed-type fixture, single-character Unicode chunks,
stop-marker boundaries, early emission, multiple calls and all three protocols.
These fixtures are separate from live-model streaming and API conformance checks.

`context-self-test --model-dir /path/to/snapshot --context-window 32768` exercises
a public arithmetic fixture longer than 4096 prompt tokens and overflow rejection.
Kernel checks independently test attention scores and causal masking near the
32768-token boundary. These checks are not a HumanEval score.

## Prefix caching

Caching is enabled by default. Each engine instance keeps one active token lineage
and at most four checkpoints. Matching uses the complete formatted token IDs,
including system text, tool schemas and chat control tokens. Checkpoints prioritize
the first and last message ends, the prompt end and the latest 4096-token boundary;
spare slots retain earlier block boundaries. A request
restores the latest checkpoint at or before its longest common prefix, then
prefills only the suffix. Identical prompts reuse the whole prefill. Appending,
editing or shortening can reuse an earlier retained checkpoint; an earlier
mismatch or an unrelated conversation can miss and replace the active lineage.
This is not a cache pool for multiple concurrent conversations.

Each checkpoint copies all 30 DeltaNet recurrent states, convolution histories
and final logits in FP32 CPU-owned arrays. The ten full-attention KV prefixes remain in their
existing buffers; decode writes only beyond the saved prompt. Restoring a branch
invalidates all later checkpoints before overwriting their KV. Snapshots use at
most 255.04 MiB in addition to the existing KV/workspace allocations (plus up to
1 MiB for token IDs). Freed checkpoint slots are reused until engine disposal.
The model, weights, tokenizer, template and RoPE configuration are fixed for the
engine instance; restarting starts empty. Changing the prefill mode, resetting,
disabling caching, cancellation or inference/stream failure invalidates the cache.
EOS policy and output limits may change: full hits select again from saved logits.
Generated output is not added to the reusable prefix in this version.

Chat Completions reports actual reuse in `usage.prompt_tokens_details.cached_tokens`;
Responses uses `usage.input_tokens_details.cached_tokens`. Their input token totals
include cached tokens. Messages reports `cache_read_input_tokens` separately from
uncached `input_tokens`, including in `message_start`. No client cache hints are
required. Keep system prompts and serialized tool definitions stable for reuse.

Use `serve --no-prefix-cache ...` for cold-only measurements or to disable reuse.
`prefix-cache-self-test --model-dir /path/to/snapshot` compares cached and uncached
real-model token sequences for exact hits, conversation append, branch, shortened
input, intermediate block hits, KV growth, EOS-policy changes and cancellation
recovery. Exact-hit checks also compare every used KV entry, recurrent/conv state
and final logits bit for bit. These implementation diagnostics are separate from
official conformance and task-quality evaluations.

## Optional DFlash2 decode (experimental)

Pass `--draft-dir` to load `incoai/Qwen3.6-35B-A3B-DFlash2` at
`51ef7b6923ad6c14cb1bb41c37a9041446496ab6`. Download `config.json` and
`model.safetensors` from that exact revision; keep them outside Git. Startup checks
both files against `draft-manifest.json`. The 1,052,513,456-byte draft checkpoint
has SHA-256 `d1f16c0ff85afbe6166fabfae998af3ec16fa11da87da4391b5e0e0a7c796a4d`.

```sh
dotnet src/Bootstrap.Engine/bin/Release/net10.0/Bootstrap.Engine.dll serve --model-dir /path/to/target --draft-dir /path/to/draft --port 17863
```

The original .NET/Metal runtime executes the six-layer BF16 draft, sharing the
unchanged target embedding and output head. Draft activations/KV, target KV,
recurrent state and accumulation remain FP32; the official FP8 target is unchanged.
The draft uses projected features from eight target layers, noncausal sliding
attention, grouped dynamic causal convolutions and the learned top-16 candidate
selector. Each round proposes up to seven tokens after a verified anchor. The
target verifies the block, accepts its matching prefix, restores the corresponding
recurrent/convolution state on GPU, and supplies a correction or bonus token.
Only verified tokens reach the existing streaming callbacks. Request budgets,
EOS, stop strings and cancellation still apply.

Draft context is bounded by the checkpoint's 2048-token sliding window; this does
not reduce the target's native context window. Prefix checkpoints additionally
save the 16 MiB projected draft context. Restoring a prefix rebuilds draft KV from
that context. The correctness-first verifier currently reserves about 503 MiB for
per-row hybrid-state snapshots plus logits; draft weights and workspace add more
memory. These allocations are in addition to growing target KV. No OS memory
settings or weight precision are changed. Omit `--draft-dir` for serial decode.

Development checks, separate from the pinned LLM Bootstrap evaluation:

```sh
dotnet src/Bootstrap.Engine/bin/Release/net10.0/Bootstrap.Engine.dll verification-self-test --model-dir /path/to/target
dotnet src/Bootstrap.Engine/bin/Release/net10.0/Bootstrap.Engine.dll draft-self-test --model-dir /path/to/target --draft-dir /path/to/draft --output-dir /path/to/public-fixture
python tests/draft_reference.py --model-dir /path/to/target --draft-dir /path/to/draft --fixture-dir /path/to/public-fixture
dotnet src/Bootstrap.Engine/bin/Release/net10.0/Bootstrap.Engine.dll speculation-self-test --model-dir /path/to/target --draft-dir /path/to/draft
```

The independent CPU draft oracle requires NumPy and uses a deterministic synthetic
feature fixture. Verification checks every retained prefix of 1/2/4/8-row blocks
against executing just that prefix, including overwriting rejected KV. Generation
checks compare fixed public prompts, cache reuse and early streaming termination.
Matching finite fixtures does not prove universal bitwise equivalence between
batched and serial FP32 paths: reduction ordering can change close logits. Official
throughput and acceptance results must come from the frozen experiment evaluator.

Architecture references: [DFlash2](https://inco.ai/blog/dflash2/),
[reference model](https://github.com/z-lab/dflash), and
[Splash](https://github.com/incoai/splash). Their published performance numbers
use different configurations and are not measurements of this engine.

## Development wall-time diagnostics

`profile-prefill --model-dir /path/to/target --draft-dir /path/to/draft
--prompt-file /path/to/public-prompt.txt --context-window 32768 --runs 3`
emits JSON lines for tokenization, one cold run, three warm uncached runs,
and one attention-counter run. Each generation produces exactly one token;
loading and tokenization are outside its wall timer. Prefix reuse is cleared
before every run, while ordinary checkpoint writes and draft feature capture
remain enabled. All runs must return the same first token. These are development
diagnostics, not endpoint-bench reports or answer-quality evaluations.

Normal measurements retain command-buffer and encoder boundaries. GPU time is
the sum of completed command buffers' `GPUEndTime - GPUStartTime`. Host encoding,
commit and blocking wait are measured separately with a monotonic clock. Process
CPU time is also reported: it measures CPU execution, not elapsed host time.
GPU execution overlaps the blocking wait; do not add those columns. Encoding
includes pipeline creation on the first run. Allocations, reset, KV growth and
checkpoint copies occur as the normal generation path requires.
Timestamp conversion follows [Apple's counter clock calibration guidance](https://developer.apple.com/documentation/metal/converting-gpu-timestamps-into-cpu-time).

Attention-counter mode uses Metal stage-boundary timestamp attachments on separate
QK, softmax and AV compute encoders, within the existing command buffers. It adds
encoder boundaries but no per-kernel CPU waits. Counter ticks are calibrated
against paired device CPU/GPU timestamps for each command buffer. The CPU value
returned by this API is nanoseconds on the tested M5 Pro; a Mach tick conversion
must not be applied again. Counter sums are checked against command-buffer GPU
time. Compare its total against ordinary runs to assess instrumentation cost.
The shared softmax kernel also includes single-token tail attention. The QK and
AV counters cover batched attention only. Projections, RoPE and DeltaNet are
outside these three counters. `BOOTSTRAP_PROFILE` must be unset because the older
serial profiler changes submissions and invalidates these measurements.

`profile-attention --runs 5` requires no model weights. It benchmarks the unchanged
QK, softmax and AV kernels both together and individually at representative
history lengths, using the engine's batch-size rule and deterministic FP32 data.
It also compares timestamp-counter measurements with ordinary command-buffer
timing. Compilation is warmed first. Softmax inputs and AV probabilities are
prepared outside each isolated timer. Causal output is checked against an
independent CPU double-precision reference at 36 output coordinates per shape.
These hot synthetic buffers do not reproduce model residency or weight traffic;
in-model counters are the basis for estimating attention's share of prefill.
Add `--all-kernels` to `profile-prefill` to sample every compute dispatch in its
last run. This uses a 4,096-entry timestamp buffer per command buffer and reports
GPU milliseconds and dispatch counts by kernel name. Command-buffer boundaries
remain unchanged; encoder boundaries change, so its total wall time must be
compared with the ordinary warm run. Blit/reset work and any gaps are outside
the per-kernel sum. The summary groups measured kernels into full attention,
routed expert projections, other matrix projections, DeltaNet recurrence, and
remaining kernels; it also reports timestamp coverage and sampled dispatch count.
Save the commands' standard output as `prefill.jsonl` and `attention.jsonl` in a
local artifact directory, then run `python3 scripts/summarize-wall-profile.py
/path/to/artifact-directory` to produce `summary.json` with warm medians and the
attention fraction. Preserve the cold and instrumented runs in the raw files.

## Strict FP32 TensorOps development path

`--fp32-tensorops` opts into MPP `matmul2d` for batched dense and routed-expert
projections. All staged operands and accumulators remain FP32, with
`relaxed_precision=false`. FP8 weight bytes and BF16 scales are decoded exactly
as before. Tile sizes, threadgroup sizes, routing and attention are unchanged;
small verification batches and single-token decoding retain their existing
kernels. Changing the option on an engine clears its reusable prefix cache.
The shader library now requires Metal Shading Language 4.0 / MPP support;
this development path is tested on the M5 Pro.

`tensorops-self-test` checks dense FP8/BF16/FP32 weights and all three expert
projections against CPU double references and legacy kernels. It covers partial
M/N tiles, expert offsets, grouped/permuted rows, output guards, and a cancellation
fixture that fails if the FP32 operands lose mantissa bits to FP16/BF16.

`compare-tensorops --model-dir /path/to/target --draft-dir /path/to/draft
--prompt-file /path/to/public-prompt.txt --context-window 32768 --runs 3`
first compares a cross-block arithmetic fixture's generated tokens and logits.
It then primes both paths on the full prompt, measures three uncached runs per
path in alternating order, and finishes with one all-kernel counter run per
path. Both paths must emit the same first token. Save stdout to `compare.jsonl`
and run `python3 scripts/summarize-tensorops.py /path/to/artifact-directory`.
Compilation/first full-prompt runs and instrumented runs are excluded from warm
medians. Finite correctness fixtures are not a full model quality evaluation;
these measurements are not official benchmark submissions.

The initial same-tile M5 Pro A/B (`artifacts/fp32-tensorops/`) measured a 10,520-token
prompt with three warm runs per path: legacy median 37,738.13 ms / 278.76 tok/s,
TensorOps median 37,716.86 ms / 278.92 tok/s. The 0.056% difference is below the
observed run variation and provides no evidence of a speedup. GPU command time
medians were 37,617.23 and 37,647.24 ms respectively. All 42,290 new numerical
checks and 1,659 existing checks passed. A 626-token cross-block model fixture
produced identical tokens and compared logits; all long-prompt runs emitted the
same first token. Full-kernel instrumentation increased wall time by over 100%
in this run and is excluded from performance conclusions. Samples confirm
9,280 dense, 3,200 small-dense and 3,840 expert MPP dispatches. This verifies API
routing, not physical Neural Accelerator utilization. It does not establish
performance for alternative tiling or BF16/FP16 operands.

## Projection precision and tiled attention experiment

`--tensorops-precision fp16` or `bf16` rounds staged projection operands to that
format while preserving FP32 accumulation/output. FP8 checkpoint bytes/scales
remain untouched; this changes computation precision, not model storage. The
legacy path remains the default. `tensorops-self-test --precision fp16|bf16`
checks each path against independently rounded CPU references, including expert
routing and tail tiles, and verifies that accumulation retains small summands
that a low-precision accumulator would lose.

`--tiled-attention` enables FP32 tiled QK and AV operations. Causal masking, FP32
softmax/gating, cache formats and batch geometry remain unchanged. It still
materializes scores and is not fused FlashAttention. It also applies to target
batched verification attention; tiny projections and single-token decode remain
on their original paths. `profile-attention
--tiled-attention --runs 3` tests seven histories, including short causal masks
and non-aligned tails, against CPU double-precision references.

`compare-prefill-variants --model-dir TARGET --draft-dir DRAFT --prompt-file PROMPT
--expected-answer ANSWER --context-window 32768 --runs 3` compares legacy, FP16,
BF16, FP32 tiled attention, and both low-precision/tiled combinations. It records
626-token arithmetic logits against the legacy reference, then checks up to 32
generated tokens against the supplied long-context answer. Those cold/full-answer
runs prime the paths and are excluded from throughput. Three warm, uncached,
first-token runs per path alternate order with no kernel counter sampling.
Use `scripts/summarize-prefill-variants.py` on stdout saved as `compare.jsonl`,
plus `attention-legacy.jsonl` and `attention-tiled.jsonl`. These finite fixtures
measure development behavior, not comprehensive quality or official evaluation.

Initial M5 Pro results for that fixed 10,520-token retrieval prompt (three warm
runs per variant, artifact directory `artifacts/lowp-attention/`):

| Projection / attention | Median wall | Prefill tok/s | Speedup |
| --- | ---: | ---: | ---: |
| Original FP32 | 37.060 s | 283.86 | 1.00x |
| FP16 / original | 31.927 s | 329.50 | 1.16x |
| BF16 / original | 31.930 s | 329.48 | 1.16x |
| Original FP32 / tiled FP32 | 25.092 s | 419.27 | 1.48x |
| FP16 / tiled FP32 | 19.919 s | 528.13 | 1.86x |
| BF16 / tiled FP32 | 19.944 s | 527.47 | 1.86x |

All six variants answered both fixed prompts correctly. This does not establish
model-wide quality equivalence: relative L2 error of the 626-token fixture's
prefill logits was 5.54% (FP16), 7.59% (BF16), 0.00839% (FP32 tiled attention),
6.26% (FP16 plus tiled attention), and 9.96% (BF16 plus tiled attention).
The independent attention microbenchmark at history 10,520 improved from 66.01
to 19.87 ms GPU time (3.32x), while application-level speedup is as shown above.
Each low-precision kernel test passed 42,290 checks, including independent
rounded-input references and FP32 accumulation checks. Seven attention shapes
passed CPU double reference checks. Experimental modes remain opt-in.
The FP16 plus tiled-attention mode also passed all 45 verifier state/rollback
checks over widths 1/2/4/8, including continuation after rejected KV is overwritten.

## Fast-prefill follow-up

`--fast-prefill` selects `--tensorops-precision fp16 --attention-mode fp16
--wide-prefill`. Explicit precision/attention options override the preset. It combines FP16
projection/attention operands with FP32 accumulators and softmax. It keeps model
batches at up to 512 tokens. Attention query slices reuse the existing bounded
128 MiB score workspace while preserving absolute causal positions and all KV
entries from the model batch. This increases projection/expert weight reuse
without expanding the attention score allocation. KV and checkpoint formats are
unchanged. Each mode/batch-setting change invalidates reusable prefix state.

`profile-fast-attention --runs 3` checks seven shapes against independent CPU
references with explicit FP16 operand/probability rounding. It also checks exact
agreement when output aliases queries, including a 512-row, 10K-history shape
that spans multiple query slices. Fused online-softmax prototypes with 64/128-key
tiles are included in the microbenchmark; only the 64-key register variant has
an engine option (`--attention-mode fused-fp16`). These are measured alternatives,
not the selected fast path. The prototypes retain FP32 maxima, normalization
sums and accumulators and avoid writing global attention scores.

`compare-prefill-variants --fast-variants ...` runs the same paired correctness /
three-warm-run harness against the preceding FP16-projection + FP32-attention
configuration, FP16 attention, wide batches, and the fused/wide alternative.
Save stdout as `compare.jsonl` and microbenchmarks as `attention-sliced.jsonl`,
then run `scripts/summarize-fast-prefill.py`. Artifacts in `artifacts/fused-fp16/`
retain the initial shared-output prototype, register-output revision, and
128-key comparison, including slower measurements.

The follow-up same-prompt run selected sliced FP16 attention with wide model
batches. Three warm-run medians:

| Configuration (all projections FP16) | Wall | Prefill tok/s |
| --- | ---: | ---: |
| Prior FP32 tiled attention, original batches | 20.186 s | 521.14 |
| FP16 tiled attention, original batches | 18.398 s | 571.79 |
| FP16 tiled attention, 512-token model batches | 16.925 s | 621.56 |
| Fused FP16 attention, 512-token model batches | 19.426 s | 541.53 |

`--fast-prefill` selects the third row: 1.193x versus the paired preceding best,
and 2.190x versus the historical original FP32 result (37.060 s). The latter is
a cross-run reference, not an additional control in this run. Selected-mode
warm times ranged from 16.9246 to 16.9405 s. GPU command time was 16.8869 s;
process CPU time was 38.3 ms. Command buffers decreased from 34 to 23 and
compute dispatches from 36,708 to 24,986. Both fixed answers were correct for
all four candidates. The faster preset preserves FP32 accumulators and softmax;
accuracy tradeoffs from rounded operands are intentional.
The final `--fast-prefill` preset passed 45 verifier state/rollback checks at
widths 1/2/4/8 and all 1,659 existing kernel checks. Verification output records
FP16 projections, FP16 attention and wide batches to confirm preset selection.

## Tuned prefill

The opt-in `--tuned-prefill` preset uses 2048-token model batches, 64-row grouped
expert tiles, and FP16 device-tensor attention. Dense projection weights are
unpacked into reusable FP16 scratch once per projection, so each output tile
does not repeat FP8 conversion. Expert projections retain inline conversion,
with row-major staging and direct accumulator writes. FP32 accumulation,
softmax, recurrent state, and the original checkpoint files are retained. KV
storage now defaults to FP16 in this preset (see below); small speculative
verification uses FP32 attention arithmetic while reading the selected KV type. The previous
`--fast-prefill` preset and the default engine configuration remain available.

At commit `a64c09a` with FP32 KV, alternating three warm uncached runs on the
M5 Pro of the same 10,520-token public prompt produced the following medians. Model loading and tokenization
are excluded, and both variants reserve the largest comparison workspace.

| Configuration | Prefill wall time | Prefill tok/s | 1K decode tok/s |
| --- | ---: | ---: | ---: |
| Previous `--fast-prefill` | 16.626 s | 632.76 | 27.74 |
| New `--tuned-prefill` | 8.307 s | 1266.42 | 27.33 |

This is **2.001x** faster prefill; the requested additional 5x is not achieved.
New warm prefill times ranged from 8.284 to 8.309 s. Both modes returned the
complete expected needle answer, `73DY-DRV5-B8FJ`, and passed an arithmetic
smoke. Decode remains around 27–28 tok/s (the new mode was 1.5% slower in this
run); this change does not resolve the earlier decode regression. Each measured
decode run generated 1000 tokens with 453 speculation cycles and 545 accepted
draft tokens. These are local development diagnostics, not a new endpoint-bench
evaluation or a leaderboard submission.

The selected kernels passed 47,258 matrix checks against independently rounded
CPU references, including FP32 accumulation checks; nine attention shapes up to
32K total context passed CPU-reference and output-alias checks. The engine also
passed 1,659 existing kernel checks, 45 verifier state/rollback checks, and 48
prefix-cache checks, including append and branching across KV-buffer growth.
These checks do not replace a broader quality evaluation of reduced precision.

Use `--tuned-prefill` with `generate` or `serve`. To reproduce the paired
diagnostics with the public prompt, set `TARGET`, `DRAFT`, and `PROMPT` to local
paths, then run:

```sh
dotnet src/Bootstrap.Engine/bin/Release/net10.0/Bootstrap.Engine.dll \
  compare-prefill-variants --model-dir "$TARGET" --draft-dir "$DRAFT" \
  --prompt-file "$PROMPT" --expected-answer 73DY-DRV5-B8FJ \
  --context-window 32768 --runs 3 --final-variants
```

This command also runs the paired decode diagnostic and selected-mode verifier
and prefix-cache checks. Raw measurements, resource samples, shader/source
snapshots and their hashes are retained locally in `artifacts/fast-prefill-next/`.
Run `python3 scripts/summarize-prefill-tuning.py artifacts/fast-prefill-next` to
summarize the saved final results. The selected run reported about 40.1 GB of
Metal buffers after cache tests. System-wide swap was active during this run;
resource samples are not evidence of zero memory pressure.

`--prefill-batch-size` (32–2048) and `--expert-tile-rows` (16/32/64) allow smaller
workspaces and tile experiments. Set the maximum batch size before the first
generation. Explicit precision/attention arguments override preset choices.
`--predecoded-experts` remains an experimental comparison path: unpacking all
expert weights was slower (about 1043 versus 1268 tok/s in candidate screening)
and is **not** enabled by the preset.

The final all-kernel profile identifies grouped expert matmul as the largest
remaining cost (3.715 s), followed by attention including conversion/softmax
(1.413 s), dense projection including conversion (1.293 s), and recurrent delta
(0.861 s). Instrumented command-buffer GPU time was 9.080 s. Encoder timestamp
sampling increased wall time from 8.315 to 9.280 s (+11.6%), so these component
times are diagnostic and are not substituted for the paired throughput above.
Large commands require multiple timestamp buffers: the profiler now uses eight
4096-sample buffers because the device rejects a larger individual buffer.
The completed run is `profile-after-multi-buffer.jsonl`; earlier capacity-error
attempts are retained separately with stderr. The profiling change leaves the
shader and model arithmetic used by the completed validation unchanged.

## FP16 KV storage

`--tuned-prefill` now defaults to `--kv-cache-precision fp16` for both the target
and DFlash draft. Other presets retain FP32 KV by default. Either precision can
be selected explicitly before engine construction; storage cannot change during
a session. FP8 KV is not implemented. All attention modes, single-token decode,
batched prefill and speculative verification read the selected cache type.
RoPE computes in FP32 and rounds on cache write. Attention accumulation, softmax,
recurrent/conv state and prefix snapshots remain FP32.

At 32,768-token capacity, target KV occupies **640 MiB instead of 1280 MiB**.
The draft's 2048-token ring occupies 48 MiB instead of 96 MiB. Device-tensor
attention consumes these half buffers directly instead of retaining another
FP16 K/V copy (up to 64 MiB at 32K). KV growth copies raw bytes and frees the old
buffer; prefix-state hashes use the actual cache element size.

The preceding FP32-KV formal run in LLM Bootstrap PR #36 remains an unchanged
partial report: Metal ran out of memory after growth to 32K, followed by failed
needle/cache requests. A local regression with FP16 KV now passes with verifier
tapes, four prefix snapshots and the 2048-token batch workspace allocated in the
same process. Two 30,044-token public access-code prompts at depths 5% and 90%
return the expected answer; each passes two exact cached repeats, and a short
cached request after 32K growth also succeeds. Metal allocation is
40,069,070,848 bytes, versus the earlier failure's last logged 40,824,045,568
bytes. These are allocations, not a measured system-wide residency limit.
The long prompts took 30.39 and 29.99 s to first token (989 and 1002 effective
input tok/s); they are different workloads from the historical 10K table above.

Validation also passes 45,160 storage/CPU-reference checks across FP32/FP16,
including target RoPE writes, prefix/tail guards, single-token decode, draft ring
wrap and mixed cached/local draft attention. All seven attention paths pass nine
shapes with output-alias checks (maximum absolute CPU-reference error 7.68e-7).
The existing 1,659 kernel checks, 45 verifier checks and 48 prefix-cache checks
pass. This is development regression coverage, not a new full evaluation or a
general quality guarantee for FP16 cache rounding.

```sh
dotnet src/Bootstrap.Engine/bin/Release/net10.0/Bootstrap.Engine.dll kv-storage-self-test
dotnet src/Bootstrap.Engine/bin/Release/net10.0/Bootstrap.Engine.dll \
  profile-fast-attention --kv-cache-precision fp16 --runs 1
dotnet src/Bootstrap.Engine/bin/Release/net10.0/Bootstrap.Engine.dll \
  kv-cache-self-test --model-dir "$TARGET" --draft-dir "$DRAFT" --tuned-prefill
```

Local raw results, resource samples and source hashes are in `artifacts/fp16-kv/`.
Comparison commands report the selected KV precision and use it for all their
candidates; they do not switch cache storage between variants. Use an explicit
`--kv-cache-precision fp32` when reproducing historical FP32-KV measurements.

## Gathered experts and causal device attention

The next `--tuned-prefill` revision packs routed expert inputs once into a
contiguous FP16 buffer, reused by gate and up projections. Expert weights stay
FP8 in memory and are unpacked by 64x64 blocks inside TensorOps kernels; the
whole expert checkpoint is not predecoded. Adaptive 16/32/64-row products avoid
computing a full 64-row tile for small expert tails. Device attention skips
fully masked QK tiles, limits AV to the valid keys for each query tile, and
writes FP16 probabilities directly from FP32 softmax. FP32 accumulation and
FP16 KV storage are unchanged. Small speculative-verification batches retain
their existing path.

On the same M5 Pro, three interleaved warm uncached runs of the 10,520-token
public prompt produced these medians. Model loading and tokenization are
excluded; all warm variants retain the largest allocated workspace.

| Configuration | Prefill wall time | Prefill tok/s |
| --- | ---: | ---: |
| Previous FP16-KV tuned path (`4187694`) | 8.480 s | 1240.62 |
| Gathered inputs + causal attention | 7.336 s | 1433.99 |
| Plus adaptive expert tails (selected) | 6.990 s | 1505.00 |

The selected path improves throughput **21.3%** (1.213x), not the requested
2.5x. Its three samples were 1494.24–1506.73 tok/s. GPU execution remains the
bottleneck: median command-buffer GPU time was 6.969 s of 6.990 s wall time;
process CPU consumption was 23.4 ms. All three variants returned the complete
needle answer and the arithmetic-smoke answer; the smoke logits matched
exactly. These are local development measurements, not a new formal evaluation
or an updated leaderboard result. No decode throughput improvement is claimed.

Independent CPU-reference tests pass 132,977 matrix checks, including all
adaptive tile sizes, partial columns, expert routing and FP32 accumulation.
Another 253,280 softmax checks use NaN-filled masked inputs and untouched output
guards. Eight attention paths pass nine shapes with output aliasing and FP16
KV (maximum absolute CPU-reference error 7.68e-7), and the 1,659 existing kernel
checks pass. Four TensorOps attention paths also pass the nine shapes with
FP32 KV (maximum absolute CPU-reference error 6.21e-7). `profile-experts` benchmarks uniform and skewed expert routing
using the production kernels, including input packing in the timing.

The selected preset also passes 45 verifier checks and 48 prefix-cache checks.
Two 30,044-token retrieval prompts at depths 5% and 90% return the correct code,
pass two exact cached repeats each, and allow a short cached request after 32K
KV growth. Metal buffers total 40,136,179,712 bytes, 64 MiB above the previous
FP16-KV regression; no OOM occurred. First-token times are 26.12 and 25.88 s
(1150 and 1161 effective input tok/s). These different workloads are not paired
with the 10K comparison above. Buffer allocation is not system-wide residency;
system swap was active during this regression.

Reproduce the paired local benchmark with:

```sh
dotnet src/Bootstrap.Engine/bin/Release/net10.0/Bootstrap.Engine.dll \
  compare-prefill-variants --model-dir "$TARGET" --draft-dir "$DRAFT" \
  --prompt-file "$PROMPT" --expected-answer 73DY-DRV5-B8FJ \
  --context-window 32768 --kv-cache-precision fp16 --runs 3 --throughput-variants
```

The public prompt SHA256 is
`d3367dad7b2bb2004af3ab0e54fe2d2d76944cf58c7ba36d18673b62a950b6ee`.
Raw paired runs are in `artifacts/expert-throughput/model-adaptive.jsonl`;
final matrix, attention, expert microbenchmarks and cache regressions use the
`final-*` prefix in the same local directory. Rejected kernel attempts and
compiler errors remain there separately. The new input workspace uses 64 MiB
at a 2048-token batch and is reused across layers. Explicit
`--predecoded-experts` selects the older experimental path instead of gathered
experts; `--gathered-experts`, `--causal-attention` and
`--adaptive-expert-tiles` also enable the individual changes outside the preset.

The preceding FP16-KV revision `4187694` already has a completed formal
[evaluation in LLM Bootstrap PR #36](https://github.com/LittleLittleCloud/LLM-Bootstrap/pull/36),
merged with the unchanged predecessor OOM report. That revision scored 1245.3
prefill tok/s, conformance 10/10, needle 10/10 and cache correctness 18/18, with
no OOM. Its report predates the optimizations in this section; it does not
validate their full evaluation performance.

## Comparison with MLX kernels

[The paired MLX 0.32.4 comparison](docs/MLX_KERNEL_COMPARISON.md) measures matched
synthetic dense, routed-expert and causal-attention workloads on the M5 Pro.
Long-context attention shows the largest per-operation gap (2.4–3.1x); dense
FP16 matmul has no comparable deficit. Quantized expert cores differ by
1.24–1.60x, shrinking to 1.12–1.28x when input packing is included. The report
separates equal-weight FP16 controls from affine-8 requantization, documents
rounding/timing differences, and provides reproduction commands. These
microbenchmarks are not an end-to-end evaluation.

## Fused online attention

`--tuned-prefill` now selects `--attention-mode fp16-online`: fused QK, online
softmax and AV, with 32 logical query rows × 256 KV rows and grouped KV reuse.
Long-context attention improves 2.26–3.13x; a paired 10,520-token full-model
check improves 1548.73→1678.54 tok/s (+8.4%). Short batches retain the previous
causal kernel. [Implementation, tuning, dedicated benchmarks and validation](docs/ONLINE_ATTENTION.md)
document the selected path and precision differences. These development
results do not replace the existing formal evaluation of the predecessor.

## Staged FP8 MoE prefill (November 2, 2026)

The next `--tuned-prefill` revision uses 32-row grouped expert tiles with
8/16/32-row tail matrices, each SIMD group owning 32 output columns and a
32-element K step. Gate and up execute in one dispatch, keep FP32 accumulators
and gate intermediates in threadgroup memory, and write the SiLU product directly
as FP16 for down. The checkpoint remains FP8 E4M3 with its existing BF16 block
scales. Shared-expert projections and the small verification/decode path retain
their previous implementation.

A cooperative grouping kernel replaces separate per-expert count/scan kernels.
It scans routes twice across 256 threads, builds prefix offsets and tiles, and
uses an inverse mapping to restore token/top-k order. Order within an expert is
unspecified. Expert dispatches still use the existing maximum grid with early
returns; this change does not implement indirect dispatch.

Paired local measurements use the same public 10,520/29,939-token fixtures as
the preceding profile, 32K context, FP16 KV, DFlash2, enabled prefix checkpoints,
and a cleared reusable prefix before every request. One cold full-answer run
checks the needle, followed by three warm one-token runs with alternating order.
These are development timings, excluding loading/tokenization and including the
first-token generation work; they do not replace endpoint-bench results.

| Configuration | 10K wall median | 30K wall median |
| --- | ---: | ---: |
| Previous gathered/adaptive experts | 6.272 s | 20.067 s |
| Staged 32-row experts | 6.175 s | Not separately measured |
| Staged + fused gate/up/SiLU | 6.118 s | Not separately measured |
| Staged + fusion + cooperative grouping | 5.834 s | 18.815 s |

The combined throughput gain is 7.5% at 10K and 6.7% at 30K in these fixtures.
Every configuration returned the complete expected needle code. The 10K smoke
also checks finite full-vocabulary logits; a few correct answers are not a broad
model-quality result. All warm runs are retained, not just the fastest.

### Rejected alternatives and numerical checks

Register prefetch with alternating FP16 stages was slower on the current FP8
layout than the simple K=32 loop. K=64 and K=128 staged alternatives were slower
as well. They remain available only in development microbenchmarks for inspection;
the serving preset does not select them. No MLX or Splash runtime is used.
The design was informed by Splash's public implementation at
`1482f6bbc2abf12375332725302019594150b6cb`, but there is no new matched Splash
measurement in this experiment.

An initial prototype returned zero accumulators when passing a computed
cooperative tensor by value. The working implementation initializes it in the
caller and passes it by reference; the failed run is retained in local evidence.
Independent CPU checks cover projection and gate/up values, varying FP8 scales,
distinct gate/up weights and gathered token indices. Full-array fusion comparisons
cover 32/257/2048-token uniform and skewed cases. Routing checks every route,
expert count, offset, tile, inverse mapping and output guard at 1/7/32/257/2048
tokens and 16/32/64-row tiles, including empty experts.

### Reproduce

Use the same model/draft arguments and context ceiling as above. With
`--tuned-prefill`, `--expert-kernel baseline --serial-expert-routing` selects the
previous expert path; `--expert-kernel staged` isolates the smaller matrix tiles,
and `--expert-kernel fused` selects fusion. Routing can be selected independently
with `--parallel-expert-routing` or `--serial-expert-routing`. Changing either
engine setting invalidates the prefix cache. The preset selects fused experts
and parallel routing, only beyond the verification-sized batches.

```sh
dotnet run -c Release --project src/Bootstrap.Engine -- profile-experts --runs 5
dotnet run -c Release --project src/Bootstrap.Engine -- profile-expert-fusion --runs 5
dotnet run -c Release --project src/Bootstrap.Engine -- profile-expert-routing --runs 5
# With --model-dir, --draft-dir, --tuned-prefill and --context-window 32768:
# compare-prefill-variants --moe-variants --runs 3 --prompt-file <public fixture> --expected-answer <code>
# compare-prefill-variants --moe-final-variants ... compares only baseline/final.
# moe-self-tests runs verification and prefix-cache regressions on the selected path.
python3 scripts/summarize-moe-experiments.py artifacts/moe-pipeline
```

Raw JSONL, shader-validation logs, failed attempts and environment evidence stay
local under `artifacts/moe-pipeline/`. Formal evaluation uses a separately frozen
derived experiment and the repository-owned evaluator.
