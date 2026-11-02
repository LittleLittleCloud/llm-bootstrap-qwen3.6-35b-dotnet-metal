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
state; see the fast-prefill measurements below. The default served window is the checkpoint's
native 262144 tokens. `--context-window` can select a smaller window (1–262144).
KV storage starts at at most 4096 tokens and grows as positions are consumed;
requesting a large output allowance does not reserve that whole allowance in memory.
Prefill attention uses a 128 MiB score workspace and smaller tiles at long positions.
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
