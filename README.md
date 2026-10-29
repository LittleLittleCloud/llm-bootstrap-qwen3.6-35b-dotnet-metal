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
scales are read directly; source BF16 tensors retain their values. Computation,
KV and recurrent state are FP32. The default served window is 4096 tokens; `--context-window` configures 512 through
32768 tokens.
Vision, MTP and speculation are excluded. Greedy text generation has thinking
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

The original Kestrel server provides the selected experiment behavior through
`/v1/models`, `/v1/chat/completions`, `/v1/responses` and `/v1/messages`, including
streaming and generic model-generated tool calls. It serializes inference and
resets each request's state. This is an experimental local endpoint, not a
claim of implementing every optional feature of those APIs.

Default prefill works in blocks of at most 512 tokens, grouping experts on GPU
and using FP32 SIMD-group matrix operations with vectorized packed-FP8 staging.
Larger dense tiles reuse decoded weights across 32 prompt tokens. Decode uses
vectorized GEMV and deterministic SIMD top-eight routing. No FP16/BF16 activation
substitution is used. A diagnostic `--sequential-prefill` option retains the
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
three formal passes, `ignore_eos=true`, no prefix reuse. Exact runtime hashes and
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

For the supplemental Pi HumanEval campaign, start the server with
`--context-window 32768`; Pi reserves up to 4096 output tokens in addition to its
input/tool history. API and engine checks share this configured total-token
window, and KV/attention workspaces are allocated accordingly. The 4096 output
budget cap is separate. A 2D attention score grid avoids overflowing a flattened
32-bit thread count at longer positions.

`context-self-test --model-dir /path/to/snapshot --context-window 32768` exercises
a public arithmetic fixture longer than 4096 prompt tokens and overflow rejection.
Kernel checks independently test attention scores and causal masking near the
32768-token boundary. These checks are not a HumanEval score.
