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
KV and recurrent state are FP32. The initial served window is 4096 tokens;
vision, MTP and speculation are excluded. Greedy text generation has thinking
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

First real-model smoke generation produced `Hello!` with token IDs
`[9419, 0, 248046]`. This establishes a working minimal generation path only;
API compatibility, formal performance and general output quality remain unverified.
The initial prefill processes tokens sequentially; batched optimization remains
work for the frozen throughput target, not an assumed result.
