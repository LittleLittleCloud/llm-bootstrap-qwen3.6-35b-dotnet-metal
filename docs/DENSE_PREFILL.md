# Dense prefill fallback paths

`--tuned-prefill` now selects `--dense-prefill all`. Use
`--dense-prefill baseline|draft|small|all` for ablations against the GDN-preparation
engine at `4d0968401440c57dc757255c8711712f2395fcba`.
Changing the mode clears reusable prefix snapshots. These paths require FP16
projection precision. Rows <=8 retain the existing tiny projection/verifier path.

## Implementation and numerical behavior

- **Draft context FC:** replace the legacy 32×64 FP32 tiled projection with the
  existing 64×64 TensorOps kernel. Decode the BF16 FC weights to FP16 and pack
  FP32 features to FP16 for each call. Accumulation, output, normalization, and
  stored draft context remain FP32. This changes FC operand rounding, not the
  BF16 checkpoint. Other draft projections, speculative verification, and the
  number of captured/committed context rows are unchanged.
- **DeltaNet a/b (N=32):** use the existing predecoded TensorOps path and reuse
  packed Norm input from the immediately preceding qkv/z projections. The
  shared scratch cannot be overwritten between these projections. Operands
  remain FP16, with FP32 accumulation/output.
- **Shared-expert gate (N=1):** use one SIMD group per row for a dot product,
  instead of a 32-column padded matrix tile. Both operands are rounded to FP16
  as before; the FP32 reduction order changes. The pinned model uses K=2048,
  a multiple of the kernel's 128-element vectorized stride.

At 2048 rows, draft packing needs 64 MiB for inputs and 64 MiB for weights.
These are dedicated reusable buffers because target-projection scratch is too
small. Input scratch is resized when feature capacity grows; all scratch is
owned and released by the engine. There is no extra allocation for the two
small target projection paths. Peak allocation, swap and actual endpoint
behavior are measured separately from the kernel benchmark.

## Development checks

```sh
dotnet build src/Bootstrap.Engine -c Release
dotnet src/Bootstrap.Engine/bin/Release/net10.0/Bootstrap.Engine.dll \
  profile-dense-prefill --runs 5
MTL_DEBUG_LAYER=1 dotnet src/Bootstrap.Engine/bin/Release/net10.0/Bootstrap.Engine.dll \
  profile-dense-prefill --runs 0
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 \
  dotnet src/Bootstrap.Engine/bin/Release/net10.0/Bootstrap.Engine.dll \
  profile-dense-prefill --runs 0 --candidate-only
```

Checks cover N=1/32 at M=1,8,9,17,127,272,1259,2048 and draft FC at
M=9,127,512,2048. They compare full outputs against the old GPU path, sampled
independent CPU/double dot products, finite values and output guards. Small
projection references use FP16-rounded operands; draft references retain FP32
activations for the original and use FP16-rounded operands for the candidate.
Benchmarks alternate order after warming, retain run 0 and report every sample.
Conversion dispatches are included in candidate GPU command durations.

`--candidate-only` skips the legacy draft kernel, whose GPU validation
instrumentation expands threadgroup scratch from 24,960 to 49,920 bytes and
exceeds this device's 32,768-byte limit. The production kernel runs normally.
In this diagnostic mode, draft candidates use the independent CPU reference;
no GPU-original equivalence is claimed. Other paths still compare to the old
small matrix kernel. The API-only pass includes all original draft paths.

Full-model checks use the existing `moe-self-tests --packed-model-check` command
with the same 10K/30K fixtures, full expected answers, uncached warm repetitions,
256-token decode runs, verifier checks and prefix-cache checks. Formal results
are from the separate pinned LLM Bootstrap Evaluate page, never these synthetic
or model development fixtures.
