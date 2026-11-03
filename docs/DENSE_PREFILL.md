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
- **Shared-expert gate (N=1):** use the same predecoded TensorOps kernel, moving
  the scalar projection before shared-expert down so it can reuse packed Norm
  from gate/up. Operands remain FP16, accumulation/output FP32. Tiny-row and
  disabled-mode dispatch order is unchanged. A SIMD-dot prototype was rejected:
  its small reduction-order differences amplified in a real-model smoke test.

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
independent CPU/double dot products, finite values and output guards. Small projection outputs must match the original exactly on these fixtures. Small
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

## Controlled model results

Final variant uses TensorOps for both N=32 and N=1. All owned GPU runs were
serial; three uncached warm observations per length are summarized below. The
baseline was remeasured after the candidate to check order drift. All earlier
controls, cold runs, and rejected SIMD-prototype results are retained locally.

| Input tokens | Baseline warm wall median | Final warm wall median | Throughput gain |
|---:|---:|---:|---:|
| 10520 | 4071.89 ms | 3916.44 ms | 3.97% |
| 29939 | 13385.30 ms | 13012.19 ms | 2.87% |

Warm sample ranges (ms):

- `baseline-before`: 10520: 4074.58–4114.85, 29939: 13343.02–13384.47.
- `draft`: 10520: 3989.03–4022.71, 29939: 13127.06–13234.74.
- `small`: 10520: 3984.82–4032.64, 29939: 13167.36–13197.88.
- `all`: 10520: 3895.00–3965.42, 29939: 12877.68–12914.48.
- `baseline-after`: 10520: 4060.20–4099.42, 29939: 13327.15–13380.84.
- `stable-all`: 10520: 3897.14–3960.31, 29939: 12975.38–13114.29.
- `baseline-final`: 10520: 4067.73–4105.68, 29939: 13348.12–13474.44.

`small` and `all` above are the rejected SIMD prototype, not the final candidate.
The draft-only ablation is inference-equivalent to the final draft path.
The final combined result cannot be assigned entirely to either sub-change.

The final engine passed both complete needle answers, repeated first-token
checks, three 256-token decode runs, 45 verifier checks and 48 prefix-cache
checks. Smoke logits are bitwise identical to the GDN baseline
(SHA-256 `5f85879f85492a1f7aa729069bd49b5df7ca8646cd4ddbfda2fdda56d4c720e2`).
The precision change in draft FC is still subject to finite-fixture limitations;
matching target smoke logits does not prove universal draft equivalence.

The SIMD prototype had only 3.34e-6 maximum scalar-projection error on actual
model inputs, but target smoke logits differed by 1.807 maximum absolute /
0.166 relative L2. Same-process isolation attributed this to the SIMD scalar
path: draft-only and N=32-only logits were identical. It was rejected, despite
passing the output-answer, synthetic-kernel, verifier and cache checks. This is
why the final N=1 implementation preserves the TensorOps reduction instead.

Final synthetic validation: 56 full-output/CPU-reference/guard cases, including
packed-input reuse; all N=1/32 outputs exactly match the original. Maximum
draft error is 3.15e-4 absolute / 1.85e-4 relative L2. All 56 pass API validation;
52 pass API plus GPU validation with only the four unsupported instrumented
legacy draft dispatches omitted, as described above.

`stable-all` maximum allocation reported during the development prefill samples: 40461910016 bytes; whole-machine new swap-outs: 0.000 MiB.

`baseline-final` maximum allocation reported during the development prefill samples: 39793098752 bytes; whole-machine new swap-outs: 0.000 MiB.

Matched profile workloads allocate 134217728 additional bytes (128 MiB), consistent with the two draft packing buffers.

Encoder intervals are diagnostic and may overlap. End-to-end performance is
reported using the uncached wall medians above. Official Full-profile endpoint
results are submitted separately and can use different generated input lengths.
