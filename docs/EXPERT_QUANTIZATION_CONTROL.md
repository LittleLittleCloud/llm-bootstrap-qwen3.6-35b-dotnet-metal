# Controlled expert quantization comparison — 2026-11-03 UTC

This experiment isolates weight representation/decoding inside one kernel
framework. It tests whether the FP8 versus affine difference can explain part
of the expert-kernel deficit without also changing activation layout, routing,
TensorOps shape, precision, output stores, or compilation mode. It is not a
whole-model evaluation or a change to production kernels.

## Controls

- Engine source: `2caf5890c3977b18a5aa137b6f3821a6966a8097`, with generated
  standalone diagnostic shaders; native source remains unchanged.
- Apple M5 Pro, 20 GPU cores, 48 GiB. All measured kernels use runtime Metal 4.0
  safe math and the same GPU command-buffer timing method.
- 256 experts, top-8 routes, source M=32/512/2048, uniform and skewed routes.
  For M=2048, uniform has 64 rows/expert and skewed alternates 1/127.
- Identical compact FP16 activations, packed four-byte weight-load layout,
  8/16/32-row adaptive TensorOps tiles with K=32, 64-thread groups, maximum
  launch grid/early returns, barriers, and non-prefetch schedule.
- FP16 matrix operands; FP32 accumulation and gate scratch; fused SiLU/up
  with final FP16 output; FP32 down output. All variants retain these contracts.
- Common group-64 controls load the same-size BF16 scale/bias pair from the same
  tile-order metadata layout once per 64 K. FP8 uses runtime zero bias; affine
  uses its runtime nonzero bias. Both use FP32 FMA then FP16 operand rounding.
- Weights are 32 signed integer levels times varying scales. FP8 stores the
  integer in E4M3; affine stores `(integer + 16) * 8`, with `scale / 8` and
  `bias = -16 * scale`. Every scalar decoded weight and rounded operand is
  checked for exact equality before GPU timing. No approximation/requantization
  error is present in this common fixture.
- Timed runs use power-of-two scales. Separate shader validation uses varying
  non-power-of-two BF16 scales and covers every shape/route.
- Two five-sample passes reverse the initial ordering and alternate order within
  each pass. Each timed sample follows at least 250 ms warm execution. Loading,
  packing, compilation and exhaustive CPU checks are excluded.

## Variants

| Variant | Payload | Metadata | Conversion |
|---|---|---|---|
| packed | FP8 | Original BF16 scale per 128×128 block | Unmodified production packed kernel |
| fp8_native | FP8 | Same original scales | Generated-skeleton FP8 arithmetic anchor |
| fp8_native_lut | FP8 | Same original scales | 256-entry exact FP32 lookup |
| fp8_g64 | FP8 | Scale/zero-bias per row × 64 K | FP8 arithmetic + FMA |
| fp8_g64_lut | FP8 | Same group-64 layout | Exact lookup + FMA |
| affine_g64_cast | Unsigned affine-8 | Scale/bias per row × 64 K | Integer-to-float + FMA |
| affine_g64_bits | Unsigned affine-8 | Same group-64 layout | Exact half-bit code conversion + FMA |

The half-bit technique adapts Splash `staged_linear`'s exact `1024 + code`
construction to this common four-consecutive-byte layout. It does not port
Splash's other layouts, BF16 epilogue, prefetch or dispatch policy. Native FP8
coefficients consume 2 bytes per 16,384 weights; affine group-64 consumes 4 bytes
per 64 weights. Consequently, comparisons to native FP8 include metadata and
coefficient-access effects; comparisons within group-64 isolate the decoder
choice more closely. The group-64 FP8 zero-bias format is an experimental
control, not the production checkpoint format.

## Results

GPU milliseconds, median of ten samples; 2048 source tokens.

| Operation | Routes | Original packed | Generated FP8 | Native FP8 LUT | G64 FP8 | G64 FP8 LUT | Affine cast | Affine bits |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gate_up | uniform | 4.861 | 4.835 | 3.932 | 4.847 | 4.044 | 3.669 | 3.869 |
| gate_up | skewed | 6.005 | 6.005 | 4.698 | 6.254 | 4.708 | 3.912 | 3.915 |
| down | uniform | 2.597 | 2.625 | 2.360 | 2.682 | 2.421 | 2.302 | 2.322 |
| down | skewed | 3.185 | 3.168 | 2.643 | 3.235 | 2.664 | 2.393 | 2.386 |

Positive percentages below mean less GPU time for affine bits.

| Operation | Routes | Saving vs G64 FP8 | Saving vs original packed | Saving vs native FP8 LUT |
|---|---|---:|---:|---:|
| gate_up | uniform | 20.2% | 20.4% | 1.6% |
| gate_up | skewed | 37.4% | 34.8% | 16.6% |
| down | uniform | 13.4% | 10.6% | 1.6% |
| down | skewed | 26.2% | 25.1% | 9.7% |

### Repeatability and smaller shapes

All variants and both pass medians are retained in local `summary.json`.
The following table includes every source-row shape; each cell is affine-bits
GPU time reduction relative to the unmodified packed FP8 kernel. Negative means
affine is slower.

| Operation | M | Routes | Affine-bits time reduction |
|---|---:|---|---:|
| gate_up | 32 | uniform | 5.3% |
| gate_up | 32 | skewed | 5.2% |
| gate_up | 512 | uniform | -2.5% |
| gate_up | 512 | skewed | 10.2% |
| gate_up | 2048 | uniform | 20.4% |
| gate_up | 2048 | skewed | 34.8% |
| down | 32 | uniform | 6.5% |
| down | 32 | skewed | 6.6% |
| down | 512 | uniform | -0.1% |
| down | 512 | skewed | 8.5% |
| down | 2048 | uniform | 10.6% |
| down | 2048 | skewed | 25.1% |

## Interpretation

The controlled 2048-token results support weight format/decoder cost as a
material contributor to the expert-kernel gap. Changing only the representation
and its decoder in the common group-64 framework changes GPU time substantially
while every output remains numerically identical. Holding metadata fixed also
shows that this effect is not merely a different scale layout.

The native FP8 lookup recovers a substantial part of the opportunity without
changing stored weights. This generated diagnostic returns the scaled FP16
operand from its decoder helper; the earlier cost-profile helper returned FP32
and scaled at the caller. Do not combine lookup timings across those generated
kernels as though the compiled code were identical. All comparisons here use
one current harness and expose the unchanged packed kernel as an anchor.

This does not show that all of the Splash gap, or the full-model prefill gap,
comes from quantization. The real model includes dense, attention, DeltaNet,
gather/combine and residency costs. Splash's native activation layout, precision,
compiler options and scheduling still differ. No new full Splash endpoint or
whole-model affine measurement was made. Smaller cases also do not show a
universal affine advantage.

A real FP8 checkpoint cannot generally be converted to affine-8 without changing
weights. The exact shared synthetic fixture here is a causal timing control, not
evidence that such a conversion preserves model quality. The immediate
precision-preserving candidate remains FP8 lookup; an affine model path would
need independent model-quality and full-evaluation checks.

## Validation and reproduction

- 840 uninstrumented timing samples; 240 matrix correctness cases.
- 13,440 independent double-precision matrix reference points.
- 13,824 affine scalar reference checks across the three passes, covering all 256 codes, signed scales and nonzero biases.
- All live outputs are finite, output guards pass, and all matrix variants have
  zero numerical difference from the reference variant.
- Every weight is checked for equality before and after operand rounding:
  536,870,912 gate/up values and 268,435,456 down values per pass.
- Shader validation uses 72 cases (six generated variants); timing adds the
  original packed anchor for 84 cases per pass. The validation record retains
  the original `broad_fp8` flag name, meaning varied BF16 scales in this harness;
  timed records use the clearer `varied_scales` name. Shader source hashes match
  all three passes; timing harness hashes match both timing passes.
- No thermal/performance warning was reported after the passes. No OS/power
  settings or model files were changed. Raw evidence remains local and ignored.

```sh
MTL_SHADER_VALIDATION=1 python3 scripts/compare-expert-quantization.py \
  --splash /path/to/splash --out artifacts/expert-quantization-control \
  --label validation --varied-scales --runs 0
python3 scripts/compare-expert-quantization.py \
  --splash /path/to/splash --out artifacts/expert-quantization-control \
  --label forward --runs 5
python3 scripts/compare-expert-quantization.py \
  --splash /path/to/splash --out artifacts/expert-quantization-control \
  --label reverse --reverse --runs 5
python3 scripts/summarize-expert-quantization.py artifacts/expert-quantization-control
```

The generator now includes the original packed anchor in validation too; for
this historical capture the retained validation contains 72 cases. The summarizer
accepts both the retained six-variant validation and the seven-variant validation
with the added original anchor. Use a new artifact directory for a new run.
Splash dependency pin: `1482f6bbc2abf12375332725302019594150b6cb`.
Original logs, generated sources/binary, commands, source hashes and environment
snapshots are under `artifacts/expert-quantization-control/`; reports are not
endpoint-bench data and do not amend a frozen experiment.
