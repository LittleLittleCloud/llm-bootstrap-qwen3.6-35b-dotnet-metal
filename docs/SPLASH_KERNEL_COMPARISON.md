# Matched-workload Splash GPU kernel comparison

Measured on Apple M5 Pro (20 GPU cores, 48 GiB), 2026-11-02 America/Los_Angeles. These are isolated development measurements, not endpoint evaluation results.

## Findings

The routed gate/up and down kernels are the measured deficit. Dense is much closer and does not show the same systematic gap. The tables use GPU command-buffer elapsed time, not host wall time. Ours / Splash greater than 1 means Splash is faster.

## Routed experts

256 experts, top-8 routes per token. Gate/up includes fused SiLU and multiplication; down is a separate projection starting from identical synthetic activations. Router, grouping, gather, combine and the shared expert are excluded. M is source tokens, so the projections process M × 8 logical routed rows.

| M | Operation K → N | Distribution | Ours GPU ms | Splash GPU ms | Ours / Splash |
| ---: | --- | --- | ---: | ---: | ---: |
| 2048 | gate_up 2048 → 512 | uniform | 5.370 | 3.369 | 1.59x |
| 2048 | gate_up 2048 → 512 | skewed | 6.529 | 3.553 | 1.84x |
| 2048 | down 512 → 2048 | uniform | 2.924 | 1.893 | 1.54x |
| 2048 | down 512 → 2048 | skewed | 3.452 | 1.953 | 1.77x |
| 512 | gate_up 2048 → 512 | uniform | 2.915 | 2.239 | 1.30x |
| 512 | gate_up 2048 → 512 | skewed | 3.060 | 2.244 | 1.36x |
| 512 | down 512 → 2048 | uniform | 1.565 | 1.201 | 1.30x |
| 512 | down 512 → 2048 | skewed | 1.625 | 1.190 | 1.37x |

At M=2048, uniform means 64 rows/expert; skewed alternates 1 and 127. At M=512 these are 16 and 1/31. Every logical route is assigned once, every source token has eight distinct experts, and both engines receive the same sorted route order. Physical padding follows each implementation: Splash stores 32 rows per tile, whereas ours uses contiguous live rows. Padding and original tile selection remain part of kernel cost.

Original dispatch behavior is retained: ours launches its maximum tile grid with early returns, Splash uses an indirect live grid. Neither side is given an artificially fixed common tile height or grid that would change its normal scheduling. Both use the original 32-row staged expert kernel for this workload; Splash additionally selects its column-dot path for an expert with only one or two rows.

## Dense projections

M × K × N shapes. Native 16-bit activation inputs are already resident for both. Our complete projection includes FP8 weight decoding into FP16 scratch followed by matmul; the extra core-only column excludes that decoding. Splash stages affine-8 weights inside the matmul. Input conversion, allocation and loading are excluded for both.

| M × K × N | Ours FP16 core ms | Ours including weight decode ms | Splash affine-8 ms | Complete ours / Splash |
| --- | ---: | ---: | ---: | ---: |
| 512 × 2048 × 512 | 0.045 | 0.052 | 0.075 | 0.70x |
| 2048 × 2048 × 512 | 0.143 | 0.152 | 0.188 | 0.81x |
| 2048 × 2048 × 2048 | 0.544 | 0.580 | 0.617 | 0.94x |
| 2048 × 2048 × 8192 | 2.120 | 2.319 | 2.347 | 0.99x |
| 2048 × 4096 × 2048 | 1.155 | 1.256 | 1.236 | 1.02x |
| 2048 × 512 × 2048 | 0.137 | 0.146 | 0.159 | 0.92x |

## Timing, inputs and precision

- Bootstrap harness commit `c5ea4e56dd722dc93d91ec683cddc4281ae7d9d9`. The production shaders are byte-for-byte the merged main implementation `96e6d774e3f7633ca381a665c368827ca94cf054`; no serving kernel or default changes in this work.
- Splash source pinned to `1482f6bbc2abf12375332725302019594150b6cb`. This is the historical comparison revision, not a claim about a newer Splash revision. Original `moe_gguf.metal`, `gguf_linear.metal` and their headers are compiled without kernel edits. This tests Splash directly, not the MLX runtime.
- Two passes with initial order reversed, nine measured calls per variant per pass (18 total). Order also alternates within each pass. Every sample follows at least 250 ms of repeated warm executions. Benchmarks run serially on the same GPU.
- `MTLCommandBuffer.GPUEndTime - GPUStartTime`, one operation per command buffer. Dense decoding and matmul share one command. Compilation, CPU references, allocation, quantization/packing and loading are outside timing. Synchronized host wall times are also retained in the raw JSONL but are not used for ratios.
- This machine has Command Line Tools but no offline `metal` compiler. Both libraries use runtime Metal 4.0 compilation: our original safe math mode; Splash fast math with its original `reassociate(off)` pragmas. Splash normally ships an offline -O3 metallib targeting macOS 26.4. Runtime compilation is a disclosed deviation from that full production build; compilation time is excluded, and no new full Splash endpoint rate is claimed.
- FP8 E4M3 with BF16 128×128 scales for ours, affine-8 group-64 with BF16 scale/bias in Splash’s native tiled planes. The fixture uses 32 signed integer levels and varying power-of-two scales, so every decoded weight is exactly representable in both formats. Gate/up weights differ, experts differ, and columns/groups differ. Activations are bounded dyadic values exactly representable in FP16 and BF16. This removes re-quantization error as a confound without replacing either weight decoder.
- Native precision differences are retained: ours uses FP16 inputs and FP32 accumulation, FP32 gate/up values before a final FP16 activation, and FP32 down/dense output. Splash uses BF16 inputs, FP16 staged weights and FP32 accumulation; gate and up are rounded to BF16 before activation, and its outputs are BF16. The benchmark does not attribute the time difference solely to quantization format, nor claim equal precision.
- A single isolated layer’s weights are resident. This is not a whole-model memory-pressure test, a real-weight distribution study or a measurement of actual model route histograms. It cannot by itself predict the overall prefill speedup.

## Validation and repeatability

- 28 numerical checks across the two measurement passes; 1268 independent double-precision CPU reference points with each implementation’s rounding epilogue. Largest sampled absolute errors: ours 0, Splash 0. These dyadic fixtures are deliberately exactly representable; this is not evidence of full-model numerical equivalence.
- Every live output was checked for finiteness; output guards passed. Dense FP8 decoding was checked over all weights and the complete decoded/core output arrays match. The reference check samples multiple experts, first/last live rows, column and scale-block boundaries.
- A separate Metal shader-validation run passed all 14 cases and 634 CPU reference points, with no diagnostic other than the validation-enabled notice. Its timings are excluded. Later harness changes only increased warmup and added metadata, with no change to kernel calls, bindings or fixtures.
- Two smoke runs, the shader-validation run, both formal microbenchmark passes and the initial unavailable offline-compiler attempts are retained in local artifacts. Only `forward.jsonl` and `reverse.jsonl` feed the result table.

| Operation / M / K → N / routing | Variant | Forward median ms | Reverse median ms | All-sample min–max ms |
| --- | --- | ---: | ---: | ---: |
| dense / 512 / 2048 → 512 / none | ours_core | 0.045 | 0.045 | 0.044–0.168 |
| dense / 512 / 2048 → 512 / none | ours_with_weight_decode | 0.052 | 0.052 | 0.052–0.055 |
| dense / 512 / 2048 → 512 / none | splash | 0.074 | 0.077 | 0.073–0.077 |
| dense / 2048 / 512 → 2048 / none | ours_core | 0.137 | 0.137 | 0.137–0.206 |
| dense / 2048 / 512 → 2048 / none | ours_with_weight_decode | 0.146 | 0.146 | 0.145–0.180 |
| dense / 2048 / 512 → 2048 / none | splash | 0.159 | 0.159 | 0.158–0.270 |
| dense / 2048 / 2048 → 512 / none | ours_core | 0.143 | 0.144 | 0.142–0.270 |
| dense / 2048 / 2048 → 512 / none | ours_with_weight_decode | 0.152 | 0.153 | 0.151–0.278 |
| dense / 2048 / 2048 → 512 / none | splash | 0.188 | 0.188 | 0.185–0.196 |
| dense / 2048 / 2048 → 2048 / none | ours_core | 0.543 | 0.544 | 0.542–0.741 |
| dense / 2048 / 2048 → 2048 / none | ours_with_weight_decode | 0.580 | 0.578 | 0.577–0.909 |
| dense / 2048 / 2048 → 2048 / none | splash | 0.616 | 0.617 | 0.614–0.779 |
| dense / 2048 / 2048 → 8192 / none | ours_core | 2.120 | 2.120 | 2.116–2.292 |
| dense / 2048 / 2048 → 8192 / none | ours_with_weight_decode | 2.313 | 2.325 | 2.296–2.621 |
| dense / 2048 / 2048 → 8192 / none | splash | 2.346 | 2.350 | 2.340–2.569 |
| dense / 2048 / 4096 → 2048 / none | ours_core | 1.163 | 1.144 | 1.135–1.308 |
| dense / 2048 / 4096 → 2048 / none | ours_with_weight_decode | 1.242 | 1.271 | 1.227–1.403 |
| dense / 2048 / 4096 → 2048 / none | splash | 1.231 | 1.238 | 1.226–1.433 |
| down / 512 / 512 → 2048 / skewed | ours | 1.621 | 1.632 | 1.616–2.200 |
| down / 512 / 512 → 2048 / skewed | splash | 1.187 | 1.193 | 1.178–1.229 |
| down / 512 / 512 → 2048 / uniform | ours | 1.562 | 1.565 | 1.558–1.844 |
| down / 512 / 512 → 2048 / uniform | splash | 1.202 | 1.200 | 1.188–1.664 |
| down / 2048 / 512 → 2048 / skewed | ours | 3.464 | 3.450 | 3.444–3.812 |
| down / 2048 / 512 → 2048 / skewed | splash | 1.969 | 1.951 | 1.942–2.269 |
| down / 2048 / 512 → 2048 / uniform | ours | 2.926 | 2.909 | 2.901–3.414 |
| down / 2048 / 512 → 2048 / uniform | splash | 1.890 | 1.895 | 1.872–2.425 |
| gate_up / 512 / 2048 → 512 / skewed | ours | 3.036 | 3.077 | 3.027–3.471 |
| gate_up / 512 / 2048 → 512 / skewed | splash | 2.246 | 2.243 | 2.224–2.651 |
| gate_up / 512 / 2048 → 512 / uniform | ours | 2.893 | 2.968 | 2.872–3.276 |
| gate_up / 512 / 2048 → 512 / uniform | splash | 2.234 | 2.242 | 2.205–2.448 |
| gate_up / 2048 / 2048 → 512 / skewed | ours | 6.531 | 6.527 | 6.455–6.801 |
| gate_up / 2048 / 2048 → 512 / skewed | splash | 3.538 | 3.562 | 3.506–3.998 |
| gate_up / 2048 / 2048 → 512 / uniform | ours | 5.364 | 5.386 | 5.284–5.840 |
| gate_up / 2048 / 2048 → 512 / uniform | splash | 3.372 | 3.339 | 3.315–3.808 |

No OS or power changes were made. Both post-pass system checks reported no thermal or performance warning. System-wide swap readings and power configuration are retained in the identity/after JSON files; they are not model-attributed memory measurements.

## Implication

Prioritize the routed expert paths. Splash’s tiled affine weight loads, staged prefetch, native BF16 epilogue and special handling of tiny experts are candidates for separate ablation. This comparison varies those together and cannot isolate one cause. Dense tile size alone is not supported as the main source of the remaining gap. Do not multiply the expert-kernel ratio by the entire model runtime.

## Reproduction

Use an unchanged checkout of Splash at the pinned commit, then:

```sh
python3 scripts/compare-splash-kernels.py --splash /path/to/splash --runs 9 --label forward
python3 scripts/compare-splash-kernels.py --splash /path/to/splash --runs 9 --reverse --label reverse
python3 scripts/summarize-splash-kernels.py artifacts/splash-kernel-comparison
```

For validation only, add `MTL_SHADER_VALIDATION=1` and use `--runs 1 --label shader-validation`; do not mix its timing into the summary. `--family experts|dense` supports narrower diagnostics. The summarizer requires both complete uninstrumented passes from a clean checkout with matching source/binary hashes.

Relevant pinned Splash sources:

- [Expert kernels](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/metal/kernels/shared/moe_gguf.metal)
- [Dense kernels](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/metal/kernels/shared/gguf_linear.metal)
- [Weight-plane layouts](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/metal/abi/QuantFormat.h)

Raw runs, source/fixture hashes, build identity, per-pass statistics and the complete evidence manifest are local-only in `artifacts/splash-kernel-comparison/`. They are accessible to workspace reviewers, not public downloads. No official evaluation report or leaderboard index is changed.

Evidence SHA256:
- `forward.jsonl`: `ec8984a4d147b141bb27854c533710d23f535b71659f174747b3aff555ff8050`
- `reverse.jsonl`: `562e2b825d9870f1c45d26228c55df27011047c1d8ffe111357777f6ed44355a`
- `forward-identity.json`: `39bdc920a301a576b134579826e34e9a075821eacdb8398ddf67841d7d422b47`
- `reverse-identity.json`: `39bdc920a301a576b134579826e34e9a075821eacdb8398ddf67841d7d422b47`
- `shader-validation.jsonl`: `0da61cc41749f513ef6dce81d31786a177667a76d77f86bf8e1193fa243159e9`
- `summary.csv`: `3052432ca5395fc0ec5c4e77ff2bad455f498086e7d1035ec4bb03452be22729`
- `evidence-manifest.json`: `c19cdfbd4827a390424a11f62c9a5cd840e939610f1dbe37aa8eaae838e043df`
