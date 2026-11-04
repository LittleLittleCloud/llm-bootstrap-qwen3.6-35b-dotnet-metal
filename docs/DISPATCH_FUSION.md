# Prefill dispatch reduction experiments

Base engine: `652ed2ad4c2040f6de104969a72029070310574c` (projection preparation PR #18). This iteration implements five independently selectable experiments. `--dispatch-fusion` is a bit mask; zero preserves the base path. Selection is based on both launch count and uncached wall time, not launch count alone.

| Bit | Experiment | Mechanism |
|---|---|---|
| 1 | Resident small projection weights | Decode the 30 GDN a/b pairs and 40 shared-expert scalar gates to their existing FP16 operand representation once per engine. Keep immutable weights resident; no target re-quantization. |
| 2 | Staged affine dense matmul | Decode affine-8 group-64 weights inside matmul. The selected diagnostic uses four independent SIMD row slices, a shared double-buffered K32 weight stage and prefetch. K64 and an initial shared-activation-stage prototype are also measured. |
| 4 | Grouped projections | Combine Q/K/V, GDN a/b and shared gate/up weight conversion and matrix launches. Each projection keeps its original independent FP32 accumulation and output. |
| 8 | Shared expert epilogue | Compute gate and up in one launch, apply the original FP32 SiLU product and emit FP16 down-projection operands directly, omitting the dead FP32 activation write. |
| 16 | Attention preparation | One launch performs Q split/gate extraction, Q/K RMSNorm, RoPE, FP16 KV writes and grouped FP16 query packing. |

The staged affine prototype is a diagnostic, not a default optimization: all three tested schedules regressed against the existing unpack-plus-matmul path. Keep the fast device-resident TensorOps core for production dense projections. Preserve the failed experiments and their observations.

## Boundaries and memory

These paths are restricted to prefill with more than eight rows, FP16 TensorOps and predecoded projection mode. Attention fusion additionally requires at least 32 rows, FP16 KV and the online attention path. Decode and the 1–8-row verifier retain their original paths. Mode changes clear reusable prefix checkpoints.

Small weights are prepared outside an active command buffer, once per engine. A cold diagnostic includes this setup; warm comparisons retain it. Grouped projection scratch grows the existing 32 MiB scratch to 36 MiB. Shared packed activation reuses routed expert activation storage after expert down has consumed it, without a separate allocation. The buffers remain alive through all encoded work. Attention uses the existing query workspace.

## Validation and measurement

`test-dispatch-fusion` compares complete outputs, packed bit patterns and untouched guards. Staged dense also has independently computed CPU dot-product samples. Fixtures cover small and partial matrix tiles, grouped projections with different output widths, shared gate/up activation, attention tails, nonzero KV offsets and untouched KV prefixes. Run it with Metal API/GPU validation as well.

`compare-dispatch-fusion` alternates variant order inside one process and clears prefix reuse before every request. Loading/tokenization are excluded from wall timing. Run zero is a warmup, and all subsequent observations are retained. Resident scratch remains allocated across variants for the warm comparison. Profile counters are collected separately: per-encoder intervals can overlap and must not be added as exclusive wall costs.

The exact same 10,520/29,939-token fixtures and 2048-token chunks are used as the previous iteration. No target weights, recurrence precision, attention precision, draft checkpoint, prefix/cache semantics or evaluator code change. Local raw evidence is under `artifacts/dispatch-fusion/`; this is not a public artifact download.

## Results at engine `172097c19a953b7d12e6b73c12dcd679a6d08896`

Default tuned affine prefill selects mask **29** (1 + 4 + 8 + 16). The affine staging bit remains off. Other configurations keep mask zero unless explicitly requested. All five ideas are implemented and measured; the rejected dense staging is not presented as an improvement.

Five warm observations per variant alternate baseline/candidate order in one process. Both retain the same allocations after warmup. All ten paired wall-time differences favor the candidate.

| Tokens | Baseline dispatches | Selected dispatches | Reduction | Baseline median ms | Selected median ms | Throughput gain |
|---|---:|---:|---:|---:|---:|---:|
| 10520 | 9438 | 7218 | 23.52% | 3863.56 | 3839.96 | 0.61% |
| 29939 | 21894 | 16344 | 25.35% | 12777.87 | 12652.09 | 0.99% |

The 10K ranges are 3845.71–3915.50 ms (baseline) and 3797.51–3862.18 ms (selected); paired savings are 9.61–55.17 ms, median 52.88 ms. The 30K ranges are 12746.13–12808.74 ms and 12630.82–12705.15 ms; paired savings are 66.32–177.93 ms, median 115.96 ms. Ratios of independent medians and medians of paired savings differ when timing drifts; both are disclosed.

### Where the 30K launch reduction comes from

| Change relative to mask 0 | Dispatches removed |
|---|---:|
| Resident small projection weights | 1500 |
| Grouped projections, after resident small weights | 2250 |
| Shared-expert activation/packing, after grouped projections | 1200 |
| Attention preparation | 600 |
| Total | 5550 |

Routed expert gate/up and down still each launch 600 times. The remaining large counts are `matrix_product_predecoded` 3315, `projection_unpack_half` 2715, `rmsnorm_pack` 1650, `projection_unpack_many` 750 and `matrix_product_many` 600. Reducing indirect threadgroup grids would not reduce these kernel launch counts.

### Independent ablations

The earlier three-warm-run 30K sweep used the larger 48 MiB grouped scratch and still materialized shared FP32 activation; it is a separate development measurement, not the final paired comparison.

| Mask | Median ms | Dispatches |
|---|---:|---:|
| 0 | 12837.16 | 21894 |
| 1 | 12771.69 | 20394 |
| 4 | 12786.88 | 19194 |
| 8 | 12802.34 | 19494 |
| 16 | 12663.30 | 21294 |
| 29 | 12653.65 | 16344 |

The 10K preliminary sweep included bit 2: its median was 4236.45 ms versus 3870.98 ms for mask 0, despite fewer launches. Three dense staging schedules were tested: initial shared X/W staging, device X with double-buffered K32 weights, and K64 weights. The best of these still regresses on every measured representative dense shape.

Final microbenchmarks below are GPU command-buffer medians over five warm observations; setup and loading are excluded. Grouped projection includes weight conversion; the shared-expert epilogue benchmark starts with decoded weights resident for both variants.

| Workload | Original ms | Candidate ms |
|---|---:|---:|
| staged-2048-256-2048 | 0.08550 | 0.12525 |
| staged-2048-512-2048 | 0.14837 | 0.22646 |
| staged-2048-2048-512 | 0.14433 | 0.24050 |
| staged-2048-2048-4096 | 1.19075 | 1.57175 |
| staged-2048-8192-2048 | 2.30892 | 3.03613 |
| many | 0.34196 | 0.32588 |
| shared | 0.32613 | 0.32038 |
| attention | 1.24087 | 0.47100 |

### Correctness and resource observations

- Release build: zero warnings/errors. All 44 kernel checks pass, also with Metal API/GPU validation. All compared FP32 outputs have zero measured error; packed query/KV/shared activation bits match. The packed-only shared path leaves the diagnostic FP32 destination untouched.
- Full 10K/30K needle answers, three 256-token decode runs, 45 verifier checks and 48 prefix-cache checks pass. Smoke logits SHA-256 `5f85879f85492a1f7aa729069bd49b5df7ca8646cd4ddbfda2fdda56d4c720e2` is identical to the base.
- Final extra retained allocation is 12,189,696 bytes (11.625 MiB): 7,995,392 bytes of small weights and a 4 MiB scratch increase. The paired comparisons retain identical allocations in both modes.
- Whole-machine new swap-outs: preliminary 10K ablations 327.8125 MiB; preliminary 30K ablations zero; final paired comparison zero; full functional/profile checks 88.625 MiB. No thermal warnings were observed. Swap-outs are not attributed solely to the engine. The functional process retains additional verifier/cache buffers.
- Instrumented profile wall times (4577.87/13339.47 ms at 10K/30K) are not used for speedup claims. Sampling changes encoder boundaries, and intervals overlap. The ordinary, uninstrumented paired timings above are primary.

## Reproduction

Build with `dotnet build src/Bootstrap.Engine -c Release`. Kernel diagnostics: `test-dispatch-fusion --runs 5`; use `--dense-kernel dense_affine_staged` or `dense_affine_pipeline64` for the rejected alternatives. Repeat with `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1` and `--runs 0` for validation.

For model tests, use the verified target/draft/cache paths and the unchanged fixture files with `--tuned-prefill --delta-mode staged8 --delta-preparation fused --dense-prefill all --projection-preparation fused --context-window 32768`. `compare-dispatch-fusion --variants 0,29 --runs 5 --prompt-file ... --long-prompt-file ...` performs the paired comparison. `--dispatch-fusion 0` restores the base path; `--dispatch-fusion 29` selects the retained combination. Full endpoint evaluation is a separate pinned LLM Bootstrap experiment, `2026-11-03-qwen36-dispatch-fusion-001`.

## Full endpoint evaluation and follow-up controls

The unchanged Full report `20261104-014304.json` completes all seven tests with no execution failures and required conformance passing. Effective 10K prefill increases from 2558.5 to 2627.5 tok/s (+2.70%); the new request has 10,415 tokens and 3963.8 ms TTFT. Each report contains only one eligible prefill request and uses a different seed, so the paired development comparisons above are stronger evidence of the implementation effect. The inherited 3113 tok/s prefill goal is missed by 485.5; the 50 tok/s decode headline goal is met by 9.67.

The report also records worse speed and short decode metrics: 100-token decode 30.82 → 26.93 tok/s, 1000-token 68.51 → 59.67, 4000-token 89.90 → 89.77. Speed is only a ten-completion-token response, 41.79 → 32.72 tok/s, not sustained generation. These observations are preserved, not replaced by the follow-up control.

To investigate, the exact original evaluated binary and shader at `652ed2a` and the selected binary/shader at `172097c` were served serially with the identical fixed public decode prompt and request parameters. Three warm observations after one excluded warmup give 30.051 → 29.783 tok/s at 100 tokens (−0.89%) and 67.089 → 66.731 at 1000 tokens (−0.53%). Output text hashes and token counts match; neither process had new swap-outs. This does not reproduce the roughly 13% formal short-decode decline, but does not establish its cause or reproduce the complete preceding evaluator workload. It is a development HTTP control, not a replacement formal report.

Both thinking modes score 9/10 with full format compliance and zero reasoning tokens. The previous report scored 10/10 on different questions. Replaying the failed question (`7c9b40ff:q10`) against the original engine produces the identical wrong answer in both modes; that particular error predates these fusions. Needle is 10/10. Formal evaluation observes 610 MiB of whole-machine new swap-outs, wired memory 39.02–39.95 GiB, compressor 2.80–4.08 GiB and no thermal warnings. These are not engine-attributed measurements. The evaluator labels the 32768 context window as assumed; the engine explicitly enforces that limit.

Local evidence manifests: `artifacts/dispatch-fusion/evidence-manifest.json` SHA-256 `ed10faabb93c55e2d5008ed8ad8798b17c2743df43bf255403146cac83f6717c`; follow-up controls `artifacts/dispatch-decode-control/evidence-manifest.json` SHA-256 `3a2761e3878e5a8adffdaa9b59b2a915daccc656b398bc4832cb938290c33f96`. These are local-only raw records available to workspace reviewers. The formal report SHA-256 is `c18cc25639f88888cbcb829849f206b0e8c57a6a75c40b912cf7f65c5d04478e`.
