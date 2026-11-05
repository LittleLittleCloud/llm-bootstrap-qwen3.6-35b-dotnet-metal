# INT8 attention: input-type and scale controls

The local controls do not support FP16 × INT8 matmul as the main reason the experimental INT8 KV path loses to FP16 KV. Direct mixed-type matmul is faster in these isolated geometries. Changing Q from FP16 to BF16 barely changes complete online attention. Removing scale handling in a unit-scale control improves that kernel by about 6–7%, making scale handling and its surrounding implementation a stronger optimization lead.

This is an attention diagnostic on the M5 Pro, not a new end-to-end evaluation. The production implementation and FP16 default are unchanged. See [the preceding INT8 experiment](INT8_KV_20261104.md) for measured whole-model regressions of 0.84% at 10,520 tokens and 1.98% at 29,939 tokens.

## Matmul controls

Inputs contain exactly the same representable values across FP16, BF16 and INT8 buffers. Outputs accumulate to FP32. Each dispatch processes 256 independent tiles; the table reports median GPU command-buffer time divided by 32 dispatches, across 15 samples. Variant order alternates forward/backward. Every variant is warmed before measurement. Input construction and shader compilation are excluded.

| Left × right input | QK, microseconds/dispatch | PV, microseconds/dispatch |
| --- | ---: | ---: |
| FP16 × FP16 | 54.61 | 67.73 |
| FP16 × INT8 | 35.17 | 38.86 |
| BF16 × BF16 | 57.38 | 65.90 |
| BF16 × INT8 | 34.28 | 38.97 |

QK uses M=32, N=128, K=256, device operands, transposed right operand and non-relaxed precision. PV uses M=32, N=256, K=128, a threadgroup left operand, transposed device right operand and relaxed precision. Geometry and descriptor flags are fixed within each family. The PV FP16 control deliberately uses the same transposed descriptor as INT8; it is not the production FP16 attention layout. These times include memory traffic and staging, so they do not establish arithmetic-unit peak throughput or explain every part of the production difference.

## Complete online-attention type controls

The probe clones the current production online template, changing only the Q and probability template types. All variants share the same INT8 K/V buffers, runtime scales, 32×128 tile, eight SIMD groups, grouped Q layout, and page-transposed V. Q values are exactly representable in both FP16 and BF16, avoiding changed input values as a confound. Each measurement contains one dispatch; medians use 15 alternating samples.

| Q / probability type | History 8192 + 2048 rows, ms | History 27900 + 2048 rows, ms |
| --- | ---: | ---: |
| FP16 / FP16 | 16.385 | 48.004 |
| FP16 / BF16 (current INT8 path) | 16.397 | 47.961 |
| BF16 / FP16 | 16.390 | 47.939 |
| BF16 / BF16 | 16.374 | 47.873 |

Changing just Q to BF16 reduces these medians by only 0.14% and 0.18%, respectively; the observed ranges overlap. An independent earlier nine-sample process also showed differences below 0.2%. These are one attention layer/chunk, not full 10K/30K prefill times. Production PV already uses BF16 probabilities to preserve tiny V scales, so “everything else is FP16” does not describe the actual implementation.

## Scale-handling ablation

A separate control fixes every K/V scale to one. One kernel still loads and applies those runtime scales; the other compiles out the scale loads and multiplications. Both retain identical INT8 data, layout, tile and FP16 Q/BF16 probability types. Outputs match bitwise in the validation run.

| Variant | History 8192 + 2048 rows, ms | History 27900 + 2048 rows, ms |
| --- | ---: | ---: |
| Runtime scale loads/multiplies, all scales = 1 | 15.909 | 47.151 |
| Scale handling compiled out | 14.856 | 44.323 |
| Latency reduction | 6.62% | 6.00% |

This isolates the effect of removing scale handling, including resulting compiler scheduling/register changes, under an artificial input distribution. Unit scales change logits and probabilities relative to the normal-scale test; do not subtract these timings from the normal-scale variant to estimate real-model costs. Real per-token scales cannot simply be removed. This result identifies a target for optimization, not a demonstrated production speedup or a proof that scale loads alone cause the entire regression.

## Validation and reproduction

All eight matmul variants produce finite outputs, preserve tail guards, and match sampled FP64 dot products exactly. Online controls cover history/rows (0,33), (8192,2048), and (27900,2048), with finite output and guard checks. Normal-scale variants pass sampled FP64 softmax/attention references; largest absolute error is 0.0001266. These checks concern arithmetic on the already-quantized cache, not real-model quantization quality. The diagnostic does not change query packing or exercise a production BF16 Q path.

```sh
dotnet build src/Bootstrap.Engine -c Release -o artifacts/attention-type-probe/bin
dotnet artifacts/attention-type-probe/bin/Bootstrap.Engine.dll \
  profile-attention-types --output-dir artifacts/attention-type-probe/new-run --runs 15 \
  > artifacts/attention-type-probe/new-results.jsonl
```

Saved evidence is under `artifacts/attention-type-probe/`: the initial `results.jsonl`, longer batched measurements in `repeat/results.jsonl`, derived `summary.json`, and a separate Metal shader-validation run under `validation/` whose timings are excluded. Each run saves its compiled `combined.metal`. `source-snapshot/` and `manifest.json` preserve the final diagnostic sources and evidence hashes. No full model evaluation or Splash rerun was performed for this type-only diagnostic.
