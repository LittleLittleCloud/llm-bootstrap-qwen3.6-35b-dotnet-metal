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
