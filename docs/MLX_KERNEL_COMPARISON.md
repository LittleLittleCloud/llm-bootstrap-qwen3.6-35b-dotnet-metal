# MLX kernel comparison — 2026-11-02

The largest measured per-operation gap is long-context attention: MLX is
2.4–3.1x faster for 2048 new queries with 8K–30K history. Dense FP16 matmul does
not show a comparable deficit. MLX affine-8 expert matmul is faster than our
current block-FP8 path, but input gathering/conversion reduces that advantage.
These isolated results do not explain the entire end-to-end leaderboard gap.

## Method

- Apple M5 Pro, 48 GiB, macOS 26.6.2. MLX/MLX Metal 0.32.4, NumPy 2.5.4,
  Python 3.13.14. Our production kernels are unchanged from `7db999b`.
- Two passes with engine order reversed: ours→MLX, then MLX→ours. Eleven
  measured calls per configuration per pass; tables show medians of all 22.
  No concurrent GPU benchmark. CPU-reference checks are followed by at least
  250 ms of GPU warmup. Exploratory pass `a` lacked that sustained warmup and
  is excluded, including its unstable short-kernel timings.
- Synchronized wall time, including host submission and completion. MLX builds
  and evaluates a fresh lazy result every call. Compilation, fixture creation,
  offline quantization and reference checks are excluded. The .NET bridge also
  records GPU time, but ratios use wall time on both sides. Sub-millisecond
  differences include runtime/allocation overhead and are not pure GPU timings.
- Identical deterministic activations and original FP8-derived FP16 weights.
  MLX affine-8 is a separate, **requantized** representation (group size 64),
  checked against its own dequantized reference. It is not bit-identical to
  our E4M3 weights with 128x128 BF16 scales.
- Both boundaries return FP32. Our TensorOps accumulate and output FP32;
  MLX's half-input matmul/SDPA produces FP16 before conversion to FP32.
  Attention's fused path also need not round normalized probabilities in the
  same way as ours. This is matched-input numerical validation, not bitwise
  equivalence or a model quality evaluation.

## Attention

16 query heads, 2 KV heads, head dimension 256, FP16 K/V, scale 1/16 and the
same lower-right causal mask. Both include FP32→FP16 query conversion and
FP32 sigmoid output gating, starting from the same token-major layout.
MLX uses `mx.fast.scaled_dot_product_attention` in a compiled wrapper; our
selected `fp16-device-causal` production path is measured.

| History + new queries | Ours ms | MLX ms | Ours / MLX |
| --- | ---: | ---: | ---: |
| 0 + 512 | 0.499 | 0.595 | 0.84x |
| 0 + 2048 | 3.886 | 2.611 | 1.49x |
| 8192 + 2048 | 36.519 | 15.184 | 2.41x |
| 10520 + 190 | 4.172 | 2.788 | 1.50x |
| 32768 + 63 | 5.011 | 3.858 | 1.30x |
| 30000 + 2048 | 169.643 | 55.392 | 3.06x |

Our path materializes FP32 QK scores and FP16 probabilities, then reads them
again for AV. At 30K+2048 it makes 97 dispatches: one query conversion and
32 workspace chunks × QK/softmax/AV. The workspace cap limits intermediate
memory, but creates repeated passes through device memory.

MLX 0.32.4's dispatch source selects NAX attention with head-dimension splitting
for the 2048-query, D=256 causal cases. A runtime graph export also shows a
single `ScaledDotProductAttention` primitive for 2048 queries, whereas 512
queries use the fallback matmul/softmax graph. This supports targeting a better
fused attention implementation. The exact shader was not GPU-captured.
[MLX source](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/scaled_dot_product_attention.cpp)

Both implementations pass independent CPU references. Maximum sampled absolute
error is 3.64e-6 for ours against its rounded-probability reference and 6.08e-5
for MLX against FP64 softmax with FP16 Q/K/V. These are different rounding
references; the maxima are not a comparative quality score. Our output-alias
checks also pass.

## Dense matrix multiplication

Shapes are M×K×N. The core comparison starts with identical resident FP16 inputs
and weights. Our complete FP8 projection column includes input conversion and
weight unpacking; MLX affine-8 includes input conversion and quantized matmul,
with affine weights prepared before timing.

| M×K×N | Ours FP16 core ms | MLX FP16 core ms | Ours FP8 projection ms | MLX affine-8 projection ms |
| --- | ---: | ---: | ---: | ---: |
| 512×2048×512 | 0.195 | 0.232 | 0.210 | 0.362 |
| 2048×2048×512 | 0.297 | 0.338 | 0.394 | 0.468 |
| 2048×2048×2048 | 0.749 | 0.823 | 0.871 | 1.007 |
| 2048×2048×8192 | 2.312 | 2.699 | 2.597 | 3.057 |
| 2048×4096×2048 | 1.301 | 1.363 | 1.612 | 1.725 |

These shapes do not support a claim that our generic FP16 TensorOps matmul is
several times slower. Results include MLX's FP16→FP32 output conversion and
runtime overhead; they do not establish universal matmul superiority.

## Routed expert projections

256 experts, 16,384 routed rows (2048 tokens × top-8). Uniform routing uses 64
rows per expert; skewed routing alternates 1 and 127. Sorted expert assignments,
tile descriptors and gather indices are prepared before timing for both engines:
routing, sorting, activation and scatter/combine are excluded. This measures
one projection, not a complete gate+up+down MoE layer.

| Projection K→N / routing | Ours FP8 incl. input pack ms | MLX affine-8 incl. input pack ms | Ours FP8 core ms | MLX affine-8 core ms |
| --- | ---: | ---: | ---: | ---: |
| 2048→512 / uniform | 3.878 | 3.456 | 3.442 | 2.145 |
| 2048→512 / skewed | 4.228 | 3.756 | 3.801 | 2.437 |
| 512→2048 / uniform | 3.891 | 3.179 | 3.704 | 2.991 |
| 512→2048 / skewed | 4.268 | 3.328 | 4.086 | 3.140 |

MLX's quantized core is 1.24–1.60x faster; including input packing reduces the
advantage to 1.12–1.28x. Our gather and FP16 conversion are fused, while this
MLX wrapper gathers FP32 rows then casts them. MLX-LM commonly starts with
low-precision activations, so this common-FP32-boundary comparison does not
reproduce every cost in an MLX-LM model. Both engines can reuse gathered gate/up
inputs; the core columns isolate that reuse case. The down path needs no gather.

A control with identical fully predecoded FP16 expert weights, including input
packing but excluding offline weight decoding, gives:

| Projection / routing | Ours ms | MLX ms |
| --- | ---: | ---: |
| 2048→512 / uniform | 4.333 | 4.425 |
| 2048→512 / skewed | 4.284 | 4.543 |
| 512→2048 / uniform | 4.406 | 3.779 |
| 512→2048 / skewed | 4.231 | 3.858 |

The FP16 control has no 2x gap. MLX's packed quantized kernel is more efficient
than either predecoded control here. Quantization format, decoding, weight
bandwidth and kernel layout all change together, so these results do not prove
that affine-8 alone causes the speedup. Keeping every layer's experts in FP16
would also use much more memory than this single-layer microbenchmark.

All dense/expert paths pass independent CPU spot references. MLX affine-8 is
validated against its dequantized values, not the original weights; this is
not a model accuracy test. The original matrix tests remain separate.

## Reproduction and evidence

```sh
uv venv artifacts/mlx-kernel-comparison/venv --python python3.13
uv pip install --python artifacts/mlx-kernel-comparison/venv/bin/python \
  mlx==0.32.4 mlx-metal==0.32.4 numpy==2.5.4
dotnet build src/Bootstrap.Engine -c Release
# Pass b: run ours, then MLX. Pass c: run MLX, then ours.
dotnet src/Bootstrap.Engine/bin/Release/net10.0/Bootstrap.Engine.dll \
  compare-mlx-kernels --runs 11 > artifacts/mlx-kernel-comparison/ours-b.jsonl
artifacts/mlx-kernel-comparison/venv/bin/python scripts/compare-mlx-kernels.py \
  --runs 11 > artifacts/mlx-kernel-comparison/mlx-b.jsonl
artifacts/mlx-kernel-comparison/venv/bin/python scripts/compare-mlx-kernels.py \
  --runs 11 > artifacts/mlx-kernel-comparison/mlx-c.jsonl
dotnet src/Bootstrap.Engine/bin/Release/net10.0/Bootstrap.Engine.dll \
  compare-mlx-kernels --runs 11 > artifacts/mlx-kernel-comparison/ours-c.jsonl
python3 scripts/summarize-mlx-kernels.py artifacts/mlx-kernel-comparison
```

Both benchmark commands accept `--family dense|experts|attention`. The ignored
local artifact directory contains raw runs, stderr, environment metadata,
summary statistics, source/result hashes, graph exports and the exact MLX
0.32.4 dispatcher sources. There is no new full-model evaluation, leaderboard
submission, inference-kernel change, or prediction of end-to-end speedup.
