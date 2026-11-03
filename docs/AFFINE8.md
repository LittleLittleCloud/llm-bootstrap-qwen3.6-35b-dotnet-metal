# Affine-8 text inference

`--affine-cache <directory>` selects unsigned affine-8 weights with group size
64 and interleaved BF16 scale/bias pairs. It requires `--tuned-prefill` and the
fused FP16 gathered-expert path. Omit it to use the original FP8 checkpoint.

This is a derivative of the pinned official FP8 checkpoint, **not** the
`mlx-community` checkpoint used by Splash. The engine has no MLX dependency.
FP8 matrices are decoded with their original BF16 block scales and then
requantized. The two BF16 vocabulary matrices (embedding and LM head) are also
quantized. Other text BF16/FP32 tensors are preserved. The unused vision tower
is omitted from this text-only cache; original checkpoint files are unchanged.
DFlash's own weights stay BF16 and it shares the target's affine vocabulary
matrices. KV storage and computational precision follow the existing preset.

## Build and run

From the engine repository, with `MODEL_DIR` pointing to the pinned FP8 source,
`DRAFT_DIR` to the pinned DFlash2 source, and `AFFINE_DIR` to a new cache directory:

```sh
dotnet build src/Bootstrap.Engine -c Release
dotnet src/Bootstrap.Engine/bin/Release/net10.0/Bootstrap.Engine.dll \
  convert-affine8 --model-dir "$MODEL_DIR" --output-dir "$AFFINE_DIR"
dotnet src/Bootstrap.Engine/bin/Release/net10.0/Bootstrap.Engine.dll \
  serve --model-dir "$MODEL_DIR" --draft-dir "$DRAFT_DIR" \
  --affine-cache "$AFFINE_DIR" --tuned-prefill --delta-mode staged8 \
  --context-window 32768 --port 17863
```

Conversion is offline, verifies the original shard hashes, and writes separate
files. It uses per-matrix scratch rather than another in-memory model; the
vocabulary matrices are the largest temporary allocations. Completed shards
have hash receipts that allow an interrupted conversion to resume. An existing
complete cache is never overwritten. Cache manifests pin the source manifest,
source shard hashes, output hashes, sizes and quantization-error aggregates.
Loading verifies these before mapping the cache. No CPU repack is performed at
runtime. Cache format v2 is required; development v1 caches are rejected.

For each row group, the converter computes min/max, rounds the bias and
`(max-min)/255` scale to BF16, and chooses the nearest clamped code using those
stored values. Reconstruction is `fma(code, scale, bias)`. Constant groups use
zero scale; non-finite inputs and overflowing metadata are rejected.

Dense matrices use row-major codes and metadata. Routed experts use packed
32x32 byte tiles and metadata ordered by row tile, column group and row lane.
The fused prefill kernel stages affine weights into FP16 TensorOps operands,
reuses each metadata pair for two K32 steps, and accumulates in FP32. Gate
scratch and down outputs remain FP32. Decode and short speculative verification
dequantize to FP32. Original FP8 execution remains available.

## Validation

```sh
MTL_SHADER_VALIDATION=1 dotnet src/Bootstrap.Engine/bin/Release/net10.0/Bootstrap.Engine.dll affine8-self-test
dotnet src/Bootstrap.Engine/bin/Release/net10.0/Bootstrap.Engine.dll packed-experts-self-test
```

The affine suite passed 187,194 checks, including independent CPU dot-product
references, every byte-code value, non-power-of-two metadata, row tails
1/2/7/8/9/16/17/31/32, K=128/512/2048, fused gate/up, down, serial decode,
speculative verification, dense and small projections, embedding, cache loading,
hash rejection, constant/non-finite inputs and source-file preservation.
Maximum normalized CPU-reference error was `1.006e-7`. The original packed-FP8
suite passed 46 checks with exact outputs.

The real v2 cache contains 34,634,465,280 quantized values and occupies
34.329 GiB. Relative L2 error against its source is 0.54975% overall:
0.54980% for formerly FP8 matrices and 0.54898% for vocabulary matrices.
These are weight errors, not a model-quality guarantee. Cache-manifest SHA-256:
`7106f2aaaa4ada10f4f3a8512d1b5ecef83f8e271b542ad6549d56ff582b4974`.

## Memory experiments retained as negative evidence

The first recipe retained BF16 vocabulary and unused vision tensors. Its cache
was 36.049 GiB and the first full-model command failed with Metal insufficient
memory. Eight-layer command submissions passed the arithmetic smoke but caused
severe paging; the development run was stopped during its first 10K prompt.

Separating embedding, transformer and output submissions completed the 10K/30K
needle answers, but waiting outweighed the GPU savings. Its three warm 10K
samples had median 5,714.80 ms wall time and 4,725.46 ms GPU time. The first warm
30K sample took 19,432.65 ms wall and 15,870.26 ms GPU. That development run was
stopped before finishing its remaining checks. No official evaluation was run
for either configuration.

The v2 recipe reduces the cache by 1.720 GiB and restores the original single
command submission for each target block. No OS memory limit or power setting
was changed. Local raw evidence is retained under ignored `artifacts/affine8/`;
weights, prompts and machine-specific logs are not published.

## Whole-model development comparison

On the 20-core M5 Pro / 48 GiB host, run affine then packed FP8 in separate
processes, with the same two prompts, 2048-token blocks, staged8 DeltaNet,
FP16 KV, DFlash2, 32768-token context and three warm uncached repetitions:

| Input tokens | Packed FP8 wall ms | Affine wall ms | FP8 tok/s | Affine tok/s | Throughput change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10,520 | 5,264.19 | 4,752.30 | 1,998.41 | 2,213.66 | +10.8% |
| 29,939 | 17,239.44 | 15,867.52 | 1,736.66 | 1,886.81 | +8.6% |

These are medians from local development checks, not endpoint-bench scores.
Affine wall ranges were 4,739.01–4,777.93 ms and 15,835.43–15,887.70 ms;
FP8 ranges were 5,261.33–5,322.63 ms and 17,239.25–17,330.01 ms.
Both returned the complete expected needle answer at both lengths, passed
45 accepted-prefix verification checks and 48 cache/state checks. FP8 smoke
logits still exactly match the previous packed-FP8 artifact.

The 256-token development decode median was 24.18 tok/s for affine versus
18.99 for FP8. Draft acceptance changed (160 versus 127 accepted tokens), so
this is not an isolated decode-kernel speedup. A single arithmetic smoke had
the same argmax, but affine logits had 0.98672 cosine similarity and 16.24%
relative L2 difference; weight error must not be presented as output error.
Broader quality claims require broader evaluation.

System-wide monitoring observed no new swap-outs during the affine run,
versus 8.765 GiB during FP8 startup and checks. Swap usage was 6,163–6,435 MiB
for affine and 6,147–13,411 MiB for FP8; system wired-memory peaks were
40.44 and 40.06 GiB respectively. No thermal/performance warning was recorded.
Fixed order, pre-existing swap, file-cache state and background applications
limit attribution. Quantization, vocabulary storage and backing-file handling
change together; use the controlled kernel study for decoder-only comparisons.
