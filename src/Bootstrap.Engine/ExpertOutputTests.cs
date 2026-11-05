using System.Diagnostics;
using System.Text.Json;
namespace Bootstrap.Engine;

// Local precision/performance diagnostic: current affine-8 weights, FP16 inputs,
// FP32 matmul accumulation, then either FP32 storage or one BF16 rounding.
internal static class ExpertOutputTests
{
    public static void Run(string source, int runs, bool scalarReference = false)
    {
        if (runs < 0) throw new ArgumentOutOfRangeException(nameof(runs));
        using var gpu = new Metal(source);
        const int experts = 256, n = 2048, k = 512, metaCount = n * k / 32, slot = n * k + metaCount * 2;
        using var raw = gpu.Allocate((ulong)experts * slot);
        using var descriptors = gpu.Allocate(experts * 48);
        for (int e = 0; e < experts; e++) {
            ulong offset = (ulong)e * slot;
            for (int choice = 0; choice < 3; choice++) {
                descriptors.Values<ulong>(experts * 6)[e * 6 + choice * 2] = offset;
                descriptors.Values<ulong>(experts * 6)[e * 6 + choice * 2 + 1] = offset + n * k;
            }
            var codes = raw.Values<byte>(n * k, offset); var metadata = raw.Values<ushort>(metaCount, offset + n * k);
            for (int row = 0; row < n; row++) for (int group = 0; group < k / 64; group++) {
                int at = AffineWeights.GroupOffset(row, group, k, true) * 2;
                metadata[at] = AffineWeights.ToBf((1 + (row + e + group) % 7) / 8192f);
                metadata[at + 1] = AffineWeights.ToBf(-0.0257f * (1 + (row + e + group) % 3));
                for (int j = 0; j < 64; j++) {
                    int col = group * 64 + j;
                    codes[AffineWeights.CodeOffset(row, col, k)] = (byte)(row * 19 + col * 37 + e * 13);
                }
            }
        }
        foreach (var (rows, layout) in new[] { (32, "tails"), (127, "uniform"), (272, "uniform"), (1259, "uniform"), (2048, "uniform"), (2048, "skew") }) {
            int m = rows * 8, size = m * n, outputSize = rows * n;
            using var x = gpu.Allocate((ulong)m * k * 2);
            using var full = gpu.Allocate((ulong)size * 4 + 16);
            using var narrow = gpu.Allocate((ulong)size * 2 + 16);
            using var rounded = gpu.Allocate((ulong)size * 4);
            using var tiles = gpu.Allocate((ulong)(m / 32 + experts + 1) * 16);
            using var tileCount = gpu.Allocate(4);
            using var residual = gpu.Allocate((ulong)outputSize * 4 + 16);
            using var shared = gpu.Allocate((ulong)outputSize * 4);
            using var gate = gpu.Allocate((ulong)rows * 4);
            using var inverse = gpu.Allocate((ulong)m * 4);
            using var rates = gpu.Allocate((ulong)m * 4);
            using var output = gpu.Allocate((ulong)outputSize * 4 + 16);
            var rng = new Random(411 + rows); float Rand() => (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < m * k; i++) x.Values<Half>(m * k)[i] = (Half)Rand();
            for (int i = 0; i < outputSize; i++) { residual.Values<float>(outputSize)[i] = Rand(); shared.Values<float>(outputSize)[i] = Rand(); }
            var initial = residual.Values<float>(outputSize).ToArray();
            for (int i = 0; i < rows; i++) gate.Values<float>(rows)[i] = i % 7 == 0 ? -80 : i % 7 == 1 ? 80 : Rand() * 8;
            var routing = Enumerable.Range(0, m).ToArray(); rng.Shuffle(routing);
            for (int i = 0; i < m; i++) { inverse.Values<uint>(m)[i] = (uint)routing[i]; rates.Values<float>(m)[i] = Math.Abs(Rand()) / 8; }
            int[] counts = new int[experts];
            if (layout == "tails") { int[] tails = [1, 2, 7, 8, 9, 16, 17, 31, 32, 133]; tails.CopyTo(counts, 0); }
            else for (int e = 0; e < experts; e++) counts[e] = layout == "skew" ? (e % 2 == 0 ? 1 : 127) : m / experts + (e < m % experts ? 1 : 0);
            int first = 0, nt = 0;
            for (int e = 0; e < experts; e++) {
                for (int r = 0; r < counts[e]; r += 32) {
                    var tile = tiles.Values<uint>((m / 32 + experts + 1) * 4).Slice(nt * 4);
                    tile[0] = (uint)e; tile[1] = (uint)(first + r); tile[2] = (uint)Math.Min(32, counts[e] - r); nt++;
                }
                first += counts[e];
            }
            if (first != m) throw new Exception("Invalid routing fixture");
            tileCount.Values<uint>(1)[0] = (uint)nt;
            var p = new KernelArgs { N = n, K = k, Kind = 2 };
            void Down(bool bf16) => gpu.Dispatch2D(bf16 ? "expert_matrix_affine_bf16_output" : scalarReference ? "expert_matrix_affine_scalar_output" : "expert_matrix_affine", p, n / 64 * 64, (uint)nt + 1, 64, x, raw, descriptors, tiles, tileCount, bf16 ? narrow : full);
            void Mix(bool bf16, bool alias = false, bool roundedReference = false) => gpu.Dispatch(bf16 ? "batch_mix_bf16_expert" : "batch_mix", new() { M = (uint)rows }, (uint)outputSize, 128,
                residual, bf16 ? narrow : roundedReference ? rounded : full, inverse, rates, shared, gate, alias ? residual : output);
            void Encode(bool bf16, string scope) {
                gpu.Begin(); if (scope != "mix") Down(bf16); if (scope != "down") Mix(bf16); gpu.End();
            }
            full.Values<byte>(size * 4 + 16).Fill(0xa5); narrow.Values<byte>(size * 2 + 16).Fill(0xa5);
            gpu.Begin(); Down(false); gpu.End();
            var firstDown = full.Values<uint>(size).ToArray();
            gpu.Begin(); Down(false); gpu.End();
            var repeatedDown = full.Values<uint>(size);
            int inconsistent = 0; for (int i = 0; i < size; i++) if (firstDown[i] != repeatedDown[i]) inconsistent++;
            Console.WriteLine(JsonSerializer.Serialize(new { type = "expert_output_repeat", rows, layout, scalar_reference = scalarReference, inconsistent }));
            if (inconsistent != 0) throw new Exception("FP32 reference is not repeatable");
            if (!scalarReference) {
                gpu.Begin(); gpu.Dispatch2D("expert_matrix_affine_scalar_output", p, n / 64 * 64, (uint)nt + 1, 64, x, raw, descriptors, tiles, tileCount, full); gpu.End();
                if (!firstDown.AsSpan().SequenceEqual(full.Values<uint>(size))) throw new Exception("Scalar FP32 store changes the reference");
            }
            gpu.Begin(); Down(true); gpu.End();
            double maxError = 0, err2 = 0, ref2 = 0;
            for (int i = 0; i < size; i++) {
                float value = full.Values<float>(size)[i]; ushort expected = AffineWeights.ToBf(value), actual = narrow.Values<ushort>(size)[i];
                if (!float.IsFinite(value) || actual != expected) throw new Exception($"BF16 rounding mismatch {rows}/{layout}/{i}: {actual:x4} vs {expected:x4}; FP32={value}, firstFP32={BitConverter.UInt32BitsToSingle(firstDown[i])}");
                float quantized = AffineWeights.Bf(expected); rounded.Values<float>(size)[i] = quantized;
                double delta = (double)quantized - value; maxError = Math.Max(maxError, Math.Abs(delta)); err2 += delta * delta; ref2 += (double)value * value;
            }
            foreach (var (buffer, bytes) in new[] { (full, size * 4), (narrow, size * 2) })
                if (buffer.Values<byte>(16, (ulong)bytes).ToArray().Any(b => b != 0xa5)) throw new Exception("Expert output guard overwritten");
            double cpuError = 0;
            foreach (int t in new[] { 0, nt / 2, nt - 1 }) foreach (int local in new[] { 0, (int)tiles.Values<uint>(nt * 4)[t * 4 + 2] - 1 }) foreach (int col in new[] { 0, 31, 32, 1023, 2047 }) {
                int e = (int)tiles.Values<uint>(nt * 4)[t * 4], row = (int)tiles.Values<uint>(nt * 4)[t * 4 + 1] + local;
                ulong offset = (ulong)e * slot; double expected = 0, magnitude = 0;
                for (int j = 0; j < k; j++) {
                    var metadata = raw.Values<ushort>(metaCount, offset + n * k); int at = AffineWeights.GroupOffset(col, j / 64, k, true) * 2;
                    float weight = MathF.FusedMultiplyAdd(raw.Values<byte>(n * k, offset)[AffineWeights.CodeOffset(col, j, k)], AffineWeights.Bf(metadata[at]), AffineWeights.Bf(metadata[at + 1]));
                    double term = (double)(float)x.Values<Half>(m * k)[row * k + j] * (float)(Half)weight; expected += term; magnitude += Math.Abs(term);
                }
                double error = Math.Abs(full.Values<float>(size)[row * n + col] - expected); cpuError = Math.Max(cpuError, error);
                if (error > 1e-6 + 3e-6 * magnitude) throw new Exception("Expert CPU dot mismatch");
            }
            Console.WriteLine(JsonSerializer.Serialize(new { type = "expert_output_down_check", rows, layout, scalar_reference = scalarReference, tiles = nt, rounded_bitwise = true, guards = true,
                max_abs_error = maxError, relative_l2_error = Math.Sqrt(err2 / ref2), cpu_max_abs_error = cpuError }));
            gpu.Begin(); Mix(false); gpu.End(); var originalMix = output.Values<float>(outputSize).ToArray();
            gpu.Begin(); Mix(false, roundedReference: true); gpu.End(); var referenceMix = output.Values<float>(outputSize).ToArray();
            foreach (bool alias in new[] { false, true }) {
                initial.CopyTo(residual.Values<float>(outputSize));
                output.Values<float>(outputSize + 4).Fill(float.NaN); residual.Values<float>(outputSize + 4)[outputSize..].Fill(float.NaN);
                gpu.Begin(); Mix(true, alias); gpu.End(); var actual = (alias ? residual : output).Values<float>(outputSize + 4);
                double combineMax = 0, combineErr2 = 0, combineRef2 = 0;
                for (int i = 0; i < outputSize; i++) {
                    if (!float.IsFinite(actual[i]) || BitConverter.SingleToUInt32Bits(actual[i]) != BitConverter.SingleToUInt32Bits(referenceMix[i])) throw new Exception("BF16 combine arithmetic mismatch");
                    double delta = (double)actual[i] - originalMix[i]; combineMax = Math.Max(combineMax, Math.Abs(delta)); combineErr2 += delta * delta; combineRef2 += (double)originalMix[i] * originalMix[i];
                }
                foreach (float guard in actual[outputSize..]) if (!float.IsNaN(guard)) throw new Exception("Combine guard overwritten");
                Console.WriteLine(JsonSerializer.Serialize(new { type = "expert_output_mix_check", rows, layout, alias, rounded_reference_bitwise = true, guards = true,
                    max_abs_error = combineMax, relative_l2_error = Math.Sqrt(combineErr2 / combineRef2) }));
            }
            initial.CopyTo(residual.Values<float>(outputSize));
            if (runs == 0 || rows < 127) continue;
            foreach (string scope in new[] { "down", "mix", "down_and_mix" })
            for (int run = 0; run <= runs; run++) foreach (bool bf16 in run % 2 == 0 ? new[] { false, true } : new[] { true, false }) {
                var warm = Stopwatch.StartNew(); while (warm.ElapsedMilliseconds < 100) Encode(bf16, scope);
                gpu.ResetWallTiming(); var watch = Stopwatch.StartNew(); Encode(bf16, scope);
                Console.WriteLine(JsonSerializer.Serialize(new { type = "expert_output_bench", rows, layout, scope, run, variant = bf16 ? "bf16" : "fp32", wall_ms = watch.Elapsed.TotalMilliseconds, timing = gpu.ReadWallTiming() }));
            }
        }
    }
}
