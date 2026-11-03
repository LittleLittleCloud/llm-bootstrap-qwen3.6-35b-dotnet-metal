using System.Diagnostics;
using System.Text.Json;

namespace Bootstrap.Engine;

// Development fixtures, separate from the repository-owned endpoint evaluator.
internal static class DeltaPrepareTests
{
    public static void Run(string source, int runs)
    {
        if (runs < 1) throw new ArgumentOutOfRangeException(nameof(runs));
        using var gpu = new Metal(source);
        using var weights = gpu.Allocate(8192 * 4 * 2);
        using var alog = gpu.Allocate(64);
        using var bias = gpu.Allocate(64);
        using var dummy = gpu.Allocate(4);
        using var carry = gpu.Allocate((24576 + 1) * 4);
        using var recurrent = gpu.Allocate((524288 + 1) * 4);
        static float Bf(ushort x) => BitConverter.UInt32BitsToSingle((uint)x << 16);
        for (int i = 0; i < 8192 * 4; i++) weights.Values<ushort>(8192 * 4)[i] = (ushort)(BitConverter.SingleToUInt32Bits((i * 7 % 17 - 8) / 16f) >> 16);
        for (int h = 0; h < 32; h++) {
            alog.Values<ushort>(32)[h] = (ushort)(h % 2 == 0 ? 0 : 0xbf00);
            bias.Values<ushort>(32)[h] = (ushort)(h % 3 == 0 ? 0x3e00 : 0);
        }
        float[] initial = Enumerable.Range(0, 24576).Select(i => i % 2048 < 128 ? 0f : (i * 13 % 101 - 50) / 128f).ToArray();
        float[] initialRecurrent = Enumerable.Range(0, 524288).Select(i => (i * 13 % 101 - 50) / 1000f).ToArray();
        foreach (int m in new[] { 1, 2, 3, 7, 17, 128, 2048 })
        {
            using var x = gpu.Allocate((ulong)m * 8192 * 4);
            using var a = gpu.Allocate((ulong)m * 32 * 4);
            using var b = gpu.Allocate((ulong)m * 32 * 4);
            using var v = gpu.Allocate(((ulong)m * 8192 + 1) * 4);
            using var q = gpu.Allocate(((ulong)m * 2048 + 1) * 4);
            using var k = gpu.Allocate(((ulong)m * 2048 + 1) * 4);
            using var gates = gpu.Allocate(((ulong)m * 64 + 1) * 4);
            using var output = gpu.Allocate(((ulong)m * 4096 + 1) * 4);
            for (int i = 0; i < m * 8192; i++) x.Values<float>(m * 8192)[i] = i % 2048 < 128 ? 0f : (i * 17L % 127 - 63) / 64f;
            for (int i = 0; i < m * 32; i++) {
                a.Values<float>(m * 32)[i] = (i * 7 % 101 - 50) / 4f;
                b.Values<float>(m * 32)[i] = (i * 13 % 101 - 50) / 4f;
            }
            void Reset()
            {
                initial.CopyTo(carry.Values<float>(24576));
                carry.Values<float>(24577)[24576] = float.NaN;
                initialRecurrent.CopyTo(recurrent.Values<float>(524288));
                recurrent.Values<float>(524289)[524288] = float.NaN;
                v.Values<float>(m * 8192 + 1).Fill(float.NaN);
                q.Values<float>(m * 2048 + 1).Fill(float.NaN);
                k.Values<float>(m * 2048 + 1).Fill(float.NaN);
                gates.Values<float>(m * 64 + 1).Fill(float.NaN);
                output.Values<float>(m * 4096 + 1).Fill(float.NaN);
            }
            void Encode(string variant, int begin = 0, int? length = null, bool scan = false)
            {
                uint n = (uint)(length ?? m);
                Binding xb = new(x, (ulong)begin * 8192 * 4), vb = new(v, (ulong)begin * 8192 * 4);
                Binding qb = new(q, (ulong)begin * 2048 * 4), kb = new(k, (ulong)begin * 2048 * 4);
                Binding ab = new(a, (ulong)begin * 32 * 4), bb = new(b, (ulong)begin * 32 * 4), gb = new(gates, (ulong)begin * 64 * 4);
                gpu.Begin();
                if (variant == "fused")
                    gpu.Dispatch("delta_prepare_fused", new() { M = n }, n * 16 * 128, 128, xb, weights, carry, vb, qb, kb, ab, bb, alog, bias, gb);
                else {
                    if (variant == "baseline") gpu.Dispatch("batch_conv", new() { M = n }, 8192, 128, xb, weights, carry, vb, dummy);
                    else gpu.Dispatch("batch_conv_parallel", new() { M = n }, n * 8192, 128, xb, weights, carry, vb);
                    gpu.Dispatch("batch_l2", new() { M = n }, n * 16 * 128, 128, vb, qb, kb);
                    gpu.Dispatch("delta_prepare_gates", new() { M = n }, n * 32, 128, ab, bb, alog, bias, gb);
                }
                if (variant != "baseline") gpu.Dispatch("batch_conv_carry", new() { M = n }, 8192, 128, xb, carry);
                if (scan) gpu.Dispatch("delta_staged8", new() { M = n }, 32 * 8 * 128, 128, qb, kb, vb, gb, recurrent, new Binding(output, (ulong)begin * 4096 * 4));
                gpu.End();
            }
            Reset(); Encode("baseline", scan: true);
            var rq = q.Values<float>(m * 2048).ToArray(); var rk = k.Values<float>(m * 2048).ToArray();
            var rv = v.Values<float>(m * 8192).ToArray(); var rg = gates.Values<float>(m * 64).ToArray();
            var rc = carry.Values<float>(24576).ToArray(); var rr = recurrent.Values<float>(524288).ToArray(); var ro = output.Values<float>(m * 4096).ToArray();
            // Independent convolution and norm reference, including zero heads,
            // signed weights, incoming carry and a block shorter than the filter.
            foreach (int h in new[] { 0, 7, 15 }) for (int t = 0; t < m; t++) {
                double[] qs = new double[128], ks = new double[128];
                double Conv(int c) {
                    double sum = 0;
                    for (int j = 0; j < 4; j++) {
                        int row = t + j - 3;
                        double value = row < 0 ? initial[(row + 3) * 8192 + c] : x.Values<float>(m * 8192)[row * 8192 + c];
                        sum += value * Bf(weights.Values<ushort>(8192 * 4)[c * 4 + j]);
                    }
                    return sum / (1 + Math.Exp(-sum));
                }
                for (int d = 0; d < 128; d++) {
                    qs[d] = Conv(h * 128 + d); ks[d] = Conv(2048 + h * 128 + d);
                    Check(rv[t * 8192 + h * 128 + d], qs[d]);
                    Check(rv[t * 8192 + 2048 + h * 128 + d], ks[d]);
                    for (int j = 0; j < 2; j++) Check(rv[t * 8192 + 4096 + (h * 2 + j) * 128 + d], Conv(4096 + (h * 2 + j) * 128 + d));
                }
                double qn = 1 / Math.Sqrt(qs.Sum(z => z * z) + 1e-6) / Math.Sqrt(128), kn = 1 / Math.Sqrt(ks.Sum(z => z * z) + 1e-6);
                for (int d = 0; d < 128; d++) { Check(rq[t * 2048 + h * 128 + d], qs[d] * qn); Check(rk[t * 2048 + h * 128 + d], ks[d] * kn); }
            }
            for (int i = 0; i < 24576; i++) {
                int row = m - 3 + i / 8192, c = i % 8192;
                if (rc[i] != (row < 0 ? initial[(row + 3) * 8192 + c] : x.Values<float>(m * 8192)[row * 8192 + c])) throw new Exception("CPU carry mismatch");
            }
            foreach (string variant in new[] { "baseline", "parallel", "fused" }) foreach (bool split in m > 1 ? new[] { false, true } : new[] { false }) {
                Reset();
                if (split) { Encode(variant, 0, m / 2, true); Encode(variant, m / 2, m - m / 2, true); }
                else Encode(variant, scan: true);
                void Equal(GpuBuffer buffer, float[] expected, string name) {
                    var actual = buffer.Values<float>(expected.Length);
                    for (int i = 0; i < expected.Length; i++) if (BitConverter.SingleToUInt32Bits(actual[i]) != BitConverter.SingleToUInt32Bits(expected[i])) throw new Exception($"{variant}/{m}/{split}/{name} bitwise mismatch at {i}: {actual[i]} vs {expected[i]}");
                    if (!float.IsNaN(buffer.Values<float>(expected.Length + 1)[expected.Length])) throw new Exception(name + " guard overwritten");
                }
                Equal(q, rq, "q"); Equal(k, rk, "k"); Equal(gates, rg, "gates"); Equal(carry, rc, "carry"); Equal(recurrent, rr, "recurrent"); Equal(output, ro, "output");
                for (int t = 0; t < m; t++) for (int c = 4096; c < 8192; c++) if (v.Values<float>(m * 8192)[t * 8192 + c] != rv[t * 8192 + c]) throw new Exception("V mismatch");
                if (!float.IsNaN(v.Values<float>(m * 8192 + 1)[m * 8192])) throw new Exception("V guard overwritten");
                Console.WriteLine(JsonSerializer.Serialize(new { type = "delta_prepare_check", variant, m, split, bitwise = true, cpu_reference = true }));
            }
            // Warm and reverse order. Measure only preparation, including carry.
            for (int run = 0; run <= runs; run++) foreach (string variant in run % 2 == 0 ? new[] { "baseline", "parallel", "fused" } : new[] { "fused", "parallel", "baseline" }) {
                Reset(); var warm = Stopwatch.StartNew(); while (warm.ElapsedMilliseconds < 100) Encode(variant);
                Reset(); gpu.ResetWallTiming(); var watch = Stopwatch.StartNew(); Encode(variant);
                Console.WriteLine(JsonSerializer.Serialize(new { type = "delta_prepare_benchmark", variant, m, run, wall_ms = watch.Elapsed.TotalMilliseconds, timing = gpu.ReadWallTiming() }));
            }
        }
    }
    static void Check(float actual, double expected) {
        if (!float.IsFinite(actual) || Math.Abs(actual - expected) > 2e-6 + 2e-5 * Math.Abs(expected)) throw new Exception($"CPU reference mismatch: {actual} vs {expected}");
    }
}
