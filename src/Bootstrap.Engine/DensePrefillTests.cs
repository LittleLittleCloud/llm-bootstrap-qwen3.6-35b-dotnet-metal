using Bootstrap.Engine;
using System.Text.Json;
using System.Diagnostics;
namespace Bootstrap.Engine;

// Development diagnostics; not part of the repository-owned endpoint evaluator.
internal static class DensePrefillTests
{
    public static void Run(string source, int runs, bool candidateOnly)
    {
        if (runs < 0) throw new ArgumentOutOfRangeException(nameof(runs));
        using var gpu = new Metal(source);
        foreach (var (m, n, k) in new[] { 1, 8, 9, 17, 127, 272, 1259, 2048 }.SelectMany(m => new[] { (m, 1, 2048), (m, 32, 2048) }).Concat(new[] { (9, 2048, 16384), (127, 2048, 16384), (512, 2048, 16384), (2048, 2048, 16384) }))
        {
            using var x = gpu.Allocate((ulong)m * (ulong)k * 4); using var w = gpu.Allocate((ulong)n * (ulong)k * 2); using var dummy = gpu.Allocate(4);
            using var xh = gpu.Allocate((ulong)m * (ulong)k * 2); using var wh = gpu.Allocate((ulong)n * (ulong)k * 2); using var y = gpu.Allocate(((ulong)m * (ulong)n + 1) * 4);
            var xs = x.Values<float>(m * k); var ws = w.Values<ushort>(n * k); var rng = new Random(491 + m + n + k);
            for (int i = 0; i < xs.Length; i++) xs[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < ws.Length; i++) { float value = (float)((rng.NextDouble() * 2 - 1) / Math.Sqrt(k)); ws[i] = (ushort)(BitConverter.SingleToUInt32Bits(value) >> 16); }
            var p = new KernelArgs { M = (uint)m, N = (uint)n, K = (uint)k, Kind = 1 };
            bool skipLegacy = candidateOnly && n > 32;
            string[] variants = n == 1 ? ["original", "predecoded_with_conversion", "n1_simd"] : ["original", "predecoded_with_conversion"];
            if (skipLegacy) variants = ["predecoded_with_conversion"];
            void Encode(string mode)
            {
                gpu.Begin();
                if (mode == "original") gpu.Dispatch2D(n <= 32 ? "matrix_product_small_fp16" : "matrix_product", p, (uint)((n + (n <= 32 ? 31 : 63)) / (n <= 32 ? 32 : 64) * (n <= 32 ? 128 : 256)), (uint)((m + (n <= 32 ? 15 : 31)) / (n <= 32 ? 16 : 32)), n <= 32 ? 128u : 256u, x, w, dummy, y);
                else if (mode == "n1_simd") gpu.Dispatch("matrix_product_scalar_fp16", p, (uint)m * 32, 128, x, w, dummy, y);
                else { gpu.Dispatch("projection_input_half", p, p.M * p.K / 4, 128, x, xh); gpu.Dispatch("projection_unpack_half", p, p.N * p.K / 4, 128, w, dummy, wh); gpu.Dispatch2D("matrix_product_predecoded", p, ((p.N + 63) / 64) * 128, (p.M + 63) / 64, 128, xh, wh, y); }
                gpu.End();
            }
            Encode(skipLegacy ? "predecoded_with_conversion" : "original"); var reference = y.Values<float>(m * n).ToArray();
            foreach (var variant in variants)
            {
                y.Values<float>(m * n + 1).Fill(float.NaN); Encode(variant); double max = 0, sq = 0, refSq = 0;
                for (int i = 0; i < m * n; i++) { float actual = y.Values<float>(m * n)[i]; if (!float.IsFinite(actual)) throw new Exception("Nonfinite output"); double error = actual - reference[i]; max = Math.Max(max, Math.Abs(error)); sq += error * error; refSq += (double)reference[i] * reference[i]; }
                double cpuMax = 0;
                foreach (int row in new[] { 0, m / 2, m - 1 }) foreach (int col in new[] { 0, n / 2, n - 1 }) { double expected = 0; for (int j = 0; j < k; j++) { float xx = xs[row * k + j], ww = BitConverter.UInt32BitsToSingle((uint)ws[col * k + j] << 16); if (n <= 32 || variant != "original") { xx = (float)(Half)xx; ww = (float)(Half)ww; } expected += (double)xx * ww; } double error = Math.Abs(y.Values<float>(m * n)[row * n + col] - expected); cpuMax = Math.Max(cpuMax, error); if (error > 1e-4 + 1e-4 * Math.Abs(expected)) throw new Exception("CPU reference mismatch"); }
                if (max > 0.002 || Math.Sqrt(sq / Math.Max(refSq, 1e-30)) > 0.001) throw new Exception("Projection output error exceeds tolerance");
                if (!float.IsNaN(y.Values<float>(m * n + 1)[m * n])) throw new Exception("Guard overwritten");
                Console.WriteLine(JsonSerializer.Serialize(new { type = "check", variant, m, n, k, max_abs_error = max, relative_l2 = Math.Sqrt(sq / refSq), cpu_max_abs_error = cpuMax, guard = true, original_gpu_reference = !skipLegacy }));
            }
            for (int run = 0; run <= (runs == 0 || m != 2048 ? -1 : runs); run++) foreach (var variant in run % 2 == 0 ? variants : variants.Reverse())
                {
                    var warm = Stopwatch.StartNew(); while (warm.ElapsedMilliseconds < 100) Encode(variant);
                    gpu.ResetWallTiming(); var watch = Stopwatch.StartNew(); Encode(variant);
                    Console.WriteLine(JsonSerializer.Serialize(new { type = "bench", variant, m, n, k, run, wall_ms = watch.Elapsed.TotalMilliseconds, timing = gpu.ReadWallTiming() }));
                }
        }
    }
}
