using System.Diagnostics;
using System.Text.Json;
namespace Bootstrap.Engine;

internal static class DenseTileTests
{
    public static void Run(string source, int runs)
    {
        if (runs < 1) throw new ArgumentOutOfRangeException(nameof(runs));
        using var gpu = new Metal(source);
        (string Name, int Rows, int Cols, int Groups)[] variants = [
            ("matrix_product_predecoded",64,64,4), ("dense_m32_n64_s1",32,64,1),
            ("dense_m64_n64_s2",64,64,2), ("dense_m64_n64_s4",64,64,4),
            ("dense_m128_n64_s4",128,64,4), ("dense_m64_n128_s4",64,128,4),
            ("dense_m128_n128_s4",128,128,4)];
        foreach (var (m, n, k) in new[] { (127, 512, 2048), (2048, 8192, 2048), (2048, 2048, 4096), (2048, 512, 2048), (2048, 2048, 512) })
        {
            using var x = gpu.Allocate((ulong)m * (ulong)k * 2);
            using var w = gpu.Allocate((ulong)n * (ulong)k * 2);
            using var y = gpu.Allocate(((ulong)m * (ulong)n + 1) * 4);
            for (int i = 0; i < m * k; i++) x.Values<Half>(m * k)[i] = (Half)((i * 7L % 127 - 63) / 256f);
            for (int i = 0; i < n * k; i++) w.Values<Half>(n * k)[i] = (Half)((i * 13L % 251 - 125) / 512f);
            void Encode((string Name, int Rows, int Cols, int Groups) variant)
            {
                gpu.Begin();
                gpu.Dispatch2D(variant.Name, new() { M = (uint)m, N = (uint)n, K = (uint)k }, (uint)((n + variant.Cols - 1) / variant.Cols * variant.Groups * 32), (uint)((m + variant.Rows - 1) / variant.Rows), (uint)variant.Groups * 32, x, w, y);
                gpu.End();
            }
            y.Values<float>(m * n + 1).Fill(float.NaN); Encode(variants[0]);
            var reference = y.Values<float>(m * n).ToArray();
            foreach (int r in new[] { 0, m / 2, m - 1 }) foreach (int c in new[] { 0, n / 2, n - 1 })
            {
                double expected = 0;
                for (int j = 0; j < k; j++) expected += (double)(float)x.Values<Half>(m * k)[r * k + j] * (float)w.Values<Half>(n * k)[c * k + j];
                Check(reference[r * n + c], expected);
            }
            foreach (var variant in variants)
            {
                y.Values<float>(m * n + 1).Fill(float.NaN); Encode(variant);
                double error = 0;
                for (int i = 0; i < m * n; i++) error = Math.Max(error, Check(y.Values<float>(m * n)[i], reference[i]));
                if (!float.IsNaN(y.Values<float>(m * n + 1)[m * n])) throw new Exception("Dense guard overwritten");
                Console.WriteLine(JsonSerializer.Serialize(new { type = "dense_check", variant = variant.Name, m, n, k, max_abs_error = error }));
            }
            for (int run = 0; run <= runs; run++) foreach (var variant in run % 2 == 0 ? variants : variants.Reverse())
            {
                var warm = Stopwatch.StartNew(); while (warm.ElapsedMilliseconds < 100) Encode(variant);
                gpu.ResetWallTiming(); var watch = Stopwatch.StartNew(); Encode(variant); double elapsed = watch.Elapsed.TotalMilliseconds;
                Console.WriteLine(JsonSerializer.Serialize(new { type = "dense_benchmark", variant = variant.Name, m, n, k, run, wall_ms = elapsed, timing = gpu.ReadWallTiming() }));
            }
        }
    }
    static double Check(float actual, double expected)
    {
        double error = Math.Abs(actual - expected);
        if (!float.IsFinite(actual) || error > 1e-4 + 1e-4 * Math.Abs(expected)) throw new Exception($"Dense mismatch: {actual} vs {expected}");
        return error;
    }
}
