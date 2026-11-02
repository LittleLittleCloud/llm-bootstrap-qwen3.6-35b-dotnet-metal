using System.Diagnostics;
using System.Text.Json;

namespace Bootstrap.Engine;

// Synthetic, matching-input microbenchmarks for scripts/compare-mlx-kernels.py.
// No model, evaluation, or leaderboard result is produced by this command.
internal static class KernelComparisonBenchmarks
{
    public static void Run(string source, string family, int runs)
    {
        if (runs < 1) throw new ArgumentOutOfRangeException(nameof(runs));
        if (family is not ("all" or "dense" or "experts" or "attention")) throw new ArgumentException("Unknown benchmark family");
        if (family is "all" or "dense") Dense(source, runs);
        if (family is "all" or "experts") ExpertThroughputTests.Run(source, runs, comparison: true);
        if (family is "all" or "attention") FastAttentionTests.Run(source, runs, halfKv: true, comparison: true);
    }

    static void Dense(string source, int runs)
    {
        using var gpu = new Metal(source);
        foreach (var (m, n, k) in new[] { (512, 512, 2048), (2048, 512, 2048), (2048, 2048, 2048), (2048, 8192, 2048), (2048, 2048, 4096) })
        {
            using var x = gpu.Allocate((ulong)m * (ulong)k * 4);
            using var packedX = gpu.Allocate((ulong)m * (ulong)k * 2);
            using var w = gpu.Allocate((ulong)n * (ulong)k);
            using var scales = gpu.Allocate((ulong)(n / 128 * (k / 128)) * 2);
            using var packedW = gpu.Allocate((ulong)n * (ulong)k * 2);
            using var y = gpu.Allocate((ulong)(m * n + 1) * 4);
            for (int i = 0; i < m * k; i++) x.Values<float>(m * k)[i] = ((i * 7L + i / 113) % 67 - 33) / 64f;
            for (int i = 0; i < n * k; i++) w.Values<byte>(n * k)[i] = (byte)(((i * 13) % 112) | (i % 3 == 0 ? 128 : 0));
            scales.Values<ushort>(n / 128 * (k / 128)).Fill(0x3b80);
            var p = new KernelArgs { M = (uint)m, N = (uint)n, K = (uint)k, Kind = 0 };
            void Pack()
            {
                gpu.Dispatch("projection_input_half", p, (uint)(m * k / 4), 128, x, packedX);
                gpu.Dispatch("projection_unpack_half", p, (uint)(n * k / 4), 128, w, scales, packedW);
            }
            gpu.Begin(); Pack(); gpu.End();
            foreach (bool includePacking in new[] { false, true })
            {
                y.Values<float>(m * n + 1).Fill(float.NaN);
                void Encode()
                {
                    gpu.Begin();
                    if (includePacking) Pack();
                    gpu.Dispatch2D("matrix_product_predecoded", p, (uint)((n + 63) / 64 * 128), (uint)((m + 63) / 64), 128, packedX, packedW, y);
                    gpu.End();
                }
                for (int run = -3; run < runs; run++)
                {
                    if (run == 0) { var warm = Stopwatch.StartNew(); while (warm.Elapsed.TotalMilliseconds < 250) Encode(); }
                    gpu.ResetWallTiming(); var watch = Stopwatch.StartNew(); Encode(); double wall = watch.Elapsed.TotalMilliseconds;
                    if (run == -1)
                    {
                        double worst = 0;
                        foreach (int row in new[] { 0, m / 2, m - 1 }) foreach (int col in new[] { 0, n / 2, n - 1 })
                        {
                            double expected = 0, magnitude = 0;
                            for (int j = 0; j < k; j++)
                            {
                                byte v = w.Values<byte>(n * k)[col * k + j]; int e = (v >> 3) & 15, mant = v & 7;
                                double decoded = (v < 128 ? 1 : -1) * (e == 0 ? mant / 512.0 : (1 + mant / 8.0) * Math.Pow(2, e - 7));
                                double term = (float)(Half)x.Values<float>(m * k)[row * k + j] * (double)(float)(Half)(decoded / 256);
                                expected += term; magnitude += Math.Abs(term);
                            }
                            float actual = y.Values<float>(m * n)[row * n + col]; double error = Math.Abs(actual - expected); worst = Math.Max(worst, error);
                            if (!float.IsFinite(actual) || error > 1e-6 + 3e-6 * magnitude) throw new Exception($"Dense reference mismatch: {row}/{col}: {actual} vs {expected}");
                        }
                        if (!float.IsNaN(y.Values<float>(m * n + 1)[m * n])) throw new Exception("Dense output guard overwritten");
                        Console.WriteLine(JsonSerializer.Serialize(new { type = "dense_check", m, n, k, include_packing = includePacking, max_abs_error = worst }));
                    }
                    if (run >= 0) Console.WriteLine(JsonSerializer.Serialize(new { type = "dense_benchmark", m, n, k, include_packing = includePacking, run, wall_ms = wall, timing = gpu.ReadWallTiming(), output_dtype = "float32" }));
                }
            }
        }
    }
}
