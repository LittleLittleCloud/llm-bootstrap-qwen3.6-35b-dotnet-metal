using System.Diagnostics;
using System.Text.Json;
namespace Bootstrap.Engine;

internal static class MixTests
{
    public static void Run(string source, int runs)
    {
        if (runs < 0) throw new ArgumentOutOfRangeException(nameof(runs));
        using var gpu = new Metal(source);
        foreach (int rows in new[] { 1, 8, 9, 17, 127, 272, 1259, 2048 })
        {
            int n = rows * 2048;
            using var residual = gpu.Allocate((ulong)(n + 4) * 4);
            using var expert = gpu.Allocate((ulong)n * 8 * 4);
            using var inverse = gpu.Allocate((ulong)rows * 8 * 4);
            using var rates = gpu.Allocate((ulong)rows * 8 * 4);
            using var shared = gpu.Allocate((ulong)n * 4);
            using var gate = gpu.Allocate((ulong)rows * 4);
            using var output = gpu.Allocate((ulong)(n + 4) * 4);
            var rng = new Random(173 + rows);
            float Rand() => (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < n; i++) { residual.Values<float>(n)[i] = Rand(); shared.Values<float>(n)[i] = Rand(); }
            for (int i = 0; i < n * 8; i++) expert.Values<float>(n * 8)[i] = Rand();
            var initial = residual.Values<float>(n).ToArray();
            var routing = Enumerable.Range(0, rows * 8).ToArray(); rng.Shuffle(routing);
            for (int i = 0; i < rows * 8; i++) { inverse.Values<uint>(rows * 8)[i] = (uint)routing[i]; rates.Values<float>(rows * 8)[i] = Math.Abs(Rand()) / 8; }
            for (int i = 0; i < rows; i++) gate.Values<float>(rows)[i] = i % 7 == 0 ? -80 : i % 7 == 1 ? 80 : Rand() * 8;
            void Encode(string name, bool alias = false)
            {
                gpu.Begin();
                gpu.Dispatch(name, new() { M = (uint)rows }, (uint)(name == "batch_mix" ? n : n / 4), 128, residual, expert, inverse, rates, shared, gate, alias ? residual : output);
                gpu.End();
            }
            Encode("batch_mix"); var reference = output.Values<float>(n).ToArray();
            double cpuError = 0;
            foreach (int row in new[] { 0, rows / 2, rows - 1 }) foreach (int col in new[] { 0, 1, 127, 1023, 2047 })
            {
                int i = row * 2048 + col;
                double expected = initial[i] + shared.Values<float>(n)[i] / (1 + Math.Exp(-gate.Values<float>(rows)[row]));
                for (int e = 0; e < 8; e++) expected += (double)rates.Values<float>(rows * 8)[row * 8 + e] * expert.Values<float>(n * 8)[routing[row * 8 + e] * 2048 + col];
                cpuError = Math.Max(cpuError, Math.Abs(expected - reference[i]));
            }
            if (cpuError > 2e-6) throw new Exception("Mix CPU reference mismatch");
            foreach (bool alias in new[] { false, true })
            {
                initial.CopyTo(residual.Values<float>(n));
                residual.Values<float>(n + 4)[n..].Fill(float.NaN); output.Values<float>(n + 4).Fill(float.NaN);
                Encode("batch_mix_vec4", alias);
                var result = (alias ? residual : output).Values<float>(n + 4);
                for (int i = 0; i < n; i++) if (!float.IsFinite(result[i]) || BitConverter.SingleToUInt32Bits(result[i]) != BitConverter.SingleToUInt32Bits(reference[i])) throw new Exception($"Mix parity mismatch at {rows}/{i}: {result[i]} vs {reference[i]}");
                foreach (float value in result[n..]) if (!float.IsNaN(value)) throw new Exception("Mix guard overwritten");
                Console.WriteLine(JsonSerializer.Serialize(new { type = "mix_check", rows, alias, bitwise = true, cpu_max_abs_error = cpuError, guards = true }));
            }
            initial.CopyTo(residual.Values<float>(n));
            if (runs == 0 || rows < 127) continue;
            for (int run = 0; run <= runs; run++) foreach (string variant in run % 2 == 0 ? new[] { "batch_mix", "batch_mix_vec4" } : new[] { "batch_mix_vec4", "batch_mix" })
            {
                var warm = Stopwatch.StartNew(); while (warm.ElapsedMilliseconds < 100) Encode(variant);
                gpu.ResetWallTiming(); var watch = Stopwatch.StartNew(); Encode(variant);
                Console.WriteLine(JsonSerializer.Serialize(new { type = "mix_bench", rows, variant, run, wall_ms = watch.Elapsed.TotalMilliseconds, timing = gpu.ReadWallTiming() }));
            }
        }
    }
}
