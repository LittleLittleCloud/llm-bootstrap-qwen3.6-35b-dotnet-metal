using System.Diagnostics;
using System.Text.Json;
namespace Bootstrap.Engine;

internal static class SharedGateTests
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
            using var readyGate = gpu.Allocate((ulong)(rows + 4) * 4);
            using var input = gpu.Allocate((ulong)n * 4);
            using var packedInput = gpu.Allocate((ulong)n * 2);
            using var weight = gpu.Allocate(2048 * 2);
            using var packedWeight = gpu.Allocate(2048 * 2);
            using var dummy = gpu.Allocate(4);
            using var output = gpu.Allocate((ulong)(n + 4) * 4);
            var rng = new Random(173 + rows);
            float Rand() => (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < n; i++) { residual.Values<float>(n)[i] = Rand(); shared.Values<float>(n)[i] = Rand(); }
            for (int i = 0; i < n * 8; i++) expert.Values<float>(n * 8)[i] = Rand();
            var initial = residual.Values<float>(n).ToArray();
            var routing = Enumerable.Range(0, rows * 8).ToArray(); rng.Shuffle(routing);
            for (int i = 0; i < rows * 8; i++) { inverse.Values<uint>(rows * 8)[i] = (uint)routing[i]; rates.Values<float>(rows * 8)[i] = Math.Abs(Rand()) / 8; }
            for (int i = 0; i < n; i++) input.Values<float>(n)[i] = Rand();
            for (int i = 0; i < 2048; i++) weight.Values<ushort>(2048)[i] = (ushort)(BitConverter.SingleToUInt32Bits(Rand() / 8) >> 16);
            // Exercise sigmoid saturation using a large signed dot product.
            for (int row = 0; row < rows; row++) if (row % 7 < 2)
                for (int col = 0; col < 2048; col++) input.Values<float>(n)[row * 2048 + col] = (row % 7 == 0 ? -1 : 1) * Math.Sign(BitConverter.UInt32BitsToSingle((uint)weight.Values<ushort>(2048)[col] << 16));
            var p = new KernelArgs { N = 1, M = (uint)rows, K = 2048, Kind = 1 };
            gpu.Begin();
            gpu.Dispatch("projection_unpack_half", p, 512, 128, weight, dummy, packedWeight);
            gpu.Dispatch("projection_input_half", p, (uint)n / 4, 128, input, packedInput);
            gpu.End();
            void Project(bool enabled)
            {
                var destination = enabled ? readyGate : gate;
                if (rows <= 8) gpu.Dispatch(enabled ? "matrix_product_tiny_sigmoid" : "matrix_product_tiny", p, 32, 128, input, weight, dummy, destination);
                else gpu.Dispatch2D(enabled ? "matrix_product_predecoded_sigmoid" : "matrix_product_predecoded", p, 128, ((uint)rows + 63) / 64, 128, packedInput, packedWeight, destination);
            }
            gpu.Begin(); Project(false); Project(true); gpu.End();
            void Encode(string name, bool alias = false, bool project = false)
            {
                bool enabled = name == "batch_mix_precomputed_gate";
                gpu.Begin();
                if (project) Project(enabled);
                gpu.Dispatch(name, new() { M = (uint)rows }, (uint)n, 128, residual, expert, inverse, rates, shared, enabled ? readyGate : gate, alias ? residual : output);
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
                readyGate.Values<float>(rows + 4)[rows..].Fill(float.NaN);
                Encode("batch_mix_precomputed_gate", alias, project: true);
                foreach (float value in readyGate.Values<float>(rows + 4)[rows..]) if (!float.IsNaN(value)) throw new Exception("Gate guard overwritten");
                var result = (alias ? residual : output).Values<float>(n + 4);
                for (int i = 0; i < n; i++) if (!float.IsFinite(result[i]) || BitConverter.SingleToUInt32Bits(result[i]) != BitConverter.SingleToUInt32Bits(reference[i])) throw new Exception($"Mix parity mismatch at {rows}/{i}: {result[i]} vs {reference[i]}");
                foreach (float value in result[n..]) if (!float.IsNaN(value)) throw new Exception("Mix guard overwritten");
                Console.WriteLine(JsonSerializer.Serialize(new { type = "shared_gate_check", rows, alias, bitwise = true, cpu_max_abs_error = cpuError, guards = true }));
            }
            initial.CopyTo(residual.Values<float>(n));
            if (runs == 0 || rows < 127) continue;
            foreach (bool project in new[] { false, true })
            for (int run = 0; run <= runs; run++) foreach (string variant in run % 2 == 0 ? new[] { "batch_mix", "batch_mix_precomputed_gate" } : new[] { "batch_mix_precomputed_gate", "batch_mix" })
            {
                var warm = Stopwatch.StartNew(); while (warm.ElapsedMilliseconds < 100) Encode(variant, project: project);
                gpu.ResetWallTiming(); var watch = Stopwatch.StartNew(); Encode(variant, project: project);
                Console.WriteLine(JsonSerializer.Serialize(new { type = "shared_gate_bench", scope = project ? "projection_and_mix" : "mix_only", rows, variant, run, wall_ms = watch.Elapsed.TotalMilliseconds, timing = gpu.ReadWallTiming() }));
            }
        }
    }
}
