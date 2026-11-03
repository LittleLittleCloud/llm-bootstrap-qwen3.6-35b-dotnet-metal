using System.Diagnostics;
using System.Text.Json;
namespace Bootstrap.Engine;

internal static class ProjectionPreparationTests
{
    public static void Run(string source, int runs)
    {
        if (runs < 0) throw new ArgumentOutOfRangeException(nameof(runs));
        using var gpu = new Metal(source);
        foreach (int rows in new[] { 1, 8, 9, 17, 127, 272, 1259, 2048 })
        foreach (string kind in new[] { "input", "residual", "gated" })
        {
            int dim = kind == "gated" ? 128 : 2048, heads = rows * (kind == "gated" ? 32 : 1), n = heads * dim;
            bool add = kind == "residual", alias = kind == "gated";
            using var input = gpu.Allocate((ulong)(n + 4) * 4); using var weights = gpu.Allocate((ulong)dim * 2);
            using var output = gpu.Allocate((ulong)(n + 4) * 4); using var gate = gpu.Allocate((ulong)n * 4);
            using var packed = gpu.Allocate((ulong)(n + 8) * 2); using var residual = gpu.Allocate((ulong)(n + 4) * 4);
            var rng = new Random(731 + rows + dim);
            float Rand() => (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < n; i++) { input.Values<float>(n)[i] = i < dim ? 0 : Rand() * (i / dim % 5 == 0 ? 0.0001f : 2); residual.Values<float>(n)[i] = i < dim ? 0 : Rand(); gate.Values<float>(n)[i] = Rand() * 8; }
            for (int i = 0; i < dim; i++) weights.Values<ushort>(dim)[i] = (ushort)(BitConverter.SingleToUInt32Bits(Rand()) >> 16);
            var initial = input.Values<float>(n).ToArray(); var originalResidual = residual.Values<float>(n).ToArray();
            uint flags = (kind == "gated" ? 2u : 1u);
            var p = new KernelArgs { M = (uint)heads, N = (uint)dim, Epsilon = 1e-6f, Flags = flags };
            void Encode(bool fused)
            {
                initial.CopyTo(input.Values<float>(n));
                gpu.Begin();
                if (fused) { var f = p; if (add) f.Flags |= 4; gpu.Dispatch("rmsnorm_pack", f, (uint)heads * (dim == 128 ? 128u : 256u), dim == 128 ? 128u : 256u, input, weights, alias ? input : output, gate, packed, residual); }
                else {
                    if (add) gpu.Dispatch("add", new() { N = (uint)n }, (uint)n, 128, input, residual, input);
                    gpu.Dispatch("rmsnorm", p, (uint)heads * 256, 256, input, weights, alias ? input : output, gate);
                    gpu.Dispatch("projection_input_half", new() { M = (uint)heads, K = (uint)dim }, (uint)n / 4, 128, alias ? input : output, packed);
                }
                gpu.End();
            }
            Encode(false); var reference = (alias ? input : output).Values<float>(n).ToArray(); var packedReference = packed.Values<ushort>(n).ToArray(); var updated = input.Values<float>(n).ToArray();
            double worst = 0;
            foreach (int head in new[] { 0, heads / 2, heads - 1 })
            {
                double sum = 0;
                for (int j = 0; j < dim; j++) { float x = initial[head * dim + j]; if (add) x += originalResidual[head * dim + j]; sum += (double)x * x; }
                double inv = 1 / Math.Sqrt(sum / dim + 1e-6);
                foreach (int j in new[] { 0, 1, dim / 2, dim - 1 }) {
                    int i = head * dim + j; float x = initial[i]; if (add) x += originalResidual[i];
                    double expected = x * inv * (BitConverter.UInt32BitsToSingle((uint)weights.Values<ushort>(dim)[j] << 16) + (alias ? 0 : 1));
                    if (alias) { float g = gate.Values<float>(n)[i]; expected *= g / (1 + Math.Exp(-g)); }
                    double error = Math.Abs(reference[i] - expected); worst = Math.Max(worst, error);
                    if (error > 3e-6 + Math.Abs(expected) * 3e-6) throw new Exception("Normalization CPU reference mismatch");
                }
            }
            input.Values<float>(n + 4)[n..].Fill(float.NaN); output.Values<float>(n + 4).Fill(float.NaN); packed.Values<ushort>(n + 8).Fill(0x7e00);
            Encode(true);
            var actual = (alias ? input : output).Values<float>(n);
            for (int i = 0; i < n; i++) {
                if (!float.IsFinite(actual[i]) || BitConverter.SingleToUInt32Bits(actual[i]) != BitConverter.SingleToUInt32Bits(reference[i])) throw new Exception($"Norm parity mismatch {rows}/{kind}/{i}: {actual[i]} vs {reference[i]}");
                if (packed.Values<ushort>(n)[i] != packedReference[i]) throw new Exception("Packed input parity mismatch");
                if (BitConverter.SingleToUInt32Bits(input.Values<float>(n)[i]) != BitConverter.SingleToUInt32Bits(updated[i])) throw new Exception("Residual state mismatch");
            }
            foreach (float v in input.Values<float>(n + 4)[n..]) if (!float.IsNaN(v)) throw new Exception("Input guard overwritten");
            foreach (float v in output.Values<float>(n + 4)[n..]) if (!float.IsNaN(v)) throw new Exception("Output guard overwritten");
            foreach (ushort v in packed.Values<ushort>(n + 8)[n..]) if (v != 0x7e00) throw new Exception("Packed guard overwritten");
            if (!residual.Values<float>(n).SequenceEqual(originalResidual)) throw new Exception("Projection source changed");
            Console.WriteLine(JsonSerializer.Serialize(new { type = "norm_check", rows, heads, dim, kind, bitwise_output = true, bitwise_packed = true, bitwise_residual = true, guards = true, cpu_max_abs_error = worst }));
            if (runs == 0 || rows != 2048) continue;
            for (int run = 0; run <= runs; run++) foreach (bool fused in run % 2 == 0 ? new[] { false, true } : new[] { true, false }) {
                for (int warm = 0; warm < 3; warm++) Encode(fused);
                gpu.ResetWallTiming(); var watch = Stopwatch.StartNew(); Encode(fused);
                Console.WriteLine(JsonSerializer.Serialize(new { type = "norm_bench", rows, heads, dim, kind, fused, run, wall_ms = watch.Elapsed.TotalMilliseconds, timing = gpu.ReadWallTiming() }));
            }
        }
    }
}
