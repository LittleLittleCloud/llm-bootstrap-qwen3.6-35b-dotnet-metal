using System.Text.Json;
namespace Bootstrap.Engine;

internal static class KvStorageTests
{
    public static void Run(string source)
    {
        using var gpu = new Metal(source);
        int checks = 0;
        float Read(GpuBuffer b, int at, bool half) => half ? (float)b.Values<Half>(at + 1)[at] : b.Values<float>(at + 1)[at];
        void Fill(GpuBuffer b, int count, bool half, int seed)
        {
            for (int i = 0; i < count; i++)
            {
                float value = ((i * 13L + seed) % 101 - 50) / 113f;
                if (half) b.Values<Half>(count)[i] = (Half)value; else b.Values<float>(count)[i] = value;
            }
        }
        void Check(double actual, double expected, double tolerance, string name)
        {
            if (!double.IsFinite(actual) || Math.Abs(actual - expected) > tolerance) throw new Exception($"{name}: {actual} vs {expected}");
            checks++;
        }
        foreach (bool half in new[] { false, true }) foreach (int rows in new[] { 1, 7 })
        {
            const int position = 4093; int length = position + rows, capacity = length + 1;
            using var q = gpu.Allocate((ulong)rows * 4096 * 4); using var k = gpu.Allocate((ulong)rows * 512 * 4);
            using var v = gpu.Allocate((ulong)rows * 512 * 4); using var qr = gpu.Allocate(q.Length);
            using var kc = gpu.Allocate((ulong)capacity * 512 * (half ? 2u : 4u)); using var vc = gpu.Allocate(kc.Length);
            using var scores = gpu.Allocate((ulong)capacity * 16 * 4); using var output = gpu.Allocate(4096 * 4); using var gate = gpu.Allocate(4096 * 4);
            Fill(q, rows * 4096, false, 7); Fill(k, rows * 512, false, 13); Fill(v, rows * 512, false, 31);
            Fill(kc, capacity * 512, half, 17); Fill(vc, capacity * 512, half, 23); Fill(gate, 4096, false, 11);
            byte[] beforeK = kc.Values<byte>((int)kc.Length).ToArray(), beforeV = vc.Values<byte>((int)vc.Length).ToArray();
            var p = new KernelArgs { M = (uint)rows, Position = position, K = (uint)capacity, Theta = 10_000_000f, Flags = half ? 256u : 0u };
            gpu.Begin(); gpu.Dispatch(rows == 1 ? "rope_cache" : "batch_rope", p, (uint)rows * 4096, 128, q, k, v, qr, kc, vc); gpu.End();
            int bytes = half ? 2 : 4;
            foreach (var pair in new[] { (kc, beforeK), (vc, beforeV) })
            {
                if (!pair.Item1.Values<byte>(position * 512 * bytes).SequenceEqual(pair.Item2.AsSpan(0, position * 512 * bytes)) ||
                    !pair.Item1.Values<byte>(512 * bytes, (ulong)length * 512 * (ulong)bytes).SequenceEqual(pair.Item2.AsSpan(length * 512 * bytes)))
                    throw new Exception("RoPE cache overwrote the prefix or tail guard");
                checks++;
            }
            double Rotate(GpuBuffer input, int at, int d, int pos)
            {
                if (d >= 64) return Read(input, at + d, false);
                int j = d % 32; double angle = pos * Math.Pow(10_000_000, -2.0 * j / 64);
                double a = Read(input, at + j, false), b = Read(input, at + 32 + j, false);
                return d < 32 ? a * Math.Cos(angle) - b * Math.Sin(angle) : b * Math.Cos(angle) + a * Math.Sin(angle);
            }
            for (int row = 0; row < rows; row++) for (int d = 0; d < 512; d++)
            {
                double expected = Rotate(k, row * 512 + d / 256 * 256, d % 256, position + row);
                if (half) expected = (float)(Half)(float)expected;
                Check(Read(kc, (position + row) * 512 + d, half), expected, 6e-4, "rotated key");
                float value = Read(v, row * 512 + d, false);
                Check(Read(vc, (position + row) * 512 + d, half), half ? (float)(Half)value : value, 0, "cached value");
            }
            // Exercise single-token decode reads from the entire grown cache.
            p.Position = (uint)(length - 1); var query = new Binding(qr, (ulong)(rows - 1) * 4096 * 4);
            gpu.Begin(); gpu.Dispatch("attention_scores", p, (uint)(16 * length * 32), 128, query, kc, scores);
            gpu.Dispatch("attention_softmax", p, 16 * 256, 256, scores);
            gpu.Dispatch("attention_values", p, 4096, 128, scores, vc, gate, output); gpu.End();
            foreach (int head in new[] { 0, 7, 8, 15 })
            {
                double[] logits = new double[length];
                for (int t = 0; t < length; t++) for (int d = 0; d < 256; d++) logits[t] += (double)Read(qr, (rows - 1) * 4096 + head * 256 + d, false) * Read(kc, t * 512 + head / 8 * 256 + d, half) / 16;
                double max = logits.Max(), sum = logits.Sum(x => Math.Exp(x - max));
                foreach (int d in new[] { 0, 127, 255 })
                {
                    double expected = 0; for (int t = 0; t < length; t++) expected += Math.Exp(logits[t] - max) / sum * Read(vc, t * 512 + head / 8 * 256 + d, half);
                    int at = head * 256 + d; expected /= 1 + Math.Exp(-Read(gate, at, false));
                    Check(Read(output, at, false), expected, 2e-5, "decode attention");
                }
            }
        }
        // Draft attention mixes a wrapping cached prefix with new FP32 local K/V.
        foreach (bool half in new[] { false, true })
        {
            const int rows = 7, position = 2051, length = 2048 + rows, start = position - 2048;
            using var q = gpu.Allocate(rows * 4096 * 4); using var k = gpu.Allocate(rows * 1024 * 4); using var v = gpu.Allocate(rows * 1024 * 4);
            using var kc = gpu.Allocate((ulong)2048 * 1024 * (half ? 2u : 4u)); using var vc = gpu.Allocate(kc.Length);
            using var scores = gpu.Allocate(rows * 32 * length * 4); using var output = gpu.Allocate(rows * 4096 * 4);
            Fill(q, rows * 4096, false, 7); Fill(k, rows * 1024, false, 17); Fill(v, rows * 1024, false, 23);
            Fill(kc, 2048 * 1024, half, 29); Fill(vc, 2048 * 1024, half, 31);
            var write = new KernelArgs { M = rows, Position = 2045, Theta = 10_000_000f, Flags = half ? 256u : 0u };
            gpu.Begin(); gpu.Dispatch("draft_context_kv", write, rows * 1024, 128, k, v, kc, vc); gpu.End();
            for (int row = 0; row < rows; row++) for (int d = 0; d < 1024; d++)
            {
                int j = d % 128, at = ((2045 + row) % 2048) * 1024 + d;
                double angle = (2045 + row) * Math.Pow(10_000_000, -(j % 64) / 64.0);
                double value = Read(k, row * 1024 + d, false) * Math.Cos(angle) + (j < 64 ? -Read(k, row * 1024 + d + 64, false) : Read(k, row * 1024 + d - 64, false)) * Math.Sin(angle);
                Check(Read(kc, at, half), half ? (float)(Half)(float)value : value, 6e-4, "draft ring key");
                float expectedV = Read(v, row * 1024 + d, false);
                Check(Read(vc, at, half), half ? (float)(Half)expectedV : expectedV, 0, "draft ring value");
            }
            var p = new KernelArgs { M = rows, Position = position, K = length, A = 2048, Flags = half ? 256u : 0u };
            gpu.Begin(); gpu.Dispatch("draft_attention_scores", p, rows * 32 * length * 32, 128, q, k, kc, scores);
            var softmax = p; softmax.Position = length - 1;
            gpu.Dispatch("attention_softmax", softmax, rows * 32 * 256, 256, scores);
            gpu.Dispatch("draft_attention_values", p, rows * 4096, 128, scores, v, vc, output); gpu.End();
            foreach (int row in new[] { 0, rows - 1 }) foreach (int head in new[] { 0, 3, 4, 31 })
            {
                double[] logits = new double[length];
                for (int slot = 0; slot < length; slot++)
                {
                    bool context = slot < 2048; int t = start + slot;
                    if (context && position + row - t >= 2048) { logits[slot] = double.NegativeInfinity; continue; }
                    for (int d = 0; d < 128; d++) logits[slot] += Read(q, (row * 32 + head) * 128 + d, false) * (double)(context ? Read(kc, ((t % 2048) * 8 + head / 4) * 128 + d, half) : Read(k, ((slot - 2048) * 8 + head / 4) * 128 + d, false)) / Math.Sqrt(128);
                }
                double max = logits.Max(), sum = logits.Sum(x => Math.Exp(x - max));
                foreach (int d in new[] { 0, 63, 127 })
                {
                    double expected = 0;
                    for (int slot = 0; slot < length; slot++) expected += Math.Exp(logits[slot] - max) / sum * (slot < 2048 ? Read(vc, (((start + slot) % 2048) * 8 + head / 4) * 128 + d, half) : Read(v, ((slot - 2048) * 8 + head / 4) * 128 + d, false));
                    Check(Read(output, (row * 32 + head) * 128 + d, false), expected, 2e-5, "draft mixed cache attention");
                }
            }
        }
        Console.WriteLine(JsonSerializer.Serialize(new { kv_storage_checks_passed = checks, precisions = new[] { "fp32", "fp16" }, reference = "CPU FP64 RoPE/softmax with storage rounding; prefix and tail guards; single-token cache reads; draft ring wrap and mixed-precision cached/local attention" }));
    }
}
