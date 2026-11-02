using System.Diagnostics;
using System.Text.Json;

namespace Bootstrap.Engine;

internal static class FastAttentionTests
{
    public static void Run(string source, int repeats)
    {
        if (repeats < 1) throw new ArgumentOutOfRangeException(nameof(repeats));
        using var gpu = new Metal(source);
        foreach (var (history, rows) in new[] { (0, 1), (17, 7), (0, 512), (1024, 512), (10520, 190), (32768, 63), (10520, 512) })
        {
            int length = history + rows;
            using var q = gpu.Allocate((ulong)rows * 4096 * 4);
            using var k = gpu.Allocate((ulong)length * 512 * 4);
            using var v = gpu.Allocate((ulong)length * 512 * 4);
            using var gate = gpu.Allocate((ulong)rows * 4096 * 4);
            using var scores = gpu.Allocate(Math.Min((ulong)rows * 16 * (ulong)length, ContextLimits.AttentionWorkspaceFloats) * 4);
            using var output = gpu.Allocate((ulong)rows * 4096 * 4);
            using var alias = gpu.Allocate((ulong)rows * 4096 * 4);
            Fill(q, rows * 4096, 17, 3); Fill(k, length * 512, 31, 3);
            Fill(v, length * 512, 23, 1); Fill(gate, rows * 4096, 11, 1);
            var p = new KernelArgs { M = (uint)rows, Position = (uint)history, K = (uint)length };
            var softmax = p; softmax.Position = (uint)(length - 1);
            foreach (string mode in new[] { "tiled", "fp16", "fused-fp16", "fused-fp16-128" })
            {
                void Encode(GpuBuffer input, GpuBuffer destination)
                {
                    if (mode.StartsWith("fused")) gpu.Dispatch2D(mode.EndsWith("128") ? "batch_attention_fused_fp16_bk128" : "batch_attention_fused_fp16", p, (uint)((rows + 15) / 16) * 128, 16, 128, input, k, v, gate, destination);
                    else
                    {
                        string suffix = mode == "fp16" ? "_fp16" : "";
                        int capacity = ContextLimits.AttentionWorkspaceFloats / 16 / length;
                        for (int first = 0; first < rows; first += capacity)
                        {
                            int count = Math.Min(capacity, rows - first);
                            var part = p; part.M = (uint)count; part.Position = (uint)(history + first);
                            var normalization = part; normalization.Position = (uint)(length - 1);
                            var inputPart = new Binding(input, (ulong)first * 4096 * 4);
                            var outputPart = new Binding(destination, (ulong)first * 4096 * 4);
                            var gatePart = new Binding(gate, (ulong)first * 4096 * 4);
                            gpu.Dispatch2D("batch_attention_scores_tiled" + suffix, part, (uint)((length + 31) / 32) * 128, (uint)((count + 15) / 16) * 16, 128, inputPart, k, scores);
                            gpu.Dispatch("attention_softmax", normalization, (uint)count * 16 * 256, 256, scores);
                            gpu.Dispatch2D("batch_attention_values_tiled" + suffix, part, 8 * 128, (uint)((count + 15) / 16) * 16, 128, scores, v, gatePart, outputPart);
                        }
                    }
                }
                gpu.Begin(); Encode(q, output); gpu.End();
                double error = Check(q, k, v, gate, output, rows, history, mode);
                // The real engine aliases the rotated queries with the final output.
                q.Values<float>(rows * 4096).CopyTo(alias.Values<float>(rows * 4096));
                gpu.Begin(); Encode(alias, alias); gpu.End();
                if (!alias.Values<float>(rows * 4096).SequenceEqual(output.Values<float>(rows * 4096)))
                    throw new Exception($"Attention in-place alias mismatch: {mode}/{history}/{rows}");
                for (int run = 0; run < repeats; run++)
                {
                    gpu.ResetWallTiming(); var watch = Stopwatch.StartNew();
                    gpu.Begin(); Encode(q, output); gpu.End();
                    Console.WriteLine(JsonSerializer.Serialize(new { type = "fast_attention_microbenchmark", history, rows, length,
                        mode, run, wall_ms = watch.Elapsed.TotalMilliseconds, timing = gpu.ReadWallTiming(),
                        max_abs_error_vs_cpu = error, reference = mode == "tiled" ? "CPU FP64" : "CPU FP64 with explicit FP16 operand/probability rounding", alias_check = "passed" }));
                }
            }
        }
    }
    static void Fill(GpuBuffer buffer, int count, uint seed, float scale)
    {
        var data = buffer.Values<float>(count);
        for (int i = 0; i < count; i++)
        {
            uint h = (uint)i + seed * 0x9e3779b9u; h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15;
            data[i] = ((h & 65535) - 32768f) / 49157f * scale;
        }
    }
    static double Check(GpuBuffer q, GpuBuffer k, GpuBuffer v, GpuBuffer gate, GpuBuffer output, int rows, int history, string mode)
    {
        int length = history + rows;
        var qs = q.Values<float>(rows * 4096); var ks = k.Values<float>(length * 512);
        var vs = v.Values<float>(length * 512); var gs = gate.Values<float>(rows * 4096); var ys = output.Values<float>(rows * 4096);
        float Round(float value) => mode == "tiled" ? value : (float)(Half)value;
        double worst = 0;
        foreach (int row in new[] { 0, rows / 2, rows - 1 }.Distinct()) foreach (int head in new[] { 0, 7, 8, 15 })
        {
            double[] logits = new double[history + row + 1];
            for (int t = 0; t < logits.Length; t++)
                for (int d = 0; d < 256; d++) logits[t] += (double)Round(qs[row * 4096 + head * 256 + d]) * Round(ks[t * 512 + head / 8 * 256 + d]) / 16;
            double max = logits.Max(), sum = logits.Sum(value => Math.Exp(value - max));
            foreach (int d in new[] { 0, 127, 255 })
            {
                double expected = 0;
                if (!mode.StartsWith("fused"))
                {
                    for (int t = 0; t < logits.Length; t++)
                    {
                        double prob = Math.Exp(logits[t] - max) / sum;
                        if (mode == "fp16") prob = (float)(Half)(float)prob;
                        expected += prob * Round(vs[t * 512 + head / 8 * 256 + d]);
                    }
                }
                else
                {
                    double runningMax = double.NegativeInfinity, denominator = 0;
                    int blockSize = mode.EndsWith("128") ? 128 : 64;
                    for (int start = 0; start < logits.Length; start += blockSize)
                    {
                        int end = Math.Min(start + blockSize, logits.Length);
                        double nextMax = runningMax;
                        for (int t = start; t < end; t++) nextMax = Math.Max(nextMax, logits[t]);
                        double decay = Math.Exp(runningMax - nextMax);
                        expected *= decay; denominator *= decay;
                        for (int t = start; t < end; t++)
                        {
                            double prob = Math.Exp(logits[t] - nextMax);
                            denominator += prob;
                            expected += (float)(Half)(float)prob * Round(vs[t * 512 + head / 8 * 256 + d]);
                        }
                        runningMax = nextMax;
                    }
                    expected /= denominator;
                }
                int at = row * 4096 + head * 256 + d;
                expected /= 1 + Math.Exp(-gs[at]);
                double error = Math.Abs(ys[at] - expected); worst = Math.Max(worst, error);
                if (!float.IsFinite(ys[at]) || error > 5e-5)
                    throw new Exception($"{mode} attention CPU mismatch {history}/{rows}/{row}/{head}/{d}: {ys[at]} vs {expected}");
            }
        }
        return worst;
    }
}
