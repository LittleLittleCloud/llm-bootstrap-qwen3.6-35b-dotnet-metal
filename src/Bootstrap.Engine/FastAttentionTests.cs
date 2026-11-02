using System.Diagnostics;
using System.Text.Json;

namespace Bootstrap.Engine;

internal static class FastAttentionTests
{
    public static void Run(string source, int repeats, bool tiles = false, bool halfKv = false)
    {
        if (repeats < 1) throw new ArgumentOutOfRangeException(nameof(repeats));
        using var gpu = new Metal(source);
        CheckCausalSoftmax(gpu);
        using var deviceAttention = new DeviceAttentionWorkspace(gpu);
        foreach (var (history, rows) in new[] { (0, 1), (17, 7), (0, 512), (1024, 512), (10520, 190), (32768, 63), (10520, 512), (10520, 2048), (30000, 2048) })
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
            using var packedK = gpu.Allocate((ulong)length * 512 * 2);
            using var packedV = gpu.Allocate((ulong)length * 512 * 2);
            for (int i = 0; i < length * 512; i++) { packedK.Values<Half>(length * 512)[i] = (Half)k.Values<float>(length * 512)[i]; packedV.Values<Half>(length * 512)[i] = (Half)v.Values<float>(length * 512)[i]; }
            var cacheK = halfKv ? packedK : k; var cacheV = halfKv ? packedV : v;
            var p = new KernelArgs { M = (uint)rows, Position = (uint)history, K = (uint)length, Flags = halfKv ? 256u : 0u };
            var softmax = p; softmax.Position = (uint)(length - 1);
            deviceAttention.Prepare(rows, length, halfKv);
            foreach (string mode in (halfKv ? new[] { "legacy", "tiled", "fp16", "fp16_opt64", "fp16-device", "fp16-device-causal", "fused-fp16", "fused-fp16-128" } : tiles ? new[] { "fp16", "fp16_opt64", "fp16-device", "fp16-device-causal" } : new[] { "tiled", "fp16", "fused-fp16", "fused-fp16-128" }))
            {
                void Encode(GpuBuffer input, GpuBuffer destination)
                {
                    if (mode.StartsWith("fp16-device")) deviceAttention.Encode(input, cacheK, cacheV, gate, scores, destination, rows, history, halfKv, mode == "fp16-device-causal");
                    else if (mode.StartsWith("fused")) gpu.Dispatch2D(mode.EndsWith("128") ? "batch_attention_fused_fp16_bk128" : "batch_attention_fused_fp16", p, (uint)((rows + 15) / 16) * 128, 16, 128, input, cacheK, cacheV, gate, destination);
                    else
                    {
                        string suffix = mode.StartsWith("fp16") ? "_" + mode : "";
                        int capacity = ContextLimits.AttentionWorkspaceFloats / 16 / length;
                        for (int first = 0; first < rows; first += capacity)
                        {
                            int count = Math.Min(capacity, rows - first);
                            var part = p; part.M = (uint)count; part.Position = (uint)(history + first);
                            var normalization = part; normalization.Position = (uint)(length - 1);
                            var inputPart = new Binding(input, (ulong)first * 4096 * 4);
                            var outputPart = new Binding(destination, (ulong)first * 4096 * 4);
                            var gatePart = new Binding(gate, (ulong)first * 4096 * 4);
                            if (mode == "legacy") gpu.Dispatch2D("batch_attention_scores", part, (uint)length * 32, (uint)count * 16, 128, inputPart, cacheK, scores);
                            else gpu.Dispatch2D("batch_attention_scores_tiled" + suffix, part, (uint)((length + 31) / 32) * 128, (uint)((count + 15) / 16) * 16, 128, inputPart, cacheK, scores);
                            gpu.Dispatch("attention_softmax", normalization, (uint)count * 16 * 256, 256, scores);
                            if (mode == "legacy") gpu.Dispatch("batch_attention_values", part, (uint)count * 4096, 128, scores, cacheV, gatePart, outputPart);
                            else gpu.Dispatch2D("batch_attention_values_tiled" + suffix, part, 8 * 128, (uint)((count + 15) / 16) * 16, 128, scores, cacheV, gatePart, outputPart);
                        }
                    }
                }
                gpu.Begin(); Encode(q, output); gpu.End();
                double error = Check(q, k, v, gate, output, rows, history, mode, halfKv);
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
                        mode, kv_cache_precision = halfKv ? "fp16" : "fp32", run, wall_ms = watch.Elapsed.TotalMilliseconds, timing = gpu.ReadWallTiming(),
                        max_abs_error_vs_cpu = error, reference = "CPU FP64 with cache and compute operand rounding per selected mode", alias_check = "passed" }));
                }
            }
        }
    }
    static void CheckCausalSoftmax(Metal gpu)
    {
        int checks = 0;
        foreach (var (history, rows) in new[] { (0, 1), (17, 7), (0, 19), (127, 63) })
        {
            int length = history + rows + 37, elements = rows * 16 * length;
            using var scores = gpu.Allocate((ulong)elements * 4);
            using var probabilities = gpu.Allocate((ulong)(elements + 1) * 2);
            scores.Values<float>(elements).Fill(float.NaN);
            probabilities.Values<Half>(elements + 1).Fill(Half.NaN);
            for (int row = 0; row < rows; row++) for (int head = 0; head < 16; head++)
                for (int t = 0; t <= history + row; t++) scores.Values<float>(elements)[(row * 16 + head) * length + t] = ((t * 31 + head * 13 + row * 7) % 67 - 33) / 8f;
            var p = new KernelArgs { M = (uint)rows, K = (uint)length, Position = (uint)history };
            gpu.Begin(); gpu.Dispatch("attention_softmax_half_causal", p, (uint)rows * 16 * 256, 256, scores, probabilities); gpu.End();
            for (int row = 0; row < rows; row++) for (int head = 0; head < 16; head++)
            {
                int valid = history + row + 1, end = history + Math.Min(rows, (row / 16 + 1) * 16);
                double Score(int t) => ((t * 31 + head * 13 + row * 7) % 67 - 33) / 8.0;
                double max = Enumerable.Range(0, valid).Max(Score), sum = Enumerable.Range(0, valid).Sum(t => Math.Exp(Score(t) - max));
                for (int t = 0; t < length; t++)
                {
                    float actual = (float)probabilities.Values<Half>(elements)[(row * 16 + head) * length + t];
                    double expected = t < valid ? (float)(Half)(Math.Exp(Score(t) - max) / sum) : 0;
                    bool correct = t >= end ? float.IsNaN(actual) : t >= valid ? actual == 0 : float.IsFinite(actual) && Math.Abs(actual - expected) <= Math.Max(1e-7, expected * .0011);
                    if (!correct) throw new Exception($"Causal softmax bounds/reference mismatch: {history}/{row}/{head}/{t}: {actual} vs {expected}");
                    checks++;
                }
            }
            if (!Half.IsNaN(probabilities.Values<Half>(elements + 1)[elements])) throw new Exception("Causal probability guard overwritten");
        }
        Console.WriteLine(JsonSerializer.Serialize(new { type = "causal_softmax_self_test", checks, masked_inputs = "NaN", unused_outputs = "NaN guards preserved" }));
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
    static double Check(GpuBuffer q, GpuBuffer k, GpuBuffer v, GpuBuffer gate, GpuBuffer output, int rows, int history, string mode, bool halfKv)
    {
        int length = history + rows;
        var qs = q.Values<float>(rows * 4096); var ks = k.Values<float>(length * 512);
        var vs = v.Values<float>(length * 512); var gs = gate.Values<float>(rows * 4096); var ys = output.Values<float>(rows * 4096);
        float Round(float value) => mode is "tiled" or "legacy" ? value : (float)(Half)value;
        double worst = 0;
        foreach (int row in new[] { 0, rows / 2, rows - 1 }.Distinct()) foreach (int head in new[] { 0, 7, 8, 15 })
        {
            double[] logits = new double[history + row + 1];
            for (int t = 0; t < logits.Length; t++)
                for (int d = 0; d < 256; d++) logits[t] += (double)Round(qs[row * 4096 + head * 256 + d]) * (halfKv ? (float)(Half)ks[t * 512 + head / 8 * 256 + d] : Round(ks[t * 512 + head / 8 * 256 + d])) / 16;
            double max = logits.Max(), sum = logits.Sum(value => Math.Exp(value - max));
            foreach (int d in new[] { 0, 127, 255 })
            {
                double expected = 0;
                if (!mode.StartsWith("fused"))
                {
                    for (int t = 0; t < logits.Length; t++)
                    {
                        double prob = Math.Exp(logits[t] - max) / sum;
                        if (mode.StartsWith("fp16")) prob = (float)(Half)(float)prob;
                        expected += prob * (halfKv ? (float)(Half)vs[t * 512 + head / 8 * 256 + d] : Round(vs[t * 512 + head / 8 * 256 + d]));
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
                            expected += (float)(Half)(float)prob * (halfKv ? (float)(Half)vs[t * 512 + head / 8 * 256 + d] : Round(vs[t * 512 + head / 8 * 256 + d]));
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
