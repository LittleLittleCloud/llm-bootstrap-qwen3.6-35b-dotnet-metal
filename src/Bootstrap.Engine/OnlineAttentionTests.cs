using System.Diagnostics;
using System.Text.Json;
namespace Bootstrap.Engine;

internal static class OnlineAttentionTests
{
    public static void Run(string source, int runs, string? selectedCase, string? selectedVariants, bool stress = false, bool halfKv = true)
    {
        if (runs < 1) throw new ArgumentOutOfRangeException(nameof(runs));
        if (selectedCase is not null) { var dims = selectedCase.Split(',').Select(int.Parse).ToArray(); if (dims.Length != 2 || dims[0] < 0 || dims[1] < 1 || (long)dims[0] + dims[1] > ContextLimits.NativeWindow) throw new ArgumentException("--case requires history,rows within the native context window"); }
        using var gpu = new Metal(source);
        using var device = new DeviceAttentionWorkspace(gpu);
        var variants = halfKv ? new[] { "baseline", "q32_k256_s8_g1_r1", "q32_k256_s8_g8_r1", "selected" } : new[] { "baseline", "selected" };
        if (selectedVariants is not null) { var requested = selectedVariants.Split(','); if (requested.Any(v => !variants.Contains(v))) throw new ArgumentException("Unknown online attention variant"); variants = requested; }
        var cases = new[] { (0, 1), (17, 7), (127, 31), (127, 32), (257, 33), (127, 63), (0, 512), (0, 2048), (8192, 2048), (10520, 190), (32768, 63), (30000, 2048) };
        if (selectedCase is not null) { var c = selectedCase.Split(',').Select(int.Parse).ToArray(); cases = [(c[0], c[1])]; }
        foreach (var (history, rows) in cases)
        {
            int length = history + rows;
            using var q = gpu.Allocate((ulong)rows * 4096 * 4); using var qhalf = gpu.Allocate((ulong)rows * 4096 * 2);
            using var k = gpu.Allocate((ulong)length * 512 * 4); using var v = gpu.Allocate((ulong)length * 512 * 4);
            using var kh = gpu.Allocate((ulong)(length + 128) * 512 * 2); using var vh = gpu.Allocate((ulong)(length + 128) * 512 * 2);
            using var gate = gpu.Allocate((ulong)rows * 4096 * 4);
            using var scores = gpu.Allocate(Math.Min((ulong)rows * 16 * (ulong)length, ContextLimits.AttentionWorkspaceFloats) * 4);
            using var output = gpu.Allocate((ulong)(rows * 4096 + 1) * 4); using var alias = gpu.Allocate((ulong)(rows * 4096 + 1) * 4);
            FastAttentionTests.Fill(q, rows * 4096, 17, 3); FastAttentionTests.Fill(k, length * 512, 31, 3);
            FastAttentionTests.Fill(v, length * 512, 23, 1); FastAttentionTests.Fill(gate, rows * 4096, 11, 1);
            if (stress) { for (int i = 0; i < rows * 4096; i++) q.Values<float>(rows * 4096)[i] *= 16; for (int i = 0; i < length * 512; i++) k.Values<float>(length * 512)[i] *= 16; }
            kh.Values<Half>((length + 128) * 512).Fill(Half.NaN); vh.Values<Half>((length + 128) * 512).Fill(Half.NaN);
            for (int i = 0; i < length * 512; i++) { kh.Values<Half>(length * 512)[i] = (Half)k.Values<float>(length * 512)[i]; vh.Values<Half>(length * 512)[i] = (Half)v.Values<float>(length * 512)[i]; }
            device.Prepare(rows, length, halfKv);
            foreach (var variant in variants)
            {
                int bq = 0, bk = 0, sg = 0, g = 0;
                if (variant == "selected") { if (rows >= 32) (bq, bk, sg, g) = (32, 256, 8, DeviceAttentionWorkspace.OnlineGroupHeads); }
                else if (variant != "baseline") { var spec = variant.Split('_').Select(s => int.Parse(s[1..])).ToArray(); (bq, bk, sg, g) = (spec[0], spec[1], spec[2], spec[3]); }
                void Encode(GpuBuffer input, GpuBuffer dest)
                {
                    gpu.Begin();
                    if (variant == "baseline") device.Encode(input, halfKv ? kh : k, halfKv ? vh : v, gate, scores, dest, rows, history, halfKv, true);
                    else if (variant == "selected") device.EncodeOnline(input, halfKv ? kh : k, halfKv ? vh : v, gate, scores, dest, rows, history, halfKv);
                    else
                    {
                        var p = new KernelArgs { M = (uint)rows, K = (uint)length, Position = (uint)history, A = (uint)g };
                        gpu.Dispatch("attention_query_grouped_half", p, (uint)rows * 1024, 128, input, qhalf);
                        gpu.Dispatch2D("attention_online_" + variant, p, (uint)((rows * g + bq - 1) / bq * sg * 32), (uint)(16 / g), (uint)(sg * 32), qhalf, kh, vh, gate, dest);
                    }
                    gpu.End();
                }
                output.Values<float>(rows * 4096 + 1).Fill(float.NaN); Encode(q, output);
                // Compare every candidate to FP64 softmax with FP16 Q/K/V.
                // The fused path rounds block-local exponentials to FP16.
                double error = Check(q, k, v, gate, output, rows, history);
                q.Values<float>(rows * 4096).CopyTo(alias.Values<float>(rows * 4096)); alias.Values<float>(rows * 4096 + 1)[rows * 4096] = float.NaN;
                Encode(alias, alias);
                if (!alias.Values<float>(rows * 4096).SequenceEqual(output.Values<float>(rows * 4096))) throw new Exception($"Alias mismatch: {variant}");
                if (!float.IsNaN(output.Values<float>(rows * 4096 + 1)[rows * 4096]) || !float.IsNaN(alias.Values<float>(rows * 4096 + 1)[rows * 4096])) throw new Exception("Output guard overwritten");
                var warm = Stopwatch.StartNew(); while (warm.Elapsed.TotalMilliseconds < 250) Encode(q, output);
                for (int run = 0; run < runs; run++)
                {
                    gpu.ResetWallTiming(); var watch = Stopwatch.StartNew(); Encode(q, output); double wall = watch.Elapsed.TotalMilliseconds;
                    Console.WriteLine(JsonSerializer.Serialize(new { type = "online_attention", history, rows, stress, kv_cache_precision = halfKv ? "fp16" : "fp32", variant, path = bq == 0 ? "causal-device" : "online", bq, bk, sg, g, run, wall_ms = wall, timing = gpu.ReadWallTiming(), max_abs_error = error, alias_check = "passed", guards = "passed" }));
                }
            }
        }
    }
    static double Check(GpuBuffer q, GpuBuffer k, GpuBuffer v, GpuBuffer gate, GpuBuffer output, int rows, int history)
    {
        int length = rows + history; double worst = 0;
        var qs = q.Values<float>(rows * 4096); var ks = k.Values<float>(length * 512); var vs = v.Values<float>(length * 512); var gs = gate.Values<float>(rows * 4096); var ys = output.Values<float>(rows * 4096);
        foreach (float value in ys) if (!float.IsFinite(value)) throw new Exception("Nonfinite attention output");
        bool exhaustiveRows = history <= 128 && rows <= 64;
        foreach (int row in (exhaustiveRows ? Enumerable.Range(0, rows) : new[] { 0, rows / 2, rows - 1 }.Distinct())) foreach (int head in (exhaustiveRows ? Enumerable.Range(0, 16) : new[] { 0, 1, 3, 7, 8, 9, 15 }))
        {
            double[] logits = new double[history + row + 1];
            for (int t = 0; t < logits.Length; t++) for (int d = 0; d < 256; d++) logits[t] += (double)(float)(Half)qs[(row * 16 + head) * 256 + d] * (float)(Half)ks[(t * 2 + head / 8) * 256 + d] / 16;
            double max = logits.Max(), sum = logits.Sum(x => Math.Exp(x - max));
            foreach (int d in new[] { 0, 127, 128, 255 })
            {
                double expected = 0; for (int t = 0; t < logits.Length; t++) expected += Math.Exp(logits[t] - max) / sum * (float)(Half)vs[(t * 2 + head / 8) * 256 + d];
                expected /= 1 + Math.Exp(-gs[(row * 16 + head) * 256 + d]);
                double error = Math.Abs(ys[(row * 16 + head) * 256 + d] - expected); worst = Math.Max(worst, error);
                if (error > 5e-4) throw new Exception($"Reference mismatch {history}/{rows}/{row}/{head}/{d}: {ys[(row * 16 + head) * 256 + d]} vs {expected}");
            }
        }
        return worst;
    }
}
