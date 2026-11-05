using System.Diagnostics;
using System.Text.Json;

namespace Bootstrap.Engine;

// Development diagnostics: independent CPU quantization/FP64 attention, guards,
// fused vs separate preparation, scalar verifier reads and in-place TensorOps.
internal static class Int8KvTests
{
    static int ValueIndex(int vector, int d) => (vector / 2 / 256 * 2 + vector % 2) * 65536 + d * 256 + vector / 2 % 256;
    static ulong ValueBytes(int length) => (ulong)((length + 255) / 256 * 256) * 512;
    static void Check(bool condition, string message) { if (!condition) throw new Exception(message); }
    static double Rotate(ReadOnlySpan<float> x, int d, int position)
    {
        if (d >= 64) return x[d];
        int j = d % 32; double angle = position * Math.Pow(10_000_000, -2.0 * j / 64);
        return d < 32 ? x[j] * Math.Cos(angle) - x[j + 32] * Math.Sin(angle) : x[j + 32] * Math.Cos(angle) + x[j] * Math.Sin(angle);
    }
    public static void Run(string source, int runs, string? selectedCase, string? tile = null)
    {
        if (runs < 1) throw new ArgumentOutOfRangeException(nameof(runs));
        using var gpu = new Metal(source);
        Store(gpu);
        var cases = new[] { (0, 1), (17, 7), (127, 31), (127, 32), (257, 33), (4093, 7), (8192, 128), (30000, 33) };
        if (selectedCase is not null)
        {
            var dims = selectedCase.Split(',').Select(int.Parse).ToArray();
            if (dims.Length != 2 || dims[0] < 0 || dims[1] < 1 || (long)dims[0] + dims[1] > ContextLimits.NativeWindow) throw new ArgumentException("--case requires history,rows within native context");
            cases = [(dims[0], dims[1])];
        }
        foreach (var (history, rows) in cases) Attention(gpu, history, rows, runs, tile);
    }
    static void Store(Metal gpu)
    {
        const int rows = 33, position = 4093, length = position + rows, capacity = length + 1;
        using var q = gpu.Allocate(rows * 4096 * 4); using var k = gpu.Allocate(rows * 512 * 4); using var v = gpu.Allocate(ValueBytes(length + 1));
        using var qr = gpu.Allocate(q.Length); using var kc = gpu.Allocate(capacity * 512); using var vc = gpu.Allocate(ValueBytes(capacity));
        using var ks = gpu.Allocate(capacity * 2 * 4); using var vs = gpu.Allocate(ks.Length);
        FastAttentionTests.Fill(q, rows * 4096, 17, 3); FastAttentionTests.Fill(k, rows * 512, 31, 3); FastAttentionTests.Fill(v, rows * 512, 23, 1);
        // Zero vectors, outliers, opposite signs and distinct K/V magnitudes.
        k.Values<float>(256).Clear(); v.Values<float>(256).Clear();
        k.Values<float>(rows * 512)[777] = -97; v.Values<float>(rows * 512)[999] = 41;
        foreach (var b in new[] { kc, vc, ks, vs }) b.Values<byte>((int)b.Length).Fill(0x55);
        var p = new KernelArgs { M = rows, Position = position, K = length, Theta = 10_000_000, Epsilon = 1e-6f, A = 8 };
        gpu.Begin(); gpu.Dispatch("batch_rope_int8", p, rows * 4096, 256, q, k, v, qr, kc, vc, ks, vs); gpu.End();
        double worst = 0;
        for (int row = 0; row < rows; row++) for (int head = 0; head < 2; head++)
        {
            int vector = row * 2 + head, dest = (position + row) * 2 + head;
            double[] key = new double[256]; for (int d = 0; d < 256; d++) key[d] = Rotate(k.Values<float>(rows * 512).Slice(vector * 256, 256), d, position + row);
            double keyMax = key.Max(Math.Abs), valMax = v.Values<float>(rows * 512).Slice(vector * 256, 256).ToArray().Max(x => Math.Abs(x));
            foreach (var (cache, scale, original, maximum) in new[] { (kc, ks, key, keyMax), (vc, vs, v.Values<float>(rows * 512).Slice(vector * 256, 256).ToArray().Select(x => (double)x).ToArray(), valMax) })
            {
                double sc = scale.Values<float>(capacity * 2)[dest]; Check(Math.Abs(sc - maximum / 127) < 2e-5, "INT8 scale mismatch");
                for (int d = 0; d < 256; d++)
                {
                    int code = cache.Values<sbyte>((int)cache.Length)[cache == vc ? ValueIndex(dest, d) : dest * 256 + d];
                    double error = Math.Abs(code * sc - original[d]); worst = Math.Max(worst, error);
                    Check(code >= -127 && error <= maximum / 254 + 3e-3, "INT8 quantization exceeds half-step + RoPE tolerance");
                    if (maximum == 0) Check(code == 0 && sc == 0, "Zero vector encoding");
                }
            }
        }
        foreach (var (buffer, bytes) in new[] { (kc, 512), (ks, 8), (vs, 8) })
        {
            Check(buffer.Values<byte>(position * bytes).ToArray().All(x => x == 0x55), "Store changed prefix");
            Check(buffer.Values<byte>(bytes, (ulong)(length * bytes)).ToArray().All(x => x == 0x55), "Store changed tail");
        }
        for (int t = 0; t < (int)vc.Length / 512; t++) if (t < position || t >= length)
            for (int h = 0; h < 2; h++) for (int d = 0; d < 256; d++) Check(vc.Values<byte>((int)vc.Length)[ValueIndex(t * 2 + h, d)] == 0x55, "V store changed prefix/tail");
        // Compare fused normalization/RoPE/quantization with the independent stages.
        using var raw = gpu.Allocate(rows * 8192 * 4); using var qw = gpu.Allocate(512); using var kw = gpu.Allocate(512);
        using var split = gpu.Allocate(q.Length); using var gate = gpu.Allocate(q.Length); using var normK = gpu.Allocate(k.Length);
        using var packed = gpu.Allocate(q.Length / 2); using var packedRef = gpu.Allocate(packed.Length);
        using var gateRef = gpu.Allocate(gate.Length); using var kRef = gpu.Allocate(kc.Length); using var vRef = gpu.Allocate(vc.Length);
        using var ksRef = gpu.Allocate(ks.Length); using var vsRef = gpu.Allocate(vs.Length); using var dummy = gpu.Allocate(4);
        FastAttentionTests.Fill(raw, rows * 8192, 31, 2);
        for (int i = 0; i < 256; i++) { qw.Values<ushort>(256)[i] = 0x3e00; kw.Values<ushort>(256)[i] = 0xbe00; }
        gpu.Begin();
        gpu.Dispatch("batch_split_q", p, rows * 4096, 128, raw, split, gateRef);
        gpu.Dispatch("rmsnorm", new() { N = 256, M = rows * 16, Flags = 1, Epsilon = 1e-6f }, rows * 16 * 256, 256, split, qw, split, dummy);
        gpu.Dispatch("rmsnorm", new() { N = 256, M = rows * 2, Flags = 1, Epsilon = 1e-6f }, rows * 2 * 256, 256, k, kw, normK, dummy);
        gpu.Dispatch("batch_rope_int8", p, rows * 4096, 256, split, normK, v, qr, kRef, vRef, ksRef, vsRef);
        gpu.Dispatch("attention_query_grouped_half", p, rows * 1024, 128, qr, packedRef);
        gpu.Dispatch("attention_prepare_fused_int8", p, rows * 18 * 256, 256, raw, k, v, qw, kw, packed, gate, kc, vc, ks, vs);
        gpu.End();
        foreach (var (a, b, offset, bytes) in new[] { (packed, packedRef, 0, rows * 4096 * 2), (gate, gateRef, 0, rows * 4096 * 4), (kc, kRef, position * 512, rows * 512), (ks, ksRef, position * 8, rows * 8), (vs, vsRef, position * 8, rows * 8) })
            Check(a.Values<byte>(bytes, (ulong)offset).SequenceEqual(b.Values<byte>(bytes, (ulong)offset)), "Fused preparation mismatch");
        for (int t = position; t < length; t++) for (int h = 0; h < 2; h++) for (int d = 0; d < 256; d++)
            Check(vc.Values<byte>((int)vc.Length)[ValueIndex(t * 2 + h, d)] == vRef.Values<byte>((int)vRef.Length)[ValueIndex(t * 2 + h, d)], "Fused V preparation mismatch");
        Console.WriteLine(JsonSerializer.Serialize(new { type = "int8_store", passed = true, max_abs_quantization_error = worst, fused_bitwise = true, prefix_and_tail_guards = true }));
    }
    static void Attention(Metal gpu, int history, int rows, int runs, string? tile)
    {
        var shape = (tile ?? "32,256").Split(',').Select(int.Parse).ToArray();
        if (shape.Length != 2 || !new[] { "32,256", "32,128" }.Contains(tile ?? "32,256")) throw new ArgumentException("Invalid INT8 tile");
        int length = history + rows;
        using var device = new DeviceAttentionWorkspace(gpu);
        using var q = gpu.Allocate((ulong)rows * 4096 * 4); using var gate = gpu.Allocate(q.Length);
        using var k = gpu.Allocate((ulong)(length + 1) * 512); using var v = gpu.Allocate(ValueBytes(length + 1));
        using var ks = gpu.Allocate((ulong)(length + 1) * 2 * 4); using var vs = gpu.Allocate(ks.Length);
        using var kh = gpu.Allocate((ulong)length * 512 * 2); using var vh = gpu.Allocate(kh.Length);
        using var output = gpu.Allocate(q.Length + 4); using var alias = gpu.Allocate(output.Length);
        using var reference = gpu.Allocate(output.Length);
        using var scores = gpu.Allocate(Math.Min((ulong)rows * 16 * (ulong)length, ContextLimits.AttentionWorkspaceFloats) * 4);
        FastAttentionTests.Fill(q, rows * 4096, 17, 3); FastAttentionTests.Fill(gate, rows * 4096, 11, 1);
        var random = new Random(2941); var keys = new float[length * 512]; var values = new float[keys.Length];
        for (int vector = 0; vector < length * 2; vector++)
        {
            for (int d = 0; d < 256; d++) { keys[vector * 256 + d] = (float)(random.NextDouble() * 6 - 3); values[vector * 256 + d] = (float)(random.NextDouble() * 2 - 1); }
            if (vector % 41 == 0) Array.Clear(values, vector * 256, 256);
            if (vector % 37 == 0) Array.Clear(keys, vector * 256, 256);
            if (vector % 43 == 1) keys[vector * 256 + 13] = -17;
            foreach (var (source, cache, scale, half) in new[] { (keys, k, ks, kh), (values, v, vs, vh) })
            {
                float max = source.AsSpan(vector * 256, 256).ToArray().Max(x => Math.Abs(x)), sc = max / 127;
                scale.Values<float>(length * 2)[vector] = sc;
                for (int d = 0; d < 256; d++) { int at = vector * 256 + d; cache.Values<sbyte>((int)cache.Length)[cache == v ? ValueIndex(vector, d) : at] = max == 0 ? (sbyte)0 : (sbyte)Math.Clamp(MathF.Round(source[at] * (127 / max)), -127, 127); half.Values<Half>(length * 512)[at] = (Half)source[at]; }
            }
        }
        k.Values<sbyte>(512, (ulong)length * 512).Fill(-128); for (int t = length; t < (int)v.Length / 512; t++) for (int h = 0; h < 2; h++) for (int d = 0; d < 256; d++) v.Values<sbyte>((int)v.Length)[ValueIndex(t * 2 + h, d)] = -128;
        ks.Values<float>(2, (ulong)length * 8).Fill(float.NaN); vs.Values<float>(2, (ulong)length * 8).Fill(float.NaN);
        device.Prepare(rows, length, true, true);
        void Encode(GpuBuffer input, GpuBuffer dest) { gpu.Begin(); device.EncodeInt8(input, k, v, ks, vs, gate, dest, rows, history, tileRows: shape[0], tileKeys: shape[1]); gpu.End(); }
        output.Values<float>(rows * 4096 + 1).Fill(float.NaN); Encode(q, output);
        Check(output.Values<float>(rows * 4096).ToArray().All(float.IsFinite), "Nonfinite INT8 attention");
        Check(float.IsNaN(output.Values<float>(rows * 4096 + 1)[rows * 4096]), "Output guard overwritten");
        q.Values<float>(rows * 4096).CopyTo(alias.Values<float>(rows * 4096)); Encode(alias, alias);
        Check(alias.Values<float>(rows * 4096).SequenceEqual(output.Values<float>(rows * 4096)), "In-place attention mismatch");
        double worst = 0, quantization = 0;
        foreach (int row in new[] { 0, rows / 2, rows - 1 }.Distinct()) foreach (int head in Enumerable.Range(0, 16))
        {
            double[] logits = new double[history + row + 1], original = new double[logits.Length];
            for (int t = 0; t < logits.Length; t++) for (int d = 0; d < 256; d++)
            {
                int at = (t * 2 + head / 8) * 256 + d; double query = (float)(Half)q.Values<float>(rows * 4096)[(row * 16 + head) * 256 + d];
                logits[t] += query * k.Values<sbyte>(length * 512)[at] * ks.Values<float>(length * 2)[t * 2 + head / 8] / 16;
                original[t] += query * (float)kh.Values<Half>(length * 512)[at] / 16;
            }
            double max = logits.Max(), sum = logits.Sum(x => Math.Exp(x - max)), originalMax = original.Max(), originalSum = original.Sum(x => Math.Exp(x - originalMax));
            foreach (int d in new[] { 0, 127, 128, 255 })
            {
                double expected = 0, unquantized = 0, absoluteWeightedValue = 0;
                for (int t = 0; t < logits.Length; t++)
                {
                    int vector = t * 2 + head / 8, at = vector * 256 + d;
                    double contribution = Math.Exp(logits[t] - max) / sum * v.Values<sbyte>((int)v.Length)[ValueIndex(vector, d)] * vs.Values<float>(length * 2)[vector];
                    expected += contribution; absoluteWeightedValue += Math.Abs(contribution);
                    unquantized += Math.Exp(original[t] - originalMax) / originalSum * (float)vh.Values<Half>(length * 512)[at];
                }
                int outAt = (row * 16 + head) * 256 + d; double gateValue = 1 + Math.Exp(-gate.Values<float>(rows * 4096)[outAt]);
                double error = Math.Abs(output.Values<float>(rows * 4096)[outAt] - expected / gateValue);
                worst = Math.Max(worst, error); quantization = Math.Max(quantization, Math.Abs(expected - unquantized) / gateValue);
                // BF16 probability rounding is bounded by 2^-8 times the weighted absolute V;
                // allow 1e-4 for FP32 dot/online reduction and exp error.
                Check(error <= absoluteWeightedValue / gateValue / 256 + 1e-4, $"INT8 TensorOps mismatch {history}/{rows}/{row}/{head}/{d}: {error}");
            }
        }
        // Every short/verifier row uses the same quantized cache, FP32 Q and FP32 softmax.
        if (rows <= 33)
        {
            var p = new KernelArgs { M = (uint)rows, K = (uint)length, Position = (uint)history };
            // Match FP16 query rounding to isolate scalar vs TensorOps arithmetic.
            for (int i = 0; i < rows * 4096; i++) alias.Values<float>(rows * 4096)[i] = (float)(Half)q.Values<float>(rows * 4096)[i];
            gpu.Begin(); gpu.Dispatch2D("batch_attention_scores_int8", p, (uint)length * 32, (uint)rows * 16, 128, alias, k, scores, ks);
            var soft = p; soft.Position = (uint)(length - 1);
            gpu.Dispatch("attention_softmax", soft, (uint)rows * 16 * 256, 256, scores);
            gpu.Dispatch("batch_attention_values_int8", p, (uint)rows * 4096, 128, scores, v, gate, reference, vs); gpu.End();
            for (int i = 0; i < rows * 4096; i++) Check(float.IsFinite(reference.Values<float>(rows * 4096)[i]) && Math.Abs(reference.Values<float>(rows * 4096)[i] - output.Values<float>(rows * 4096)[i]) < 4.1e-3, "Scalar/verifier attention mismatch");
        }
        if (history == 0 && rows == 1)
        {
            var originalScales = vs.Values<float>(length * 2).ToArray();
            for (int i = 0; i < originalScales.Length; i++) vs.Values<float>(length * 2)[i] = originalScales[i] * 1e-6f;
            Encode(q, alias);
            double magnitude = output.Values<float>(rows * 4096).ToArray().Max(x => Math.Abs(x));
            for (int i = 0; i < rows * 4096; i++)
                Check(Math.Abs(alias.Values<float>(rows * 4096)[i] / 1e-6 - output.Values<float>(rows * 4096)[i]) < magnitude * .008 + 1e-6, "Small V scale underflow");
            originalScales.CopyTo(vs.Values<float>(length * 2));
            Console.WriteLine(JsonSerializer.Serialize(new { type = "int8_small_values", passed = true, scale_factor = 1e-6 }));
        }
        // Same Q and raw K/V, with FP16 vs INT8 storage; kernel-only warm wall time.
        foreach (bool quantized in new[] { false, true })
        {
            void Run() { gpu.Begin(); if (quantized) device.EncodeInt8(q, k, v, ks, vs, gate, reference, rows, history, tileRows: shape[0], tileKeys: shape[1]); else device.EncodeOnline(q, kh, vh, gate, scores, reference, rows, history, true); gpu.End(); }
            var warm = Stopwatch.StartNew(); while (warm.ElapsedMilliseconds < 200) Run();
            double[] times = new double[runs]; for (int r = 0; r < runs; r++) { var timer = Stopwatch.StartNew(); Run(); times[r] = timer.Elapsed.TotalMilliseconds; }
            Console.WriteLine(JsonSerializer.Serialize(new { type = "int8_attention", history, rows, tile, kv = quantized ? "int8" : "fp16", wall_ms = times, max_abs_vs_fp64_quantized = worst, max_abs_quantization_vs_fp16 = quantization, guards_and_alias = true }));
        }
    }
}
