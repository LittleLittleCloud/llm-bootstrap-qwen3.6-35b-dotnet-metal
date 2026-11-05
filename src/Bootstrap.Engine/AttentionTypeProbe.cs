using System.Diagnostics;
using System.Text.Json;
namespace Bootstrap.Engine;

// Local type-only controls. The production kernels and runtime defaults are unchanged.
internal static class AttentionTypeProbe
{
    static ushort Bf(float x) => (ushort)(BitConverter.SingleToUInt32Bits(x) >> 16);
    static void Check(bool pass, string message) { if (!pass) throw new Exception(message); }
    static string Replace(string source, string old, string replacement)
    {
        Check(source.Contains(old), "Probe source anchor missing: " + old);
        return source.Replace(old, replacement);
    }
    public static void Run(string source, string extra, string outputDirectory, int runs, string? candidateSource = null, string? selectedVariants = null)
    {
        Check(runs >= 3, "Use at least three alternating runs"); Directory.CreateDirectory(outputDirectory);
        int start = source.IndexOf("template<int BQ,int BK,int SG,int G,bool Interior=false", StringComparison.Ordinal);
        Check(start >= 0, "Cannot find production attention template");
        int end = source.IndexOf("#define ONLINE_REUSE", start, StringComparison.Ordinal);
        Check(start >= 0 && end > start, "Cannot isolate production attention template");
        string clone = source[start..end];
        clone = Replace(clone, "typename Probability=half>", "typename Probability=half,typename Query=half,bool Scale=true>");
        clone = Replace(clone, "attention_online_reuse(device half*q", "attention_online_type_probe(device Query*q");
        clone = Replace(clone, "if constexpr(Quantized)a*=ks", "if constexpr(Quantized&&Scale)a*=ks");
        clone = Replace(clone, "if constexpr(Quantized)scale=t<p.k?ks", "if constexpr(Quantized&&Scale)scale=t<p.k?ks");
        clone = Replace(clone, "if constexpr(Quantized){uint t=key+lane+j*32;scaled*=", "if constexpr(Quantized&&Scale){uint t=key+lane+j*32;scaled*=");
        string combined = source + "\n" + clone + "\n" + extra + "\n" + candidateSource;
        File.WriteAllText(Path.Combine(outputDirectory, "combined.metal"), combined);
        using var gpu = new Metal(combined);
        if (candidateSource is null) Matmul(gpu, runs);
        foreach (var (history, rows) in new[] { (0, 33), (8192, 2048), (27900, 2048) }) Online(gpu, history, rows, runs, selectedVariants);
    }
    static void Time(Metal gpu, string family, int history, int rows, string[] variants, int runs, Action<string> encode)
    {
        foreach (string name in variants) { var warm = Stopwatch.StartNew(); while (warm.ElapsedMilliseconds < 100) { gpu.Begin(); encode(name); gpu.End(); } }
        for (int run = 0; run < runs; run++) foreach (string name in run % 2 == 0 ? variants : variants.Reverse())
        {
            int iterations = family == "online" ? 1 : 32;
            gpu.ResetWallTiming(); var watch = Stopwatch.StartNew(); gpu.Begin();
            for (int i = 0; i < iterations; i++) encode(name);
            gpu.End(); double wall = watch.Elapsed.TotalMilliseconds;
            Console.WriteLine(JsonSerializer.Serialize(new { type = "type_probe_timing", family, history, rows, variant = name, run, iterations, wall_ms = wall, timing = gpu.ReadWallTiming() }));
        }
    }
    static void Matmul(Metal gpu, int runs)
    {
        const int groups = 256, m = 32;
        foreach (bool pv in new[] { false, true })
        {
            int k = pv ? 128 : 256, n = pv ? 256 : 128;
            using var lh = gpu.Allocate((ulong)groups * m * (ulong)k * 2); using var lb = gpu.Allocate(lh.Length);
            using var ri = gpu.Allocate((ulong)groups * (ulong)n * (ulong)k); using var rh = gpu.Allocate(ri.Length * 2); using var rb = gpu.Allocate(rh.Length);
            using var output = gpu.Allocate((ulong)groups * m * (ulong)n * 4 + 4);
            var random = new Random(726);
            for (int i = 0; i < groups * m * k; i++) { float value = pv ? random.Next(0, 17) / 256f : random.Next(-16, 17) / 16f; lh.Values<Half>(groups * m * k)[i] = (Half)value; lb.Values<ushort>(groups * m * k)[i] = Bf(value); }
            for (int i = 0; i < groups * n * k; i++) { int value = random.Next(-127, 128); ri.Values<sbyte>(groups * n * k)[i] = (sbyte)value; rh.Values<Half>(groups * n * k)[i] = (Half)value; rb.Values<ushort>(groups * n * k)[i] = Bf(value); }
            string family = pv ? "pv" : "qk";
            string[] names = ["hh", "hi", "bb", "bi"];
            void Encode(string name) => gpu.Dispatch("probe_" + family + "_" + name, new(), groups * 256, 256, name[0] == 'h' ? lh : lb, name[1] == 'i' ? ri : name[1] == 'h' ? rh : rb, output);
            foreach (string name in names)
            {
                output.Values<float>(groups * m * n + 1).Fill(float.NaN); gpu.Begin(); Encode(name); gpu.End();
                Check(output.Values<float>(groups * m * n).ToArray().All(float.IsFinite), "Nonfinite matmul/invalid output extent");
                Check(float.IsNaN(output.Values<float>(groups * m * n + 1)[groups * m * n]), "Matmul guard overwritten");
                foreach (int g in new[] { 0, 7, groups - 1 }) foreach (int row in new[] { 0, 17, 31 }) foreach (int col in new[] { 0, n / 2, n - 1 })
                {
                    double expected = 0; for (int d = 0; d < k; d++) expected += (double)(float)lh.Values<Half>(groups * m * k)[(g * m + row) * k + d] * ri.Values<sbyte>(groups * n * k)[(g * n + col) * k + d];
                    Check(output.Values<float>(groups * m * n)[(g * m + row) * n + col] == expected, $"Matmul type check failed: {family}/{name}");
                }
            }
            Console.WriteLine(JsonSerializer.Serialize(new { type = "type_probe_check", family, m, n, k, groups, equal_representable_inputs = true, fp64_sampled_reference = "exact", guards = true }));
            Time(gpu, family, 0, m, names, runs, Encode);
        }
    }
    static void Online(Metal gpu, int history, int rows, int runs, string? selectedVariants)
    {
        int length = history + rows, padded = (length + 255) / 256 * 256;
        using var qh = gpu.Allocate((ulong)rows * 4096 * 2); using var qb = gpu.Allocate(qh.Length);
        using var ki = gpu.Allocate((ulong)length * 512); using var vi = gpu.Allocate((ulong)padded * 512);
        using var ks = gpu.Allocate((ulong)length * 2 * 4); using var vs = gpu.Allocate(ks.Length); using var ones = gpu.Allocate(ks.Length);
        using var gate = gpu.Allocate((ulong)rows * 4096 * 4); using var output = gpu.Allocate(gate.Length + 4);
        var random = new Random(913);
        for (int i = 0; i < rows * 4096; i++) { float x = random.Next(-16, 17) / 16f; qh.Values<Half>(rows * 4096)[i] = (Half)x; qb.Values<ushort>(rows * 4096)[i] = Bf(x); }
        for (int t = 0; t < length; t++) for (int h = 0; h < 2; h++)
        {
            int vector = t * 2 + h; ks.Values<float>(length * 2)[vector] = 1 / 64f; vs.Values<float>(length * 2)[vector] = 1 / 128f;
            for (int d = 0; d < 256; d++) { ki.Values<sbyte>(length * 512)[vector * 256 + d] = (sbyte)random.Next(-127, 128); vi.Values<sbyte>(padded * 512)[(t / 256 * 2 + h) * 65536 + d * 256 + t % 256] = (sbyte)random.Next(-127, 128); }
        }
        ones.Values<float>(length * 2).Fill(1); gate.Values<float>(rows * 4096).Clear();
        var p = new KernelArgs { M = (uint)rows, K = (uint)length, Position = (uint)history };
        string[] names = selectedVariants?.Split(',') ?? ["hh", "hb", "bh", "bb", "hb_unit", "hb_noscale"];
        void Encode(string name)
        {
            bool candidate = name.StartsWith("opt_", StringComparison.Ordinal);
            bool unit = name is "hb_unit" or "hb_noscale";
            int tileRows = candidate && name.Contains("q64") ? 64 : 32;
            gpu.Dispatch2D(candidate ? name : "probe_online_" + name, p, (uint)((rows * 8 + tileRows - 1) / tileRows) * 256, 2, 256, candidate || name[0] == 'h' ? qh : qb, ki, vi, gate, output, unit ? ones : ks, unit ? ones : vs);
        }
        byte[]? unitReference = null; double maximumError = 0;
        foreach (string name in names)
        {
            output.Values<float>(rows * 4096 + 1).Fill(float.NaN); gpu.Begin(); Encode(name); gpu.End();
            Check(output.Values<float>(rows * 4096).ToArray().All(float.IsFinite), "Nonfinite online type probe");
            Check(float.IsNaN(output.Values<float>(rows * 4096 + 1)[rows * 4096]), "Online guard overwritten");
            if (name == "hb_unit") unitReference = output.Values<byte>(rows * 4096 * 4).ToArray();
            if (name == "hb_noscale") Check(output.Values<byte>(rows * 4096 * 4).SequenceEqual(unitReference), "Scale=1 removal changed output");
            if (name is "hb_unit" or "hb_noscale") continue;
            foreach (int row in new[] { 0, rows - 1 }.Distinct()) foreach (int head in new[] { 0, 7, 8, 15 })
            {
                var logits = new double[history + row + 1];
                for (int t = 0; t < logits.Length; t++) for (int d = 0; d < 256; d++) logits[t] += (double)(float)qh.Values<Half>(rows * 4096)[((head / 8 * rows + row) * 8 + head % 8) * 256 + d] * ki.Values<sbyte>(length * 512)[(t * 2 + head / 8) * 256 + d] / (16 * 64);
                double max = logits.Max(), sum = logits.Sum(x => Math.Exp(x - max));
                foreach (int d in new[] { 0, 127, 255 })
                {
                    double expected = 0, absolute = 0;
                    for (int t = 0; t < logits.Length; t++) { double value = Math.Exp(logits[t] - max) / sum * vi.Values<sbyte>(padded * 512)[(t / 256 * 2 + head / 8) * 65536 + d * 256 + t % 256] / 256; expected += value; absolute += Math.Abs(value); }
                    double error = Math.Abs(output.Values<float>(rows * 4096)[(row * 16 + head) * 256 + d] - expected); maximumError = Math.Max(maximumError, error);
                    Check(error <= absolute / 256 + 1e-4, "Online FP64 type probe mismatch");
                }
            }
        }
        Console.WriteLine(JsonSerializer.Serialize(new { type = "type_probe_check", family = "online", history, rows, equal_representable_queries = true, shared_int8_cache = true, max_abs_error = maximumError, unit_scale_removal_bitwise = unitReference is not null }));
        Time(gpu, "online", history, rows, names, runs, Encode);
    }
}
