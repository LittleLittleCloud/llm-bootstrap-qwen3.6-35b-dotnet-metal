using System.Diagnostics;
using System.Text.Json;

namespace Bootstrap.Engine;

// Development benchmark, not an endpoint evaluation. Every variant includes
// gate preparation and is checked against the original recurrence and CPU math.
internal static class DeltaScanTests
{
    public static void Run(string source, int runs)
    {
        if (runs < 1) throw new ArgumentOutOfRangeException(nameof(runs));
        using var gpu = new Metal(source);
        const int stateCount = 32 * 128 * 128;
        using var state = gpu.Allocate((stateCount + 1UL) * 4);
        using var alog = gpu.Allocate(64);
        using var bias = gpu.Allocate(64);
        using var dummy = gpu.Allocate(4);
        for (int h = 0; h < 32; h++)
        {
            alog.Values<ushort>(32)[h] = (ushort)(h % 2 == 0 ? 0 : 0xbf00);
            bias.Values<ushort>(32)[h] = (ushort)(h % 3 == 0 ? 0x3e00 : 0);
        }
        float[] initial = new float[stateCount];
        for (int i = 0; i < initial.Length; i++) initial[i] = (i * 13L % 101 - 50) / 1000f;
        foreach (int m in new[] { 1, 7, 17, 128, 2048 })
        {
            using var q = gpu.Allocate((ulong)m * 2048 * 4);
            using var k = gpu.Allocate((ulong)m * 2048 * 4);
            using var v = gpu.Allocate((ulong)m * 8192 * 4);
            using var a = gpu.Allocate((ulong)m * 32 * 4);
            using var b = gpu.Allocate((ulong)m * 32 * 4);
            using var gates = gpu.Allocate((ulong)m * 64 * 4);
            using var y = gpu.Allocate(((ulong)m * 4096 + 1) * 4);
            for (int i = 0; i < m * 2048; i++)
            {
                q.Values<float>(m * 2048)[i] = (i * 7L % 67 - 33) / 256f;
                k.Values<float>(m * 2048)[i] = (i * 11L % 71 - 35) / 256f;
            }
            for (int i = 0; i < m * 8192; i++) v.Values<float>(m * 8192)[i] = (i * 17L % 127 - 63) / 128f;
            for (int i = 0; i < m * 32; i++)
            {
                a.Values<float>(m * 32)[i] = -2 + (i * 7L % 31 - 15) / 32f;
                b.Values<float>(m * 32)[i] = (i * 13L % 43 - 21) / 16f;
            }
            void Encode(string variant, int begin = 0, int? length = null)
            {
                uint count = (uint)(length ?? m);
                Binding qb = new(q, (ulong)begin * 2048 * 4), kb = new(k, (ulong)begin * 2048 * 4);
                Binding vb = new(v, (ulong)begin * 8192 * 4), ab = new(a, (ulong)begin * 32 * 4), bb = new(b, (ulong)begin * 32 * 4);
                Binding yb = new(y, (ulong)begin * 4096 * 4);
                gpu.Begin();
                if (variant != "baseline") gpu.Dispatch("delta_prepare_gates", new() { M = count }, count * 32, 128, ab, bb, alog, bias, gates);
                if (variant == "baseline") gpu.Dispatch("batch_delta", new() { M = count }, 32 * 128 * 32, 128, qb, kb, vb, ab, bb, alog, bias, state, yb, dummy);
                else if (variant == "gates") gpu.Dispatch("batch_delta_gates", new() { M = count }, 32 * 128 * 32, 128, qb, kb, vb, ab, bb, alog, bias, state, yb, dummy, gates);
                else gpu.Dispatch(variant, new() { M = count }, 32 * 8 * 128, 128, qb, kb, vb, gates, state, yb);
                gpu.End();
            }
            void Reset()
            {
                initial.CopyTo(state.Values<float>(stateCount));
                state.Values<float>(stateCount + 1)[stateCount] = float.NaN;
                y.Values<float>(m * 4096 + 1).Fill(float.NaN);
            }
            string[] variants = ["baseline", "gates", "delta_direct8", "delta_staged8_reference", "delta_staged8", "delta_staged16"];
            Reset(); Encode("baseline");
            float[] reference = y.Values<float>(m * 4096).ToArray(), referenceState = state.Values<float>(stateCount).ToArray();
            // Independent double recurrence, including nonzero incoming state.
            foreach (int h in new[] { 0, 17, 31 }) foreach (int d in new[] { 0, 127 })
            {
                double[] cells = initial.AsSpan((h * 128 + d) * 128, 128).ToArray().Select(x => (double)x).ToArray();
                double gain = Math.Exp(BitConverter.UInt32BitsToSingle((uint)alog.Values<ushort>(32)[h] << 16));
                double shift = BitConverter.UInt32BitsToSingle((uint)bias.Values<ushort>(32)[h] << 16);
                for (int t = 0; t < m; t++)
                {
                    double alpha = a.Values<float>(m * 32)[t * 32 + h] + shift;
                    double decay = Math.Exp(-gain * (Math.Max(alpha, 0) + Math.Log(1 + Math.Exp(-Math.Abs(alpha)))));
                    double beta = 1 / (1 + Math.Exp(-b.Values<float>(m * 32)[t * 32 + h]));
                    double dot = 0;
                    for (int c = 0; c < 128; c++) { cells[c] *= decay; dot += cells[c] * k.Values<float>(m * 2048)[t * 2048 + h / 2 * 128 + c]; }
                    double correction = beta * (v.Values<float>(m * 8192)[t * 8192 + 4096 + h * 128 + d] - dot), expected = 0;
                    for (int c = 0; c < 128; c++) { cells[c] += correction * k.Values<float>(m * 2048)[t * 2048 + h / 2 * 128 + c]; expected += cells[c] * q.Values<float>(m * 2048)[t * 2048 + h / 2 * 128 + c]; }
                    Check(reference[t * 4096 + h * 128 + d], expected);
                }
                for (int c = 0; c < 128; c++) Check(referenceState[(h * 128 + d) * 128 + c], cells[c]);
            }
            Reset(); Encode("delta_staged8_reference");
            float[] originalStaged = y.Values<float>(m * 4096).ToArray(), originalStagedState = state.Values<float>(stateCount).ToArray();
            foreach (string variant in variants)
            {
                Reset(); Encode(variant);
                double worst = 0;
                for (int i = 0; i < reference.Length; i++) worst = Math.Max(worst, Check(y.Values<float>(reference.Length)[i], reference[i]));
                for (int i = 0; i < stateCount; i++) worst = Math.Max(worst, Check(state.Values<float>(stateCount)[i], referenceState[i]));
                if (variant == "delta_staged8" && (!y.Values<float>(m * 4096).SequenceEqual(originalStaged) || !state.Values<float>(stateCount).SequenceEqual(originalStagedState)))
                    throw new Exception("Prefetched DeltaNet differs from original staged recurrence");
                if (!float.IsNaN(y.Values<float>(m * 4096 + 1)[m * 4096]) || !float.IsNaN(state.Values<float>(stateCount + 1)[stateCount])) throw new Exception("Delta guard overwritten");
                // Splitting a chunk must carry the same state, including a tail.
                if (m > 1)
                {
                    Reset(); Encode(variant, 0, m / 2); Encode(variant, m / 2, m - m / 2);
                    for (int i = 0; i < reference.Length; i++) Check(y.Values<float>(reference.Length)[i], reference[i]);
                    for (int i = 0; i < stateCount; i++) Check(state.Values<float>(stateCount)[i], referenceState[i]);
                }
                Console.WriteLine(JsonSerializer.Serialize(new { type = "delta_check", variant, m, max_abs_error = worst, split_state = m > 1 }));
            }
            // Alternate order and warm after host checking. Reset is outside timing.
            for (int run = 0; run <= runs; run++) foreach (string variant in run % 2 == 0 ? variants : variants.Reverse())
            {
                Reset();
                var warm = Stopwatch.StartNew(); while (warm.ElapsedMilliseconds < 100) Encode(variant);
                Reset(); gpu.ResetWallTiming(); var watch = Stopwatch.StartNew(); Encode(variant); double elapsed = watch.Elapsed.TotalMilliseconds;
                Console.WriteLine(JsonSerializer.Serialize(new { type = "delta_benchmark", variant, m, run, wall_ms = elapsed, timing = gpu.ReadWallTiming() }));
            }
        }
    }
    static double Check(float actual, double expected)
    {
        double error = Math.Abs(actual - expected);
        if (!float.IsFinite(actual) || error > 2e-5 + 2e-4 * Math.Abs(expected)) throw new Exception($"Delta recurrence mismatch: {actual} vs {expected}");
        return error;
    }
}
