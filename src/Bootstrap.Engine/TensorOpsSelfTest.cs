using System.Text.Json;

namespace Bootstrap.Engine;

internal static class TensorOpsSelfTest
{
    static double Decode(byte b)
    {
        int e = (b >> 3) & 15, m = b & 7;
        return (b < 128 ? 1 : -1) * (e == 0 ? m / 512.0 : (1 + m / 8.0) * Math.Pow(2, e - 7));
    }
    static float Input(int i) => MathF.Sin(i * .731f) * .7f + (i % 7) * .000013f;
    public static void Run(string source, string precision = "fp32", bool optimized = false, int tileRows = 16, bool predecoded = false, bool expertPredecoded = false)
    {
        if (precision is not ("fp32" or "fp16" or "bf16")) throw new ArgumentException("Invalid test precision");
        if (tileRows is not (16 or 32 or 64) || ((optimized || predecoded || expertPredecoded) && precision != "fp16")) throw new ArgumentException("Invalid optimized test configuration");
        float Round(float value) => precision == "fp16" ? (float)(Half)value : precision == "bf16"
            ? BitConverter.UInt32BitsToSingle((BitConverter.SingleToUInt32Bits(value) + 0x7fffu + ((BitConverter.SingleToUInt32Bits(value) >> 16) & 1)) & 0xffff0000u) : value;
        string suffix = precision == "fp32" ? "_mpp" : "_" + precision;
        using var gpu = new Metal(source);
        int checks = 0;
        double worst = 0;
        void Check(float actual, double expected, double magnitude, string label)
        {
            double error = Math.Abs(actual - expected);
            if (!float.IsFinite(actual) || error > 1e-6 + 3e-6 * magnitude)
                throw new Exception($"{label}: {actual} != {expected}; error={error}");
            worst = Math.Max(worst, error); checks++;
        }
        foreach (bool small in new[] { false, true }) foreach (uint kind in new[] { 0u, 1u, 2u })
        {
            int m = small ? 19 : 37, n = small ? 17 : 137, k = 256;
            using var x = gpu.Allocate((ulong)m * (ulong)k * 4);
            using var w = gpu.Allocate((ulong)n * (ulong)k * 4);
            using var s = gpu.Allocate((ulong)((n + 127) / 128 * (k / 128)) * 2);
            using var y = gpu.Allocate((ulong)(m * n + 1) * 4);
            using var legacy = gpu.Allocate((ulong)(m * n + 1) * 4);
            for (int i = 0; i < m * k; i++) x.Values<float>(m * k)[i] = Input(i);
            for (int i = 0; i < (int)s.Length / 2; i++) s.Values<ushort>((int)s.Length / 2)[i] = (ushort)(0x3c00 + i * 128);
            for (int i = 0; i < n * k; i++)
            {
                if (kind == 0) w.Values<byte>(n * k)[i] = (byte)((i * 13 % 112) | ((i % 3 == 0) ? 128 : 0));
                else if (kind == 1) w.Values<ushort>(n * k)[i] = (ushort)(BitConverter.SingleToUInt32Bits(Input(i + 93)) >> 16);
                else w.Values<float>(n * k)[i] = Input(i + 93);
            }
            string name = small ? "matrix_product_small" : "matrix_product";
            y.Values<float>(m * n + 1).Fill(float.NaN);
            var p = new KernelArgs { M = (uint)m, N = (uint)n, K = (uint)k, Kind = kind };
            uint tx = small || optimized ? 128u : 256u, tm = small ? 16u : 32u, tn = small ? 32u : 64u;
            gpu.Begin();
            gpu.Dispatch2D(name, p, ((uint)n + tn - 1) / tn * (small ? 128u : 256u), ((uint)m + tm - 1) / tm, small ? 128u : 256u, x, w, s, legacy);
            if (!small && predecoded) {
                using var xhalf=gpu.Allocate((ulong)m*(ulong)k*2);using var whalf=gpu.Allocate((ulong)n*(ulong)k*2);
                gpu.Dispatch("projection_unpack_half",p,(uint)(n*k/4),128,w,s,whalf);
                gpu.Dispatch("projection_input_half",p,(uint)(m*k/4),128,x,xhalf);
                gpu.Dispatch2D("matrix_product_predecoded",p,(uint)((n+63)/64)*128,(uint)((m+63)/64),128,xhalf,whalf,y);
                gpu.End(); gpu.Begin();
            } else gpu.Dispatch2D(name + suffix + (!small && optimized ? "_tile" : ""), p, ((uint)n + tn - 1) / tn * tx, ((uint)m + tm - 1) / tm, tx, x, w, s, y);
            gpu.End();
            for (int row = 0; row < m; row++) for (int col = 0; col < n; col++)
            {
                double expected = 0, magnitude = 0, originalExpected = 0, originalMagnitude = 0;
                for (int j = 0; j < k; j++)
                {
                    int at = col * k + j;
                    double weight = kind == 0 ? Decode(w.Values<byte>(n * k)[at]) * BitConverter.UInt32BitsToSingle((uint)s.Values<ushort>((int)s.Length / 2)[col / 128 * (k / 128) + j / 128] << 16)
                        : kind == 1 ? BitConverter.UInt32BitsToSingle((uint)w.Values<ushort>(n * k)[at] << 16) : w.Values<float>(n * k)[at];
                    double originalTerm = x.Values<float>(m * k)[row * k + j] * weight;
                    originalExpected += originalTerm; originalMagnitude += Math.Abs(originalTerm);
                    double term = (double)Round(x.Values<float>(m * k)[row * k + j]) * Round((float)weight);
                    expected += term; magnitude += Math.Abs(term);
                }
                Check(y.Values<float>(m * n)[row * n + col], expected, magnitude, name + " CPU");
                Check(legacy.Values<float>(m * n)[row * n + col], originalExpected, originalMagnitude, name + " legacy CPU");
            }
            if (!float.IsNaN(y.Values<float>(m * n + 1)[m * n])) throw new Exception("Output guard overwritten");
        }
        // Cancellation fixture: losing FP32 mantissa bits to FP16/BF16 would yield zero.
        if (precision == "fp32") foreach (bool small in new[] { false, true })
        {
            const int k = 128;
            using var x = gpu.Allocate(k * 4); using var w = gpu.Allocate(k * 4); using var y = gpu.Allocate(4);
            for (int j = 0; j < k; j++) { x.Values<float>(k)[j] = j % 2 == 0 ? 1 + MathF.Pow(2, -20) : 1; w.Values<float>(k)[j] = j % 2 == 0 ? 1 : -1; }
            gpu.Begin(); gpu.Dispatch2D(small ? "matrix_product_small_mpp" : "matrix_product_mpp", new() { M = 1, N = 1, K = k, Kind = 2 }, small ? 128u : 256u, 1, small ? 128u : 256u, x, w, w, y); gpu.End();
            float expected = 64 * MathF.Pow(2, -20);
            if (y.Values<float>(1)[0] != expected) throw new Exception("FP32 precision cancellation fixture failed");
            checks++;
        }
        // Distinct expert/choice offsets, partial tiles, nontrivial token permutation,
        // grouped down input, tail output columns and an excess dispatched tile.
        {
            int m = tileRows + 8; const int n = 69, k = 256, scaleCount = 2, slotBytes = n * k + scaleCount * 2;
            using var x = gpu.Allocate((ulong)m * k * 4); using var raw = gpu.Allocate(6 * slotBytes);
            using var desc = gpu.Allocate(2 * 6 * 8); using var tiles = gpu.Allocate(3 * 4 * 4);
            using var count = gpu.Allocate(4); using var tokens = gpu.Allocate((ulong)m * 4);
            using var y = gpu.Allocate((ulong)(m * n + 1) * 4); using var legacy = gpu.Allocate((ulong)m * n * 4);
            for (int i = 0; i < m * k; i++) x.Values<float>(m * k)[i] = Input(i);
            for (int i = 0; i < m; i++) tokens.Values<uint>(m)[i] = (uint)((i * 7 + 5) % m);
            for (int slot = 0; slot < 6; slot++)
            {
                desc.Values<ulong>(12)[slot * 2] = (ulong)(slot * slotBytes);
                desc.Values<ulong>(12)[slot * 2 + 1] = (ulong)(slot * slotBytes + n * k);
                for (int i = 0; i < n * k; i++) raw.Values<byte>(n * k, (ulong)(slot * slotBytes))[i] = (byte)(((i * 13 + slot * 3) % 112) | (i % 3 == 0 ? 128 : 0));
                raw.Values<ushort>(2, (ulong)(slot * slotBytes + n * k))[0] = (ushort)(0x3c00 + slot * 128);
                raw.Values<ushort>(2, (ulong)(slot * slotBytes + n * k))[1] = (ushort)(0x3b80 + slot * 128);
            }
            uint[] tileData = [0, 0, 1, 0, 1, 1, 7, 0, 0, 8, (uint)tileRows, 0]; tileData.CopyTo(tiles.Values<uint>(12)); count.Values<uint>(1)[0] = 3;
            for (uint choice = 0; choice < 3; choice++)
            {
                y.Values<float>(m * n + 1).Fill(float.NaN);
                var p = new KernelArgs { N = n, K = k, Kind = choice };
                gpu.Begin();
                if (tileRows == 16) gpu.Dispatch2D("expert_matrix", p, 512, 4, 256, x, raw, desc, tiles, count, tokens, legacy);
                if (expertPredecoded) {
                    using var xhalf=gpu.Allocate((ulong)m*k*2);using var whalf=gpu.Allocate(2*n*k*2);
                    p.M=(uint)m;p.A=2;
                    gpu.Dispatch("expert_unpack_half",p,2*n*k/4,128,raw,desc,whalf);
                    gpu.Dispatch("expert_input_half",p,(uint)(m*k/4),128,x,tokens,xhalf);
                    gpu.Dispatch2D("expert_matrix_predecoded",p,256,4,128,xhalf,whalf,tiles,count,y);
                    gpu.End();gpu.Begin();
                } else gpu.Dispatch2D("expert_matrix" + suffix + (optimized ? "_tile" + (tileRows == 16 ? "" : tileRows.ToString()) + (choice == 2 ? "_down" : "_up") : ""), p, optimized && choice != 2 && tileRows != 64 ? 256u : 512u, 4, optimized && choice != 2 && tileRows != 64 ? 128u : 256u, x, raw, desc, tiles, count, tokens, y); gpu.End();
                for (int row = 0; row < m; row++) for (int col = 0; col < n; col++)
                {
                    int slot = (row is >= 1 and < 8 ? 3 : 0) + (int)choice;
                    int inputRow = choice == 2 ? row : (int)tokens.Values<uint>(m)[row];
                    double expected = 0, magnitude = 0, originalExpected = 0, originalMagnitude = 0;
                    for (int j = 0; j < k; j++)
                    {
                        double scale = BitConverter.UInt32BitsToSingle((uint)raw.Values<ushort>(2, (ulong)(slot * slotBytes + n * k))[j / 128] << 16);
                        double weight = Decode(raw.Values<byte>(n * k, (ulong)(slot * slotBytes))[col * k + j]) * scale;
                        double originalTerm = x.Values<float>(m * k)[inputRow * k + j] * weight;
                        originalExpected += originalTerm; originalMagnitude += Math.Abs(originalTerm);
                        double term = (double)Round(x.Values<float>(m * k)[inputRow * k + j]) * Round((float)weight);
                        expected += term; magnitude += Math.Abs(term);
                    }
                    Check(y.Values<float>(m * n)[row * n + col], expected, magnitude, "expert CPU");
                    if (tileRows == 16) Check(legacy.Values<float>(m * n)[row * n + col], originalExpected, originalMagnitude, "expert legacy CPU");
                }
                if (!float.IsNaN(y.Values<float>(m * n + 1)[m * n])) throw new Exception("Expert output guard overwritten");
            }
        }
        // A low-precision accumulator would lose the small summands at this scale.
        foreach (bool small in new[] { false, true })
        {
            const int k = 4096;
            using var x = gpu.Allocate(k * 4); using var w = gpu.Allocate(k * 4); using var y = gpu.Allocate(4);
            x.Values<float>(k).Fill(1);
            for (int j = 0; j < k; j++) w.Values<float>(k)[j] = j % 2 == 0 ? 1 : 1f / 1024;
            if (!small && predecoded)
            {
                using var xhalf=gpu.Allocate(k*2);using var whalf=gpu.Allocate(k*2);
                var p=new KernelArgs { M=1,N=1,K=k,Kind=2 };
                gpu.Begin();gpu.Dispatch("projection_unpack_half",p,k/4,128,w,w,whalf);
                gpu.Dispatch("projection_input_half",p,k/4,128,x,xhalf);
                gpu.Dispatch2D("matrix_product_predecoded",p,128,1,128,xhalf,whalf,y);gpu.End();
            }
            else
            {
            gpu.Begin(); gpu.Dispatch2D((small ? "matrix_product_small" : "matrix_product") + suffix + (!small && optimized ? "_tile" : ""), new() { M = 1, N = 1, K = k, Kind = 2 }, small || optimized ? 128u : 256u, 1, small || optimized ? 128u : 256u, x, w, w, y); gpu.End();
            }
            if (y.Values<float>(1)[0] != 2050) throw new Exception("FP32 accumulation fixture failed");
            checks++;
        }
        Console.WriteLine(JsonSerializer.Serialize(new { type = "tensorops_self_test", precision, optimized, tileRows, predecoded, expertPredecoded, checks,
            max_abs_error_vs_rounded_reference = worst, fp32_accumulation = "passed", strict_fp32_cancellation = precision == "fp32" ? "passed" : "not applicable" }));
    }
}
