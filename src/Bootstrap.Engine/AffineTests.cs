using System.Runtime.InteropServices;
using System.Text.Json;

namespace Bootstrap.Engine;

internal static class AffineTests
{
    static int checks;
    static double maximumRelativeError;
    static void Check(float got, double expected, double tolerance = 0.002)
    {
        double relative = Math.Abs(got - expected) / Math.Max(1, Math.Abs(expected));
        if (!float.IsFinite(got) || relative > tolerance) throw new Exception($"Affine reference mismatch: {got} vs {expected}, error={relative}");
        maximumRelativeError = Math.Max(maximumRelativeError, relative); checks++;
    }
    static float Input(int i) => ((i * 19 + i / 113) % 67 - 33) / 65f;
    static float Silu(float v) => v / (1 + MathF.Exp(-v));
    public static void Run(string source)
    {
        using var gpu = new Metal(source);
        Conversion(gpu);
        foreach (int k in new[] { 128, 512, 2048 })
        {
            const int n = 64, experts = 3, choices = 3, rows = 32;
            int metaCount = n * k / 32, slot = n * k + metaCount * 2;
            using var raw = gpu.Allocate((ulong)(experts * choices * slot));
            using var desc = gpu.Allocate(experts * 48);
            var logical = new float[experts * choices][];
            for (int e = 0; e < experts; e++) for (int h = 0; h < choices; h++)
            {
                int index = e * choices + h; ulong off = (ulong)(index * slot);
                desc.Values<ulong>(experts * 6)[e * 6 + h * 2] = off;
                desc.Values<ulong>(experts * 6)[e * 6 + h * 2 + 1] = off + (ulong)(n * k);
                logical[index] = new float[n * k];
                for (int r = 0; r < n; r++) for (int g = 0; g < k / 64; g++)
                {
                    ushort s = AffineWeights.ToBf((1 + (r + e + h + g) % 7) / 65536f);
                    ushort b = AffineWeights.ToBf(-0.004f * (1 + (r + g + e) % 3));
                    int at = AffineWeights.GroupOffset(r, g, k, true) * 2;
                    var meta = raw.Values<ushort>(metaCount, off + (ulong)(n * k)); meta[at] = s; meta[at + 1] = b;
                    for (int j = 0; j < 64; j++)
                    {
                        int col = g * 64 + j; byte q = (byte)(r * 19 + col * 37 + e * 13 + h * 67);
                        raw.Values<byte>(n * k, off)[AffineWeights.CodeOffset(r, col, k)] = q;
                        logical[index][r * k + col] = MathF.FusedMultiplyAdd(q, AffineWeights.Bf(s), AffineWeights.Bf(b));
                    }
                }
            }
            using var x = gpu.Allocate(experts * rows * (ulong)k * 4);
            using var hx = gpu.Allocate(experts * rows * (ulong)k * 2);
            using var y = gpu.Allocate(experts * rows * n * 4 + 16);
            using var tiles = gpu.Allocate(experts * 16);
            using var count = gpu.Allocate(4);
            using var ids = gpu.Allocate(experts * 4);
            using var tokens = gpu.Allocate(experts * rows * 4);
            for (int i = 0; i < experts * rows * k; i++) { x.Values<float>(experts * rows * k)[i] = Input(i); hx.Values<Half>(experts * rows * k)[i] = (Half)Input(i); }
            for (int i = 0; i < experts; i++) ids.Values<int>(experts)[i] = i;
            for (int i = 0; i < experts * rows; i++) tokens.Values<uint>(experts * rows)[i] = (uint)i;
            double Dot(int expert, int choice, int row, int col, bool half)
            {
                double sum = 0;
                for (int j = 0; j < k; j++) { float a = Input(row * k + j), b = logical[expert * choices + choice][col * k + j]; sum += half ? (double)(float)(Half)a * (float)(Half)b : (double)a * b; }
                return sum;
            }
            foreach (bool fused in new[] { true, false })
            {
                var p = new KernelArgs { N = n, K = (uint)k, M = experts, Kind = fused ? 0u : 2u, Flags = 3072 };
                gpu.Begin(); gpu.Dispatch(fused ? "expert_gate_up" : "expert_down", p, experts * n * 32, 128, x, raw, desc, ids, y); gpu.End();
                for (int e = 0; e < experts; e++) for (int c = 0; c < n; c++)
                {
                    double expected = fused ? Silu((float)Dot(e, 0, 0, c, false)) * Dot(e, 1, 0, c, false) : Dot(e, 2, e, c, false);
                    Check(y.Values<float>(experts * n)[e * n + c], expected, 0.00002);
                }
                foreach (int live in new[] { 1, 2, 7, 8, 9, 16, 17, 31, 32 })
                {
                    for (int e = 0; e < experts; e++) { var tile = tiles.Values<uint>(experts * 4).Slice(e * 4); tile[0] = (uint)e; tile[1] = (uint)(e * live); tile[2] = (uint)live; }
                    count.Values<uint>(1)[0] = experts;
                    int bytes = experts * live * n * (fused ? 2 : 4);
                    y.Values<byte>(bytes + 16).Fill(0xa5);
                    gpu.Begin(); gpu.Dispatch2D(fused ? "expert_gate_up_affine" : "expert_matrix_affine", p, (uint)(n / (fused ? 32 : 64) * 64), experts + 1, 64, hx, raw, desc, tiles, count, y); gpu.End();
                    if (!y.Values<byte>(16, (ulong)bytes).ToArray().All(b => b == 0xa5)) throw new Exception("Affine output overrun"); checks++;
                    for (int e = 0; e < experts; e++) foreach (int r in new[] { 0, live - 1 }) foreach (int c in new[] { 0, 7, 31, 32, 63 })
                    {
                        int row = e * live + r, at = row * n + c;
                        double expected = fused ? (float)(Half)(Silu((float)Dot(e, 0, row, c, true)) * (float)Dot(e, 1, row, c, true)) : Dot(e, 2, row, c, true);
                        Check(fused ? (float)y.Values<Half>(experts * live * n)[at] : y.Values<float>(experts * live * n)[at], expected, 0.0001);
                    }
                    if (live <= 8) for (uint choice = 0; choice < (fused ? 2 : 1); choice++)
                    {
                        p.Kind = fused ? choice : 2;
                        gpu.Begin(); gpu.Dispatch2D("expert_matrix_tiny", p, n * 32, experts + 1, 128, x, raw, desc, tiles, count, tokens, y); gpu.End();
                        for (int e = 0; e < experts; e++) for (int r = 0; r < live; r++) foreach (int c in new[] { 0, 31, 32, 63 }) Check(y.Values<float>(experts * live * n)[(e * live + r) * n + c], Dot(e, (int)p.Kind, e * live + r, c, false), 0.00002);
                    }
                    p.Kind = fused ? 0u : 2u;
                }
            }
            // Row-major affine is also used by dense and shared-expert projections.
            using var dense = gpu.Allocate((ulong)(n * k));
            using var metadata = gpu.Allocate((ulong)(metaCount * 2));
            for (int r = 0; r < n; r++) for (int j = 0; j < k; j++) dense.Values<byte>(n * k)[r * k + j] = raw.Values<byte>(n * k)[AffineWeights.CodeOffset(r, j, k)];
            for (int r = 0; r < n; r++) for (int g = 0; g < k / 64; g++) raw.Values<ushort>(metaCount, (ulong)(n * k)).Slice(AffineWeights.GroupOffset(r, g, k, true) * 2, 2).CopyTo(metadata.Values<ushort>(metaCount).Slice((r * (k / 64) + g) * 2, 2));
            var dp = new KernelArgs { N = n, K = (uint)k, Kind = 3, M = 7 };
            gpu.Begin(); gpu.Dispatch("gemv", dp, n * 32, 128, x, dense, metadata, y); gpu.End();
            for (int c = 0; c < n; c++) Check(y.Values<float>(n)[c], Dot(0, 0, 0, c, false), 0.00002);
            gpu.Begin(); gpu.Dispatch("matrix_product_tiny", dp, n * 32, 128, x, dense, metadata, y); gpu.End();
            for (int r = 0; r < 7; r++) for (int c = 0; c < n; c++) Check(y.Values<float>(7 * n)[r * n + c], Dot(0, 0, r, c, false), 0.00002);
            using var unpacked = gpu.Allocate((ulong)(n * k * 2));
            gpu.Begin(); gpu.Dispatch("projection_unpack_half", dp, (uint)(n * k / 4), 128, dense, metadata, unpacked); gpu.End();
            for (int i = 0; i < n * k; i++) Check((float)unpacked.Values<Half>(n * k)[i], (float)(Half)logical[0][i], 0);
            dp.M = 31;
            gpu.Begin(); gpu.Dispatch2D("matrix_product_predecoded", dp, 128, 1, 128, hx, unpacked, y); gpu.End();
            for (int r = 0; r < 31; r++) foreach (int c in new[] { 0, 31, 32, 63 }) Check(y.Values<float>(31 * n)[r * n + c], Dot(0, 0, r, c, true), 0.0001);
            dp.N = 32;
            gpu.Begin(); gpu.Dispatch2D("matrix_product_small_fp16", dp, 128, 2, 128, x, dense, metadata, y); gpu.End();
            for (int r = 0; r < 31; r++) foreach (int c in new[] { 0, 7, 31 }) Check(y.Values<float>(31 * 32)[r * 32 + c], Dot(0, 0, r, c, true), 0.0001);
            if (k == 2048)
            {
                gpu.Begin(); gpu.Dispatch("embedding_affine", new() { N = 2048, Position = 63 }, 2048, 128, dense, y, metadata); gpu.End();
                for (int j = 0; j < 2048; j++) Check(y.Values<float>(2048)[j], logical[0][63 * 2048 + j], 0);
                int[] selected = { 0, 31, 63 }; selected.CopyTo(ids.Values<int>(3));
                gpu.Begin(); gpu.Dispatch("batch_embedding_affine", new() { M = 3 }, 3 * 2048, 128, dense, ids, y, metadata); gpu.End();
                for (int r = 0; r < 3; r++) for (int j = 0; j < 2048; j++) Check(y.Values<float>(3 * 2048)[r * 2048 + j], logical[0][selected[r] * 2048 + j], 0);
            }
        }
        Console.WriteLine(JsonSerializer.Serialize(new { type = "affine_checks", checks, maximum_relative_error = maximumRelativeError, passed = true }));
    }
    static void Conversion(Metal gpu)
    {
        string root = Path.Combine(Path.GetTempPath(), "bootstrap-affine-test-" + Guid.NewGuid().ToString("N")); Directory.CreateDirectory(root);
        try
        {
            string key = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight";
            byte[] weights = Enumerable.Range(0, 32 * 128).Select(i => (byte)(i % 126)).ToArray();
            ushort[] scales = { 0x3b91 };
            var quant = AffineWeights.Quantize(weights, scales, 32, 128, true);
            if (Math.Sqrt(quant.Error / quant.Norm) > 0.01) throw new Exception("Quantization error exceeds expected bound"); checks++;
            foreach (byte constant in new byte[] { 0, 1, 127, 128, 185, 254 })
            {
                try
                {
                    var q = AffineWeights.Quantize(Enumerable.Repeat(constant, 32 * 128).ToArray(), scales, 32, 128, true);
                    if (constant == 127) throw new Exception("NaN source accepted");
                    if (!double.IsFinite(q.Error)) throw new Exception("Constant group is not finite"); checks++;
                }
                catch (AggregateException e) when (constant == 127 && e.InnerException is InvalidDataException) { checks++; }
            }
            var header = new Dictionary<string, object> {
                [key] = new { dtype = "F8_E4M3", shape = new[] {32,128}, data_offsets = new[]{0,4096} },
                [key+"_scale_inv"] = new { dtype = "BF16", shape = new[]{1,1}, data_offsets = new[]{4096,4098} },
                ["norm.weight"] = new { dtype = "BF16", shape = new[]{32}, data_offsets = new[]{4098,4162} },
                ["lm_head.weight"] = new { dtype = "BF16", shape = new[]{32,128}, data_offsets = new[]{4162,12354} },
                ["model.visual.test.weight"] = new { dtype = "BF16", shape = new[]{32}, data_offsets = new[]{12354,12418} }
            };
            string shard = Path.Combine(root, "layers-0.safetensors"), manifest = Path.Combine(root, "manifest.json"), cache = Path.Combine(root, "cache");
            byte[] json = JsonSerializer.SerializeToUtf8Bytes(header);
            ushort[] vocabulary = Enumerable.Range(0, 4096).Select(i => AffineWeights.ToBf(MathF.Sin(i * 0.37f) / 32)).ToArray();
            byte[] vocabularyBytes = MemoryMarshal.AsBytes(vocabulary.AsSpan()).ToArray();
            var expectedVocabulary = AffineWeights.Quantize(vocabularyBytes, Array.Empty<ushort>(), 32, 128, false, true);
            using (var file = File.Create(shard)) { file.Write(BitConverter.GetBytes((ulong)json.Length)); file.Write(json); file.Write(weights); file.Write(MemoryMarshal.AsBytes(scales.AsSpan())); file.Write(Enumerable.Repeat((byte)0x3f, 64).ToArray()); file.Write(vocabularyBytes); file.Write(new byte[64]); }
            string hash = AffineWeights.Hash(shard);
            File.WriteAllText(manifest, JsonSerializer.Serialize(new { files = new[] { new { name = "layers-0.safetensors", sha256 = hash } } }));
            AffineWeights.ConvertCheckpoint(root, cache, manifest);
            using (var checkpoint = new Checkpoint(gpu, root, manifest, affineCache: cache))
            {
                var tensor = checkpoint[key]; if (tensor.Kind != 3 || !tensor.Storage.Values<byte>(4096, tensor.Offset).SequenceEqual(quant.Codes)) throw new Exception("Cached affine bytes differ"); checks++;
                var norm = checkpoint["norm.weight"]; if (!norm.Storage.Values<byte>(64, norm.Offset).ToArray().All(x => x == 0x3f)) throw new Exception("Nonquantized weights changed"); checks++;
                var vocab = checkpoint["lm_head.weight"]; if (vocab.Kind != 3 || !vocab.Storage.Values<byte>(4096, vocab.Offset).SequenceEqual(expectedVocabulary.Codes) || Math.Sqrt(expectedVocabulary.Error / expectedVocabulary.Norm) > 0.01) throw new Exception("Vocabulary conversion failed"); checks++;
                if (checkpoint.Contains("model.visual.test.weight")) throw new Exception("Unused vision tensor retained in text cache"); checks++;
            }
            if (AffineWeights.Hash(shard) != hash) throw new Exception("Original checkpoint changed"); checks++;
            using (var file = new FileStream(Path.Combine(cache, "layers-0.safetensors"), FileMode.Open, FileAccess.Write)) { file.Position = file.Length - 1; file.WriteByte(0); }
            try { using var checkpoint = new Checkpoint(gpu, root, manifest, affineCache: cache); throw new Exception("Corrupt cache accepted"); }
            catch (InvalidDataException) { checks++; }
        }
        finally { Directory.Delete(root, true); }
    }
}
