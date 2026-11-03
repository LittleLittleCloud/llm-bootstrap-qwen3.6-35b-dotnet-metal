using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;

namespace Bootstrap.Engine;

// A derived cache, not an MLX checkpoint. Original FP8/BF16 files are never modified.
// Every FP8 text matrix and the two BF16 vocabulary matrices become unsigned
// affine-8 with 64 values per row group. Unused vision tensors are omitted.
// Scale/bias are BF16 pairs; experts use the existing packed 32x32 byte layout
// and metadata ordered [row/32, column/64, row%32, scale/bias].
internal static class AffineWeights
{
    public const string Format = "bootstrap-affine8-g64-v2";
    internal sealed record Entry(string Name, string SourceSha256, string Sha256, long Bytes, long Values, double SquaredError, double SquaredNorm, double MaxError);
    internal sealed record Manifest(string Format, string SourceManifestSha256, Entry[] Files);
    internal static string Hash(string path) { using var s = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(s)); }
    internal static float Bf(ushort x) => BitConverter.UInt32BitsToSingle((uint)x << 16);
    internal static ushort ToBf(float x)
    {
        uint bits = BitConverter.SingleToUInt32Bits(x);
        return (ushort)((bits + 0x7fffu + ((bits >> 16) & 1)) >> 16);
    }
    internal static float F8(byte b)
    {
        int e = (b >> 3) & 15, m = b & 7;
        if (e == 15 && m == 7) return float.NaN;
        return (b < 128 ? 1 : -1) * (e == 0 ? m / 512f : MathF.ScaleB(1 + m / 8f, e - 7));
    }
    internal static int CodeOffset(int row, int col, int k) => (row / 32) * k * 32 + (col / 32) * 1024 + (col % 32 / 4) * 128 + (row % 32) * 4 + col % 4;
    internal static int GroupOffset(int row, int group, int k, bool packed) => packed ? (row / 32) * (k / 64) * 32 + group * 32 + row % 32 : row * (k / 64) + group;

    internal static (byte[] Codes, ushort[] Metadata, double Error, double Norm, double Max) Quantize(byte[] fp8, ushort[] scales, int n, int k, bool packed, bool bf16 = false)
    {
        if (n <= 0 || k <= 0 || k % 64 != 0 || (packed && n % 32 != 0) || fp8.Length != checked(n * k * (bf16 ? 2 : 1)) || (!bf16 && scales.Length != ((n + 127) / 128) * ((k + 127) / 128)))
            throw new InvalidDataException("Unsupported affine matrix geometry");
        var codes = new byte[n * k]; var metadata = new ushort[n * k / 32];
        var errors = new double[n]; var norms = new double[n]; var maxima = new double[n];
        float[] lookup = Enumerable.Range(0, 256).Select(i => F8((byte)i)).ToArray();
        Parallel.For(0, n, new ParallelOptions { MaxDegreeOfParallelism = 8 }, row =>
        {
            Span<float> values = stackalloc float[64];
            double error = 0, norm = 0, maxError = 0;
            for (int group = 0; group < k / 64; group++)
            {
                float sourceScale = bf16 ? 1 : Bf(scales[(row / 128) * ((k + 127) / 128) + group / 2]);
                float lo = float.PositiveInfinity, hi = float.NegativeInfinity;
                for (int j = 0; j < 64; j++)
                {
                    int at = row * k + group * 64 + j;
                    float v = bf16 ? Bf(BinaryPrimitives.ReadUInt16LittleEndian(fp8.AsSpan(at * 2, 2))) : lookup[fp8[at]] * sourceScale;
                    if (!float.IsFinite(v)) throw new InvalidDataException("Non-finite source weight");
                    values[j] = v; lo = Math.Min(lo, v); hi = Math.Max(hi, v);
                }
                ushort biasBits = ToBf(lo), scaleBits = ToBf((hi - lo) / 255f);
                float bias = Bf(biasBits), scale = Bf(scaleBits);
                if (!float.IsFinite(bias) || !float.IsFinite(scale)) throw new InvalidDataException("Affine metadata overflow");
                int meta = GroupOffset(row, group, k, packed) * 2;
                metadata[meta] = scaleBits; metadata[meta + 1] = biasBits;
                for (int j = 0; j < 64; j++)
                {
                    float v = values[j];
                    // Quantize using the stored (rounded) metadata, not its FP32 precursor.
                    byte q = scale == 0 ? (byte)0 : (byte)Math.Clamp(MathF.Round((v - bias) / scale, MidpointRounding.ToEven), 0, 255);
                    int col = group * 64 + j;
                    codes[packed ? CodeOffset(row, col, k) : row * k + col] = q;
                    double d = MathF.FusedMultiplyAdd(q, scale, bias) - (double)v;
                    error += d * d; norm += (double)v * v; maxError = Math.Max(maxError, Math.Abs(d));
                }
            }
            errors[row] = error; norms[row] = norm; maxima[row] = maxError;
        });
        return (codes, metadata, errors.Sum(), norms.Sum(), maxima.Max());
    }

    public static Manifest ReadManifest(string cache, string sourceManifest)
    {
        var result = JsonSerializer.Deserialize<Manifest>(File.ReadAllBytes(Path.Combine(cache, "affine-manifest.json")))!;
        if (result.Format != Format || result.SourceManifestSha256 != Hash(sourceManifest)) throw new InvalidDataException("Affine cache provenance mismatch; regenerate it");
        using var source = JsonDocument.Parse(File.ReadAllBytes(sourceManifest));
        var expected = source.RootElement.GetProperty("files").EnumerateArray().Where(e => e.GetProperty("name").GetString() != "mtp.safetensors").ToArray();
        if (result.Files.Length != expected.Length || result.Files.Select(e => e.Name).Distinct().Count() != expected.Length) throw new InvalidDataException("Affine cache shard set mismatch");
        foreach (var item in expected)
        {
            string name = item.GetProperty("name").GetString()!;
            if (Path.GetFileName(name) != name || result.Files.SingleOrDefault(e => e.Name == name)?.SourceSha256 != item.GetProperty("sha256").GetString()) throw new InvalidDataException("Affine source hash mismatch: " + name);
        }
        return result;
    }

    public static void ConvertCheckpoint(string directory, string output, string manifestPath)
    {
        if (Path.GetFullPath(directory) == Path.GetFullPath(output)) throw new ArgumentException("Affine output must be separate from the source");
        Directory.CreateDirectory(output);
        // Do not overwrite an existing cache, even if conversion is interrupted.
        using var conversionLock = new FileStream(Path.Combine(output, ".convert.lock"), FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);
        if (File.Exists(Path.Combine(output, "affine-manifest.json"))) throw new IOException("Affine cache already exists; use it or choose a new output directory");
        using var source = JsonDocument.Parse(File.ReadAllBytes(manifestPath));
        var entries = new List<Entry>();
        foreach (var item in source.RootElement.GetProperty("files").EnumerateArray())
        {
            string name = item.GetProperty("name").GetString()!;
            if (name == "mtp.safetensors") continue;
            string path = Path.Combine(directory, name), hash = item.GetProperty("sha256").GetString()!;
            if (Hash(path) != hash) throw new InvalidDataException("Source hash mismatch: " + name);
            string destination = Path.Combine(output, name), receipt = destination + ".json";
            if (File.Exists(destination))
            {
                var previous = JsonSerializer.Deserialize<Entry>(File.ReadAllBytes(receipt))!;
                if (previous.Name != name || previous.SourceSha256 != hash || Hash(destination) != previous.Sha256) throw new InvalidDataException("Partial cache mismatch: " + name);
                entries.Add(previous); continue;
            }
            var entry = ConvertShard(path, destination, name, hash);
            File.WriteAllText(receipt, JsonSerializer.Serialize(entry)); entries.Add(entry);
            Console.WriteLine(JsonSerializer.Serialize(new { type = "affine_shard", entry, relative_l2 = entry.SquaredNorm == 0 ? 0 : Math.Sqrt(entry.SquaredError / entry.SquaredNorm) }));
        }
        var manifest = new Manifest(Format, Hash(manifestPath), entries.ToArray());
        string temp = Path.Combine(output, "affine-manifest.json.tmp");
        File.WriteAllText(temp, JsonSerializer.Serialize(manifest, new JsonSerializerOptions { WriteIndented = true }));
        File.Move(temp, Path.Combine(output, "affine-manifest.json"));
    }

    static Entry ConvertShard(string source, string destination, string name, string sourceHash)
    {
        var tensors = Checkpoint.Header(source);
        var header = new Dictionary<string, object> { ["__metadata__"] = new Dictionary<string, string> { ["format"] = Format } };
        var data = new Dictionary<string, (string Dtype, int[] Shape, long Offset, long Length)>();
        long cursor = 0;
        foreach (var (key, tensor) in tensors)
        {
            if (key.StartsWith("model.visual.", StringComparison.Ordinal)) continue;
            if (key.EndsWith(".weight_scale_inv", StringComparison.Ordinal)) continue;
            long size = tensor.Shape.Aggregate(1L, (a, b) => checked(a * b));
            bool quantized = tensor.Dtype == "F8_E4M3" || IsVocabulary(key);
            if (quantized && (tensor.Shape.Length != 2 || tensor.Shape[1] % 64 != 0)) throw new InvalidDataException("Unsupported affine weight: " + key);
            long bytes = size * (quantized ? 1 : tensor.Dtype == "BF16" ? 2 : tensor.Dtype == "F32" ? 4 : throw new InvalidDataException("Unsupported source dtype"));
            Add(key, quantized ? "U8" : tensor.Dtype, tensor.Shape, bytes);
            if (quantized) Add(key + "_scale_inv", "BF16", new[] { tensor.Shape[0], tensor.Shape[1] / 64, 2 }, size / 16);
        }
        void Add(string key, string dtype, int[] shape, long length)
        {
            cursor = (cursor + 15) & ~15L;
            data.Add(key, (dtype, shape, cursor, length));
            header.Add(key, new { dtype, shape, data_offsets = new[] { cursor, cursor + length } }); cursor += length;
        }
        byte[] json = JsonSerializer.SerializeToUtf8Bytes(header);
        int headerSize = (json.Length + 15) / 16 * 16 + 8; // data starts at a 16-byte boundary
        string temp = destination + ".tmp";
        double error = 0, norm = 0, max = 0; long values = 0;
        using (var input = File.OpenRead(source))
        using (var output = new FileStream(temp, FileMode.Create, FileAccess.ReadWrite, FileShare.None))
        {
            Span<byte> size = stackalloc byte[8]; BinaryPrimitives.WriteUInt64LittleEndian(size, (ulong)headerSize); output.Write(size);
            output.Write(json); output.Write(Enumerable.Repeat((byte)' ', headerSize - json.Length).ToArray());
            output.SetLength(8L + headerSize + cursor);
            foreach (var (key, tensor) in tensors)
            {
                if (key.StartsWith("model.visual.", StringComparison.Ordinal)) continue;
                if (key.EndsWith(".weight_scale_inv", StringComparison.Ordinal)) continue;
                var target = data[key];
                if (tensor.Dtype == "F8_E4M3" || IsVocabulary(key))
                {
                    bool bf16 = tensor.Dtype == "BF16";
                    if (IsVocabulary(key) && !bf16) throw new InvalidDataException("Expected BF16 vocabulary matrix");
                    byte[] bytes = new byte[checked((int)target.Length * (bf16 ? 2 : 1))]; input.Position = (long)tensor.Offset; input.ReadExactly(bytes);
                    byte[] scaleBytes = Array.Empty<byte>();
                    if (!bf16) {
                        var scale = tensors[key + "_scale_inv"]; scaleBytes = new byte[scale.Shape.Aggregate(2, (a, b) => checked(a * b))];
                        input.Position = (long)scale.Offset; input.ReadExactly(scaleBytes);
                    }
                    var quant = Quantize(bytes, MemoryMarshal.Cast<byte, ushort>(scaleBytes).ToArray(), tensor.Shape[0], tensor.Shape[1], key.Contains(".mlp.experts.", StringComparison.Ordinal), bf16);
                    output.Position = 8L + headerSize + target.Offset; output.Write(quant.Codes);
                    output.Position = 8L + headerSize + data[key + "_scale_inv"].Offset; output.Write(MemoryMarshal.AsBytes(quant.Metadata.AsSpan()));
                    values += quant.Codes.Length; error += quant.Error; norm += quant.Norm; max = Math.Max(max, quant.Max);
                }
                else
                {
                    input.Position = (long)tensor.Offset; output.Position = 8L + headerSize + target.Offset;
                    byte[] buffer = new byte[1024 * 1024]; long left = target.Length;
                    while (left > 0) { int count = (int)Math.Min(buffer.Length, left); input.ReadExactly(buffer.AsSpan(0, count)); output.Write(buffer.AsSpan(0, count)); left -= count; }
                }
            }
            output.Flush(true);
        }
        File.Move(temp, destination);
        return new(name, sourceHash, Hash(destination), new FileInfo(destination).Length, values, error, norm, max);
    }
    static bool IsVocabulary(string key) => key is "model.language_model.embed_tokens.weight" or "lm_head.weight";
}
