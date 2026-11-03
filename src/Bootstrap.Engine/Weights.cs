using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text.Json;

namespace Bootstrap.Engine;

internal sealed record Tensor(GpuBuffer Storage, ulong Offset, string Dtype, int[] Shape)
{
    public Binding Binding => new(Storage, Offset);
    public uint Kind => Dtype switch { "F8_E4M3" => 0, "BF16" => 1, "F32" => 2, _ => throw new NotSupportedException("Unsupported tensor dtype " + Dtype) };
}

internal sealed class Checkpoint : IDisposable
{
    readonly Dictionary<string, Tensor> tensors = new(StringComparer.Ordinal);
    readonly Dictionary<string, GpuBuffer> files = new(StringComparer.Ordinal);
    public Tensor this[string key] => tensors.TryGetValue(key, out var value) ? value : throw new InvalidDataException("Missing checkpoint tensor: " + key);
    public bool Contains(string key) => tensors.ContainsKey(key);
    public GpuBuffer Shard(string name) => files[name];
    public Checkpoint(Metal device, string directory, string manifestPath, bool packedExperts = false)
    {
        using var manifest = JsonDocument.Parse(File.ReadAllBytes(manifestPath));
        foreach (var entry in manifest.RootElement.GetProperty("files").EnumerateArray())
        {
            string name = entry.GetProperty("name").GetString()!;
            if (name == "mtp.safetensors") continue; // DFlash2 uses a separate pinned draft, not the target MTP head.
            string path = Path.Combine(directory, name);
            using (var stream = File.OpenRead(path))
            {
                string actual = Convert.ToHexStringLower(SHA256.HashData(stream));
                if (actual != entry.GetProperty("sha256").GetString()) throw new InvalidDataException("Weight hash mismatch: " + name);
            }
            Load(device, path, name, packedExperts);
            Console.Error.WriteLine("Verified and mapped " + name);
        }
    }
    void Load(Metal device, string path, string name, bool packedExperts)
    {
        using var file = File.OpenRead(path);
        Span<byte> size = stackalloc byte[8]; file.ReadExactly(size);
        ulong headerSize = BinaryPrimitives.ReadUInt64LittleEndian(size);
        if (headerSize > 32 * 1024 * 1024) throw new InvalidDataException("Invalid safetensors header size");
        byte[] header = new byte[checked((int)headerSize)]; file.ReadExactly(header);
        var storage = device.MapFile(path); files.Add(name, storage);
        using var json = JsonDocument.Parse(header);
        foreach (var property in json.RootElement.EnumerateObject())
        {
            if (property.Name == "__metadata__") continue;
            var value = property.Value;
            var offsets = value.GetProperty("data_offsets").EnumerateArray().Select(x => x.GetUInt64()).ToArray();
            if (offsets.Length != 2 || offsets[1] < offsets[0] || 8 + headerSize + offsets[1] > (ulong)file.Length) throw new InvalidDataException("Invalid tensor offsets");
            var shape = value.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray();
            tensors.Add(property.Name, new(storage, 8 + headerSize + offsets[0], value.GetProperty("dtype").GetString()!, shape));
            if (packedExperts && property.Name.Contains(".mlp.experts.", StringComparison.Ordinal) && property.Name.EndsWith(".weight", StringComparison.Ordinal))
            {
                if (value.GetProperty("dtype").GetString() != "F8_E4M3" || shape.Length != 2 || shape[0] % 32 != 0 || shape[1] % 32 != 0
                    || offsets[1] - offsets[0] != checked((ulong)shape[0] * (ulong)shape[1]))
                    throw new InvalidDataException("Unsupported packed expert tensor: " + property.Name);
                RepackExpert(storage.Values<byte>(checked(shape[0] * shape[1]), 8 + headerSize + offsets[0]), shape[0], shape[1]);
            }
        }
    }
    // The native mapping is MAP_PRIVATE. Reorder each verified matrix in that
    // same mapping, retaining only one matrix of temporary storage. Original
    // checkpoint files and BF16 scales are unchanged; no second model is kept.
    internal static void RepackExpert(Span<byte> bytes, int n, int k)
    {
        if (n <= 0 || k <= 0 || n % 32 != 0 || k % 32 != 0 || bytes.Length != checked(n * k))
            throw new ArgumentException("Packed experts require whole 32x32 tiles");
        byte[] scratch = System.Buffers.ArrayPool<byte>.Shared.Rent(bytes.Length);
        try
        {
            bytes.CopyTo(scratch);
            var source = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, uint>(scratch.AsSpan(0, bytes.Length));
            var output = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, uint>(bytes);
            int to = 0;
            for (int c = 0; c < n; c += 32)
                for (int j = 0; j < k; j += 32)
                    for (int q = 0; q < 8; q++)
                        for (int lane = 0; lane < 32; lane++)
                            output[to++] = source[(c + lane) * (k / 4) + j / 4 + q];
        }
        finally { System.Buffers.ArrayPool<byte>.Shared.Return(scratch); }
    }
    public static Dictionary<string, (ulong Offset, string Dtype, int[] Shape)> Header(string path)
    {
        using var stream = File.OpenRead(path); Span<byte> size = stackalloc byte[8]; stream.ReadExactly(size);
        byte[] bytes = new byte[checked((int)BinaryPrimitives.ReadUInt64LittleEndian(size))]; stream.ReadExactly(bytes);
        using var json = JsonDocument.Parse(bytes); var result = new Dictionary<string, (ulong, string, int[])>();
        foreach (var item in json.RootElement.EnumerateObject())
            if (item.Name != "__metadata__") result[item.Name] = (8UL + (ulong)bytes.Length + item.Value.GetProperty("data_offsets")[0].GetUInt64(), item.Value.GetProperty("dtype").GetString()!, item.Value.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray());
        return result;
    }
    public void Dispose() { foreach (var buffer in files.Values) buffer.Dispose(); }
}
