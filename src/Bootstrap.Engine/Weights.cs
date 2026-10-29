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
    public Checkpoint(Metal device, string directory, string manifestPath)
    {
        using var manifest = JsonDocument.Parse(File.ReadAllBytes(manifestPath));
        foreach (var entry in manifest.RootElement.GetProperty("files").EnumerateArray())
        {
            string name = entry.GetProperty("name").GetString()!;
            if (name == "mtp.safetensors") continue; // Speculation and MTP are explicitly out of scope.
            string path = Path.Combine(directory, name);
            using (var stream = File.OpenRead(path))
            {
                string actual = Convert.ToHexStringLower(SHA256.HashData(stream));
                if (actual != entry.GetProperty("sha256").GetString()) throw new InvalidDataException("Weight hash mismatch: " + name);
            }
            Load(device, path, name);
            Console.Error.WriteLine("Verified and mapped " + name);
        }
    }
    void Load(Metal device, string path, string name)
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
        }
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
