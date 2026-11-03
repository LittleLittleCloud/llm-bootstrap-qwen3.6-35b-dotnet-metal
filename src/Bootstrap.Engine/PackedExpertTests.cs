using System.Security.Cryptography;
using System.Text.Json;
namespace Bootstrap.Engine;

internal static class PackedExpertTests
{
    public static void Run(string source)
    {
        using var gpu = new Metal(source);
        int checks = 0;
        // This exercises the actual MAP_PRIVATE loader contract, not just a heap copy.
        string file = Path.GetTempFileName();
        try
        {
            byte[] original = Enumerable.Range(0, 32 * 128).Select(i => (byte)(i * 37 + i / 127)).ToArray();
            File.WriteAllBytes(file, original);
            using (var mapped = gpu.MapFile(file))
            {
                Checkpoint.RepackExpert(mapped.Values<byte>(original.Length), 32, 128);
                CheckBytes(original, mapped.Values<byte>(original.Length), 32, 128); checks++;
            }
            if (!SHA256.HashData(original).SequenceEqual(SHA256.HashData(File.ReadAllBytes(file)))) throw new Exception("Checkpoint file changed");
            checks++;
        }
        finally { File.Delete(file); }
        foreach (bool fused in new[] { true, false })
        {
            int k = fused ? 2048 : 512, n = fused ? 512 : 2048, choices = fused ? 2 : 1;
            const int experts = 3;
            int scaleCount = n / 128 * (k / 128), slot = n * k + scaleCount * 2;
            using var raw = gpu.Allocate((ulong)(experts * choices * slot));
            using var packed = gpu.Allocate(raw.Length);
            using var desc = gpu.Allocate(experts * 48);
            for (int e = 0; e < experts; e++) for (int h = 0; h < choices; h++)
            {
                ulong off = (ulong)((e * choices + h) * slot);
                for (int d = 0; d < 3; d++) if (choices == 1 || h == d) { desc.Values<ulong>(experts * 6)[e * 6 + d * 2] = off; desc.Values<ulong>(experts * 6)[e * 6 + d * 2 + 1] = off + (ulong)(n * k); }
                for (int i = 0; i < n * k; i++) { byte b = (byte)(i * 37 + i / 127 + e * 13 + h * 67); raw.Values<byte>(n * k, off)[i] = (b & 127) == 127 ? (byte)126 : b; }
                for (int i = 0; i < scaleCount; i++) raw.Values<ushort>(scaleCount, off + (ulong)(n * k))[i] = (ushort)(0x3a10 + (i % 3) * 128);
            }
            raw.Values<byte>((int)raw.Length).CopyTo(packed.Values<byte>((int)packed.Length));
            for (int e = 0; e < experts; e++) for (int h = 0; h < choices; h++)
            {
                ulong off = (ulong)((e * choices + h) * slot);
                Checkpoint.RepackExpert(packed.Values<byte>(n * k, off), n, k);
                CheckBytes(raw.Values<byte>(n * k, off), packed.Values<byte>(n * k, off), n, k); checks++;
                if (!raw.Values<ushort>(scaleCount, off + (ulong)(n * k)).SequenceEqual(packed.Values<ushort>(scaleCount, off + (ulong)(n * k)))) throw new Exception("Scales changed");
            }
            using var x = gpu.Allocate((ulong)(experts * 32 * k * 4));
            using var hx = gpu.Allocate((ulong)(experts * 32 * k * 2));
            using var y = gpu.Allocate((ulong)(experts * 32 * n + 1) * 4);
            using var reference = gpu.Allocate(y.Length);
            using var tiles = gpu.Allocate(experts * 16);
            using var count = gpu.Allocate(4);
            using var args = gpu.Allocate(16);
            using var tokens = gpu.Allocate(experts * 32 * 4);
            using var ids = gpu.Allocate(experts * 4);
            for (int i = 0; i < experts * 32 * k; i++) { float v = ((i * 19 + i / 113) % 67 - 33) / 65f; x.Values<float>(experts * 32 * k)[i] = v; hx.Values<Half>(experts * 32 * k)[i] = (Half)v; }
            for (int i = 0; i < experts; i++) ids.Values<int>(experts)[i] = i;
            for (int i = 0; i < experts * 32; i++) tokens.Values<uint>(experts * 32)[i] = (uint)i;
            var p = new KernelArgs { N = (uint)n, K = (uint)k, M = experts, Kind = fused ? 0u : 2u };
            // Decode reads FP32 activations; preserve its original precision/order.
            foreach (bool layout in new[] { false, true })
            {
                p.Flags = layout ? 1024u : 0u; gpu.Begin();
                gpu.Dispatch(fused ? "expert_gate_up" : "expert_down", p, (uint)(experts * n * 32), 128, x, layout ? packed : raw, desc, ids, layout ? y : reference); gpu.End();
            }
            Equal(reference, y, experts * n * 4); checks++;
            foreach (int live in new[] { 1, 2, 7, 8, 9, 16, 17, 31, 32 })
            {
                for (int e = 0; e < experts; e++) { var t = tiles.Values<uint>(experts * 4).Slice(e * 4); t[0] = (uint)e; t[1] = (uint)(e * live); t[2] = (uint)live; }
                count.Values<uint>(1)[0] = experts;
                int bytes = experts * live * n * (fused ? 2 : 4);
                reference.Values<byte>(bytes + 4).Fill(0xa5); y.Values<byte>(bytes + 4).Fill(0xa5);
                p.Flags = 0;
                gpu.Begin(); gpu.Dispatch2D(fused ? "expert_gate_up_staged" : "expert_matrix_staged", p, (uint)(n / (fused ? 32 : 64) * 64), experts + 1, 64, hx, raw, desc, tiles, count, reference); gpu.End();
                gpu.Begin(); gpu.Dispatch("expert_dispatch_args", new() { N = (uint)(n / (fused ? 32 : 64)) }, 1, 1, count, args);
                gpu.DispatchIndirect(fused ? "ablation_gate_up_packed" : "ablation_matrix_packed", p, args, 4, 64, hx, packed, desc, tiles, count, y); gpu.End();
                Equal(reference, y, bytes + 4); checks++;
                if (live <= 8) for (uint choice = 0; choice < (fused ? 2 : 1); choice++)
                {
                    p.Kind = fused ? choice : 2;
                    foreach (bool layout in new[] { false, true })
                    {
                        p.Flags = layout ? 1024u : 0u; gpu.Begin();
                        gpu.Dispatch2D("expert_matrix_tiny", p, (uint)n * 32, experts + 1, 128, x, layout ? packed : raw, desc, tiles, count, tokens, layout ? y : reference); gpu.End();
                    }
                    Equal(reference, y, experts * live * n * 4); checks++;
                }
                p.Kind = fused ? 0u : 2u;
            }
            try { gpu.DispatchIndirect("unused", p, args, 8, 64); throw new Exception("Invalid argument range accepted"); }
            catch (ArgumentOutOfRangeException) { checks++; }
        }
        Console.WriteLine(JsonSerializer.Serialize(new { type = "packed_expert_checks", checks, exact_outputs = true, checkpoint_unchanged = true, tails = new[] { 1, 2, 7, 8, 9, 16, 17, 31, 32 } }));
    }
    static void Equal(GpuBuffer a, GpuBuffer b, int bytes)
    {
        if (!a.Values<byte>(bytes).SequenceEqual(b.Values<byte>(bytes))) throw new Exception("Packed expert output differs");
    }
    static void CheckBytes(ReadOnlySpan<byte> source, ReadOnlySpan<byte> packed, int n, int k)
    {
        for (int r = 0; r < n; r++) for (int j = 0; j < k; j++)
        {
            int at = (r / 32) * k * 32 + (j / 32) * 1024 + (j % 32 / 4) * 128 + (r % 32) * 4 + j % 4;
            if (source[r * k + j] != packed[at]) throw new Exception("Packed byte differs");
        }
    }
}
