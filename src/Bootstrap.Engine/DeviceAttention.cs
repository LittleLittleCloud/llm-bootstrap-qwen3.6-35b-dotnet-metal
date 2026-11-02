namespace Bootstrap.Engine;

// Scratch conversion is shared by all tiles of one full-attention layer.
// Prepare runs before Begin; Encode does not allocate or synchronize on the CPU.
internal sealed class DeviceAttentionWorkspace(Metal gpu) : IDisposable
{
    GpuBuffer? query, key, value, probability;
    void Ensure(ref GpuBuffer? buffer, ulong bytes)
    {
        if (buffer is not null && buffer.Length >= bytes) return;
        buffer?.Dispose(); buffer = gpu.Allocate(bytes);
    }
    public void Prepare(int rows, int length, bool halfKv = false)
    {
        Ensure(ref query, (ulong)rows * 4096 * 2);
        if (!halfKv) { Ensure(ref key, (ulong)length * 512 * 2); Ensure(ref value, (ulong)length * 512 * 2); }
        else { key?.Dispose(); value?.Dispose(); key = value = null; }
        Ensure(ref probability, Math.Min((ulong)rows * 16 * (ulong)length, ContextLimits.AttentionWorkspaceFloats) * 2);
    }
    public void Encode(GpuBuffer q, GpuBuffer k, GpuBuffer v, GpuBuffer gate, GpuBuffer scores, GpuBuffer output, int rows, int position, bool halfKv = false)
    {
        int length = position + rows;
        void Convert(Binding source, GpuBuffer target, int floats) => gpu.Dispatch("attention_input_half", new() { N = (uint)floats }, (uint)floats / 4, 128, source, target);
        Convert(q, query!, rows * 4096);
        if (!halfKv) { Convert(k, key!, length * 512); Convert(v, value!, length * 512); }
        var cachedKey = halfKv ? k : key!; var cachedValue = halfKv ? v : value!;
        int capacity = ContextLimits.AttentionWorkspaceFloats / 16 / length;
        for (int first = 0; first < rows; first += capacity)
        {
            int count = Math.Min(capacity, rows - first);
            var p = new KernelArgs { M = (uint)count, Position = (uint)(position + first), K = (uint)length };
            var input = new Binding(query!, (ulong)first * 4096 * 2);
            var result = new Binding(output, (ulong)first * 4096 * 4);
            var g = new Binding(gate, (ulong)first * 4096 * 4);
            gpu.Dispatch2D("attention_scores_device", p, (uint)((length + 63) / 64) * 128, (uint)((count + 15) / 16) * 16, 128, input, cachedKey, scores);
            var normalization = p; normalization.Position = (uint)(length - 1);
            gpu.Dispatch("attention_softmax", normalization, (uint)count * 16 * 256, 256, scores);
            Convert(scores, probability!, count * 16 * length);
            gpu.Dispatch2D("attention_values_device", p, 4 * 128, (uint)((count + 15) / 16) * 16, 128, probability!, cachedValue, g, result);
        }
    }
    public void Dispose() { query?.Dispose(); key?.Dispose(); value?.Dispose(); probability?.Dispose(); }
}
