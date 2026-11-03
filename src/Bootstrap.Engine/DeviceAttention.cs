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
    public void Prepare(int rows, int length, bool halfKv = false, bool online = false)
    {
        Ensure(ref query, (ulong)rows * 4096 * 2);
        if (!halfKv) { Ensure(ref key, (ulong)length * 512 * 2); Ensure(ref value, (ulong)length * 512 * 2); }
        else { key?.Dispose(); value?.Dispose(); key = value = null; }
        if (!online || rows < 32) Ensure(ref probability, Math.Min((ulong)rows * 16 * (ulong)length, ContextLimits.AttentionWorkspaceFloats) * 2);
    }
    public void Encode(GpuBuffer q, GpuBuffer k, GpuBuffer v, GpuBuffer gate, GpuBuffer scores, GpuBuffer output, int rows, int position, bool halfKv = false, bool causal = false)
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
            gpu.Dispatch2D(causal ? "attention_scores_device_causal" : "attention_scores_device", p, (uint)((length + 63) / 64) * 128, (uint)((count + 15) / 16) * 16, 128, input, cachedKey, scores);
            if (causal) gpu.Dispatch("attention_softmax_half_causal", p, (uint)count * 16 * 256, 256, scores, probability!);
            else
            {
                var normalization = p; normalization.Position = (uint)(length - 1);
                gpu.Dispatch("attention_softmax", normalization, (uint)count * 16 * 256, 256, scores);
                Convert(scores, probability!, count * 16 * length);
            }
            gpu.Dispatch2D(causal ? "attention_values_device_causal" : "attention_values_device", p, 4 * 128, (uint)((count + 15) / 16) * 16, 128, probability!, cachedValue, g, result);
        }
    }
    // Group eight adjacent query heads around their shared KV head. Below 32
    // queries the measured causal device path is faster than the fused kernel.
    public const int OnlineGroupHeads = 8;
    public void EncodeOnline(GpuBuffer q, GpuBuffer k, GpuBuffer v, GpuBuffer gate, GpuBuffer scores, GpuBuffer output, int rows, int position, bool halfKv = false)
    {
        if (rows < 32) { Encode(q, k, v, gate, scores, output, rows, position, halfKv, causal: true); return; }
        int length = position + rows, heads = OnlineGroupHeads;
        var p = new KernelArgs { M = (uint)rows, K = (uint)length, Position = (uint)position, A = (uint)heads };
        gpu.Dispatch("attention_query_grouped_half", p, (uint)rows * 1024, 128, q, query!);
        if (!halfKv)
        {
            var convert = new KernelArgs { N = (uint)length * 512 };
            gpu.Dispatch("attention_input_half", convert, (uint)length * 128, 128, k, key!);
            gpu.Dispatch("attention_input_half", convert, (uint)length * 128, 128, v, value!);
        }
        gpu.Dispatch2D("attention_online_q32_k256_s8_g8_r2", p, (uint)((rows * heads + 31) / 32) * 256, (uint)(16 / heads), 256,
            query!, halfKv ? k : key!, halfKv ? v : value!, gate, output);
    }
    public void Dispose() { query?.Dispose(); key?.Dispose(); value?.Dispose(); probability?.Dispose(); }
}
