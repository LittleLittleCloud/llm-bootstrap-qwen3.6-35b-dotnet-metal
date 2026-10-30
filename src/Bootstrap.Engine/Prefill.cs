namespace Bootstrap.Engine;

internal sealed partial class QwenEngine
{
    BatchBuffers? batch;
    sealed class BatchBuffers
    {
        public const int Capacity = 512;
        public readonly GpuBuffer Tokens, Hidden, Norm, Projection, Qkv, Z, A, B, Conv, Q, K, V, Rotated, Qgate, Scores, Router, Ids, Rates, Counts, Offsets, Tiles, TileCount, SortedTokens, Inverse, ExpertGate, ExpertUp, ExpertActivation, ExpertOutput, SharedGate, SharedUp, SharedActivation, SharedOutput, SharedWeight;
        public BatchBuffers(QwenEngine engine)
        {
            GpuBuffer B(int size) => engine.Buffer(size);
            Tokens = B(Capacity); Hidden = B(Capacity * 2048); Norm = B(Capacity * 2048); Projection = B(Capacity * 2048);
            Qkv = B(Capacity * 8192); Z = B(Capacity * 4096); A = B(Capacity * 32); this.B = B(Capacity * 32); Conv = B(Capacity * 8192);
            Q = B(Capacity * 4096); K = B(Capacity * 2048); V = B(Capacity * 512); Rotated = B(Capacity * 4096); Qgate = B(Capacity * 4096);
            Scores = B(ContextLimits.AttentionWorkspaceFloats); Router = B(Capacity * 256); Ids = B(Capacity * 8); Rates = B(Capacity * 8);
            Counts = B(256); Offsets = B(257); Tiles = B(512 * 4); TileCount = B(1); SortedTokens = B(Capacity * 8); Inverse = B(Capacity * 8);
            ExpertGate = B(Capacity * 8 * 512); ExpertUp = B(Capacity * 8 * 512); ExpertActivation = B(Capacity * 8 * 512); ExpertOutput = B(Capacity * 8 * 2048);
            SharedGate = B(Capacity * 512); SharedUp = B(Capacity * 512); SharedActivation = B(Capacity * 512); SharedOutput = B(Capacity * 2048); SharedWeight = B(Capacity);
        }
    }
    void BatchProject(string name, GpuBuffer input, GpuBuffer output, int count)
    {
        var tensor = weights[name + ".weight"];
        var scales = tensor.Kind == 0 ? weights[name + ".weight_scale_inv"].Binding : (Binding)dummy;
        bool small = tensor.Shape[0] <= 32;
        gpu.Dispatch2D(small ? "matrix_product_small" : "matrix_product", new() { N = (uint)tensor.Shape[0], M = (uint)count, K = (uint)tensor.Shape[1], Kind = tensor.Kind }, checked((uint)((tensor.Shape[0] + (small ? 31 : 63)) / (small ? 32 : 64) * (small ? 128 : 256))), checked((uint)((count + (small ? 15 : 31)) / (small ? 16 : 32))), small ? 128u : 256u, input, tensor.Binding, scales, output);
    }
    int PrefillBlock(ReadOnlySpan<int> tokens, int position, bool output, bool ignoreEos)
    {
        if (batch is null) { batch = new BatchBuffers(this); Console.Error.WriteLine("Batched FP32 prefill workspace ready; Metal buffer bytes=" + gpu.AllocatedBytes); }
        var b = batch; int count = tokens.Length; tokens.CopyTo(b.Tokens.Values<int>(count));
        EnsureCacheCapacity(position + count);
        if ((long)count * 16 * (position + count) > ContextLimits.AttentionWorkspaceFloats) throw new ArgumentException("Prefill tile exceeds attention workspace");
        gpu.Begin();
        gpu.Dispatch("batch_embedding", new() { M = (uint)count }, checked((uint)count * Hidden), 128, weights["model.language_model.embed_tokens.weight"].Binding, b.Tokens, b.Hidden);
        for (int layer = 0; layer < 40; layer++)
        {
            string prefix = Prefix(layer); Normalize(b.Hidden, prefix + ".input_layernorm", b.Norm, Hidden, (uint)count);
            if ((layer + 1) % 4 != 0)
            {
                string path = prefix + ".linear_attn";
                BatchProject(path + ".in_proj_qkv", b.Norm, b.Qkv, count); BatchProject(path + ".in_proj_z", b.Norm, b.Z, count); BatchProject(path + ".in_proj_a", b.Norm, b.A, count); BatchProject(path + ".in_proj_b", b.Norm, b.B, count);
                gpu.Dispatch("batch_conv", new() { M = (uint)count }, 8192, 128, b.Qkv, weights[path + ".conv1d.weight"].Binding, convolution[layer], b.Conv);
                gpu.Dispatch("batch_l2", new() { M = (uint)count }, checked((uint)count * 16 * 128), 128, b.Conv, b.Q, b.K);
                gpu.Dispatch("batch_delta", new() { M = (uint)count }, 32 * 128 * 32, 128, b.Q, b.K, b.Conv, b.A, b.B, weights[path + ".A_log"].Binding, weights[path + ".dt_bias"].Binding, state[layer], b.Rotated);
                Normalize(b.Rotated, path + ".norm", b.Rotated, 128, checked((uint)count * 32), false, b.Z);
                BatchProject(path + ".out_proj", b.Rotated, b.Projection, count);
            }
            else
            {
                string path = prefix + ".self_attn";
                BatchProject(path + ".q_proj", b.Norm, b.Qkv, count); BatchProject(path + ".k_proj", b.Norm, b.K, count); BatchProject(path + ".v_proj", b.Norm, b.V, count);
                gpu.Dispatch("batch_split_q", new() { M = (uint)count }, checked((uint)count * 4096), 128, b.Qkv, b.Q, b.Qgate);
                Normalize(b.Q, path + ".q_norm", b.Q, 256, checked((uint)count * 16)); Normalize(b.K, path + ".k_norm", b.K, 256, checked((uint)count * 2));
                var p = new KernelArgs { M = (uint)count, Position = (uint)position, K = (uint)(position + count), Theta = 10_000_000f };
                gpu.Dispatch("batch_rope", p, checked((uint)count * 4096), 128, b.Q, b.K, b.V, b.Rotated, keyCache[layer], valueCache[layer]);
                gpu.Dispatch2D("batch_attention_scores", p, checked((uint)(position + count) * 32), checked((uint)count * 16), 128, b.Rotated, keyCache[layer], b.Scores);
                var normalization = p; normalization.Position = (uint)(position + count - 1);
                gpu.Dispatch("attention_softmax", normalization, checked((uint)count * 16 * 256), 256, b.Scores);
                gpu.Dispatch("batch_attention_values", p, checked((uint)count * 4096), 128, b.Scores, valueCache[layer], b.Qgate, b.Rotated);
                BatchProject(path + ".o_proj", b.Rotated, b.Projection, count);
            }
            gpu.Dispatch("add", new() { N = checked((uint)count * Hidden) }, checked((uint)count * Hidden), 128, b.Hidden, b.Projection, b.Hidden);
            Normalize(b.Hidden, prefix + ".post_attention_layernorm", b.Norm, Hidden, (uint)count);
            BatchProject(prefix + ".mlp.gate", b.Norm, b.Router, count);
            gpu.Dispatch("batch_router", new() { K = (uint)count }, (uint)count * 32, 128, b.Router, b.Ids, b.Rates);
            gpu.Dispatch("expert_counts", new() { M = (uint)count }, 256, 128, b.Ids, b.Counts);
            gpu.Dispatch("expert_tiles", new() { M = (uint)count }, 1, 1, b.Counts, b.Offsets, b.Tiles, b.TileCount);
            gpu.Dispatch("expert_group", new() { M = (uint)count }, 256, 128, b.Ids, b.Offsets, b.SortedTokens, b.Inverse);
            var raw = weights.Shard("layers-" + layer + ".safetensors");
            uint tileLimit = (uint)Math.Min(count * 8, (count * 8 + 15) / 16 + 256);
            void ExpertProjection(uint choice, uint rows, uint columns, GpuBuffer input, GpuBuffer result) => gpu.Dispatch2D("expert_matrix", new() { N = rows, K = columns, Kind = choice }, ((rows + 63) / 64) * 256, tileLimit, 256, input, raw, descriptors[layer], b.Tiles, b.TileCount, b.SortedTokens, result);
            ExpertProjection(0, 512, Hidden, b.Norm, b.ExpertGate); ExpertProjection(1, 512, Hidden, b.Norm, b.ExpertUp);
            gpu.Dispatch("silu_product", new() { N = checked((uint)count * 8 * 512) }, checked((uint)count * 8 * 512), 128, b.ExpertGate, b.ExpertUp, b.ExpertActivation);
            ExpertProjection(2, Hidden, 512, b.ExpertActivation, b.ExpertOutput);
            BatchProject(prefix + ".mlp.shared_expert.gate_proj", b.Norm, b.SharedGate, count); BatchProject(prefix + ".mlp.shared_expert.up_proj", b.Norm, b.SharedUp, count);
            gpu.Dispatch("silu_product", new() { N = checked((uint)count * 512) }, checked((uint)count * 512), 128, b.SharedGate, b.SharedUp, b.SharedActivation);
            BatchProject(prefix + ".mlp.shared_expert.down_proj", b.SharedActivation, b.SharedOutput, count); BatchProject(prefix + ".mlp.shared_expert_gate", b.Norm, b.SharedWeight, count);
            gpu.Dispatch("batch_mix", new() { M = (uint)count }, checked((uint)count * Hidden), 128, b.Hidden, b.ExpertOutput, b.Inverse, b.Rates, b.SharedOutput, b.SharedWeight, b.Hidden);
        }
        if (output)
        {
            gpu.Dispatch("rmsnorm", new() { N = Hidden, M = 1, Epsilon = 1e-6f, Flags = 1 }, 256, 256, new Binding(b.Hidden, checked((ulong)(count - 1) * Hidden * 4)), weights["model.language_model.norm.weight"].Binding, norm, dummy);
            Project("lm_head", norm, logits);
            gpu.Dispatch("argmax", new() { N = Vocabulary, Flags = ignoreEos ? 1u : 0u, A = (uint)Tokenizer.End, B = (uint)Tokenizer.EndOfText }, 256, 256, logits, next);
        }
        gpu.End(); return output ? next.Values<int>(1)[0] : -1;
    }
}
