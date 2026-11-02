namespace Bootstrap.Engine;

internal sealed partial class QwenEngine
{
    string tensorOpsPrecision = "legacy";
    public string TensorOpsPrecision
    {
        get => tensorOpsPrecision;
        set
        {
            if (value is not ("legacy" or "fp32" or "fp16" or "bf16")) throw new ArgumentException("Precision must be legacy, fp32, fp16 or bf16");
            if (tensorOpsPrecision != value) { ClearPrefixCache(); tensorOpsPrecision = value; }
        }
    }
    public bool Fp32TensorOps { get => TensorOpsPrecision == "fp32"; set => TensorOpsPrecision = value ? "fp32" : "legacy"; }
    bool causalDeviceAttention;
    public bool CausalDeviceAttention { get => causalDeviceAttention; set { if (causalDeviceAttention != value) { ClearPrefixCache(); causalDeviceAttention = value; } } }
    bool adaptiveExpertTiles;
    public bool AdaptiveExpertTiles { get => adaptiveExpertTiles; set { if (adaptiveExpertTiles != value) { ClearPrefixCache(); adaptiveExpertTiles = value; } } }
    bool gatheredExperts;
    public bool GatheredExperts { get => gatheredExperts; set { if (gatheredExperts != value) { ClearPrefixCache(); gatheredExperts = value; } } }
    string expertKernel = "baseline";
    public string ExpertKernel
    {
        get => expertKernel;
        set {
            if (value is not ("baseline" or "staged" or "fused")) throw new ArgumentException("Invalid expert kernel");
            if (expertKernel != value) { ClearPrefixCache(); expertKernel = value; }
        }
    }
    bool parallelExpertRouting;
    public bool ParallelExpertRouting { get => parallelExpertRouting; set { if (parallelExpertRouting != value) { ClearPrefixCache(); parallelExpertRouting = value; } } }
    bool predecodedExperts;
    public bool PredecodedExperts { get => predecodedExperts; set { if (predecodedExperts != value) { ClearPrefixCache(); predecodedExperts = value; } } }
    bool predecodedProjections;
    public bool PredecodedProjections { get => predecodedProjections; set { if (predecodedProjections != value) { ClearPrefixCache(); predecodedProjections = value; } } }
    bool optimizedProjections;
    public bool OptimizedProjections { get => optimizedProjections; set { if (optimizedProjections != value) { ClearPrefixCache(); optimizedProjections = value; } } }
    string ProjectionKernel(string kernel) => TensorOpsPrecision == "legacy" ? kernel : kernel + (TensorOpsPrecision == "fp32" ? "_mpp" : "_" + TensorOpsPrecision);
    string attentionMode = "legacy";
    bool widePrefill;
    public string AttentionMode
    {
        get => attentionMode;
        set
        {
            if (value is not ("legacy" or "tiled" or "fp16" or "fp16-opt" or "fp16-device" or "fp16-online" or "fused-fp16")) throw new ArgumentException("Invalid attention mode");
            if (attentionMode != value) { ClearPrefixCache(); attentionMode = value; }
        }
    }
    public bool TiledAttention { get => AttentionMode != "legacy"; set => AttentionMode = value ? "tiled" : "legacy"; }
    public bool WidePrefill
    {
        get => widePrefill;
        set { if (widePrefill != value) { ClearPrefixCache(); widePrefill = value; } }
    }
    string AttentionKernel(string kernel) => kernel + (AttentionMode == "fp16-opt" ? "_fp16_opt64" : AttentionMode == "fp16" ? "_fp16" : "");
    int prefillBatchSize = 512, prefillWorkspaceCapacity = 512, expertTileRows = 16;
    public int PrefillBatchSize
    {
        get => prefillBatchSize;
        set {
            if (value < 32 || value > 2048) throw new ArgumentOutOfRangeException(nameof(value));
            if (batch is not null && value > batch.Capacity) throw new InvalidOperationException("Reserve the maximum batch size before first generation");
            if (value != prefillBatchSize) ClearPrefixCache();
            prefillBatchSize = value; prefillWorkspaceCapacity = Math.Max(prefillWorkspaceCapacity, value);
        }
    }
    public int ExpertTileRows { get => expertTileRows; set { if (value is not (16 or 32 or 64)) throw new ArgumentOutOfRangeException(nameof(value)); if (value != expertTileRows) ClearPrefixCache(); expertTileRows = value; } }
    DeviceAttentionWorkspace? deviceAttention;
    BatchBuffers? batch;
    sealed class BatchBuffers
    {
        public readonly int Capacity;
        public GpuBuffer? PackedInputs, PackedWeights, PackedExpertInputs, PackedExpertWeights;
        public readonly GpuBuffer Tokens, Hidden, Norm, Projection, Qkv, Z, A, B, Conv, Q, K, V, Rotated, Qgate, Scores, Router, Ids, Rates, Counts, Offsets, Tiles, TileCount, SortedTokens, Inverse, ExpertGate, ExpertUp, ExpertActivation, ExpertOutput, SharedGate, SharedUp, SharedActivation, SharedOutput, SharedWeight;
        public BatchBuffers(QwenEngine engine)
        {
            Capacity = engine.prefillWorkspaceCapacity;
            GpuBuffer B(int size) => engine.Buffer(size);
            Tokens = B(Capacity); Hidden = B(Capacity * 2048); Norm = B(Capacity * 2048); Projection = B(Capacity * 2048);
            Qkv = B(Capacity * 8192); Z = B(Capacity * 4096); A = B(Capacity * 32); this.B = B(Capacity * 32); Conv = B(Capacity * 8192);
            Q = B(Capacity * 4096); K = B(Capacity * 2048); V = B(Capacity * 512); Rotated = B(Capacity * 4096); Qgate = B(Capacity * 4096);
            Scores = B(ContextLimits.AttentionWorkspaceFloats); Router = B(Capacity * 256); Ids = B(Capacity * 8); Rates = B(Capacity * 8);
            Counts = B(256); Offsets = B(257); Tiles = B((Capacity / 2 + 256) * 4); TileCount = B(1); SortedTokens = B(Capacity * 8); Inverse = B(Capacity * 8);
            ExpertGate = B(Capacity * 8 * 512); ExpertUp = B(Capacity * 8 * 512); ExpertActivation = B(Capacity * 8 * 512); ExpertOutput = B(Capacity * 8 * 2048);
            SharedGate = B(Capacity * 512); SharedUp = B(Capacity * 512); SharedActivation = B(Capacity * 512); SharedOutput = B(Capacity * 2048); SharedWeight = B(Capacity);
        }
    }
    void BatchProject(string name, GpuBuffer input, GpuBuffer output, int count)
    {
        var tensor = weights[name + ".weight"];
        var scales = tensor.Kind == 0 ? weights[name + ".weight_scale_inv"].Binding : (Binding)dummy;
        if (count <= VerificationCapacity)
        {
            gpu.Dispatch("matrix_product_tiny", new() { N = (uint)tensor.Shape[0], M = (uint)count, K = (uint)tensor.Shape[1], Kind = tensor.Kind }, (uint)tensor.Shape[0] * 32, 128, input, tensor.Binding, scales, output);
            return;
        }
        if (PredecodedProjections && TensorOpsPrecision == "fp16" && tensor.Shape[0] > 32)
        {
            var b = batch!;
            b.PackedInputs ??= Buffer(b.Capacity * 8192 / 2);
            b.PackedWeights ??= Buffer(8192 * 2048 / 2);
            if ((ulong)tensor.Shape[0] * (ulong)tensor.Shape[1] * 2 > b.PackedWeights.Length || (ulong)count * (ulong)tensor.Shape[1] * 2 > b.PackedInputs.Length)
                throw new ArgumentException("Projection exceeds predecoded scratch workspace");
            var p = new KernelArgs { N = (uint)tensor.Shape[0], M = (uint)count, K = (uint)tensor.Shape[1], Kind = tensor.Kind };
            gpu.Dispatch("projection_unpack_half", p, p.N * p.K / 4, 128, tensor.Binding, scales, b.PackedWeights);
            gpu.Dispatch("projection_input_half", p, p.M * p.K / 4, 128, input, b.PackedInputs);
            gpu.Dispatch2D("matrix_product_predecoded", p, ((p.N + 63) / 64) * 128, (p.M + 63) / 64, 128, b.PackedInputs, b.PackedWeights, output);
            return;
        }
        bool small = tensor.Shape[0] <= 32;
        uint threads = small || (OptimizedProjections && TensorOpsPrecision == "fp16") ? 128u : 256u;
        string kernel = ProjectionKernel(small ? "matrix_product_small" : "matrix_product") + (!small && OptimizedProjections && TensorOpsPrecision == "fp16" ? "_tile" : "");
        gpu.Dispatch2D(kernel, new() { N = (uint)tensor.Shape[0], M = (uint)count, K = (uint)tensor.Shape[1], Kind = tensor.Kind }, checked((uint)((tensor.Shape[0] + (small ? 31 : 63)) / (small ? 32 : 64) * threads)), checked((uint)((count + (small ? 15 : 31)) / (small ? 16 : 32))), threads, input, tensor.Binding, scales, output);
    }
    int PrefillBlock(ReadOnlySpan<int> tokens, int position, bool output, bool ignoreEos, bool verify = false)
    {
        if (batch is null) { batch = new BatchBuffers(this); Console.Error.WriteLine("Batched FP32 prefill workspace ready; Metal buffer bytes=" + gpu.AllocatedBytes); }
        // Large prefill tiles are inefficient for the 1-8-row speculative verifier.
        bool useTiledAttention = TiledAttention && !(verify && AttentionMode is "fp16-opt" or "fp16-device");
        var b = batch; int count = tokens.Length; tokens.CopyTo(b.Tokens.Values<int>(count));
        if (count > b.Capacity) throw new ArgumentException("Batch exceeds allocated workspace");
        EnsureCacheCapacity(position + count);
        ActiveDraft?.EnsureFeatureCapacity(count);
        if (!verify && (AttentionMode is "fp16-device" or "fp16-online")) { deviceAttention ??= new(gpu); deviceAttention.Prepare(count, position + count, HalfKv, AttentionMode == "fp16-online"); }
        gpu.Begin();
        gpu.Dispatch("batch_embedding", new() { M = (uint)count }, checked((uint)count * Hidden), 128, weights["model.language_model.embed_tokens.weight"].Binding, b.Tokens, b.Hidden);
        for (int layer = 0; layer < 40; layer++)
        {
            string prefix = Prefix(layer); Normalize(b.Hidden, prefix + ".input_layernorm", b.Norm, Hidden, (uint)count);
            if ((layer + 1) % 4 != 0)
            {
                string path = prefix + ".linear_attn";
                BatchProject(path + ".in_proj_qkv", b.Norm, b.Qkv, count); BatchProject(path + ".in_proj_z", b.Norm, b.Z, count); BatchProject(path + ".in_proj_a", b.Norm, b.A, count); BatchProject(path + ".in_proj_b", b.Norm, b.B, count);
                gpu.Dispatch("batch_conv", new() { M = (uint)count, Flags = verify ? 1u : 0u }, 8192, 128, b.Qkv, weights[path + ".conv1d.weight"].Binding, convolution[layer], b.Conv, verify ? verification!.Convolution[layer] : dummy);
                gpu.Dispatch("batch_l2", new() { M = (uint)count }, checked((uint)count * 16 * 128), 128, b.Conv, b.Q, b.K);
                gpu.Dispatch("batch_delta", new() { M = (uint)count, Flags = verify ? 1u : 0u }, 32 * 128 * 32, 128, b.Q, b.K, b.Conv, b.A, b.B, weights[path + ".A_log"].Binding, weights[path + ".dt_bias"].Binding, state[layer], b.Rotated, verify ? verification!.State[layer] : dummy);
                Normalize(b.Rotated, path + ".norm", b.Rotated, 128, checked((uint)count * 32), false, b.Z);
                BatchProject(path + ".out_proj", b.Rotated, b.Projection, count);
            }
            else
            {
                string path = prefix + ".self_attn";
                BatchProject(path + ".q_proj", b.Norm, b.Qkv, count); BatchProject(path + ".k_proj", b.Norm, b.K, count); BatchProject(path + ".v_proj", b.Norm, b.V, count);
                gpu.Dispatch("batch_split_q", new() { M = (uint)count }, checked((uint)count * 4096), 128, b.Qkv, b.Q, b.Qgate);
                Normalize(b.Q, path + ".q_norm", b.Q, 256, checked((uint)count * 16)); Normalize(b.K, path + ".k_norm", b.K, 256, checked((uint)count * 2));
                var p = new KernelArgs { M = (uint)count, Position = (uint)position, K = (uint)(position + count), Theta = 10_000_000f, Flags = KvFlags };
                gpu.Dispatch("batch_rope", p, checked((uint)count * 4096), 128, b.Q, b.K, b.V, b.Rotated, keyCache[layer], valueCache[layer]);
                if (!verify && AttentionMode == "fp16-online")
                    deviceAttention!.EncodeOnline(b.Rotated, keyCache[layer], valueCache[layer], b.Qgate, b.Scores, b.Rotated, count, position, HalfKv);
                else if (!verify && AttentionMode == "fp16-device")
                    deviceAttention!.Encode(b.Rotated, keyCache[layer], valueCache[layer], b.Qgate, b.Scores, b.Rotated, count, position, HalfKv, CausalDeviceAttention);
                else if (AttentionMode == "fused-fp16")
                    gpu.Dispatch2D("batch_attention_fused_fp16", p, (uint)((count + 15) / 16) * 128, 16, 128, b.Rotated, keyCache[layer], valueCache[layer], b.Qgate, b.Rotated);
                else
                {
                    // Keep the model batch wide while reusing the bounded attention
                    // score workspace across query slices. KV includes the entire
                    // batch; each slice retains its absolute causal position.
                    int capacity = ContextLimits.AttentionWorkspaceFloats / 16 / (position + count);
                    for (int first = 0; first < count; first += capacity)
                    {
                        int rows = Math.Min(capacity, count - first);
                        var attention = p; attention.M = (uint)rows; attention.Position = (uint)(position + first);
                        var query = new Binding(b.Rotated, (ulong)first * 4096 * 4);
                        var gate = new Binding(b.Qgate, (ulong)first * 4096 * 4);
                        if (useTiledAttention) gpu.Dispatch2D(AttentionKernel("batch_attention_scores_tiled"), attention, (uint)((position + count + 31) / 32) * 128, (uint)((rows + 15) / 16) * 16, 128, query, keyCache[layer], b.Scores);
                        else gpu.Dispatch2D("batch_attention_scores", attention, (uint)(position + count) * 32, (uint)rows * 16, 128, query, keyCache[layer], b.Scores);
                        var normalization = attention; normalization.Position = (uint)(position + count - 1);
                        gpu.Dispatch("attention_softmax", normalization, (uint)rows * 16 * 256, 256, b.Scores);
                        if (useTiledAttention) gpu.Dispatch2D(AttentionKernel("batch_attention_values_tiled"), attention, 8 * 128, (uint)((rows + 15) / 16) * 16, 128, b.Scores, valueCache[layer], gate, query);
                        else gpu.Dispatch("batch_attention_values", attention, (uint)rows * 4096, 128, b.Scores, valueCache[layer], gate, query);
                    }
                }
                BatchProject(path + ".o_proj", b.Rotated, b.Projection, count);
            }
            gpu.Dispatch("add", new() { N = checked((uint)count * Hidden) }, checked((uint)count * Hidden), 128, b.Hidden, b.Projection, b.Hidden);
            Normalize(b.Hidden, prefix + ".post_attention_layernorm", b.Norm, Hidden, (uint)count);
            BatchProject(prefix + ".mlp.gate", b.Norm, b.Router, count);
            gpu.Dispatch("batch_router", new() { K = (uint)count }, (uint)count * 32, 128, b.Router, b.Ids, b.Rates);
            bool stagedExperts = count > VerificationCapacity && GatheredExperts && TensorOpsPrecision == "fp16" && ExpertKernel != "baseline";
            int tileRows = stagedExperts ? 32 : count > VerificationCapacity && OptimizedProjections && TensorOpsPrecision == "fp16" ? ExpertTileRows : 16;
            if (ParallelExpertRouting && count > VerificationCapacity)
                gpu.Dispatch("expert_group_parallel", new() { M = (uint)count, A = (uint)tileRows }, 256, 256, b.Ids, b.Counts, b.Offsets, b.Tiles, b.TileCount, b.SortedTokens, b.Inverse);
            else {
                gpu.Dispatch("expert_counts", new() { M = (uint)count }, 256, 128, b.Ids, b.Counts);
                gpu.Dispatch("expert_tiles", new() { M = (uint)count, A = (uint)tileRows }, 1, 1, b.Counts, b.Offsets, b.Tiles, b.TileCount);
                gpu.Dispatch("expert_group", new() { M = (uint)count }, 256, 128, b.Ids, b.Offsets, b.SortedTokens, b.Inverse);
            }
            var raw = weights.Shard("layers-" + layer + ".safetensors");
            uint tileLimit = (uint)Math.Min(count * 8, (count * 8 + tileRows - 1) / tileRows + 256);
            void ExpertProjection(uint choice, uint rows, uint columns, GpuBuffer input, GpuBuffer result)
            {
                bool tiny = count <= VerificationCapacity;
                if (!tiny && GatheredExperts && TensorOpsPrecision == "fp16")
                {
                    b.PackedExpertInputs ??= Buffer(b.Capacity * 8 * Hidden / 2);
                    var p = new KernelArgs { M = (uint)count * 8, N = rows, K = columns, Kind = choice, Flags = AdaptiveExpertTiles ? 1u : 0u };
                    if (choice != 1) gpu.Dispatch("expert_input_half", p, p.M * columns / 4, 128, input, b.SortedTokens, b.PackedExpertInputs);
                    gpu.Dispatch2D(stagedExperts ? "expert_matrix_staged" : "expert_matrix_gathered", p, ((rows + 63) / 64) * (stagedExperts ? 64u : 128u), tileLimit, stagedExperts ? 64u : 128u, b.PackedExpertInputs, raw, descriptors[layer], b.Tiles, b.TileCount, result);
                    return;
                }
                if (!tiny && PredecodedExperts && TensorOpsPrecision == "fp16")
                {
                    b.PackedExpertInputs ??= Buffer(b.Capacity * 8 * Hidden / 2);
                    b.PackedExpertWeights ??= Buffer(256 * 512 * Hidden / 2);
                    var p = new KernelArgs { M = (uint)count * 8, N = rows, K = columns, Kind = choice, A = 256 };
                    gpu.Dispatch("expert_unpack_half", p, 256 * rows * columns / 4, 128, raw, descriptors[layer], b.PackedExpertWeights);
                    gpu.Dispatch("expert_input_half", p, p.M * columns / 4, 128, input, b.SortedTokens, b.PackedExpertInputs);
                    gpu.Dispatch2D("expert_matrix_predecoded", p, ((rows + 63) / 64) * 128, tileLimit, 128, b.PackedExpertInputs, b.PackedExpertWeights, b.Tiles, b.TileCount, result);
                    return;
                }
                bool optimized = !tiny && OptimizedProjections && TensorOpsPrecision == "fp16";
                uint threads = tiny || (optimized && choice != 2 && tileRows != 64) ? 128u : 256u;
                string kernel = tiny ? "expert_matrix_tiny" : ProjectionKernel("expert_matrix") + (optimized ? "_tile" + (tileRows == 16 ? "" : tileRows.ToString()) + (choice == 2 ? "_down" : "_up") : "");
                gpu.Dispatch2D(kernel, new() { N = rows, K = columns, Kind = choice }, tiny ? rows * 32 : ((rows + 63) / 64) * threads, tileLimit, threads, input, raw, descriptors[layer], b.Tiles, b.TileCount, b.SortedTokens, result);
            }
            if (stagedExperts && ExpertKernel == "fused") {
                b.PackedExpertInputs ??= Buffer(b.Capacity * 8 * Hidden / 2);
                var p = new KernelArgs { M = (uint)count * 8, N = 512, K = Hidden };
                gpu.Dispatch("expert_input_half", p, p.M * Hidden / 4, 128, b.Norm, b.SortedTokens, b.PackedExpertInputs);
                // The activation buffer is deliberately FP16 on this path. Down reads
                // it directly; no global gate/up outputs or activation repack needed.
                gpu.Dispatch2D("expert_gate_up_staged", p, 16 * 64, tileLimit, 64, b.PackedExpertInputs, raw, descriptors[layer], b.Tiles, b.TileCount, b.ExpertActivation);
                gpu.Dispatch2D("expert_matrix_staged", new() { N = Hidden, K = 512, Kind = 2 }, (Hidden / 64) * 64, tileLimit, 64, b.ExpertActivation, raw, descriptors[layer], b.Tiles, b.TileCount, b.ExpertOutput);
            } else {
                ExpertProjection(0, 512, Hidden, b.Norm, b.ExpertGate); ExpertProjection(1, 512, Hidden, b.Norm, b.ExpertUp);
                gpu.Dispatch("silu_product", new() { N = checked((uint)count * 8 * 512) }, checked((uint)count * 8 * 512), 128, b.ExpertGate, b.ExpertUp, b.ExpertActivation);
                ExpertProjection(2, Hidden, 512, b.ExpertActivation, b.ExpertOutput);
            }
            BatchProject(prefix + ".mlp.shared_expert.gate_proj", b.Norm, b.SharedGate, count); BatchProject(prefix + ".mlp.shared_expert.up_proj", b.Norm, b.SharedUp, count);
            gpu.Dispatch("silu_product", new() { N = checked((uint)count * 512) }, checked((uint)count * 512), 128, b.SharedGate, b.SharedUp, b.SharedActivation);
            BatchProject(prefix + ".mlp.shared_expert.down_proj", b.SharedActivation, b.SharedOutput, count); BatchProject(prefix + ".mlp.shared_expert_gate", b.Norm, b.SharedWeight, count);
            gpu.Dispatch("batch_mix", new() { M = (uint)count }, checked((uint)count * Hidden), 128, b.Hidden, b.ExpertOutput, b.Inverse, b.Rates, b.SharedOutput, b.SharedWeight, b.Hidden);
            ActiveDraft?.Capture(layer, b.Hidden, count);
        }
        if (!verify) ActiveDraft?.CommitFeatures(count, position);
        if (verify)
        {
            var verificationBuffers = verification!;
            Normalize(b.Hidden, "model.language_model.norm", b.Norm, Hidden, (uint)count);
            BatchProject("lm_head", b.Norm, verificationBuffers.Logits, count);
            for (int row = 0; row < count; row++)
                gpu.Dispatch("argmax", new() { N = Vocabulary, Flags = ignoreEos ? 1u : 0u, A = (uint)Tokenizer.End, B = (uint)Tokenizer.EndOfText }, 256, 256,
                    new Binding(verificationBuffers.Logits, (ulong)row * Vocabulary * 4), new Binding(verificationBuffers.Next, (ulong)row * 4));
        }
        else if (output)
        {
            gpu.Dispatch("rmsnorm", new() { N = Hidden, M = 1, Epsilon = 1e-6f, Flags = 1 }, 256, 256, new Binding(b.Hidden, checked((ulong)(count - 1) * Hidden * 4)), weights["model.language_model.norm.weight"].Binding, norm, dummy);
            Project("lm_head", norm, logits);
            SelectNext(ignoreEos);
        }
        gpu.End(); return output && !verify ? next.Values<int>(1)[0] : -1;
    }
}
