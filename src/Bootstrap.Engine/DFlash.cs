using System.Security.Cryptography;
using System.Text.Json;

namespace Bootstrap.Engine;

internal sealed partial class QwenEngine
{
    DFlashDraft? draft;
    bool speculativeDecoding = true;
    public bool SpeculativeDecoding
    {
        get => speculativeDecoding;
        set { if (speculativeDecoding != value) ClearPrefixCache(); speculativeDecoding = value; }
    }
    DFlashDraft? ActiveDraft => SpeculativeDecoding ? draft : null;
    public object? LastSpeculationStats { get; private set; }
    public void LoadDraft(string directory, string manifest)
    {
        if (draft is not null) throw new InvalidOperationException("Draft already loaded");
        ClearPrefixCache(); sparePrefixBuffers.Clear();
        draft = new DFlashDraft(this, directory, manifest);
    }

    public bool DraftLoaded => draft is not null;
    bool windowedDraftPrefill = true;
    public bool WindowedDraftPrefill
    {
        get => windowedDraftPrefill;
        set { if (windowedDraftPrefill != value) ClearPrefixCache(); windowedDraftPrefill = value; }
    }
    public int LastDraftPrefillRows { get; private set; }

    public void DraftSelfTest(string directory, int context = 16)
    {
        if (draft is null) throw new InvalidOperationException("--draft-dir is required");
        Directory.CreateDirectory(directory);
        draft.ExportSyntheticFixture(directory, context);
    }

    // Draft checkpoint weights stay BF16; stored activations stay FP32. The optional
    // prefill FC path rounds matrix operands to FP16, accumulating/outputting FP32.
    // Cached KV follows the engine cache precision. The target
    // Embedding and output head share the target's selected storage format;
    // the pinned draft's own weights remain BF16.
    sealed class DFlashDraft : IDisposable
    {
        public const int Window = 2048, ContextFloats = Window * Hidden;
        static readonly int[] FeatureLayers = [1, 6, 11, 16, 22, 27, 32, 37];
        readonly QwenEngine e;
        readonly Checkpoint weights;
        public readonly GpuBuffer Context;
        GpuBuffer features, projected;
        GpuBuffer? packedFcInput, packedFcWeights;
        int featureCapacity = 512;
        readonly GpuBuffer contextInput, contextK, contextV;
        readonly GpuBuffer hidden, norm, prepared, dynamic, output, q, k, v, attention, scores, gate, up, activation, logits, selector, tokens, proposals;
        readonly GpuBuffer[] keys = new GpuBuffer[6], values = new GpuBuffer[6];
        int contextEnd, cachedThrough;
        public DFlashDraft(QwenEngine engine, string directory, string manifest)
        {
            e = engine;
            using var metadata = JsonDocument.Parse(File.ReadAllBytes(manifest));
            if (Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(Path.Combine(directory, "config.json")))) != metadata.RootElement.GetProperty("config_sha256").GetString())
                throw new InvalidDataException("Draft config hash mismatch");
            weights = new Checkpoint(e.gpu, directory, manifest);
            GpuBuffer B(int count) => e.Buffer(count);
            Context = B(ContextFloats); features = B(512 * Hidden * 8); projected = B(512 * Hidden);
            contextInput = B(512 * Hidden); contextK = B(512 * 1024); contextV = B(512 * 1024);
            hidden = B(8 * Hidden); norm = B(8 * Hidden); prepared = B(8 * Hidden); dynamic = B(8 * 512); output = B(8 * Hidden);
            q = B(8 * 4096); k = B(8 * 1024); v = B(8 * 1024); attention = B(8 * 4096); scores = B(8 * 32 * (Window + 8));
            gate = B(8 * 6144); up = B(8 * 6144); activation = B(8 * 6144); logits = B(8 * Vocabulary); selector = B(8 * 256); tokens = B(8); proposals = B(8);
            for (int l = 0; l < 6; l++) { keys[l] = B(Window * 1024 * e.DraftKvElementBytes / 4); values[l] = B(Window * 1024 * e.DraftKvElementBytes / 4); }
        }
        public void Reset(int position = 0) { contextEnd = position; cachedThrough = Math.Max(0, position - Window); }
        public void EnsureFeatureCapacity(int count)
        {
            if (count <= featureCapacity) return;
            features.Dispose(); projected.Dispose();
            features = e.Buffer(count * Hidden * 8); projected = e.Buffer(count * Hidden); featureCapacity = count;
            // Resized lazily on the next optimized FC dispatch; tiny draft paths need no packing.
            packedFcInput?.Dispose(); packedFcInput = null;
        }
        public void Capture(int layer, GpuBuffer source, int count)
        {
            int feature = Array.IndexOf(FeatureLayers, layer);
            if (feature < 0) return;
            e.gpu.Dispatch("draft_capture", new() { M = (uint)count, A = (uint)feature }, (uint)(count * Hidden), 128, source, features);
        }
        public void CommitFeatures(int count, int position)
        {
            Project("fc", features, projected, count);
            Normalize("hidden_norm", projected, projected, Hidden, count);
            e.gpu.Dispatch("draft_context_store", new() { M = (uint)count, Position = (uint)position }, (uint)(count * Hidden), 128, projected, Context);
            contextEnd = position + count;
        }
        void Project(string name, GpuBuffer input, GpuBuffer destination, int count)
        {
            var tensor = weights[name + ".weight"];
            if (tensor.Dtype != "BF16") throw new InvalidDataException("Draft projections must remain BF16");
            if (count <= VerificationCapacity)
            {
                e.gpu.Dispatch("matrix_product_tiny", new() { N = (uint)tensor.Shape[0], M = (uint)count, K = (uint)tensor.Shape[1], Kind = 1 }, (uint)tensor.Shape[0] * 32, 128, input, tensor.Binding, e.dummy, destination);
                return;
            }
            if (name == "fc" && e.FastDraftFc)
            {
                int n = tensor.Shape[0], k = tensor.Shape[1];
                packedFcInput ??= e.Buffer(checked(featureCapacity * k / 2));
                packedFcWeights ??= e.Buffer(checked(n * k / 2));
                var p = new KernelArgs { N = (uint)n, M = (uint)count, K = (uint)k, Kind = 1 };
                // Reuse bounded scratch, keeping the BF16 checkpoint and complete context work.
                e.gpu.Dispatch("projection_unpack_half", p, p.N * p.K / 4, 128, tensor.Binding, e.dummy, packedFcWeights);
                e.gpu.Dispatch("projection_input_half", p, p.M * p.K / 4, 128, input, packedFcInput);
                e.gpu.Dispatch2D("matrix_product_predecoded", p, ((p.N + 63) / 64) * 128, (p.M + 63) / 64, 128, packedFcInput, packedFcWeights, destination);
                return;
            }
            e.gpu.Dispatch2D("matrix_product", new() { N = (uint)tensor.Shape[0], M = (uint)count, K = (uint)tensor.Shape[1], Kind = 1 }, (uint)((tensor.Shape[0] + 63) / 64 * 256), (uint)((count + 31) / 32), 256, input, tensor.Binding, e.dummy, destination);
        }
        void Normalize(string name, GpuBuffer input, GpuBuffer destination, int size, int rows)
        {
            e.gpu.Dispatch("rmsnorm", new() { N = (uint)size, M = (uint)rows, Epsilon = 1e-6f }, (uint)(rows * 256), 256, input, weights[name + ".weight"].Binding, destination, e.dummy);
        }
        void Convolve(string name, GpuBuffer input, GpuBuffer destination, int count, bool finish)
        {
            if (!finish) Project(name + ".kernel_projection", input, dynamic, count);
            e.gpu.Dispatch("draft_dynamic_conv", new() { M = (uint)count, A = finish ? 1u : 0u }, (uint)(count * Hidden), 128, input, dynamic, weights[name + ".base_kernel"].Binding, destination);
        }
        void UpdateContext(int layer)
        {
            string path = $"layers.{layer}.self_attn";
            for (int start = Math.Max(cachedThrough, Math.Max(0, contextEnd - Window)); start < contextEnd; start += 512)
            {
                int count = Math.Min(512, contextEnd - start);
                e.gpu.Dispatch("draft_context_load", new() { M = (uint)count, Position = (uint)start }, (uint)(count * Hidden), 128, Context, contextInput);
                Project(path + ".k_proj", contextInput, contextK, count); Project(path + ".v_proj", contextInput, contextV, count);
                Normalize(path + ".k_norm", contextK, contextK, 128, count * 8);
                e.gpu.Dispatch("draft_context_kv", new() { M = (uint)count, Position = (uint)start, Theta = 10_000_000f, Flags = e.DraftKvFlags }, (uint)(count * 1024), 128, contextK, contextV, keys[layer], values[layer]);
            }
        }
        public int[] Propose(int anchor, int position, int count)
        {
            if (position != contextEnd || count is < 2 or > 8) throw new InvalidOperationException("Draft context/position mismatch");
            var tokenValues = tokens.Values<int>(count); tokenValues.Fill(248077); tokenValues[0] = anchor;
            e.gpu.Begin();
            e.BatchEmbed(tokens, hidden, count);
            for (int layer = 0; layer < 6; layer++)
            {
                string path = $"layers.{layer}";
                UpdateContext(layer);
                Normalize(path + ".input_layernorm", hidden, norm, Hidden, count);
                Convolve(path + ".attention_conv", norm, prepared, count, false);
                Project(path + ".self_attn.q_proj", prepared, q, count); Project(path + ".self_attn.k_proj", prepared, k, count); Project(path + ".self_attn.v_proj", prepared, v, count);
                Normalize(path + ".self_attn.q_norm", q, q, 128, count * 32); Normalize(path + ".self_attn.k_norm", k, k, 128, count * 8);
                var p = new KernelArgs { M = (uint)count, Position = (uint)position, A = (uint)Math.Min(Window, position), K = (uint)(Math.Min(Window, position) + count), Theta = 10_000_000f, Flags = e.DraftKvFlags };
                e.gpu.Dispatch("draft_rotate", p, (uint)(count * 2048), 128, q, k);
                e.gpu.Dispatch("draft_attention_scores", p, (uint)(count * 32) * p.K * 32, 128, q, k, keys[layer], scores);
                var softmax = p; softmax.Position = p.K - 1;
                e.gpu.Dispatch("attention_softmax", softmax, (uint)(count * 32 * 256), 256, scores);
                e.gpu.Dispatch("draft_attention_values", p, (uint)(count * 4096), 128, scores, v, values[layer], attention);
                Project(path + ".self_attn.o_proj", attention, output, count);
                Convolve(path + ".attention_conv", output, prepared, count, true);
                Add(hidden, prepared, count);
                Normalize(path + ".post_attention_layernorm", hidden, norm, Hidden, count);
                Convolve(path + ".mlp_conv", norm, prepared, count, false);
                Project(path + ".mlp.gate_proj", prepared, gate, count); Project(path + ".mlp.up_proj", prepared, up, count);
                e.gpu.Dispatch("silu_product", new() { N = (uint)(count * 6144) }, (uint)(count * 6144), 128, gate, up, activation);
                Project(path + ".mlp.down_proj", activation, output, count);
                Convolve(path + ".mlp_conv", output, prepared, count, true); Add(hidden, prepared, count);
            }
            Normalize("norm", hidden, norm, Hidden, count);
            e.BatchProject("lm_head", norm, logits, count);
            Project("candidate_selector.hidden_projection", norm, selector, count);
            e.gpu.Dispatch("draft_select", new() { M = (uint)count, Position = (uint)anchor, N = Vocabulary }, 256, 256, logits, selector, weights["candidate_selector.predecessor_codebook"].Binding, weights["candidate_selector.successor_codebook"].Binding, proposals);
            e.gpu.End(); cachedThrough = position;
            return proposals.Values<int>(count - 1).ToArray();
        }
        void Add(GpuBuffer residual, GpuBuffer update, int count) => e.gpu.Dispatch("add", new() { N = (uint)(count * Hidden) }, (uint)(count * Hidden), 128, residual, update, residual);
        public void ExportSyntheticFixture(string directory, int context)
        {
            Reset();
            if (context is < 1 or > 4096) throw new ArgumentOutOfRangeException(nameof(context));
            const int width = 8, anchor = 1234;
            for (int start = 0; start < context; start += 512)
            {
                int count = Math.Min(512, context - start);
                var data = features.Values<float>(count * Hidden * 8);
                for (int i = 0; i < data.Length; i++) data[i] = ((((start * Hidden * 8 + i) * 13) % 2001) - 1000) * 0.001f;
                e.gpu.Begin(); CommitFeatures(count, start); e.gpu.End();
            }
            int[] first = Propose(anchor, context, width);
            float[] hiddenOutput = norm.Values<float>(width * Hidden).ToArray();
            int[] cached = Propose(anchor, context, width);
            if (!first.SequenceEqual(cached) || !hiddenOutput.SequenceEqual(norm.Values<float>(width * Hidden).ToArray()))
                throw new Exception("Draft context-KV reuse changed output");
            void Write(string name, ReadOnlySpan<float> values) => File.WriteAllBytes(Path.Combine(directory, name), System.Runtime.InteropServices.MemoryMarshal.AsBytes(values).ToArray());
            Write("draft-hidden.f32", hiddenOutput);
            int retained = Math.Min(Window, context);
            var logicalContext = new float[retained * Hidden];
            for (int row = 0; row < retained; row++) Context.Values<float>(Hidden, (ulong)((context - retained + row) % Window) * Hidden * 4).CopyTo(logicalContext.AsSpan(row * Hidden, Hidden));
            Write("draft-context.f32", logicalContext);
            File.WriteAllText(Path.Combine(directory, "draft-fixture.json"), JsonSerializer.Serialize(new { context, width, anchor, proposals = first, context_kv_reuse_bitwise = true }));
            Console.WriteLine(JsonSerializer.Serialize(new { draft_fixture = directory, proposals = first, context_kv_reuse_bitwise = true }));
        }
        public void Dispose() => weights.Dispose();
    }
}
