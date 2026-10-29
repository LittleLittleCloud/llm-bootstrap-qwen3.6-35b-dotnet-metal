using System.Text.Json;
using System.Text;
using System.Security.Cryptography;

namespace Bootstrap.Engine;

internal sealed partial class QwenEngine : IDisposable
{
    const int Hidden = 2048, Context = 4096, Vocabulary = 248320;
    readonly Metal gpu;
    readonly Checkpoint weights;
    readonly List<GpuBuffer> owned = new();
    readonly GpuBuffer x, norm, attention, projection, qkv, z, a, b, conv, q, k, rotated, qgate, v, scores, router, ids, rates, expertActivation, expertOutput, sharedGate, sharedUp, sharedActivation, sharedOutput, sharedWeight, logits, next, dummy;
    readonly GpuBuffer[] state = new GpuBuffer[40], convolution = new GpuBuffer[40], keyCache = new GpuBuffer[40], valueCache = new GpuBuffer[40], descriptors = new GpuBuffer[40];
    public ByteBpe Tokenizer { get; }
    public ulong AllocatedBytes => gpu.AllocatedBytes;
    public bool BatchedPrefill { get; set; } = true;
    GpuBuffer Buffer(int floats) { var result = gpu.Allocate(checked((ulong)floats * 4)); owned.Add(result); return result; }
    public QwenEngine(string directory, string kernelPath, string manifestPath)
    {
        using var manifest = JsonDocument.Parse(File.ReadAllBytes(manifestPath));
        Tokenizer = new ByteBpe(Path.Combine(directory, "tokenizer.json"), manifest.RootElement.GetProperty("tokenizer_sha256").GetString()!);
        if (Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(Path.Combine(directory, "chat_template.jinja")))) != manifest.RootElement.GetProperty("chat_template_sha256").GetString()) throw new InvalidDataException("Chat-template hash mismatch");
        using var config = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(directory, "config.json")));
        var geometry = config.RootElement.GetProperty("text_config");
        if (geometry.GetProperty("hidden_size").GetInt32() != Hidden || geometry.GetProperty("num_hidden_layers").GetInt32() != 40 || geometry.GetProperty("num_experts").GetInt32() != 256 || geometry.GetProperty("num_experts_per_tok").GetInt32() != 8 || geometry.GetProperty("vocab_size").GetInt32() != Vocabulary) throw new NotSupportedException("Unexpected model geometry");
        gpu = new Metal(File.ReadAllText(kernelPath));
        weights = new Checkpoint(gpu, directory, manifestPath);
        x = Buffer(Hidden); norm = Buffer(Hidden); attention = Buffer(4096); projection = Buffer(Hidden); qkv = Buffer(8192); z = Buffer(4096); a = Buffer(32); b = Buffer(32); conv = Buffer(8192); q = Buffer(4096); k = Buffer(512 * 4); rotated = Buffer(4096); qgate = Buffer(4096); v = Buffer(512); scores = Buffer(16 * Context); router = Buffer(256); ids = Buffer(8); rates = Buffer(8); expertActivation = Buffer(8 * 512); expertOutput = Buffer(8 * Hidden); sharedGate = Buffer(512); sharedUp = Buffer(512); sharedActivation = Buffer(512); sharedOutput = Buffer(Hidden); sharedWeight = Buffer(1); logits = Buffer(Vocabulary); next = Buffer(1); dummy = Buffer(1);
        for (int layer = 0; layer < 40; layer++)
        {
            if ((layer + 1) % 4 != 0) { state[layer] = Buffer(32 * 128 * 128); convolution[layer] = Buffer(3 * 8192); }
            else { keyCache[layer] = Buffer(Context * 512); valueCache[layer] = Buffer(Context * 512); }
            descriptors[layer] = Buffer(256 * 6 * 2);
            var table = descriptors[layer].Values<ulong>(256 * 6);
            string prefix = Prefix(layer) + ".mlp.experts.";
            for (int e = 0; e < 256; e++)
            {
                var gate = weights[prefix + e + ".gate_proj.weight"]; var up = weights[prefix + e + ".up_proj.weight"]; var down = weights[prefix + e + ".down_proj.weight"];
                if (gate.Dtype != "F8_E4M3" || up.Dtype != "F8_E4M3" || down.Dtype != "F8_E4M3") throw new InvalidDataException("Experts must preserve official FP8 storage");
                table[e * 6] = gate.Offset; table[e * 6 + 1] = weights[prefix + e + ".gate_proj.weight_scale_inv"].Offset;
                table[e * 6 + 2] = up.Offset; table[e * 6 + 3] = weights[prefix + e + ".up_proj.weight_scale_inv"].Offset;
                table[e * 6 + 4] = down.Offset; table[e * 6 + 5] = weights[prefix + e + ".down_proj.weight_scale_inv"].Offset;
            }
        }
    }
    static string Prefix(int layer) => "model.language_model.layers." + layer;
    void Project(string name, GpuBuffer input, GpuBuffer output)
    {
        var tensor = weights[name + ".weight"];
        if (tensor.Shape.Length != 2) throw new InvalidDataException("Projection weight must be a matrix: " + name);
        var scaling = tensor.Kind == 0 ? weights[name + ".weight_scale_inv"].Binding : (Binding)dummy;
        gpu.Dispatch("gemv", new() { N = (uint)tensor.Shape[0], K = (uint)tensor.Shape[1], Kind = tensor.Kind }, checked((uint)tensor.Shape[0] * 32), 128, input, tensor.Binding, scaling, output);
    }
    void Normalize(GpuBuffer input, string name, GpuBuffer output, uint dimension = Hidden, uint heads = 1, bool offset = true, GpuBuffer? gate = null)
    {
        var tensor = weights[name + ".weight"];
        if (tensor.Dtype != "BF16") throw new InvalidDataException("Unsupported norm weight precision");
        gpu.Dispatch("rmsnorm", new() { N = dimension, M = heads, Epsilon = 1e-6f, Flags = (offset ? 1u : 0u) | (gate is not null ? 2u : 0u) }, heads * 256, 256, input, tensor.Binding, output, gate ?? dummy);
    }
    void Add(GpuBuffer residual, GpuBuffer output) => gpu.Dispatch("add", new() { N = Hidden }, Hidden, 128, residual, output, residual);
    public void Reset()
    {
        gpu.Begin();
        for (int l = 0; l < 40; l++)
            if ((l + 1) % 4 != 0) { gpu.Clear(state[l]); gpu.Clear(convolution[l]); }
            else { gpu.Clear(keyCache[l]); gpu.Clear(valueCache[l]); }
        gpu.End();
    }
    public int Forward(int token, int position, bool output, bool ignoreEos)
    {
        if (position >= Context) throw new ArgumentOutOfRangeException(nameof(position));
        gpu.Begin(); gpu.Dispatch("embedding", new() { N = Hidden, Position = (uint)token }, Hidden, 128, weights["model.language_model.embed_tokens.weight"].Binding, x);
        for (int layer = 0; layer < 40; layer++)
        {
            string prefix = Prefix(layer); Normalize(x, prefix + ".input_layernorm", norm);
            if ((layer + 1) % 4 != 0)
            {
                string path = prefix + ".linear_attn";
                Project(path + ".in_proj_qkv", norm, qkv); Project(path + ".in_proj_z", norm, z); Project(path + ".in_proj_a", norm, a); Project(path + ".in_proj_b", norm, b);
                gpu.Dispatch("conv", new() { N = 8192 }, 8192, 128, qkv, weights[path + ".conv1d.weight"].Binding, convolution[layer], conv);
                gpu.Dispatch("l2_qk", new(), 16 * 128, 128, conv, q, k);
                gpu.Dispatch("delta", new(), 32 * 128 * 32, 128, q, k, new Binding(conv, 4096 * 4), a, b, weights[path + ".A_log"].Binding, weights[path + ".dt_bias"].Binding, state[layer], attention);
                Normalize(attention, path + ".norm", attention, 128, 32, false, z);
                Project(path + ".out_proj", attention, projection);
            }
            else
            {
                string path = prefix + ".self_attn";
                Project(path + ".q_proj", norm, qkv); Project(path + ".k_proj", norm, k); Project(path + ".v_proj", norm, v);
                gpu.Dispatch("split_q", new(), 4096, 128, qkv, q, qgate);
                Normalize(q, path + ".q_norm", q, 256, 16); Normalize(k, path + ".k_norm", k, 256, 2);
                var attentionArgs = new KernelArgs { Position = (uint)position, K = Context, Theta = 10_000_000f };
                gpu.Dispatch("rope_cache", attentionArgs, 4096, 128, q, k, v, rotated, keyCache[layer], valueCache[layer]);
                gpu.Dispatch("attention_scores", attentionArgs, checked((uint)(16 * (position + 1) * 32)), 128, rotated, keyCache[layer], scores);
                gpu.Dispatch("attention_softmax", attentionArgs, 16 * 256, 256, scores);
                gpu.Dispatch("attention_values", attentionArgs, 4096, 128, scores, valueCache[layer], qgate, attention);
                Project(path + ".o_proj", attention, projection);
            }
            Add(x, projection); Normalize(x, prefix + ".post_attention_layernorm", norm);
            Project(prefix + ".mlp.gate", norm, router);
            gpu.Dispatch("router", new() { N = 256, M = 8 }, 32, 32, router, ids, rates);
            Project(prefix + ".mlp.shared_expert.gate_proj", norm, sharedGate); Project(prefix + ".mlp.shared_expert.up_proj", norm, sharedUp);
            gpu.Dispatch("silu_product", new() { N = 512 }, 512, 128, sharedGate, sharedUp, sharedActivation);
            Project(prefix + ".mlp.shared_expert.down_proj", sharedActivation, sharedOutput); Project(prefix + ".mlp.shared_expert_gate", norm, sharedWeight);
            gpu.Dispatch("expert_gate_up", new() { N = 512, K = Hidden, M = 8 }, 8 * 512 * 32, 128, norm, weights.Shard("layers-" + layer + ".safetensors"), descriptors[layer], ids, expertActivation);
            gpu.Dispatch("expert_down", new() { N = Hidden, K = 512, M = 8 }, 8 * Hidden * 32, 128, expertActivation, weights.Shard("layers-" + layer + ".safetensors"), descriptors[layer], ids, expertOutput);
            gpu.Dispatch("mix", new() { N = Hidden, M = 8 }, Hidden, 128, x, expertOutput, rates, sharedOutput, sharedWeight, x);
        }
        if (output)
        {
            Normalize(x, "model.language_model.norm", norm); Project("lm_head", norm, logits);
            gpu.Dispatch("argmax", new() { N = Vocabulary, Flags = ignoreEos ? 1u : 0u, A = (uint)Tokenizer.End, B = (uint)Tokenizer.EndOfText }, 256, 256, logits, next);
        }
        gpu.End(); return output ? next.Values<int>(1)[0] : -1;
    }
    public Generation Generate(int[] prompt, int maximum, bool ignoreEos, Func<string, bool>? onText = null, CancellationToken cancellation = default)
    {
        if (prompt.Length == 0 || maximum < 1 || prompt.Length + maximum > Context) throw new ArgumentException("Context window exceeded or empty request");
        Reset(); int predicted = -1; int position = 0;
        while (position < prompt.Length)
        {
            cancellation.ThrowIfCancellationRequested();
            int count = Math.Min(512, prompt.Length - position);
            if (BatchedPrefill && count >= 8)
            {
                predicted = PrefillBlock(prompt.AsSpan(position, count), position, position + count == prompt.Length, ignoreEos);
                position += count;
            }
            else predicted = Forward(prompt[position], position++, position == prompt.Length, ignoreEos);
        }
        var generated = new List<int>(); var text = new StringBuilder(); Decoder decoder = Encoding.UTF8.GetDecoder(); string finish = "length";
        for (int i = 0; i < maximum; i++)
        {
            cancellation.ThrowIfCancellationRequested(); generated.Add(predicted);
            if (!ignoreEos && (predicted == Tokenizer.End || predicted == Tokenizer.EndOfText)) { finish = "stop"; break; }
            byte[] bytes = Tokenizer.TokenBytes(predicted); char[] chars = new char[bytes.Length + 4]; int count = decoder.GetChars(bytes, chars, flush: false);
            string piece = new(chars, 0, count); text.Append(piece); if (piece.Length > 0 && onText is not null && !onText(piece)) { finish = "stop"; break; }
            if (i + 1 < maximum) predicted = Forward(predicted, position++, true, ignoreEos);
        }
        return new(text.ToString(), prompt.Length, generated.ToArray(), finish);
    }
    public void Dispose() { foreach (var buffer in owned) buffer.Dispose(); weights.Dispose(); gpu.Dispose(); }
}
internal sealed record Generation(string Text, int PromptTokens, int[] Tokens, string FinishReason);
