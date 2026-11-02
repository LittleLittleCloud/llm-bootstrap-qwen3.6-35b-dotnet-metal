using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace Bootstrap.Engine;

internal sealed partial class QwenEngine
{
    sealed record PrefillVariant(string Name, string Precision, string Attention, bool Wide,
        bool Optimized = false, int Batch = 512, int Tile = 16, bool Predecoded = false, bool ExpertPredecoded = false, bool Gathered = false, bool Causal = false, bool Adaptive = false);

    // Alternate configuration order; warm timings exclude model load and tokenization.
    public void ComparePrefillVariants(string prompt, int runs, string expectedAnswer, bool fast = false,
        bool projections = false, bool batches = false, bool tuned = false, bool experts = false, bool final = false, bool throughput = false)
    {
        if (runs < 1) throw new ArgumentOutOfRangeException(nameof(runs));
        if (new[] { fast, projections, batches, tuned, experts, final, throughput }.Count(x => x) > 1) throw new ArgumentException("Select one comparison family");
        PrefillVariant[] variants = throughput ? [
            new("baseline_fp16_kv", "fp16", "fp16-device", true, true, 2048, 64, true),
            new("gathered_causal", "fp16", "fp16-device", true, true, 2048, 64, true, false, true, true),
            new("gathered_causal_adaptive", "fp16", "fp16-device", true, true, 2048, 64, true, false, true, true, true)
        ] : final ? [
            new("previous_fast_prefill", "fp16", "fp16", true),
            new("tuned_prefill", "fp16", "fp16-device", true, true, 2048, 64, true)
        ] : experts ? [
            new("predecoded_dense", "fp16", "fp16-opt", true, true, 2048, 64, true),
            new("device_attention_dense", "fp16", "fp16-device", true, true, 2048, 64, true),
            new("device_attention_dense_experts", "fp16", "fp16-device", true, true, 2048, 64, true, true)
        ] : tuned ? [
            new("previous_fast_prefill", "fp16", "fp16", true),
            new("tuned512", "fp16", "fp16-opt", true, true),
            new("tuned1024", "fp16", "fp16-opt", true, true, 1024, 32),
            new("tuned2048", "fp16", "fp16-opt", true, true, 2048, 64),
            new("predecoded2048", "fp16", "fp16-opt", true, true, 2048, 64, true)
        ] : batches ? [
            new("batch512_tile16", "fp16", "fp16", true, true),
            new("batch1024_tile32", "fp16", "fp16", true, true, 1024, 32),
            new("batch2048_tile64", "fp16", "fp16", true, true, 2048, 64)
        ] : projections ? [
            new("previous_fast_prefill", "fp16", "fp16", true),
            new("optimized_projections", "fp16", "fp16", true, true)
        ] : fast ? [
            new("fp16+tiled_attention", "fp16", "tiled", false),
            new("fp16+fp16_attention", "fp16", "fp16", false),
            new("fp16+fp16_attention+wide", "fp16", "fp16", true),
            new("fp16+fused-fp16_attention+wide", "fp16", "fused-fp16", true)
        ] : [
            new("legacy", "legacy", "legacy", false),
            new("fp16", "fp16", "legacy", false), new("bf16", "bf16", "legacy", false),
            new("legacy+tiled_attention", "legacy", "tiled", false),
            new("fp16+tiled_attention", "fp16", "tiled", false), new("bf16+tiled_attention", "bf16", "tiled", false)
        ];
        PrefillBatchSize = variants.Max(v => v.Batch); // Reserve once for all paired candidates.
        string Name(PrefillVariant variant) => variant.Name;
        void Configure(PrefillVariant variant)
        {
            AdaptiveExpertTiles = variant.Adaptive;
            CausalDeviceAttention = variant.Causal;
            GatheredExperts = variant.Gathered;
            TensorOpsPrecision = variant.Precision; AttentionMode = variant.Attention; WidePrefill = variant.Wide;
            OptimizedProjections = variant.Optimized; PredecodedProjections = variant.Predecoded; PredecodedExperts = variant.ExpertPredecoded; PrefillBatchSize = variant.Batch; ExpertTileRows = variant.Tile;
            ClearPrefixCache();
        }
        var tokens = Tokenizer.SimpleChat(prompt);
        Console.WriteLine(JsonSerializer.Serialize(new { type = "fixture", kv_cache_precision = KvCachePrecision, prompt_tokens = tokens.Length,
            context_window = ContextWindow, prefix_caching = PrefixCachingEnabled, draft_enabled = draft is not null,
            sha256 = Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(prompt))),
            expected_answer = expectedAnswer, variants = variants.Select(Name), runs,
            configurations = variants, accumulation = "FP32" }));
        var smoke = Tokenizer.SimpleChat(string.Concat(Enumerable.Repeat("This sentence is harmless context. ", 100)) + "Reply with only the number: what is 2 + 2?");
        float[]? reference = null;
        foreach (var variant in variants)
        {
            Configure(variant);
            var result = Generate(smoke, 1, false);
            if (throughput && result.Text.Trim() != "4") throw new Exception($"{variant.Name} failed arithmetic smoke");
            var current = logits.Values<float>(Vocabulary).ToArray(); reference ??= current;
            double squaredError = 0, squaredReference = 0, maxError = 0;
            for (int i = 0; i < Vocabulary; i++)
            {
                if (!float.IsFinite(current[i])) throw new Exception($"{Name(variant)} produced nonfinite smoke logits");
                double error = (double)current[i] - reference[i];
                maxError = Math.Max(maxError, Math.Abs(error)); squaredError += error * error; squaredReference += (double)reference[i] * reference[i];
            }
            Console.WriteLine(JsonSerializer.Serialize(new { type = "model_smoke", variant = Name(variant), prompt_tokens = smoke.Length,
                text = result.Text, answer_correct = result.Text.Trim() == "4", tokens = result.Tokens,
                logit_max_abs_error = maxError, logit_relative_l2_error = Math.Sqrt(squaredError / squaredReference) }));
        }
        using var process = Process.GetCurrentProcess();
        var firstTokens = new Dictionary<string, int>();
        // Run zero checks the complete long-context answer and primes every path.
        // Warm measurements generate only the first token and exclude all samples.
        for (int run = 0; run <= runs; run++)
        {
            int[] order = Enumerable.Range(0, variants.Length).ToArray();
            if (run % 2 == 1) Array.Reverse(order);
            else if (run != 0) order = order.Skip(2).Concat(order.Take(2)).ToArray();
            foreach (int index in order)
            {
                var variant = variants[index]; Configure(variant); gpu.ResetWallTiming();
                var cpu = process.TotalProcessorTime; var timer = Stopwatch.StartNew();
                var result = Generate(tokens, run == 0 ? 32 : 1, run != 0);
                double wall = timer.Elapsed.TotalMilliseconds, cpuMs = (process.TotalProcessorTime - cpu).TotalMilliseconds;
                if (throughput && run == 0 && result.Text.Trim() != expectedAnswer) throw new Exception($"{variant.Name} failed full needle answer");
                int first = result.Tokens[0]; string name = Name(variant);
                if (run == 0) firstTokens.Add(name, first);
                if (first != firstTokens[name]) throw new Exception($"{name} changed first token across repeated runs");
                int nonfinite = 0; foreach (float value in logits.Values<float>(Vocabulary)) if (!float.IsFinite(value)) nonfinite++;
                if (nonfinite != 0) throw new Exception($"{name} produced nonfinite long-prompt logits");
                Console.WriteLine(JsonSerializer.Serialize(new { type = "prefill_variant", run, variant = name,
                    mode = run == 0 ? "cold_full_answer" : "warm_uncached", projection_precision = variant.Precision, tiled_attention = variant.Attention != "legacy", attention_mode = variant.Attention, wide_prefill = WidePrefill, optimized_projections = OptimizedProjections, predecoded_projections = PredecodedProjections, predecoded_experts = PredecodedExperts, gathered_experts = GatheredExperts, causal_device_attention = CausalDeviceAttention, adaptive_expert_tiles = AdaptiveExpertTiles, prefill_batch_size = PrefillBatchSize, expert_tile_rows = ExpertTileRows,
                    wall_ms = wall, process_cpu_ms = cpuMs, prompt_tokens = tokens.Length,
                    prefill_tokens_per_second = run == 0 ? (double?)null : tokens.Length / (wall / 1000),
                    first_token = first, first_token_matches_baseline = first == firstTokens[Name(variants[0])],
                    text = run == 0 ? result.Text : null, answer_correct = run == 0 ? result.Text.Trim() == expectedAnswer : (bool?)null,
                    generated_tokens = result.Tokens.Length, metal_buffer_bytes = AllocatedBytes, timing = gpu.ReadWallTiming() }));
            }
        }
        if (final)
        {
            CompareDecodeVariants(runs);
            Configure(variants[1]); VerificationSelfTest();
            PrefixCachingEnabled = true; ClearPrefixCache(); PrefixCacheSelfTest.Run(this);
        }
    }
}
