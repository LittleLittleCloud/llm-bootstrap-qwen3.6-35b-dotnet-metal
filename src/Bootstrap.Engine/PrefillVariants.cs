using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace Bootstrap.Engine;

internal sealed partial class QwenEngine
{
    // Paired development experiment: keep checkpoints, batch geometry and model
    // weights fixed; vary projection operand precision and attention tiling only.
    public void ComparePrefillVariants(string prompt, int runs, string expectedAnswer, bool fast = false)
    {
        if (runs < 1) throw new ArgumentOutOfRangeException(nameof(runs));
        var variants = fast
            ? new[] { ("fp16", "tiled", false), ("fp16", "fp16", false), ("fp16", "fp16", true), ("fp16", "fused-fp16", true) }
            : new[] { ("legacy", "legacy", false), ("fp16", "legacy", false), ("bf16", "legacy", false), ("legacy", "tiled", false), ("fp16", "tiled", false), ("bf16", "tiled", false) };
        string Name((string precision, string attention, bool wide) variant) => variant.precision + (variant.attention == "legacy" ? "" : "+" + variant.attention + "_attention") + (variant.wide ? "+wide" : "");
        void Configure((string precision, string attention, bool wide) variant)
        {
            TensorOpsPrecision = variant.precision; AttentionMode = variant.attention; WidePrefill = variant.wide; ClearPrefixCache();
        }
        var tokens = Tokenizer.SimpleChat(prompt);
        Console.WriteLine(JsonSerializer.Serialize(new { type = "fixture", prompt_tokens = tokens.Length,
            context_window = ContextWindow, prefix_caching = PrefixCachingEnabled, draft_enabled = draft is not null,
            sha256 = Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(prompt))),
            expected_answer = expectedAnswer, variants = variants.Select(Name), runs,
            projection_tiles = "unchanged", attention_precision = fast ? "varies by variant" : "FP32", accumulation = "FP32" }));
        var smoke = Tokenizer.SimpleChat(string.Concat(Enumerable.Repeat("This sentence is harmless context. ", 100)) + "Reply with only the number: what is 2 + 2?");
        float[]? reference = null;
        foreach (var variant in variants)
        {
            Configure(variant);
            var result = Generate(smoke, 1, false);
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
                int first = result.Tokens[0]; string name = Name(variant);
                if (run == 0) firstTokens.Add(name, first);
                if (first != firstTokens[name]) throw new Exception($"{name} changed first token across repeated runs");
                int nonfinite = 0; foreach (float value in logits.Values<float>(Vocabulary)) if (!float.IsFinite(value)) nonfinite++;
                if (nonfinite != 0) throw new Exception($"{name} produced nonfinite long-prompt logits");
                Console.WriteLine(JsonSerializer.Serialize(new { type = "prefill_variant", run, variant = name,
                    mode = run == 0 ? "cold_full_answer" : "warm_uncached", projection_precision = variant.Item1, tiled_attention = variant.Item2 != "legacy", attention_mode = variant.Item2, wide_prefill = variant.Item3,
                    wall_ms = wall, process_cpu_ms = cpuMs, prompt_tokens = tokens.Length,
                    prefill_tokens_per_second = run == 0 ? (double?)null : tokens.Length / (wall / 1000),
                    first_token = first, first_token_matches_baseline = first == firstTokens[Name(variants[0])],
                    text = run == 0 ? result.Text : null, answer_correct = run == 0 ? result.Text.Trim() == expectedAnswer : (bool?)null,
                    generated_tokens = result.Tokens.Length, metal_buffer_bytes = AllocatedBytes, timing = gpu.ReadWallTiming() }));
            }
        }
    }
}
