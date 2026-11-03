using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
namespace Bootstrap.Engine;

internal sealed partial class QwenEngine
{
    // Local development checks, separate from the pinned browser evaluation.
    public void PackedExpertModelCheck(string shortPrompt, string longPrompt, string expected, int runs, string outputDirectory)
    {
        if (runs < 1) throw new ArgumentOutOfRangeException(nameof(runs));
        Directory.CreateDirectory(outputDirectory);
        var smoke = Tokenizer.SimpleChat(string.Concat(Enumerable.Repeat("This sentence is harmless context. ", 100)) + "Reply with only the number: what is 2 + 2?");
        ClearPrefixCache(); var smokeResult = Generate(smoke, 1, false);
        if (smokeResult.Text.Trim() != "4") throw new Exception("Arithmetic smoke failed");
        byte[] logitBytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes(logits.Values<float>(Vocabulary)).ToArray();
        foreach (float v in logits.Values<float>(Vocabulary)) if (!float.IsFinite(v)) throw new Exception("Nonfinite smoke logits");
        File.WriteAllBytes(Path.Combine(outputDirectory, "smoke-logits.f32"), logitBytes);
        Console.WriteLine(JsonSerializer.Serialize(new { type = "packed_model_smoke", packed_experts = PackedExperts, text = smokeResult.Text, logits_sha256 = Convert.ToHexStringLower(SHA256.HashData(logitBytes)) }));
        foreach (string prompt in new[] { shortPrompt, longPrompt })
        {
            var tokens = Tokenizer.SimpleChat(prompt);
            Console.WriteLine(JsonSerializer.Serialize(new { type = "fixture", prompt_tokens = tokens.Length, sha256 = Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(prompt))), packed_experts = PackedExperts, delta_mode = DeltaMode, delta_preparation = DeltaPreparation }));
            int? first = null;
            for (int run = 0; run <= runs; run++)
            {
                ClearPrefixCache(); gpu.ResetWallTiming(); var watch = Stopwatch.StartNew();
                var result = Generate(tokens, run == 0 ? 32 : 1, run != 0);
                double ms = watch.Elapsed.TotalMilliseconds;
                if (run == 0 && result.Text.Trim() != expected) throw new Exception("Full needle answer failed: " + result.Text);
                if (first is not null && result.Tokens[0] != first) throw new Exception("First token changed");
                first = result.Tokens[0];
                Console.WriteLine(JsonSerializer.Serialize(new { type = "packed_model_prefill", packed_experts = PackedExperts, run, mode = run == 0 ? "cold_full_answer" : "warm_uncached", prompt_tokens = tokens.Length, wall_ms = ms, prefill_tokens_per_second = run == 0 ? (double?)null : tokens.Length * 1000 / ms, text = run == 0 ? result.Text : null, first_token = first, metal_buffer_bytes = AllocatedBytes, timing = gpu.ReadWallTiming() }));
            }
        }
        var decode = Tokenizer.SimpleChat("Generate a continuous stream of lowercase English words separated by single spaces. Do not use punctuation, numbers, headings, explanations, or formatting. Begin immediately and continue generating until stopped.");
        for (int run = 0; run <= 2; run++)
        {
            ClearPrefixCache(); var watch = Stopwatch.StartNew(); double first = -1, last = 0;
            var result = Generate(decode, 256, true, _ => { last = watch.Elapsed.TotalMilliseconds; if (first < 0) first = last; return true; });
            if (result.Tokens.Length != 256) throw new Exception("Short decode ended early");
            Console.WriteLine(JsonSerializer.Serialize(new { type = "packed_model_decode", packed_experts = PackedExperts, run, output_tokens = result.Tokens.Length, tokens_per_second = 255000 / (last - first), speculation = LastSpeculationStats }));
        }
    }
}
