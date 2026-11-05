using System.Runtime.InteropServices;
using System.Text.Json;

namespace Bootstrap.Engine;

internal sealed partial class QwenEngine
{
    // Paired processes keep exactly one full model resident. Dump first-token
    // logits for an independent probability comparison across cache precisions.
    public void CheckKvQuality(string prompt10k, string prompt30k, string outputDirectory)
    {
        Directory.CreateDirectory(outputDirectory);
        string padding = string.Concat(Enumerable.Repeat("This sentence is harmless context. ", 100));
        var fixtures = new[] {
            ("needle10k", prompt10k, "73DY-DRV5-B8FJ"),
            ("needle30k", prompt30k, "73DY-DRV5-B8FJ"),
            ("arithmetic", padding + "Return only the integer result of 17 * 23.", "391"),
            ("sort", padding + "Sort these integers in ascending order: 19, -4, 7, 0, 12. Return only comma-separated integers with no spaces.", "-4,0,7,12,19"),
            ("extract", padding + "Record: name=Iris; city=Oslo; id=Q7M2. Return only the id value.", "Q7M2"),
            ("count", padding + "Count the exact word red in this list: red blue red green red. Return only the count as an integer.", "3"),
            ("override", padding + "An earlier note says the active code is OLD-17. The final correction says the active code is NEW-42. Return only the active code after the correction.", "NEW-42")
        };
        bool previousSpeculation = SpeculativeDecoding, previousCaching = PrefixCachingEnabled;
        try
        {
            PrefixCachingEnabled = true;
            foreach (var (name, text, expected) in fixtures)
            {
                int[] prompt = Tokenizer.SimpleChat(text);
                foreach (bool speculative in draft is null ? new[] { false } : new[] { false, true })
                {
                    SpeculativeDecoding = speculative; Reset();
                    Generate(prompt, 1, true);
                    var firstLogits = logits.Values<float>(Vocabulary).ToArray();
                    if (firstLogits.Any(x => !float.IsFinite(x))) throw new Exception("Nonfinite KV quality logits");
                    string stem = name + (speculative ? "-draft" : "-serial");
                    File.WriteAllBytes(Path.Combine(outputDirectory, stem + ".f32"), MemoryMarshal.AsBytes(firstLogits.AsSpan()).ToArray());
                    var result = Generate(prompt, 48, false);
                    bool passed = result.Text.Trim() == expected;
                    Console.WriteLine(JsonSerializer.Serialize(new { type = "kv_quality", fixture = name, kv = KvCachePrecision, speculative, prompt_tokens = prompt.Length,
                        expected, text = result.Text, passed, tokens = result.Tokens, cached_tokens = result.CachedTokens,
                        target_kv_bytes = TargetKvBytes, draft_kv_precision = DraftKvElementBytes == 2 ? "fp16" : "fp32", logits_file = stem + ".f32" }));
                    if (!passed || result.CachedTokens != prompt.Length) throw new Exception("KV quality/cache check failed: " + stem);
                }
            }
        }
        finally { SpeculativeDecoding = previousSpeculation; PrefixCachingEnabled = previousCaching; Reset(); }
    }
}
