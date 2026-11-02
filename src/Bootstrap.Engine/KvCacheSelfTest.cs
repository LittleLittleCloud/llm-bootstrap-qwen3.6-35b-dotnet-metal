using System.Diagnostics;
using System.Text.Json;
namespace Bootstrap.Engine;

internal sealed partial class QwenEngine
{
    // Development regression for the memory failure: verifier tapes + prefix
    // snapshots must coexist with full-model 32K KV and the large prefill workspace.
    public void KvCacheSelfTest()
    {
        if (ContextWindow < 32768 || draft is null) throw new ArgumentException("32K context and a draft are required");
        VerificationSelfTest();
        PrefixCachingEnabled = true; ClearPrefixCache(); PrefixCacheSelfTest.Run(this);
        const string code = "M7Q2-R4T8-V9K3", padding = "This sentence is harmless context. ";
        foreach (int depth in new[] { 5, 90 })
        {
            int before = 5000 * depth / 100;
            string text = string.Concat(Enumerable.Repeat(padding, before)) + "\nThe access code is " + code + ".\n" +
                string.Concat(Enumerable.Repeat(padding, 5000 - before)) + "\nReply with only the access code, exactly as written above.";
            int[] tokens = Tokenizer.SimpleChat(text);
            if (tokens.Length < 30000 || tokens.Length + 64 > 32768) throw new Exception("Unexpected long-context fixture size");
            ClearPrefixCache(); gpu.ResetWallTiming(); var timer = Stopwatch.StartNew(); double firstMs = 0;
            var result = Generate(tokens, 64, false, _ => { if (firstMs == 0) firstMs = timer.Elapsed.TotalMilliseconds; return true; });
            double wall = timer.Elapsed.TotalMilliseconds;
            if (result.Text.Trim() != code || result.CachedTokens != 0 || CacheCapacity < 32768) throw new Exception($"Long KV fixture failed: {depth}%: {result.Text}");
            for (int repeat = 0; repeat < 2; repeat++)
            {
                var cached = Generate(tokens, 64, false);
                if (cached.CachedTokens != tokens.Length || !cached.Tokens.SequenceEqual(result.Tokens)) throw new Exception("Long KV prefix reuse changed output");
            }
            Console.WriteLine(JsonSerializer.Serialize(new { type = "long_kv_fixture", kv_cache_precision = KvCachePrecision, depth_percent = depth,
                prompt_tokens = tokens.Length, answer = result.Text, cached_repeats_passed = 2, wall_ms = wall, ttft_ms = firstMs,
                effective_prefill_tok_s = tokens.Length * 1000 / firstMs, cache_capacity = CacheCapacity, target_kv_bytes = TargetKvBytes,
                metal_buffer_bytes = AllocatedBytes, prefix_snapshot_bytes = PrefixCacheBytes }));
        }
        int[] shortPrompt = Tokenizer.SimpleChat("Reply with only the number: what is 2 + 2?");
        var first = Generate(shortPrompt, 16, false); var second = Generate(shortPrompt, 16, false);
        if (first.Text.Trim() != "4" || second.CachedTokens != shortPrompt.Length || !first.Tokens.SequenceEqual(second.Tokens)) throw new Exception("Short cache request after 32K growth failed");
        Console.WriteLine(JsonSerializer.Serialize(new { type = "kv_cache_self_test", status = "passed", kv_cache_precision = KvCachePrecision,
            target_kv_bytes = TargetKvBytes, metal_buffer_bytes = AllocatedBytes, short_cache_after_32k = "passed" }));
    }
}
