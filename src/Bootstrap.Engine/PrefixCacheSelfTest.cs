using System.Text.Json;

namespace Bootstrap.Engine;

internal static class PrefixCacheSelfTest
{
    public static void Run(QwenEngine engine)
    {
        int checks = 0;
        var cases = new List<object>();
        void Check(bool condition, string name) { if (!condition) throw new Exception("Prefix cache: " + name); checks++; }
        int[] Prompt(string text) => engine.Tokenizer.SimpleChat(text);
        Generation Run(int[] prompt, bool ignoreEos = false, int maximum = 12) => engine.Generate(prompt, maximum, ignoreEos);
        void CompareCold(string name, int[] prompt, Generation warm, bool ignoreEos = false, int maximum = 12)
        {
            engine.PrefixCachingEnabled = false;
            var cold = Run(prompt, ignoreEos, maximum);
            Check(cold.CachedTokens == 0, name + " disabled-cache accounting");
            Check(warm.Tokens.SequenceEqual(cold.Tokens) && warm.Text == cold.Text && warm.FinishReason == cold.FinishReason, name + " warm/cold generated tokens");
            cases.Add(new { name, prompt_tokens = prompt.Length, cached_tokens = warm.CachedTokens, output_tokens = warm.Tokens.Length, matched = true });
            engine.PrefixCachingEnabled = true;
            Console.Error.WriteLine("Prefix cache case passed: " + name);
        }
        var simple = Prompt("Reply with only the number: what is 2 + 2?");
        var initial = Run(simple);
        Check(initial.CachedTokens == 0, "initial miss");
        string initialState = engine.PrefixTestStateHash(simple.Length + initial.Tokens.Length - 1);
        var exact = Run(simple);
        Check(engine.PrefixTestStateHash(simple.Length + exact.Tokens.Length - 1) == initialState, "exact-hit recurrent/conv/KV/logits bitwise equality after decode");
        Check(exact.CachedTokens == simple.Length, "exact full hit");
        CompareCold("identical", simple, exact);

        Run(simple);
        var noEos = Run(simple, true, 16);
        Check(noEos.CachedTokens == simple.Length, "ignore_eos reuses logits");
        CompareCold("changed-ignore-eos", simple, noEos, true, 16);
        Run(simple, true, 16);
        CompareCold("changed-back-to-eos", simple, Run(simple));

        int[] conversation = ChatFormatting.Format(engine.Tokenizer, new ChatInput("chat",
            [new ChatMessage("user", "Reply with only the number: what is 2 + 2?"), new ChatMessage("assistant", "4"), new ChatMessage("user", "Now what is 3 + 3? Reply with only the number.")], new(), 12, false, false, [], false, false));
        Run(simple);
        int expected = engine.CachedPrefixTokens(conversation);
        Check(expected > 0 && expected < simple.Length, "conversation reuses message checkpoint despite assistant template change");
        var appended = Run(conversation);
        Check(appended.CachedTokens == expected, "append accounting");
        CompareCold("conversation-append", conversation, appended);

        // Longer than the first KV allocation: exercise a checkpoint through growth.
        string padding = string.Concat(Enumerable.Repeat("This sentence is harmless context. ", 700));
        var longPrompt = Prompt(padding + "Reply with only the number: what is 2 + 2?");
        Check(longPrompt.Length > 4096, "fixture crosses KV growth");
        var growthSeed = longPrompt[..4090];
        Run(growthSeed);
        var grown = Run(longPrompt);
        Check(grown.CachedTokens == growthSeed.Length && engine.CacheCapacity > 4096, "restored prefix survives KV growth");
        CompareCold("append-through-kv-growth", longPrompt, grown);
        var longCold = Run(longPrompt);
        string longState = engine.PrefixTestStateHash(longPrompt.Length + longCold.Tokens.Length - 1);
        var longHit = Run(longPrompt);
        Check(longHit.CachedTokens == longPrompt.Length, "long exact hit");
        Check(engine.PrefixTestStateHash(longPrompt.Length + longHit.Tokens.Length - 1) == longState, "long exact-hit state bitwise equality");
        CompareCold("long-identical", longPrompt, longHit);

        Run(longPrompt);
        var branch = Prompt(padding + "Reply with only the number: what is 3 + 3?");
        var branched = Run(branch);
        Check(branched.CachedTokens == 4096, "branch falls back to block checkpoint");
        CompareCold("branch", branch, branched);

        Run(longPrompt);
        var shortPrefix = longPrompt[..4100];
        var shortened = Run(shortPrefix);
        Check(shortened.CachedTokens == 4096, "shorten restores earlier state");
        CompareCold("shortened", shortPrefix, shortened);

        Run(longPrompt);
        var blockPrefix = longPrompt[..4096];
        var blockHit = Run(blockPrefix);
        Check(blockHit.CachedTokens == 4096, "exact intermediate checkpoint logits");
        CompareCold("exact-block-boundary", blockPrefix, blockHit);

        Run(simple);
        var busted = (int[])simple.Clone(); busted[0] = engine.Tokenizer.EndOfText;
        Check(engine.CachedPrefixTokens(busted) == 0, "first-token change misses");
        CompareCold("busted", busted, Run(busted));

        Run(simple);
        using (var cancelled = new CancellationTokenSource())
        {
            cancelled.Cancel();
            try { engine.Generate(simple, 12, false, cancellation: cancelled.Token); throw new Exception("Cancellation accepted"); }
            catch (OperationCanceledException) { checks++; }
        }
        Check(engine.CachedPrefixTokens(simple) == 0, "pre-cancelled request invalidates cache");
        Run(simple);
        using (var cancelled = new CancellationTokenSource())
        {
            try { engine.Generate(simple, 32, true, _ => { cancelled.Cancel(); return true; }, cancelled.Token); throw new Exception("Decode cancellation accepted"); }
            catch (OperationCanceledException) { checks++; }
        }
        Check(engine.CachedPrefixTokens(simple) == 0, "decode cancellation invalidates cache");
        CompareCold("after-cancellation", simple, Run(simple));

        Run(simple);
        try { engine.Generate(simple, 16, true, _ => throw new IOException("synthetic disconnected writer")); throw new Exception("Writer failure swallowed"); }
        catch (IOException) { checks++; }
        Check(engine.CachedPrefixTokens(simple) == 0, "callback failure invalidates cache");
        Run(simple);
        engine.Reset();
        Check(engine.CachedPrefixTokens(simple) == 0, "explicit reset invalidates cache");
        Run(simple);
        engine.BatchedPrefill = false;
        Check(engine.CachedPrefixTokens(simple) == 0, "execution-mode change invalidates cache");
        Run(simple);
        CompareCold("sequential-exact", simple, Run(simple));
        engine.BatchedPrefill = true;
        int[] mutable = (int[])simple.Clone(); Run(mutable); mutable[0] = engine.Tokenizer.EndOfText;
        Check(engine.CachedPrefixTokens(simple) == simple.Length && engine.CachedPrefixTokens(mutable) == 0, "token key owns its array");
        Check(engine.PrefixCheckpointCount <= 4 && engine.PrefixCacheBytes <= 256UL * 1024 * 1024, "snapshot memory bounded");
        Console.WriteLine(JsonSerializer.Serialize(new { passed = checks, cases, snapshot_bytes = engine.PrefixCacheBytes, metal_buffer_bytes = engine.AllocatedBytes }));
    }
}

// Only used by the real-model diagnostic command; no state leaves the process.
internal sealed partial class QwenEngine
{
    internal string PrefixTestStateHash(int consumed)
    {
        using var hash = System.Security.Cryptography.IncrementalHash.CreateHash(System.Security.Cryptography.HashAlgorithmName.SHA256);
        void Add(GpuBuffer buffer, int bytes) => hash.AppendData(buffer.Values<byte>(bytes));
        for (int layer = 0; layer < 40; layer++)
            if ((layer + 1) % 4 != 0)
            {
                Add(state[layer], RecurrentFloats * sizeof(float)); Add(convolution[layer], ConvolutionFloats * sizeof(float));
            }
            else
            {
                Add(keyCache[layer], consumed * 512 * sizeof(float)); Add(valueCache[layer], consumed * 512 * sizeof(float));
            }
        Add(logits, Vocabulary * sizeof(float));
        return Convert.ToHexStringLower(hash.GetHashAndReset());
    }
}
