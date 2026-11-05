using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;

namespace Bootstrap.Engine;

internal sealed partial class QwenEngine
{
    // Hash only live ring rows; unused slots are not part of a short prefix.
    string DraftPrefillCheckpointHash()
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        int stateFloats = SnapshotFloats - (draft is null ? 0 : DFlashDraft.ContextFloats);
        foreach (var checkpoint in prefixCheckpoints.OrderBy(c => c.Position))
        {
            hash.AppendData(BitConverter.GetBytes(checkpoint.Position));
            hash.AppendData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(checkpoint.Data.AsSpan(0, stateFloats)));
            if (draft is not null)
                for (int row = Math.Max(0, checkpoint.Position - DFlashDraft.Window); row < checkpoint.Position; row++)
                    hash.AppendData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(
                        checkpoint.Data.AsSpan(stateFloats + row % DFlashDraft.Window * Hidden, Hidden)));
        }
        return Convert.ToHexStringLower(hash.GetHashAndReset());
    }

    public void CompareDraftPrefill(string prompt, int runs, bool reverse)
    {
        if (draft is null || !BatchedPrefill) throw new InvalidOperationException("Batched prefill and a draft are required");
        if (runs < 1) throw new ArgumentOutOfRangeException(nameof(runs));
        var tokens = Tokenizer.SimpleChat(prompt);
        bool original = WindowedDraftPrefill;
        string? expectedState = null, expectedCheckpoints = null;
        int[]? expectedTokens = null;
        Console.WriteLine(JsonSerializer.Serialize(new { type = "draft_prefill_fixture", prompt_tokens = tokens.Length,
            kv_cache_precision = KvCachePrecision, prefix_cache = PrefixCachingEnabled,
            boundaries = PrefixBoundaries(tokens), sha256 = Convert.ToHexStringLower(SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(prompt))) }));
        try
        {
            // First pair warms both paths; last pair is counter-only evidence.
            for (int run = 0; run < runs + 2; run++)
                foreach (bool windowed in (run % 2 == 0) != reverse ? new[] { false, true } : new[] { true, false })
                {
                    WindowedDraftPrefill = windowed; ClearPrefixCache();
                    bool sampled = run == runs + 1;
                    gpu.ResetWallTiming(sampled, sampled);
                    var started = DateTimeOffset.UtcNow;
                    var watch = Stopwatch.StartNew();
                    var output = Generate(tokens, 1, true);
                    double wall = watch.Elapsed.TotalMilliseconds;
                    var completed = DateTimeOffset.UtcNow;
                    var timing = gpu.ReadWallTiming();
                    string state = PrefixTestStateHash(tokens.Length), checkpoints = DraftPrefillCheckpointHash();
                    expectedState ??= state; expectedCheckpoints ??= checkpoints; expectedTokens ??= output.Tokens;
                    if (state != expectedState || checkpoints != expectedCheckpoints || !output.Tokens.SequenceEqual(expectedTokens))
                        throw new Exception("Windowed draft prefill changed target state, live draft context, checkpoint or first token");
                    Console.WriteLine(JsonSerializer.Serialize(new { type = "draft_prefill", run,
                        mode = sampled ? "counters" : run == 0 ? "warmup" : "warm", windowed,
                        prompt_tokens = tokens.Length, projected_rows = LastDraftPrefillRows,
                        wall_ms = wall, started_utc = started, completed_utc = completed,
                        state_sha256 = state, checkpoint_sha256 = checkpoints, output_tokens = output.Tokens,
                        bitwise_equal = true, timing }));
                }
        }
        finally { WindowedDraftPrefill = original; ClearPrefixCache(); }
    }

    public void CheckDraftPrefill()
    {
        if (draft is null) throw new InvalidOperationException("A draft is required");
        bool original = WindowedDraftPrefill, originalCache = PrefixCachingEnabled;
        string padding = string.Concat(Enumerable.Repeat("This sentence is harmless context. ", 1800));
        var prompt = Tokenizer.SimpleChat(padding + "Write a Python function that returns the sum of squares of a list. Give examples.");
        var branch = Tokenizer.SimpleChat(padding + "Write a Python function that returns the product of a list. Give examples.");
        var expected = new Dictionary<string, (int[] Tokens, string State, string Checkpoints, int Cached, string Speculation)>();
        try
        {
            foreach (bool windowed in new[] { false, true })
            {
                WindowedDraftPrefill = windowed; PrefixCachingEnabled = true; ClearPrefixCache();
                void Check(string name, int[] input, int maximum = 64)
                {
                    var output = Generate(input, maximum, true);
                    string state = PrefixTestStateHash(input.Length + output.Tokens.Length - 1);
                    string checkpoints = DraftPrefillCheckpointHash();
                    using var stats = JsonDocument.Parse(JsonSerializer.Serialize(LastSpeculationStats));
                    string speculation = stats.RootElement.GetProperty("cycles").ToString() + ":" + stats.RootElement.GetProperty("accepted_draft_tokens").ToString();
                    if (!windowed) expected[name] = (output.Tokens, state, checkpoints, output.CachedTokens, speculation);
                    else
                    {
                        var reference = expected[name];
                        if (!output.Tokens.SequenceEqual(reference.Tokens) || state != reference.State || checkpoints != reference.Checkpoints || output.CachedTokens != reference.Cached || speculation != reference.Speculation)
                            throw new Exception("Draft prefill control failed: " + name);
                    }
                    Console.WriteLine(JsonSerializer.Serialize(new { type = "draft_prefill_check", name, windowed,
                        prompt_tokens = input.Length, output_tokens = output.Tokens.Length, cached_tokens = output.CachedTokens,
                        projected_rows = LastDraftPrefillRows, state_sha256 = state, checkpoint_sha256 = checkpoints,
                        speculation, bitwise_equal_to_full = windowed }));
                }
                Check("cold-long", prompt);
                Check("exact-long", prompt);
                Check("branch", branch);
                int block = PrefixBoundaries(prompt).Where(p => p % PrefixCheckpointInterval == 0).Max();
                Check("shorten-to-block", prompt[..block]);
                Check("append-after-shortening", prompt);
                PrefixCachingEnabled = false;
                Check("no-prefix-cache", prompt);
                PrefixCachingEnabled = true;
                // Cancellation must not leave a reusable lineage.
                using var cancelled = new CancellationTokenSource();
                cancelled.Cancel();
                try { Generate(prompt, 1, true, cancellation: cancelled.Token); throw new Exception("Cancellation ignored"); }
                catch (OperationCanceledException) { }
                if (CachedPrefixTokens(prompt) != 0) throw new Exception("Cancelled request retained a prefix");
                Check("after-cancellation", prompt);
            }
        }
        finally { PrefixCachingEnabled = originalCache; WindowedDraftPrefill = original; ClearPrefixCache(); }
    }
}
