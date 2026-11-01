namespace Bootstrap.Engine;

internal sealed partial class QwenEngine
{
    // A single active KV lineage, not a multi-session KV pool. Checkpoints share the
    // untouched KV prefix and own exact FP32 copies of every recurrent/conv state.
    // Snapshots are CPU-owned arrays: they never bind to a Metal command buffer.
    const int PrefixCheckpointLimit = 4, PrefixCheckpointInterval = 4096;
    const int RecurrentFloats = 32 * 128 * 128, ConvolutionFloats = 3 * 8192;
    const int SnapshotFloats = 30 * (RecurrentFloats + ConvolutionFloats) + Vocabulary;
    sealed record PrefixCheckpoint(int Position, float[] Data);
    readonly List<PrefixCheckpoint> prefixCheckpoints = new();
    readonly Stack<float[]> sparePrefixBuffers = new();
    int[] prefixTokens = [];
    bool prefixCachingEnabled = true, batchedPrefill = true;
    public bool PrefixCachingEnabled
    {
        get => prefixCachingEnabled;
        set { if (prefixCachingEnabled != value) ClearPrefixCache(); prefixCachingEnabled = value; }
    }
    public bool BatchedPrefill
    {
        get => batchedPrefill;
        set { if (batchedPrefill != value) ClearPrefixCache(); batchedPrefill = value; }
    }
    public int PrefixCheckpointCount => prefixCheckpoints.Count;
    public ulong PrefixCacheBytes => (ulong)(prefixCheckpoints.Count + sparePrefixBuffers.Count) * SnapshotFloats * sizeof(float);

    public void ClearPrefixCache()
    {
        foreach (var checkpoint in prefixCheckpoints) sparePrefixBuffers.Push(checkpoint.Data);
        prefixCheckpoints.Clear(); prefixTokens = [];
    }

    PrefixCheckpoint? FindPrefix(int[] prompt)
    {
        if (!PrefixCachingEnabled) return null;
        int common = 0, limit = Math.Min(prompt.Length, prefixTokens.Length);
        while (common < limit && prompt[common] == prefixTokens[common]) common++;
        return prefixCheckpoints.LastOrDefault(c => c.Position <= common);
    }
    // The API calls this while holding its inference semaphore, before message_start.
    public int CachedPrefixTokens(int[] prompt) => FindPrefix(prompt)?.Position ?? 0;

    int RestorePrefix(int[] prompt, bool ignoreEos, out int predicted)
    {
        predicted = -1;
        var checkpoint = FindPrefix(prompt);
        if (checkpoint is null) { Reset(); return 0; }
        // Anything beyond this point will be overwritten by the new suffix. Drop its
        // checkpoints before the first write, including when branching/shortening.
        for (int i = prefixCheckpoints.Count - 1; i >= 0 && prefixCheckpoints[i].Position > checkpoint.Position; i--)
        {
            sparePrefixBuffers.Push(prefixCheckpoints[i].Data); prefixCheckpoints.RemoveAt(i);
        }
        int offset = 0;
        var data = checkpoint.Data.AsSpan();
        for (int layer = 0; layer < 40; layer++) if ((layer + 1) % 4 != 0)
        {
            data.Slice(offset, RecurrentFloats).CopyTo(state[layer].Values<float>(RecurrentFloats)); offset += RecurrentFloats;
            data.Slice(offset, ConvolutionFloats).CopyTo(convolution[layer].Values<float>(ConvolutionFloats)); offset += ConvolutionFloats;
        }
        // End() waits for completion; CPU copies occur only between GPU submissions.
        // Positions at/after the checkpoint need no clear: attention is causal and
        // every newly consumed position is overwritten before it is read.
        if (checkpoint.Position == prompt.Length)
        {
            data.Slice(offset, Vocabulary).CopyTo(logits.Values<float>(Vocabulary));
            gpu.Begin(); SelectNext(ignoreEos); gpu.End(); predicted = next.Values<int>(1)[0];
        }
        return checkpoint.Position;
    }

    void SavePrefix(int position, int[] boundaries)
    {
        if (!PrefixCachingEnabled) return;
        if (prefixCheckpoints.Any(c => c.Position == position)) return;
        float[] storage;
        if (sparePrefixBuffers.TryPop(out var spare)) storage = spare;
        else if (prefixCheckpoints.Count == PrefixCheckpointLimit)
        {
            // Keep this request's chosen anchors; many short messages must not
            // evict the only block checkpoint that can serve an edited long turn.
            int victim = prefixCheckpoints.FindIndex(c => !boundaries.Contains(c.Position));
            if (victim < 0) victim = 0;
            storage = prefixCheckpoints[victim].Data; prefixCheckpoints.RemoveAt(victim);
        }
        else storage = new float[SnapshotFloats];
        int offset = 0;
        var data = storage.AsSpan();
        for (int layer = 0; layer < 40; layer++) if ((layer + 1) % 4 != 0)
        {
            state[layer].Values<float>(RecurrentFloats).CopyTo(data.Slice(offset, RecurrentFloats)); offset += RecurrentFloats;
            convolution[layer].Values<float>(ConvolutionFloats).CopyTo(data.Slice(offset, ConvolutionFloats)); offset += ConvolutionFloats;
        }
        logits.Values<float>(Vocabulary).CopyTo(data.Slice(offset, Vocabulary));
        prefixCheckpoints.Add(new(position, storage));
    }

    int[] PrefixBoundaries(int[] prompt)
    {
        // Keep the first message (often a stable system/tools prefix), last
        // message, prompt end and latest block boundary. Fill spare slots with
        // earlier blocks. Copy at most four snapshots even for a very long chat.
        var boundaries = new SortedSet<int> { prompt.Length };
        int firstMessage = Array.IndexOf(prompt, Tokenizer.End) + 1;
        int lastMessage = Array.LastIndexOf(prompt, Tokenizer.End) + 1;
        if (firstMessage > 0) boundaries.Add(firstMessage);
        if (lastMessage > 0) boundaries.Add(lastMessage);
        int block = (prompt.Length - 1) / PrefixCheckpointInterval * PrefixCheckpointInterval;
        while (block > 0 && boundaries.Count < PrefixCheckpointLimit)
        {
            boundaries.Add(block); block -= PrefixCheckpointInterval;
        }
        return boundaries.ToArray();
    }

    void SelectNext(bool ignoreEos) => gpu.Dispatch("argmax", new() { N = Vocabulary, Flags = ignoreEos ? 1u : 0u, A = (uint)Tokenizer.End, B = (uint)Tokenizer.EndOfText }, 256, 256, logits, next);
}
