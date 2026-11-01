namespace Bootstrap.Engine;

internal sealed partial class QwenEngine
{
    // Correctness-first verifier: retain every row's recurrent state on the GPU.
    // KV is append-only; discarded positions are overwritten before causal reads.
    // Row zero consumes the already verified anchor, so committing one row means
    // rejecting all draft proposals. This never commits an unverified proposal.
    const int VerificationCapacity = 8;
    VerificationBuffers? verification;
    int pendingVerificationRows, pendingVerificationPosition;

    sealed class VerificationBuffers
    {
        public readonly GpuBuffer[] State = new GpuBuffer[40], Convolution = new GpuBuffer[40];
        public readonly GpuBuffer Logits, Next;
        public VerificationBuffers(QwenEngine engine)
        {
            for (int layer = 0; layer < 40; layer++) if ((layer + 1) % 4 != 0)
            {
                State[layer] = engine.Buffer(VerificationCapacity * RecurrentFloats);
                Convolution[layer] = engine.Buffer(VerificationCapacity * ConvolutionFloats);
            }
            Logits = engine.Buffer(VerificationCapacity * Vocabulary);
            Next = engine.Buffer(VerificationCapacity);
        }
    }

    int[] VerifyBlock(ReadOnlySpan<int> tokens, int position, bool ignoreEos)
    {
        if (tokens.Length is < 1 or > VerificationCapacity || position < 0 || position + tokens.Length > Context)
            throw new ArgumentOutOfRangeException(nameof(tokens));
        if (pendingVerificationRows != 0) throw new InvalidOperationException("Commit the preceding verification before starting another");
        verification ??= new VerificationBuffers(this);
        PrefillBlock(tokens, position, false, ignoreEos, verify: true);
        pendingVerificationRows = tokens.Length; pendingVerificationPosition = position;
        return verification.Next.Values<int>(tokens.Length).ToArray();
    }

    int CommitVerification(int consumedRows)
    {
        if (consumedRows < 1 || consumedRows > pendingVerificationRows)
            throw new ArgumentOutOfRangeException(nameof(consumedRows));
        var buffers = verification!;
        gpu.Begin();
        for (int layer = 0; layer < 40; layer++) if ((layer + 1) % 4 != 0)
        {
            CopyFloats(new Binding(buffers.State[layer], (ulong)(consumedRows - 1) * RecurrentFloats * 4), state[layer], RecurrentFloats);
            CopyFloats(new Binding(buffers.Convolution[layer], (ulong)(consumedRows - 1) * ConvolutionFloats * 4), convolution[layer], ConvolutionFloats);
        }
        CopyFloats(new Binding(buffers.Logits, (ulong)(consumedRows - 1) * Vocabulary * 4), logits, Vocabulary);
        ActiveDraft?.CommitFeatures(consumedRows, pendingVerificationPosition);
        gpu.End();
        pendingVerificationRows = 0;
        return buffers.Next.Values<int>(VerificationCapacity)[consumedRows - 1];
    }

    void CopyFloats(Binding source, Binding destination, int count) =>
        gpu.Dispatch("copy_floats", new() { N = (uint)count }, (uint)count, 128, source, destination);

    public void SpeculationSelfTest()
    {
        if (draft is null) throw new InvalidOperationException("--draft-dir is required");
        string[] fixtures = [
            "Write a Python function that returns the sum of squares of a list of integers. Explain the function and give three examples.",
            "Write a Python function to merge two sorted lists, with a docstring and tests.",
            string.Concat(Enumerable.Repeat("This sentence is harmless context. ", 400)) + "Write a Python function to reverse the words in a string, with examples."
        ];
        var cases = new List<object>();
        foreach (string fixture in fixtures)
        {
            int[] prompt = Tokenizer.SimpleChat(fixture);
            SpeculativeDecoding = false; Reset();
            Generate(prompt, 8, true);
            var watch = System.Diagnostics.Stopwatch.StartNew();
            var baseline = Generate(prompt, 128, true); double baselineMs = watch.Elapsed.TotalMilliseconds;
            SpeculativeDecoding = true; Reset();
            Generate(prompt, 8, true);
            watch.Restart(); var candidate = Generate(prompt, 128, true); double candidateMs = watch.Elapsed.TotalMilliseconds;
            var statistics = LastSpeculationStats;
            watch.Restart(); var cached = Generate(prompt, 128, true); double cachedMs = watch.Elapsed.TotalMilliseconds;
            if (!candidate.Tokens.SequenceEqual(cached.Tokens)) throw new Exception("Draft prefix restore changed output");
            int firstMismatch = Enumerable.Range(0, Math.Min(baseline.Tokens.Length, candidate.Tokens.Length)).FirstOrDefault(i => baseline.Tokens[i] != candidate.Tokens[i], -1);
            cases.Add(new { prompt_tokens = prompt.Length, generated_tokens = candidate.Tokens.Length, baseline_ms = baselineMs, speculative_ms = candidateMs, repeat_ms = cachedMs, first_mismatch = firstMismatch, cached_tokens = cached.CachedTokens, speculation = statistics });
            int pieces = 0; Generate(prompt, 128, true, _ => ++pieces < 3);
            var afterStop = Generate(prompt, 128, true);
            if (!candidate.Tokens.SequenceEqual(afterStop.Tokens)) throw new Exception("Early streaming stop corrupted draft prefix restore");
        }
        Console.WriteLine(System.Text.Json.JsonSerializer.Serialize(new { development_only = true, cases, stream_stop_and_prefix_restore = "passed", metal_buffer_bytes = AllocatedBytes }));
    }

    public void ProfileDecode()
    {
        gpu.SetProfiling(false);
        int[] prompt = Tokenizer.SimpleChat("Write a Python function to sum the squares of integers.");
        Generate(prompt, 8, true);
        gpu.SetProfiling(true);
        var result = Generate(prompt, 4, true);
        Console.WriteLine(System.Text.Json.JsonSerializer.Serialize(new { diagnostic_only = true, generated_tokens = result.Tokens, cached_tokens = result.CachedTokens }));
    }

    // Verifier tests use identical supplied inputs, independent of draft accuracy.
    // Compare full speculative execution + rollback against only consuming the
    // accepted prefix. Then overwrite rejected KV with an unrelated continuation.
    public void VerificationSelfTest()
    {
        PrefixCachingEnabled = false;
        int[] prompt = Tokenizer.SimpleChat("Write a Python function to sum the squares of integers.");
        int[] candidates = Tokenizer.Encode("def sum_squares(values):\n    return sum(x * x for x in values)").Take(8).ToArray();
        int checks = 0;
        var timings = new List<object>();
        foreach (int width in new[] { 1, 2, 4, 8 })
        {
            foreach (int retained in Enumerable.Range(1, width))
            {
                Reset(); PrefillBlock(prompt, 0, true, false);
                var watch = System.Diagnostics.Stopwatch.StartNew();
                int[] predictions = VerifyBlock(candidates.AsSpan(0, width), prompt.Length, false);
                int predicted = CommitVerification(retained); watch.Stop();
                string actual = PrefixTestStateHash(prompt.Length + retained);
                int continuation = Forward(Tokenizer.Encode("! ")[0], prompt.Length + retained, true, false);
                string continued = PrefixTestStateHash(prompt.Length + retained + 1);

                Reset(); PrefillBlock(prompt, 0, true, false);
                int[] reference = VerifyBlock(candidates.AsSpan(0, retained), prompt.Length, false);
                int expected = CommitVerification(retained);
                if (!predictions.Take(retained).SequenceEqual(reference) || predicted != expected)
                    throw new Exception($"Verifier row predictions differ: width={width}, retained={retained}");
                if (actual != PrefixTestStateHash(prompt.Length + retained))
                    throw new Exception($"Verifier accepted-prefix state mismatch: width={width}, retained={retained}");
                if (continuation != Forward(Tokenizer.Encode("! ")[0], prompt.Length + retained, true, false) || continued != PrefixTestStateHash(prompt.Length + retained + 1))
                    throw new Exception($"Rejected KV contaminated continuation: width={width}, retained={retained}");
                checks += 3;
                timings.Add(new { width, retained, verification_and_commit_ms = watch.Elapsed.TotalMilliseconds });
            }
        }
        Reset();
        Console.WriteLine(System.Text.Json.JsonSerializer.Serialize(new { verifier_checks_passed = checks, metal_buffer_bytes = AllocatedBytes, timings }));
    }
}
