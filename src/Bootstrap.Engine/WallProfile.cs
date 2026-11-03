using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace Bootstrap.Engine;

internal sealed partial class QwenEngine
{
    public void CompareTensorOps(string prompt, int runs)
    {
        if (runs < 1) throw new ArgumentOutOfRangeException(nameof(runs));
        var tokens = Tokenizer.SimpleChat(prompt);
        Console.WriteLine(JsonSerializer.Serialize(new { type = "fixture", prompt_tokens = tokens.Length,
            context_window = ContextWindow, prefix_caching = PrefixCachingEnabled, draft_enabled = draft is not null,
            sha256 = Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(prompt))),
            precision = "FP32 operands/accumulation, relaxed_precision=false", tiles = "unchanged" }));
        string fixture = string.Concat(Enumerable.Repeat("This sentence is harmless context. ", 100)) + "Reply with only the number: what is 2 + 2?";
        var small = Tokenizer.SimpleChat(fixture);
        Fp32TensorOps = false; ClearPrefixCache(); var original = Generate(small, 8, false);
        var originalLogits = logits.Values<float>(Vocabulary).ToArray();
        Fp32TensorOps = true; ClearPrefixCache(); var candidate = Generate(small, 8, false);
        if (!original.Tokens.SequenceEqual(candidate.Tokens) || candidate.Text.Trim() != "4")
            throw new Exception("TensorOps model smoke output mismatch");
        double maxError = 0, squaredError = 0, squaredReference = 0;
        for (int i = 0; i < Vocabulary; i++)
        {
            float actual = logits.Values<float>(Vocabulary)[i];
            if (!float.IsFinite(actual)) throw new Exception("Nonfinite TensorOps logits");
            double error = Math.Abs(actual - originalLogits[i]);
            maxError = Math.Max(maxError, error); squaredError += error * error; squaredReference += (double)originalLogits[i] * originalLogits[i];
        }
        double relativeL2 = Math.Sqrt(squaredError / squaredReference);
        if (maxError > .05 || relativeL2 > .001) throw new Exception($"TensorOps logit error too large: {maxError}, {relativeL2}");
        Console.WriteLine(JsonSerializer.Serialize(new { type = "model_correctness", prompt_tokens = small.Length,
            tokens = candidate.Tokens, text = candidate.Text, logit_max_abs_error = maxError, logit_relative_l2_error = relativeL2 }));
        using var process = Process.GetCurrentProcess();
        int? expected = null;
        for (int run = 0; run < runs + 2; run++)
        {
            // Warm both implementations, then alternate order to reduce thermal/order bias.
            bool sampled = run == runs + 1;
            foreach (bool enabled in run % 2 == 0 ? new[] { false, true } : new[] { true, false })
            {
                Fp32TensorOps = enabled; ClearPrefixCache(); gpu.ResetWallTiming(sampled, sampled);
                var cpu = process.TotalProcessorTime; var timer = Stopwatch.StartNew();
                var result = Generate(tokens, 1, true); double wall = timer.Elapsed.TotalMilliseconds;
                double cpuMs = (process.TotalProcessorTime - cpu).TotalMilliseconds;
                int first = result.Tokens[0];
                if (expected is not null && first != expected) throw new Exception("TensorOps full-prompt first-token mismatch");
                expected = first;
                Console.WriteLine(JsonSerializer.Serialize(new { type = "prefill", run, fp32_tensorops = enabled,
                    mode = sampled ? "all_kernel_encoder_counters" : run == 0 ? "cold_full_prompt" : "warm_uncached",
                    wall_ms = wall, process_cpu_ms = cpuMs, prompt_tokens = tokens.Length,
                    tokens_per_second = tokens.Length / (wall / 1000), first_token = first,
                    metal_buffer_bytes = AllocatedBytes, timing = gpu.ReadWallTiming() }));
            }
        }
    }

    // Development diagnostics: excludes loading/tokenization from generation timing.
    // Clear only the reusable prefix before each run. Generate resets model state;
    // retained allocations and compiled pipelines are intentionally warm after run 0.
    public void ProfilePrefill(string prompt, int runs, bool allKernels = false)
    {
        if (runs < 1) throw new ArgumentOutOfRangeException(nameof(runs));
        var timer = Stopwatch.StartNew();
        var tokens = Tokenizer.SimpleChat(prompt);
        Console.WriteLine(JsonSerializer.Serialize(new { type = "fixture", packed_experts = PackedExperts, affine8 = Affine8, kv_cache_precision = KvCachePrecision, fp32_tensorops = Fp32TensorOps, projection_precision = TensorOpsPrecision, delta_mode = DeltaMode, delta_preparation = DeltaPreparation, dense_prefill = DensePrefill, expert_kernel = ExpertKernel, parallel_expert_routing = ParallelExpertRouting, attention_mode = AttentionMode, wide_prefill = WidePrefill, prompt_tokens = tokens.Length,
            tokenizer_wall_ms = timer.Elapsed.TotalMilliseconds, context_window = ContextWindow,
            prefix_caching = PrefixCachingEnabled, draft_enabled = draft is not null,
            sha256 = Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(prompt))) }));
        using var process = Process.GetCurrentProcess();
        int? expected = null;
        for (int run = 0; run < runs + 2; run++)
        {
            bool samples = run == runs + 1;
            ClearPrefixCache();
            gpu.ResetWallTiming(samples, samples && allKernels);
            var started = DateTimeOffset.UtcNow;
            var cpu = process.TotalProcessorTime;
            timer.Restart();
            var result = Generate(tokens, 1, true);
            double wall = timer.Elapsed.TotalMilliseconds;
            double cpuMs = (process.TotalProcessorTime - cpu).TotalMilliseconds;
            int first = result.Tokens[0];
            if (expected is not null && first != expected) throw new Exception("Profiling changed first-token output");
            expected = first;
            Console.WriteLine(JsonSerializer.Serialize(new { type = "prefill", run, started_utc = started, completed_utc = DateTimeOffset.UtcNow,
                mode = samples ? (allKernels ? "all_kernel_encoder_counters" : "attention_encoder_counters") : run == 0 ? "cold_first_run" : "warm_uncached",
                wall_ms = wall, process_cpu_ms = cpuMs, prompt_tokens = tokens.Length,
                tokens_per_second = tokens.Length / (wall / 1000), first_token = first,
                metal_buffer_bytes = AllocatedBytes, timing = gpu.ReadWallTiming() }));
        }
    }
}

internal static class AttentionWallProfile
{
    public static void Run(string source, int repeats, bool tiled = false)
    {
        if (repeats < 1) throw new ArgumentOutOfRangeException(nameof(repeats));
        using var gpu = new Metal(source);
        // Actual prefill batch geometry at four representative history lengths.
        foreach (int history in new[] { 0, 1, 17, 1024, 4096, 10520, 32768 })
        {
            int rows = ContextLimits.PrefillCount(history, 512);
            int length = history + rows;
            using var q = gpu.Allocate((ulong)rows * 4096 * 4);
            using var k = gpu.Allocate((ulong)length * 512 * 4);
            using var v = gpu.Allocate((ulong)length * 512 * 4);
            using var gate = gpu.Allocate((ulong)rows * 4096 * 4);
            using var scores = gpu.Allocate((ulong)rows * 16 * (ulong)length * 4);
            using var output = gpu.Allocate((ulong)rows * 4096 * 4);
            Fill(q, rows * 4096, 17); Fill(k, length * 512, 31); Fill(v, length * 512, 23); Fill(gate, rows * 4096, 11);
            var p = new KernelArgs { M = (uint)rows, Position = (uint)history, K = (uint)length };
            var softmax = p; softmax.Position = (uint)(length - 1);
            void Encode(string phase)
            {
                if (phase is "qk" or "combined") gpu.Dispatch2D(tiled ? "batch_attention_scores_tiled" : "batch_attention_scores", p, tiled ? (uint)((length + 31) / 32) * 128 : (uint)length * 32, tiled ? (uint)((rows + 15) / 16) * 16 : (uint)rows * 16, 128, q, k, scores);
                if (phase is "softmax" or "combined") gpu.Dispatch("attention_softmax", softmax, (uint)rows * 16 * 256, 256, scores);
                if (phase is "av" or "combined")
                {
                    if (tiled) gpu.Dispatch2D("batch_attention_values_tiled", p, 8 * 128, (uint)((rows + 15) / 16) * 16, 128, scores, v, gate, output);
                    else gpu.Dispatch("batch_attention_values", p, (uint)rows * 4096, 128, scores, v, gate, output);
                }
            }
            // Prime compilation/data, and independently check output including causal masking.
            gpu.Begin(); Encode("combined"); gpu.End();
            Check(q, k, v, gate, output, rows, history);
            foreach (string phase in new[] { "combined", "combined_counters", "qk", "softmax", "av" })
            {
                // Each isolated softmax needs fresh raw scores; AV needs probabilities.
                for (int run = 0; run < repeats; run++)
                {
                    if (phase is "softmax" or "av")
                    {
                        gpu.Begin(); Encode("qk"); if (phase == "av") Encode("softmax"); gpu.End();
                    }
                    gpu.ResetWallTiming(phase == "combined_counters"); var watch = Stopwatch.StartNew();
                    gpu.Begin(); Encode(phase == "combined_counters" ? "combined" : phase); gpu.End();
                    double elapsed = watch.Elapsed.TotalMilliseconds;
                    Console.WriteLine(JsonSerializer.Serialize(new { type = "attention_microbenchmark", tiled, history, rows, length,
                        phase, run, wall_ms = elapsed, timing = gpu.ReadWallTiming(), correctness = "CPU double reference passed" }));
                }
            }
        }
    }
    static void Fill(GpuBuffer buffer, int count, int modulus)
    {
        var values = buffer.Values<float>(count);
        for (int i = 0; i < count; i++) values[i] = ((i * 7L + i / 256 * 13L) % modulus - modulus / 2) / 32f;
    }
    static void Check(GpuBuffer q, GpuBuffer k, GpuBuffer v, GpuBuffer gate, GpuBuffer output, int rows, int history)
    {
        int length = history + rows;
        var qs = q.Values<float>(rows * 4096); var ks = k.Values<float>(length * 512);
        var vs = v.Values<float>(length * 512); var gs = gate.Values<float>(rows * 4096); var ys = output.Values<float>(rows * 4096);
        foreach (int row in new[] { 0, rows / 2, rows - 1 }) foreach (int head in new[] { 0, 7, 8, 15 })
        {
            double[] logits = new double[history + row + 1];
            for (int t = 0; t < logits.Length; t++)
                for (int d = 0; d < 256; d++) logits[t] += (double)qs[row * 4096 + head * 256 + d] * ks[t * 512 + head / 8 * 256 + d] / 16;
            double max = logits.Max(), sum = 0;
            for (int t = 0; t < logits.Length; t++) sum += logits[t] = Math.Exp(logits[t] - max);
            foreach (int d in new[] { 0, 127, 255 })
            {
                double expected = 0;
                for (int t = 0; t < logits.Length; t++) expected += logits[t] / sum * vs[t * 512 + head / 8 * 256 + d];
                int at = row * 4096 + head * 256 + d;
                expected /= 1 + Math.Exp(-gs[at]);
                if (!float.IsFinite(ys[at]) || Math.Abs(ys[at] - expected) > 2e-5) throw new Exception($"Attention CPU reference mismatch at {row}/{head}/{d}: {ys[at]} vs {expected}");
            }
        }
    }
}
