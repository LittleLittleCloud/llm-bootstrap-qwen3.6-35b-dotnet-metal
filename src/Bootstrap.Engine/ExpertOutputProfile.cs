using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace Bootstrap.Engine;

internal sealed partial class QwenEngine
{
    // Local acceptance probes, not a broad benchmark or a bitwise requirement.
    public void CheckExpertOutputQuality(string longPrompt)
    {
        if (!Affine8 || !GatheredExperts || TensorOpsPrecision != "fp16" || ExpertKernel != "fused" || PrecomputeSharedGate)
            throw new InvalidOperationException("Expert-output quality check requires tuned affine experts and baseline shared gate");
        string padding = string.Concat(Enumerable.Repeat("This sentence is harmless context. ", 100));
        bool original = Bf16ExpertOutput;
        try {
            foreach (var (name, prompt, expected) in new[] {
                ("long_needle", longPrompt, "73DY-DRV5-B8FJ"),
                ("arithmetic", padding + "Return only the integer result of 17 * 23.", "391"),
                ("sort", padding + "Sort these integers in ascending order: 19, -4, 7, 0, 12. Return only comma-separated integers with no spaces.", "-4,0,7,12,19"),
                ("extract", padding + "Record: name=Iris; city=Oslo; id=Q7M2. Return only the id value.", "Q7M2"),
                ("count", padding + "Count the exact word red in this list: red blue red green red. Return only the count as an integer.", "3"),
                ("override", padding + "An earlier note says the active code is OLD-17. The final correction says the active code is NEW-42. Return only the active code after the correction.", "NEW-42") }) {
                float[]? reference = null; int[]? referenceTokens = null; int first = -1;
                foreach (bool enabled in new[] { false, true }) {
                    Bf16ExpertOutput = enabled; ClearPrefixCache(); float[]? firstLogits = null;
                    var result = Generate(Tokenizer.SimpleChat(prompt), 48, false, _ => { firstLogits ??= logits.Values<float>(Vocabulary).ToArray(); return true; });
                    if (firstLogits is null || firstLogits.Any(v => !float.IsFinite(v))) throw new Exception("Missing or nonfinite first-token logits");
                    reference ??= firstLogits; referenceTokens ??= result.Tokens; if (first < 0) first = result.Tokens[0];
                    double maxA = reference.Max(), maxB = firstLogits.Max();
                    double logZa = maxA + Math.Log(reference.Sum(v => Math.Exp(v - maxA)));
                    double logZb = maxB + Math.Log(firstLogits.Sum(v => Math.Exp(v - maxB)));
                    double kl = 0, tv = 0, meanA = reference.Average(v => (double)v), meanB = firstLogits.Average(v => (double)v), centeredErr = 0, centeredRef = 0;
                    for (int i = 0; i < Vocabulary; i++) {
                        double logP = reference[i] - logZa, logQ = firstLogits[i] - logZb, probA = Math.Exp(logP), probB = Math.Exp(logQ);
                        kl += probA * (logP - logQ); tv += Math.Abs(probA - probB) / 2;
                        double delta = (firstLogits[i] - meanB) - (reference[i] - meanA); centeredErr += delta * delta; centeredRef += (reference[i] - meanA) * (reference[i] - meanA);
                    }
                    var topA = Enumerable.Range(0, Vocabulary).OrderByDescending(i => reference[i]).Take(10).ToArray();
                    var topB = Enumerable.Range(0, Vocabulary).OrderByDescending(i => firstLogits[i]).Take(10).ToArray();
                    Console.WriteLine(JsonSerializer.Serialize(new { type = "expert_output_quality", fixture = name, variant = enabled ? "bf16" : "fp32", prompt_tokens = result.PromptTokens,
                        expected, text = result.Text, passed = result.Text.Trim() == expected, tokens_match_baseline = result.Tokens.SequenceEqual(referenceTokens),
                        first_token = result.Tokens[0], first_token_matches = result.Tokens[0] == first, top1_matches = topA[0] == topB[0], top10_overlap = topA.Intersect(topB).Count(),
                        softmax_kl_nats = Math.Max(0, kl), softmax_total_variation = tv, mean_logit_shift = meanB - meanA,
                        centered_logit_relative_l2_error = Math.Sqrt(centeredErr / centeredRef),
                        baseline_first_token_probability = Math.Exp(reference[first] - logZa), candidate_first_token_probability = Math.Exp(firstLogits[first] - logZb) }));
                }
            }
        } finally { Bf16ExpertOutput = original; ClearPrefixCache(); }
    }

    public void CompareExpertOutput(string prompt, int runs)
    {
        if (runs < 1) throw new ArgumentOutOfRangeException(nameof(runs));
        if (!Affine8 || !GatheredExperts || TensorOpsPrecision != "fp16" || ExpertKernel != "fused" || PrecomputeSharedGate)
            throw new InvalidOperationException("Expert-output comparison requires tuned affine experts and baseline shared gate");
        var tokens = Tokenizer.SimpleChat(prompt); bool original = Bf16ExpertOutput;
        int? first = null; float[]? reference = null;
        var repeated = new Dictionary<bool, uint[]>();
        Console.WriteLine(JsonSerializer.Serialize(new { type = "expert_output_fixture", prompt_tokens = tokens.Length,
            sha256 = Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(prompt))) }));
        try {
            for (int run = 0; run <= runs; run++) foreach (bool enabled in run % 2 == 0 ? new[] { false, true } : new[] { true, false }) {
                Bf16ExpertOutput = enabled; ClearPrefixCache(); gpu.ResetWallTiming();
                var watch = Stopwatch.StartNew(); var result = Generate(tokens, 1, true); double wall = watch.Elapsed.TotalMilliseconds;
                first ??= result.Tokens[0];
                var actual = logits.Values<float>(Vocabulary).ToArray(); reference ??= actual;
                var bits = logits.Values<uint>(Vocabulary).ToArray();
                if (repeated.TryGetValue(enabled, out var previous) && !previous.SequenceEqual(bits)) throw new Exception("Nondeterministic expert-output variant");
                repeated[enabled] = bits;
                double maxError = 0, squaredError = 0, squaredReference = 0, squaredActual = 0, dot = 0;
                for (int i = 0; i < Vocabulary; i++) {
                    if (!float.IsFinite(actual[i])) throw new Exception("Nonfinite expert-output logits");
                    double delta = (double)actual[i] - reference[i]; maxError = Math.Max(maxError, Math.Abs(delta));
                    squaredError += delta * delta; squaredReference += (double)reference[i] * reference[i];
                    squaredActual += (double)actual[i] * actual[i]; dot += (double)actual[i] * reference[i];
                }
                Console.WriteLine(JsonSerializer.Serialize(new { type = "expert_output_pair", run, variant = enabled ? "bf16" : "fp32", prompt_tokens = tokens.Length,
                    wall_ms = wall, first_token = result.Tokens[0], first_token_matches = result.Tokens[0] == first,
                    logit_max_abs_error = maxError, logit_relative_l2_error = Math.Sqrt(squaredError / squaredReference),
                    logit_cosine = dot / Math.Sqrt(squaredReference * squaredActual), repeatable = true,
                    metal_buffer_bytes = AllocatedBytes, timing = gpu.ReadWallTiming() }));
            }
            // Untimed checks exercise ordinary decode/speculation after changed prefill.
            foreach (var (name, fixture, maximum) in new[] {
                ("arithmetic", string.Concat(Enumerable.Repeat("This sentence is harmless context. ", 100)) + "Reply with only the number: what is 2 + 2?", 8),
                ("full_prompt", prompt, 32) }) {
                int[]? expectedTokens = null; string? expectedText = null;
                foreach (bool enabled in new[] { false, true }) {
                    Bf16ExpertOutput = enabled; ClearPrefixCache(); var result = Generate(Tokenizer.SimpleChat(fixture), maximum, false);
                    expectedTokens ??= result.Tokens; expectedText ??= result.Text;
                    Console.WriteLine(JsonSerializer.Serialize(new { type = "expert_output_generation", fixture = name, variant = enabled ? "bf16" : "fp32",
                        text = result.Text, tokens = result.Tokens, tokens_match_baseline = result.Tokens.SequenceEqual(expectedTokens), text_matches_baseline = result.Text == expectedText,
                        arithmetic_passed = name == "arithmetic" ? result.Text.Trim() == "4" : (bool?)null }));
                    if (name == "arithmetic" && result.Text.Trim() != "4") throw new Exception("Arithmetic smoke failed");
                }
            }
        } finally { Bf16ExpertOutput = original; ClearPrefixCache(); }
    }
}
