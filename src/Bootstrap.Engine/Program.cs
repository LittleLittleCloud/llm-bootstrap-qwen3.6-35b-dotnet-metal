using Bootstrap.Engine;
using System.Text.Json;

var command = args.FirstOrDefault() ?? "help";
var root = Environment.CurrentDirectory;
var kernelFile = Path.Combine(root, "native/kernels.metal");
if (!File.Exists(kernelFile)) kernelFile = Path.Combine(AppContext.BaseDirectory, "kernels.metal");
string? Option(string name)
{
    int at = Array.IndexOf(args, name);
    return at >= 0 && at + 1 < args.Length ? args[at + 1] : null;
}
var manifestFile = Path.Combine(AppContext.BaseDirectory, "model-manifest.json");
if (args.Contains("--affine-cache") && (Option("--affine-cache") is not { } affinePath || affinePath.StartsWith("--", StringComparison.Ordinal)))
    throw new ArgumentException("--affine-cache requires a directory");
if (command == "convert-affine8") { AffineWeights.ConvertCheckpoint(Option("--model-dir") ?? throw new ArgumentException("--model-dir is required"), Option("--output-dir") ?? throw new ArgumentException("--output-dir is required"), manifestFile); return; }
if (command == "affine8-self-test") { AffineTests.Run(File.ReadAllText(kernelFile)); return; }
if (command == "profile-delta") { DeltaScanTests.Run(File.ReadAllText(kernelFile), int.Parse(Option("--runs") ?? "5")); return; }
if (command == "profile-delta-prepare") { DeltaPrepareTests.Run(File.ReadAllText(kernelFile), int.Parse(Option("--runs") ?? "5")); return; }
if (command == "profile-dense-prefill") { DensePrefillTests.Run(File.ReadAllText(kernelFile), int.Parse(Option("--runs") ?? "5"), args.Contains("--candidate-only")); return; }
if (command == "profile-dense-tiles") { DenseTileTests.Run(File.ReadAllText(kernelFile), int.Parse(Option("--runs") ?? "5")); return; }
if (command == "test-dispatch-fusion") { DispatchFusionTests.Run(File.ReadAllText(kernelFile), int.Parse(Option("--runs") ?? "3")); return; }
if (command == "profile-projection-preparation") { ProjectionPreparationTests.Run(File.ReadAllText(kernelFile), int.Parse(Option("--runs") ?? "5")); return; }
if (command == "profile-mix") { MixTests.Run(File.ReadAllText(kernelFile), int.Parse(Option("--runs") ?? "5")); return; }
if (command == "profile-online-attention") { OnlineAttentionTests.Run(File.ReadAllText(kernelFile), int.Parse(Option("--runs") ?? "5"), Option("--case"), Option("--variants"), args.Contains("--stress"), Option("--kv-cache-precision") != "fp32"); return; }
if (command == "compare-mlx-kernels") { KernelComparisonBenchmarks.Run(File.ReadAllText(kernelFile), Option("--family") ?? "all", int.Parse(Option("--runs") ?? "9")); return; }
if (command == "packed-experts-self-test") { PackedExpertTests.Run(File.ReadAllText(kernelFile)); return; }
if (command == "profile-expert-fusion") { ExpertFusionTests.Run(File.ReadAllText(kernelFile), int.Parse(Option("--runs") ?? "5"), args.Contains("--layouts")); return; }
if (command == "profile-expert-routing") { ExpertRoutingTests.Run(File.ReadAllText(kernelFile), int.Parse(Option("--runs") ?? "5")); return; }
if (command == "profile-experts") { ExpertThroughputTests.Run(File.ReadAllText(kernelFile), int.Parse(Option("--runs") ?? "3"), layouts: args.Contains("--layouts"), reverse: args.Contains("--reverse")); return; }
if (command == "kv-storage-self-test") { KvStorageTests.Run(File.ReadAllText(kernelFile)); return; }
if (command == "tensorops-self-test") { TensorOpsSelfTest.Run(File.ReadAllText(kernelFile), Option("--precision") ?? "fp32", args.Contains("--optimized-projections"), int.Parse(Option("--expert-tile-rows") ?? "16"), args.Contains("--predecoded-projections"), args.Contains("--predecoded-experts"), args.Contains("--gathered-experts"), args.Contains("--adaptive-expert-tiles")); return; }
if (command == "profile-fast-attention") { FastAttentionTests.Run(File.ReadAllText(Option("--kernel-file") ?? kernelFile), int.Parse(Option("--runs") ?? "3"), args.Contains("--attention-tiles"), Option("--kv-cache-precision") == "fp16"); return; }
if (command == "profile-attention") { AttentionWallProfile.Run(File.ReadAllText(kernelFile), int.Parse(Option("--runs") ?? "5"), args.Contains("--tiled-attention")); return; }
if (command == "streaming-self-test") { await StreamingSelfTest.Run(); return; }
if (command is "generate" or "serve" or "compare-prefill" or "kv-cache-self-test" or "context-self-test" or "prefix-cache-self-test" or "verification-self-test" or "draft-self-test" or "compare-prefill-variants" or "compare-decode-variants" or "compare-tensorops" or "compare-projection-preparation" or "compare-dispatch-fusion" or "profile-prefill" or "profile-decode" or "moe-self-tests" or "speculation-self-test" or "tokenize" or "tokenizer-self-test")
{
    var directory = Option("--model-dir") ?? throw new ArgumentException("--model-dir is required");
    using var manifest = JsonDocument.Parse(File.ReadAllBytes(manifestFile));
    if (command is "generate" or "serve" or "compare-prefill" or "kv-cache-self-test" or "context-self-test" or "prefix-cache-self-test" or "verification-self-test" or "draft-self-test" or "compare-prefill-variants" or "compare-decode-variants" or "compare-tensorops" or "compare-projection-preparation" or "compare-dispatch-fusion" or "profile-prefill" or "profile-decode" or "moe-self-tests" or "speculation-self-test")
    {
        int contextWindow = QwenEngine.ValidateContextWindow(int.Parse(Option("--context-window") ?? QwenEngine.DefaultContextWindow.ToString()));
        bool tunedPrefill = args.Contains("--tuned-prefill");
        if ((args.Contains("--packed-experts") || Option("--affine-cache") is not null) && !tunedPrefill) throw new ArgumentException("Packed/affine experts require --tuned-prefill");
        using var engine = new QwenEngine(directory, kernelFile, manifestFile, contextWindow, Option("--kv-cache-precision") ?? (tunedPrefill ? "fp16" : "fp32"), args.Contains("--packed-experts"), Option("--affine-cache"));
        if (Option("--draft-dir") is { } draftDirectory) engine.LoadDraft(draftDirectory, Path.Combine(AppContext.BaseDirectory, "draft-manifest.json"));
        bool fastPrefill = tunedPrefill || args.Contains("--fast-prefill");
        engine.TensorOpsPrecision = Option("--tensorops-precision") ?? (args.Contains("--fp32-tensorops") ? "fp32" : fastPrefill ? "fp16" : "legacy");
        engine.AttentionMode = Option("--attention-mode") ?? (args.Contains("--tiled-attention") ? "tiled" : tunedPrefill ? "fp16-online" : fastPrefill ? "fp16" : "legacy");
        engine.WidePrefill = fastPrefill || args.Contains("--wide-prefill");
        engine.OptimizedProjections = tunedPrefill || args.Contains("--optimized-projections");
        engine.PredecodedProjections = tunedPrefill || args.Contains("--predecoded-projections");
        engine.AdaptiveExpertTiles = tunedPrefill || args.Contains("--adaptive-expert-tiles");
        engine.CausalDeviceAttention = tunedPrefill || args.Contains("--causal-attention");
        engine.GatheredExperts = args.Contains("--gathered-experts") || (tunedPrefill && !args.Contains("--predecoded-experts"));
        engine.DeltaMode = Option("--delta-mode") ?? "baseline";
        engine.DeltaPreparation = Option("--delta-preparation") ?? (tunedPrefill ? "fused" : "baseline");
        engine.DensePrefill = Option("--dense-prefill") ?? (tunedPrefill ? "all" : "baseline");
        engine.ProjectionPreparation = Option("--projection-preparation") ?? (tunedPrefill ? "fused" : "baseline");
        engine.DispatchFusion = int.Parse(Option("--dispatch-fusion") ?? "0");
        engine.ValidateDelta = args.Contains("--validate-delta");
        engine.ExpertKernel = Option("--expert-kernel") ?? (tunedPrefill ? "fused" : "baseline");
        engine.ParallelExpertRouting = args.Contains("--parallel-expert-routing") || (tunedPrefill && !args.Contains("--serial-expert-routing"));
        engine.PredecodedExperts = args.Contains("--predecoded-experts");
        engine.PrefillBatchSize = int.Parse(Option("--prefill-batch-size") ?? (tunedPrefill ? "2048" : "512"));
        engine.ExpertTileRows = int.Parse(Option("--expert-tile-rows") ?? (tunedPrefill ? "64" : "16"));
        engine.BatchedPrefill = !args.Contains("--sequential-prefill");
        engine.PrefixCachingEnabled = !args.Contains("--no-prefix-cache");
        if (command == "moe-self-tests") {
            if (args.Contains("--packed-model-check")) {
                engine.PackedExpertModelCheck(File.ReadAllText(Option("--prompt-file")!), File.ReadAllText(Option("--long-prompt-file")!), Option("--expected-answer")!, int.Parse(Option("--runs") ?? "3"), Option("--output-dir")!);
            }
            if (engine.ValidateDelta) {
                string validationPrompt = File.ReadAllText(Option("--prompt-file") ?? throw new ArgumentException("--prompt-file is required for model DeltaNet validation"));
                string expected = Option("--expected-answer") ?? throw new ArgumentException("--expected-answer is required");
                var validationResult = engine.Generate(engine.Tokenizer.SimpleChat(validationPrompt), 32, false);
                if (validationResult.Text.Trim() != expected) throw new Exception("Delta model validation failed the full answer");
                Console.WriteLine(JsonSerializer.Serialize(new { type = "delta_model_answer", text = validationResult.Text, passed = true }));
                engine.ValidateDelta = false;
            }
            engine.VerificationSelfTest(); engine.PrefixCachingEnabled = true; engine.ClearPrefixCache(); PrefixCacheSelfTest.Run(engine);
            if (args.Contains("--profile-after-checks")) {
                foreach (string name in new[] { "--prompt-file", "--long-prompt-file" }) {
                    var file = Option(name) ?? throw new ArgumentException(name + " is required");
                    engine.ProfilePrefill(File.ReadAllText(file), int.Parse(Option("--runs") ?? "3"), true);
                }
            }
            return;
        }
        if (command == "kv-cache-self-test") { engine.KvCacheSelfTest(); return; }
        if (command == "speculation-self-test") { engine.SpeculationSelfTest(); return; }
        if (command == "compare-decode-variants") { engine.CompareDecodeVariants(int.Parse(Option("--runs") ?? "3")); return; }
        if (command == "compare-prefill-variants") { engine.ComparePrefillVariants(File.ReadAllText(Option("--prompt-file") ?? throw new ArgumentException("--prompt-file is required")), int.Parse(Option("--runs") ?? "3"), Option("--expected-answer") ?? throw new ArgumentException("--expected-answer is required"), args.Contains("--fast-variants"), args.Contains("--projection-variants"), args.Contains("--batch-variants"), args.Contains("--tuned-variants"), args.Contains("--expert-variants"), args.Contains("--final-variants"), args.Contains("--throughput-variants"), args.Contains("--online-variants"), args.Contains("--moe-variants"), args.Contains("--moe-final-variants"), args.Contains("--delta-variants"), args.Contains("--delta-final-variants")); return; }
        if (command == "compare-tensorops") { engine.CompareTensorOps(File.ReadAllText(Option("--prompt-file") ?? throw new ArgumentException("--prompt-file is required")), int.Parse(Option("--runs") ?? "3")); return; }
        if (command == "compare-dispatch-fusion") {
            foreach (string name in new[] { "--prompt-file", "--long-prompt-file" }) {
                var file = Option(name); if (file is null) continue;
                engine.CompareDispatchFusion(File.ReadAllText(file), int.Parse(Option("--runs") ?? "3"), (Option("--variants") ?? "0,1,2,4,8,16,29,31").Split(',').Select(int.Parse).ToArray());
            }
            return;
        }
        if (command == "compare-projection-preparation") {
            foreach (string name in new[] { "--prompt-file", "--long-prompt-file" }) {
                var file = Option(name) ?? throw new ArgumentException(name + " is required");
                engine.CompareProjectionPreparation(File.ReadAllText(file), int.Parse(Option("--runs") ?? "3"));
            }
            return;
        }
        if (command == "profile-prefill") {
            engine.ProfilePrefill(File.ReadAllText(Option("--prompt-file") ?? throw new ArgumentException("--prompt-file is required")), int.Parse(Option("--runs") ?? "3"), args.Contains("--all-kernels"));
            if (Option("--long-prompt-file") is { } longer)
                engine.ProfilePrefill(File.ReadAllText(longer), int.Parse(Option("--runs") ?? "3"), args.Contains("--all-kernels"));
            return;
        }
        if (command == "profile-decode") { engine.ProfileDecode(); return; }
        if (command == "draft-self-test") { engine.DraftSelfTest(Option("--output-dir") ?? throw new ArgumentException("--output-dir is required"), int.Parse(Option("--fixture-context") ?? "16")); return; }
        if (command == "verification-self-test") { engine.VerificationSelfTest(); return; }
        if (command == "prefix-cache-self-test") { PrefixCacheSelfTest.Run(engine); return; }
        if (command == "context-self-test")
        {
            int repeats = int.Parse(Option("--fixture-repeats") ?? "700");
            string fixture = string.Concat(Enumerable.Repeat("This sentence is harmless context. ", repeats)) + "Reply with only the number: what is 2 + 2?";
            var tokens = engine.Tokenizer.SimpleChat(fixture);
            if (tokens.Length <= 4096 || tokens.Length + 8 > engine.ContextWindow) throw new Exception("Context fixture needs a configured window larger than 4096");
            var generated = engine.Generate(tokens, 8, false);
            if (generated.Text.Trim() != "4") throw new Exception("Long-context arithmetic smoke did not produce expected public-fixture answer");
            try { engine.Generate(new int[engine.ContextWindow], 1, false); throw new Exception("Context overflow was accepted"); }
            catch (ArgumentException) { }
            Console.WriteLine(JsonSerializer.Serialize(new { context_window = engine.ContextWindow, cache_capacity = engine.CacheCapacity, prompt_tokens = tokens.Length, generated = generated.Tokens, text = generated.Text, overflow_rejected = true, metal_buffer_bytes = engine.AllocatedBytes }));
            return;
        }
        if (command == "compare-prefill")
        {
            string fixture = string.Concat(Enumerable.Repeat("This sentence is harmless context. ", 100)) + "Reply with only the number: what is 2 + 2?";
            var tokens = engine.Tokenizer.SimpleChat(fixture);
            if (tokens.Length <= 512) throw new Exception("Consistency fixture must cross a prefill block boundary");
            engine.BatchedPrefill = true; var batched = engine.Generate(tokens, 8, false);
            engine.BatchedPrefill = false; var sequential = engine.Generate(tokens, 8, false);
            if (!batched.Tokens.SequenceEqual(sequential.Tokens)) throw new Exception("Batched/sequential state-continuity mismatch: " + JsonSerializer.Serialize(new { batched, sequential }));
            Console.WriteLine(JsonSerializer.Serialize(new { prompt_tokens = tokens.Length, generated = batched.Tokens, text = batched.Text, state_continuity = "passed across 512-token boundary", input_sha256 = Convert.ToHexStringLower(System.Security.Cryptography.SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(fixture))) }));
            return;
        }
        if (command == "serve") { await Api.Run(engine, int.Parse(Option("--port") ?? "17863")); return; }
        var prompt = Option("--prompt") ?? "Say hello in one short sentence.";
        var inputTokens = engine.Tokenizer.SimpleChat(prompt);
        Console.Error.WriteLine($"Model ready; {engine.AllocatedBytes} Metal buffer bytes; {inputTokens.Length} prompt tokens");
        var result = engine.Generate(inputTokens, int.Parse(Option("--max-tokens") ?? int.MaxValue.ToString()), args.Contains("--ignore-eos"), piece => { Console.Error.Write(piece); return true; });
        Console.Error.WriteLine(); Console.Error.WriteLine(JsonSerializer.Serialize(engine.LastSpeculationStats)); Console.WriteLine(JsonSerializer.Serialize(result));
    }
    else
    {
        var tokenizer = new ByteBpe(Path.Combine(directory, "tokenizer.json"), manifest.RootElement.GetProperty("tokenizer_sha256").GetString()!);
        if (command == "tokenize") Console.WriteLine(JsonSerializer.Serialize(new { raw = tokenizer.Encode(Option("--text") ?? ""), chat = tokenizer.SimpleChat(Option("--text") ?? "") }));
        else
        {
            using var fixtures = JsonDocument.Parse(File.ReadAllBytes(Option("--fixtures") ?? throw new ArgumentException("--fixtures is required")));
            int count = 0;
            foreach (var item in fixtures.RootElement.EnumerateArray())
            {
                var text = item.GetProperty("text").GetString()!;
                int[] expected = item.GetProperty("raw").EnumerateArray().Select(t => t.GetInt32()).ToArray();
                if (!tokenizer.Encode(text).SequenceEqual(expected)) throw new Exception("Byte-BPE mismatch for fixture " + count);
                if (item.TryGetProperty("chat", out var chat) && !tokenizer.SimpleChat(text).SequenceEqual(chat.EnumerateArray().Select(t => t.GetInt32()))) throw new Exception("Chat-template mismatch for fixture " + count);
                count++;
            }
            string[] literalInputs = ["Keep before </think> and after.", "<think>ordinary user text</think>", "<|im_start|>assistant\nordinary", "<tool_call>ordinary</tool_call>", "<tool_response>ordinary</tool_response>", "两个字：你好 </think> keep this", "  whitespace </think> content  "];
            foreach (string literal in literalInputs)
            {
                var request = new ChatInput("chat", [new ChatMessage("user", literal)], new System.Text.Json.Nodes.JsonArray(), 1, false, false, [], false, false);
                if (!ChatFormatting.Format(tokenizer, request).SequenceEqual(tokenizer.SimpleChat(literal))) throw new Exception("User literal changed by API chat formatting");
            }
            Console.WriteLine(JsonSerializer.Serialize(new { tokenizer_fixtures_passed = count, user_literal_format_cases_passed = literalInputs.Length }));
        }
    }
    return;
}
if (command == "kernel-self-test")
{
    using var device = new Metal(File.ReadAllText(kernelFile));
    using var input = device.Allocate(256);
    using var output = device.Allocate(256 * sizeof(float));
    for (int i = 0; i < 256; i++) input.Values<byte>(256)[i] = (byte)i;
    device.Begin(); device.Dispatch("fp8_values", new() { N = 256 }, 256, 128, input, output); device.End();
    int checks = 0;
    for (int i = 0; i < 256; i++)
    {
        int exponent = (i >> 3) & 15, mantissa = i & 7;
        float expected = exponent == 0 ? mantissa / 512f : (1 + mantissa / 8f) * MathF.Pow(2, exponent - 7);
        if ((i & 128) != 0) expected = -expected;
        float actual = output.Values<float>(256)[i];
        if (exponent == 15 && mantissa == 7) { if (!float.IsNaN(actual)) throw new Exception("FP8 NaN mismatch"); }
        else if (actual != expected) throw new Exception($"FP8 mismatch byte={i}: {actual} != {expected}");
        checks++;
    }
    const int rows = 8, cols = 256;
    using var x = device.Allocate(cols * 4); using var w = device.Allocate(rows * cols);
    using var scales = device.Allocate((rows + 127) / 128 * (cols / 128) * 2); using var y = device.Allocate(rows * 4);
    for (int i = 0; i < cols; i++) x.Values<float>(cols)[i] = (i % 17 - 8) / 32f;
    for (int i = 0; i < rows * cols; i++) { byte value = (byte)(i * 13 + (i / cols) * 3); if ((value & 127) == 127) value--; w.Values<byte>(rows * cols)[i] = value; }
    scales.Values<ushort>(2)[0] = 0x3f80; scales.Values<ushort>(2)[1] = 0x3f00;
    device.Begin(); device.Dispatch("gemv", new() { N = rows, K = cols, Kind = 0 }, rows * 32, 128, x, w, scales, y); device.End();
    for (int r = 0; r < rows; r++)
    {
        double expected = 0;
        for (int j = 0; j < cols; j++) { int b = w.Values<byte>(rows * cols)[r * cols + j], e = (b >> 3) & 15, m = b & 7; double value = e == 0 ? m / 512.0 : (1 + m / 8.0) * Math.Pow(2, e - 7); if ((b & 128) != 0) value = -value; expected += value * (j < 128 ? 1 : .5) * x.Values<float>(cols)[j]; }
        if (Math.Abs(y.Values<float>(rows)[r] - expected) > 1e-3) throw new Exception("FP8 projection mismatch");
        checks++;
    }
    using var bfWeights = device.Allocate(rows * cols * 2);
    for (int i = 0; i < rows * cols; i++) bfWeights.Values<ushort>(rows * cols)[i] = (ushort)(BitConverter.SingleToUInt32Bits((i % 23 - 11) / 16f) >> 16);
    device.Begin(); device.Dispatch("gemv", new() { N = rows, K = cols, Kind = 1 }, rows * 32, 128, x, bfWeights, scales, y); device.End();
    for (int r = 0; r < rows; r++)
    {
        double expected = 0;
        for (int j = 0; j < cols; j++) expected += BitConverter.UInt32BitsToSingle((uint)bfWeights.Values<ushort>(rows * cols)[r * cols + j] << 16) * (double)x.Values<float>(cols)[j];
        if (Math.Abs(y.Values<float>(rows)[r] - expected) > 1e-5) throw new Exception("BF16 projection mismatch");
        checks++;
    }
    const int matrixRows = 19, matrixColumns = 40;
    using var matrixX = device.Allocate(matrixRows * cols * 4); using var matrixW = device.Allocate(matrixColumns * cols); using var matrixY = device.Allocate(matrixRows * matrixColumns * 4);
    for (int i = 0; i < matrixRows * cols; i++) matrixX.Values<float>(matrixRows * cols)[i] = ((i * 7 + i / cols * 11) % 31 - 15) / 32f;
    for (int i = 0; i < matrixColumns * cols; i++) { byte value = (byte)(i * 13 + i / cols * 29); if ((value & 127) == 127) value--; matrixW.Values<byte>(matrixColumns * cols)[i] = value; }
    device.Begin(); device.Dispatch2D("matrix_product", new() { N = matrixColumns, M = matrixRows, K = cols, Kind = 0 }, ((matrixColumns + 63) / 64) * 256, (matrixRows + 31) / 32, 256, matrixX, matrixW, scales, matrixY); device.End();
    for (int r = 0; r < matrixRows; r++) for (int c = 0; c < matrixColumns; c++)
    {
        double expected = 0;
        for (int j = 0; j < cols; j++) { int b = matrixW.Values<byte>(matrixColumns * cols)[c * cols + j], e = (b >> 3) & 15, m = b & 7; double value = e == 0 ? m / 512.0 : (1 + m / 8.0) * Math.Pow(2, e - 7); if ((b & 128) != 0) value = -value; expected += value * (j < 128 ? 1 : .5) * matrixX.Values<float>(matrixRows * cols)[r * cols + j]; }
        float actual = matrixY.Values<float>(matrixRows * matrixColumns)[r * matrixColumns + c];
        if (Math.Abs(actual - expected) > .002) throw new Exception($"Batched FP8 projection mismatch row={r} col={c}: {actual} != {expected}");
        checks++;
    }
    foreach (int tinyRows in new[] { 1, 2, 3, 4, 7, 8 }) foreach (uint kind in new[] { 0u, 1u })
    {
        device.Begin(); device.Dispatch("matrix_product_tiny", new() { N = rows, M = (uint)tinyRows, K = cols, Kind = kind }, rows * 32, 128, matrixX, kind == 0 ? w : bfWeights, scales, matrixY); device.End();
        for (int r = 0; r < tinyRows; r++) for (int c = 0; c < rows; c++)
        {
            double expected = 0;
            for (int j = 0; j < cols; j++)
            {
                double weight;
                if (kind == 1) weight = BitConverter.UInt32BitsToSingle((uint)bfWeights.Values<ushort>(rows * cols)[c * cols + j] << 16);
                else
                {
                    int bits = w.Values<byte>(rows * cols)[c * cols + j], exponent = (bits >> 3) & 15, mantissa = bits & 7;
                    weight = exponent == 0 ? mantissa / 512.0 : (1 + mantissa / 8.0) * Math.Pow(2, exponent - 7);
                    if ((bits & 128) != 0) weight = -weight;
                    weight *= j < 128 ? 1 : .5;
                }
                expected += weight * matrixX.Values<float>(matrixRows * cols)[r * cols + j];
            }
            if (Math.Abs(matrixY.Values<float>(tinyRows * rows)[r * rows + c] - expected) > (kind == 0 ? .002 : .00001)) throw new Exception($"Tiny matrix reference mismatch: rows={tinyRows}, kind={kind}, row={r}, column={c}");
            checks++;
        }
    }
    const int routeTokens = 5;
    using var routeX = device.Allocate(routeTokens * 256 * 4); using var routeIds = device.Allocate(routeTokens * 8 * 4); using var routeProb = device.Allocate(routeTokens * 8 * 4);
    for (int t = 0; t < routeTokens; t++) for (int i = 0; i < 256; i++) routeX.Values<float>(routeTokens * 256)[t * 256 + i] = t == 0 ? 0 : ((i * 31 + t * 29) % (t * 17 + 11) - 15) / 8f;
    device.Begin(); device.Dispatch("batch_router", new() { K = routeTokens }, routeTokens * 32, 128, routeX, routeIds, routeProb); device.End();
    for (int t = 0; t < routeTokens; t++)
    {
        float[] values = routeX.Values<float>(routeTokens * 256).Slice(t * 256, 256).ToArray();
        int[] expected = Enumerable.Range(0, 256).OrderByDescending(i => values[i]).ThenBy(i => i).Take(8).ToArray();
        double denominator = expected.Sum(i => Math.Exp(values[i] - values[expected[0]]));
        for (int i = 0; i < 8; i++)
        {
            if (routeIds.Values<int>(routeTokens * 8)[t * 8 + i] != expected[i]) throw new Exception("Parallel router tie/selection mismatch");
            if (Math.Abs(routeProb.Values<float>(routeTokens * 8)[t * 8 + i] - Math.Exp(values[expected[i]] - values[expected[0]]) / denominator) > 1e-6) throw new Exception("Parallel router probability mismatch");
            checks += 2;
        }
    }
    const int longContext = ContextLimits.NativeWindow, queryTokens = 2;
    using var longQ = device.Allocate(queryTokens * 4096 * 4); using var longK = device.Allocate((ulong)longContext * 512 * 4); using var longScores = device.Allocate((ulong)queryTokens * 16 * longContext * 4);
    for (int i = 0; i < queryTokens * 4096; i++) longQ.Values<float>(queryTokens * 4096)[i] = (i % 19 - 9) / 32f;
    for (int i = 0; i < longContext * 512; i++) longK.Values<float>(longContext * 512)[i] = (i % 31 - 15) / 32f;
    device.Begin(); device.Dispatch2D("batch_attention_scores", new() { M = queryTokens, Position = longContext - queryTokens, K = longContext }, longContext * 32, queryTokens * 16, 128, longQ, longK, longScores); device.End();
    for (int t = 0; t < queryTokens; t++) for (int h = 0; h < 16; h++) foreach (int position in new[] { 0, 13123, longContext - 2, longContext - 1 })
    {
        float actual = longScores.Values<float>(queryTokens * 16 * longContext)[(t * 16 + h) * longContext + position];
        if (position > longContext - queryTokens + t) { if (!float.IsNegativeInfinity(actual)) throw new Exception("Long-window causal mask mismatch"); }
        else
        {
            double expected = 0;
            for (int d = 0; d < 256; d++) expected += (double)longQ.Values<float>(queryTokens * 4096)[t * 4096 + h * 256 + d] * longK.Values<float>(longContext * 512)[(position * 2 + h / 8) * 256 + d] / 16;
            if (Math.Abs(actual - expected) > 1e-6) throw new Exception("Long-window 2D attention score mismatch");
        }
        checks++;
    }
    foreach (int valid in new[] { 1, 512, 4096, 8192, 32768, 32769, ContextLimits.NativeWindow }) { if (QwenEngine.ValidateContextWindow(valid) != valid) throw new Exception("Valid context rejected"); checks++; }
    foreach (int invalid in new[] { -1, 0, ContextLimits.NativeWindow + 1, int.MaxValue })
    {
        try { QwenEngine.ValidateContextWindow(invalid); throw new Exception("Invalid context accepted"); }
        catch (ArgumentOutOfRangeException) { checks++; }
    }
    foreach (int position in new[] { 0, 4090, 32760, 65530, ContextLimits.NativeWindow - 513 })
    {
        int count = ContextLimits.PrefillCount(position, 512);
        if (count < 1 || count > 512 || (long)count * 16 * (position + count) > ContextLimits.AttentionWorkspaceFloats) throw new Exception("Attention workspace budget exceeded");
        checks++;
    }
    if (ContextLimits.OutputBudget(10, 65536, ContextLimits.NativeWindow) != 65536 || ContextLimits.OutputBudget(10, int.MaxValue, 32768) != 32758 || ContextLimits.OutputBudget(32767, 99999, 32768) != 1) throw new Exception("Output budget clamping mismatch");
    checks += 3;
    Console.WriteLine(JsonSerializer.Serialize(new { passed = checks, allocated_bytes = device.AllocatedBytes, native_model_backend = "original Metal kernels" }));
    return;
}
Console.WriteLine("Bootstrap .NET + Metal engine. Commands: serve, generate, convert-affine8, affine8-self-test, kernel-self-test. See README.md for options and diagnostics.");
