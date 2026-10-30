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
if (command is "generate" or "serve" or "compare-prefill" or "context-self-test" or "tokenize" or "tokenizer-self-test")
{
    var directory = Option("--model-dir") ?? throw new ArgumentException("--model-dir is required");
    using var manifest = JsonDocument.Parse(File.ReadAllBytes(manifestFile));
    if (command is "generate" or "serve" or "compare-prefill" or "context-self-test")
    {
        int contextWindow = QwenEngine.ValidateContextWindow(int.Parse(Option("--context-window") ?? QwenEngine.DefaultContextWindow.ToString()));
        using var engine = new QwenEngine(directory, kernelFile, manifestFile, contextWindow);
        engine.BatchedPrefill = !args.Contains("--sequential-prefill");
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
        Console.Error.WriteLine(); Console.WriteLine(JsonSerializer.Serialize(result));
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
Console.WriteLine("Bootstrap .NET + Metal engine. Commands: kernel-self-test");
