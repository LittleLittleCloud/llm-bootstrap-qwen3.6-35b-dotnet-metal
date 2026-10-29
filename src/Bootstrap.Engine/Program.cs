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
if (command is "generate" or "tokenize" or "tokenizer-self-test")
{
    var directory = Option("--model-dir") ?? throw new ArgumentException("--model-dir is required");
    using var manifest = JsonDocument.Parse(File.ReadAllBytes(manifestFile));
    if (command == "generate")
    {
        using var engine = new QwenEngine(directory, kernelFile, manifestFile);
        var prompt = Option("--prompt") ?? "Say hello in one short sentence.";
        var inputTokens = engine.Tokenizer.SimpleChat(prompt);
        Console.Error.WriteLine($"Model ready; {engine.AllocatedBytes} Metal buffer bytes; {inputTokens.Length} prompt tokens");
        var result = engine.Generate(inputTokens, int.Parse(Option("--max-tokens") ?? "32"), args.Contains("--ignore-eos"), piece => Console.Error.Write(piece));
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
            Console.WriteLine(JsonSerializer.Serialize(new { tokenizer_fixtures_passed = count }));
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
    for (int i = 0; i < rows * cols; i++) w.Values<byte>(rows * cols)[i] = (byte)(i % 64);
    scales.Values<ushort>(2).Fill(0x3f80);
    device.Begin(); device.Dispatch("gemv", new() { N = rows, K = cols, Kind = 0 }, rows * 32, 128, x, w, scales, y); device.End();
    for (int r = 0; r < rows; r++)
    {
        double expected = 0;
        for (int j = 0; j < cols; j++) { int b = w.Values<byte>(rows * cols)[r * cols + j], e = (b >> 3) & 15, m = b & 7; double value = e == 0 ? m / 512.0 : (1 + m / 8.0) * Math.Pow(2, e - 7); expected += value * x.Values<float>(cols)[j]; }
        if (Math.Abs(y.Values<float>(rows)[r] - expected) > 1e-5) throw new Exception("FP8 projection mismatch");
        checks++;
    }
    Console.WriteLine(JsonSerializer.Serialize(new { passed = checks, allocated_bytes = device.AllocatedBytes, native_model_backend = "original Metal kernels" }));
    return;
}
Console.WriteLine("Bootstrap .NET + Metal engine. Commands: kernel-self-test");
