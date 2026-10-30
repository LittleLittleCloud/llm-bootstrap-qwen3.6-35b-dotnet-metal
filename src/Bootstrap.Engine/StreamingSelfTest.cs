using System.Text;
using System.Text.Json.Nodes;

namespace Bootstrap.Engine;

internal static class StreamingSelfTest
{
    static JsonArray Tools() => JsonNode.Parse("""
        [{"type":"function","function":{"name":"record","parameters":{"type":"object","properties":{
        "text":{"type":"string"},"n":{"type":"integer"},"x":{"type":"number"},"flag":{"type":"boolean"},
        "obj":{"type":"object"},"items":{"type":"array"}}}}},
        {"type":"function","function":{"name":"other","parameters":{"type":"object","properties":{}}}}]
        """)!.AsArray();
    const string Fixture = "Hello 😀\n<tool_call>\n<function=record>\n<parameter=text>\n  A \"quote\" \\ slash\n你好😀\u0001  \n</parameter>\n<parameter=n> +12 </parameter>\n<parameter=x> 1.25e2 </parameter>\n<parameter=flag> True </parameter>\n<parameter=obj> { \"nested\": [1, {\"v\": \"x\\ny\"}] } </parameter>\n<parameter=items> [true, null, \"😀\"] </parameter>\n</function>\n</tool_call>\n<tool_call><function=other></function></tool_call>ignored tail";
    static void Expect(bool condition, string message) { if (!condition) throw new Exception(message); }

    public static async Task Run()
    {
        int checks = 0;
        foreach (int split in Enumerable.Range(0, Fixture.Length + 1))
        {
            var rendered = new StringBuilder(); var fragments = new Dictionary<int, StringBuilder>(); var completed = new List<FunctionCall>();
            var parser = new ToolStreamParser(Tools(), s => rendered.Append(s), (i, id, name) => fragments.Add(i, new()),
                (i, s) => fragments[i].Append(s), (i, call) => completed.Add(call));
            parser.Append(Fixture[..split]); parser.Append(Fixture[split..]); var result = parser.Finish();
            Expect(result.Text == "Hello 😀\n" && rendered.ToString() == result.Text, "Split text reconstruction failed");
            Expect(result.Calls.Count == 2 && completed.Count == 2, "Split call count failed");
            var args = result.Calls[0].Arguments;
            Expect(args["text"]!.GetValue<string>() == "A \"quote\" \\ slash\n你好😀\u0001", "String escaping/trim failed");
            Expect(args["n"]!.GetValue<int>() == 12 && args["x"]!.GetValue<double>() == 125 && args["flag"]!.GetValue<bool>(), "Typed scalar conversion failed");
            Expect(args["obj"]?["nested"]?[1]?["v"]?.GetValue<string>() == "x\ny" && args["items"]?[2]?.GetValue<string>() == "😀", "Nested JSON failed");
            for (int i = 0; i < result.Calls.Count; i++) Expect(fragments[i].ToString() == result.Calls[i].JsonArguments, "Final arguments differ from fragments");
            checks += 7;
        }
        foreach (int size in new[] { 1, 2, 7, 53 })
        {
            var parser = new ToolStreamParser(Tools(), _ => { }, (_, _, _) => { }, (_, _) => { }, (_, _) => { });
            for (int at = 0; at < Fixture.Length; at += size) parser.Append(Fixture.Substring(at, Math.Min(size, Fixture.Length - at)));
            var result = parser.Finish(); Expect(result.Calls[0].Arguments["text"]!.GetValue<string>().Contains("你好😀"), "Unicode chunking failed"); checks++;
        }
        foreach (string literal in new[] { "ordinary text 😀", "literal <tool", "literal <function", "<not_a_tool>" })
        {
            var output = new StringBuilder(); var parser = new ToolStreamParser(Tools(), s => output.Append(s), (_, _, _) => throw new Exception("Unexpected call"), (_, _) => { }, (_, _) => { });
            foreach (char c in literal) parser.Append(c.ToString());
            Expect(parser.Finish().Text == literal && output.ToString() == literal, "Literal prefix lost"); checks++;
        }
        foreach (string broken in new[] { "<tool_call>", "<function=record>", "<function=record><parameter=text>unfinished", "<function=unknown>", "<function=record><parameter=n>NaN</parameter>", "<function=record><parameter=text>a</parameter><parameter=text>b</parameter>" })
        {
            int completed = 0; bool failed = false;
            try
            {
                var parser = new ToolStreamParser(Tools(), _ => { }, (_, _, _) => { }, (_, _) => { }, (_, _) => completed++);
                parser.Append(broken); parser.Finish();
            }
            catch (Exception e) when (e is InvalidDataException or FormatException) { failed = true; }
            Expect(failed && completed == 0, "Malformed output became an executable call"); checks++;
        }
        const string stopped = "hé😀 startSTOPignored";
        foreach (int split in Enumerable.Range(0, stopped.Length + 1))
        {
            var text = new StringBuilder(); var buffer = new StopSequenceBuffer(["STOP", "STOPLONG"], s => text.Append(s));
            if (buffer.Append(stopped[..split])) buffer.Append(stopped[split..]); buffer.Finish();
            Expect(text.ToString() == "hé😀 start" && buffer.Matched == "STOP", "Split stop marker leaked"); checks++;
        }
        foreach (string api in new[] { "chat", "responses", "messages" })
        {
            var context = new DefaultHttpContext(); using var stream = new MemoryStream(); context.Response.Body = stream;
            var input = new ChatInput(api, [new("user", "fixture")], Tools(), 100, true, false, [], false, false);
            var writer = new ApiWriter(context, input, new JsonObject { ["stream_options"] = new JsonObject { ["include_usage"] = true } }, 10);
            await writer.Start();
            var parser = new ToolStreamParser(input.Tools, s => writer.Text(s).GetAwaiter().GetResult(),
                (i, id, name) => writer.ToolStart(i, id, name).GetAwaiter().GetResult(),
                (i, s) => writer.ToolArguments(i, s).GetAwaiter().GetResult(), (i, c) => writer.ToolEnd(i, c).GetAwaiter().GetResult());
            parser.Append("Hello 😀\n");
            Expect(Encoding.UTF8.GetString(stream.ToArray()).Contains(api == "chat" ? "chat.completion.chunk" : api == "responses" ? "output_text.delta" : "text_delta"), "Tool-enabled text did not stream early");
            long before = stream.Length;
            parser.Append("<tool_call><function=record><parameter=text>");
            parser.Append(new string('a', 2048));
            Expect(stream.Length > before + 2048, "Arguments were buffered until parameter completion");
            parser.Append("</parameter></function></tool_call><function=other></function>");
            var result = parser.Finish(); await writer.Complete(result.Text, result.Calls, new(result.Text, 10, [1, 2, 3], "stop"), null);
            var events = Encoding.UTF8.GetString(stream.ToArray()).Split('\n').Where(s => s.StartsWith("data: ") && s != "data: [DONE]").Select(s => JsonNode.Parse(s[6..])!.AsObject()).ToList();
            for (int i = 0; i < 2; i++)
            {
                string wire;
                if (api == "chat") wire = string.Concat(events.SelectMany(e => e["choices"]?.AsArray() ?? []).SelectMany(c => c?["delta"]?["tool_calls"]?.AsArray() ?? []).Where(c => c?["index"]?.GetValue<int>() == i).Select(c => c?["function"]?["arguments"]?.GetValue<string>() ?? ""));
                else if (api == "responses") wire = string.Concat(events.Where(e => e["type"]?.GetValue<string>() == "response.function_call_arguments.delta" && e["output_index"]?.GetValue<int>() == i + 1).Select(e => e["delta"]!.GetValue<string>()));
                else wire = string.Concat(events.Where(e => e["type"]?.GetValue<string>() == "content_block_delta" && e["index"]?.GetValue<int>() == i + 1).Select(e => e["delta"]?["partial_json"]?.GetValue<string>() ?? ""));
                Expect(wire == result.Calls[i].JsonArguments, "API argument deltas were duplicated or lost");
            }
            if (api == "responses") Expect(events.Last()["response"]?["output"]?.AsArray().Count == 3, "Final response output differs from stream");
            if (api == "messages") Expect(events.Count(e => e["type"]?.GetValue<string>() == "content_block_stop") == 3, "Blocks not closed once");
            checks += 6;
        }
        Console.WriteLine(System.Text.Json.JsonSerializer.Serialize(new { streaming_checks_passed = checks, apis = new[] { "chat", "responses", "messages" }, scope = "synthetic parser/writer fixtures, no model inference" }));
    }
}
