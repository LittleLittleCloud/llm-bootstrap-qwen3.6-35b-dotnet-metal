using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;

namespace Bootstrap.Engine;

internal sealed record FunctionCall(string Id, string Name, JsonObject Arguments);
internal sealed record ChatMessage(string Role, string Text, IReadOnlyList<FunctionCall>? Calls = null);
internal sealed record ChatInput(string Api, List<ChatMessage> Messages, JsonArray Tools, int Maximum, bool Stream, bool IgnoreEos, string[] Stops, bool ForceTool, bool JsonObject);

internal static class ChatParsing
{
    public const string Model = "qwen3.6-35b-a3b";
    static string Text(JsonNode? node)
    {
        if (node is null) return "";
        if (node is JsonValue value) return value.GetValue<string>();
        if (node is not JsonArray array) throw new ArgumentException("Content must be text or content blocks");
        return string.Concat(array.Where(p => p?["type"]?.GetValue<string>() is "text" or "input_text" or "output_text").Select(p => p?["text"]?.GetValue<string>() ?? ""));
    }
    static FunctionCall OpenAiCall(JsonNode node)
    {
        var function = node["function"] ?? throw new ArgumentException("Missing function definition");
        var arguments = function["arguments"];
        return new(node["id"]?.GetValue<string>() ?? "call_" + Guid.NewGuid().ToString("N"), function["name"]!.GetValue<string>(), arguments is JsonObject obj ? (JsonObject)obj.DeepClone() : JsonNode.Parse(arguments?.GetValue<string>() ?? "{}") as JsonObject ?? throw new ArgumentException("Function arguments must be an object"));
    }
    static void AddMessage(List<ChatMessage> output, JsonNode item, string api)
    {
        string role = item["role"]?.GetValue<string>() ?? throw new ArgumentException("Missing role");
        if (role is not ("system" or "developer" or "user" or "assistant" or "tool")) throw new ArgumentException("Unknown role");
        if (role == "developer") role = "system";
        var content = item["content"];
        if (api == "messages" && content is JsonArray blocks)
        {
            var calls = blocks.Where(p => p?["type"]?.GetValue<string>() == "tool_use").Select(p => new FunctionCall(p!["id"]!.GetValue<string>(), p["name"]!.GetValue<string>(), (JsonObject)p["input"]!.DeepClone())).ToArray();
            string text = Text(content);
            if (calls.Length > 0 || text.Length > 0) output.Add(new(role, text, calls.Length > 0 ? calls : null));
            foreach (var block in blocks.Where(p => p?["type"]?.GetValue<string>() == "tool_result")) output.Add(new("tool", Text(block!["content"])));
        }
        else
        {
            var calls = (item["tool_calls"] as JsonArray)?.Select(p => OpenAiCall(p!)).ToArray();
            output.Add(new(role, Text(content), calls));
        }
    }
    public static ChatInput Parse(JsonObject body, string api)
    {
        if (body["model"]?.GetValue<string>() != Model) throw new ArgumentException("Unknown model");
        string budgetName = api == "responses" ? "max_output_tokens" : "max_tokens";
        if (api == "messages" && body[budgetName] is null) throw new ArgumentException("max_tokens is required");
        int maximum = body[budgetName]?.GetValue<int>() ?? 64;
        if (maximum < 1 || maximum > 4096) throw new ArgumentException("Invalid output token budget");
        var messages = new List<ChatMessage>();
        if (api == "responses")
        {
            var input = body["input"] ?? throw new ArgumentException("input is required");
            if (body["instructions"] is not null) messages.Add(new("system", body["instructions"]!.GetValue<string>()));
            if (input is JsonValue) messages.Add(new("user", input.GetValue<string>()));
            else if (input is JsonArray array)
                foreach (var item in array)
                {
                    switch (item?["type"]?.GetValue<string>() ?? "message")
                    {
                        case "message": AddMessage(messages, item!, api); break;
                        case "function_call": messages.Add(new("assistant", "", [new FunctionCall(item!["call_id"]!.GetValue<string>(), item["name"]!.GetValue<string>(), JsonNode.Parse(item["arguments"]!.GetValue<string>()) as JsonObject ?? throw new ArgumentException("Function arguments must be an object"))])); break;
                        case "function_call_output": messages.Add(new("tool", Text(item!["output"]))); break;
                        default: throw new ArgumentException("Unsupported input item");
                    }
                }
            else throw new ArgumentException("Invalid input");
        }
        else
        {
            if (body["messages"] is not JsonArray array || array.Count == 0) throw new ArgumentException("messages must be a nonempty array");
            if (api == "messages" && body["system"] is not null) messages.Add(new("system", Text(body["system"])));
            foreach (var item in array) AddMessage(messages, item ?? throw new ArgumentException("Invalid message"), api);
        }
        if (!messages.Any(m => m.Role == "user")) throw new ArgumentException("At least one user message is required");
        var tools = new JsonArray();
        foreach (var tool in body["tools"] as JsonArray ?? [])
        {
            if (tool is null) throw new ArgumentException("Invalid tool");
            if (api == "chat") tools.Add(tool.DeepClone());
            else tools.Add(new JsonObject { ["type"] = "function", ["function"] = new JsonObject { ["name"] = tool["name"]?.DeepClone(), ["description"] = tool["description"]?.DeepClone() ?? "", ["parameters"] = tool[api == "messages" ? "input_schema" : "parameters"]?.DeepClone() ?? new JsonObject { ["type"] = "object", ["properties"] = new JsonObject() } } });
        }
        string? choice = body["tool_choice"] is JsonValue ? body["tool_choice"]!.GetValue<string>() : null;
        string? selected = body["tool_choice"] is JsonObject select ? select["name"]?.GetValue<string>() ?? select["function"]?["name"]?.GetValue<string>() : null;
        if (body["tool_choice"] is JsonObject choiceObject && choiceObject["type"]?.GetValue<string>() == "none") choice = "none";
        if (choice == "none") tools.Clear();
        if (selected is not null)
        {
            tools = new JsonArray(tools.Where(t => t?["function"]?["name"]?.GetValue<string>() == selected).Select(t => t!.DeepClone()).ToArray());
            if (tools.Count == 0) throw new ArgumentException("Selected tool is not defined");
        }
        bool force = selected is not null || choice is "required" || body["tool_choice"] is JsonObject any && any["type"]?.GetValue<string>() == "any";
        JsonNode? stop = body[api == "messages" ? "stop_sequences" : "stop"];
        string[] stops = stop is JsonArray list ? list.Select(s => s!.GetValue<string>()).ToArray() : stop is JsonValue ? [stop.GetValue<string>()] : [];
        if (stops.Any(s => s.Length == 0)) throw new ArgumentException("Stop strings cannot be empty");
        return new(api, messages, tools, maximum, body["stream"]?.GetValue<bool>() ?? false, body["ignore_eos"]?.GetValue<bool>() ?? false, stops, force, body["response_format"]?["type"]?.GetValue<string>() == "json_object");
    }
    static readonly Regex Functions = new(@"<function=([^>]+)>(.*?)</function>", RegexOptions.Singleline | RegexOptions.Compiled, TimeSpan.FromSeconds(1));
    static readonly Regex Parameters = new(@"<parameter=([^>]+)>(.*?)</parameter>", RegexOptions.Singleline | RegexOptions.Compiled, TimeSpan.FromSeconds(1));
    public static (string Text, List<FunctionCall> Calls) ParseOutput(string text, JsonArray tools)
    {
        var calls = new List<FunctionCall>();
        foreach (Match function in Functions.Matches(text))
        {
            string name = function.Groups[1].Value.Trim();
            var definition = tools.FirstOrDefault(t => t?["function"]?["name"]?.GetValue<string>() == name)?["function"];
            if (definition is null) continue;
            var parameters = new JsonObject();
            foreach (Match parameter in Parameters.Matches(function.Groups[2].Value))
            {
                string key = parameter.Groups[1].Value.Trim(), value = parameter.Groups[2].Value.Trim();
                string? type = definition["parameters"]?["properties"]?[key]?["type"]?.GetValue<string>();
                parameters[key] = type switch
                {
                    "integer" => JsonValue.Create(long.Parse(value, System.Globalization.CultureInfo.InvariantCulture)),
                    "number" => JsonValue.Create(double.Parse(value, System.Globalization.CultureInfo.InvariantCulture)),
                    "boolean" => JsonValue.Create(bool.Parse(value)),
                    "object" or "array" => JsonNode.Parse(value),
                    _ => JsonValue.Create(value)
                };
            }
            calls.Add(new("call_" + Guid.NewGuid().ToString("N"), name, parameters));
        }
        if (calls.Count > 0)
        {
            int start = text.IndexOf("<tool_call>", StringComparison.Ordinal);
            if (start < 0) start = text.IndexOf("<function=", StringComparison.Ordinal);
            text = text[..start].Trim();
        }
        return (text, calls);
    }
}

internal static class ChatFormatting
{
    const string ToolInstructions = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";
    static readonly System.Text.Json.JsonSerializerOptions JsonOptions = new() { Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping };
    static string Json(JsonNode? value) => value switch
    {
        JsonObject obj => "{" + string.Join(", ", obj.Select(p => System.Text.Json.JsonSerializer.Serialize(p.Key, JsonOptions) + ": " + Json(p.Value))) + "}",
        JsonArray array => "[" + string.Join(", ", array.Select(Json)) + "]",
        null => "null",
        _ => value.ToJsonString(JsonOptions)
    };
    public static int[] Format(ByteBpe tokenizer, ChatInput input)
    {
        var ids = new List<int>();
        void Plain(string value) => ids.AddRange(tokenizer.Encode(value));
        void Control(string value) => ids.Add(tokenizer.Special(value));
        void Markup(string value)
        {
            string[] tags = ["<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>", "<think>", "</think>"];
            int at = 0;
            while (at < value.Length)
            {
                int closest = value.Length; string? tag = null;
                foreach (var candidate in tags) { int found = value.IndexOf(candidate, at, StringComparison.Ordinal); if (found >= 0 && found < closest) { closest = found; tag = candidate; } }
                Plain(value[at..closest]);
                if (tag is null) break;
                Control(tag); at = closest + tag.Length;
            }
        }
        string system = string.Join("\n\n", input.Messages.Where(m => m.Role == "system").Select(m => m.Text.Trim()).Where(t => t.Length > 0));
        if (input.JsonObject) system += "\n\nReturn a valid JSON object without a Markdown code fence.";
        if (input.ForceTool) system += "\n\nYou must call one of the provided functions; do not answer directly.";
        if (input.Tools.Count > 0 || system.Length > 0)
        {
            Control("<|im_start|>"); Plain("system\n");
            if (input.Tools.Count > 0) { Plain("# Tools\n\nYou have access to the following functions:\n\n<tools>"); foreach (var tool in input.Tools) Plain("\n" + Json(tool)); Plain("\n</tools>"); Markup(ToolInstructions); if (system.Length > 0) Plain("\n\n"); }
            Plain(system.Trim()); Control("<|im_end|>"); Plain("\n");
        }
        int lastUser = input.Messages.FindLastIndex(m => m.Role == "user");
        for (int i = 0; i < input.Messages.Count; i++)
        {
            var message = input.Messages[i]; if (message.Role == "system") continue;
            if (message.Role == "tool")
            {
                if (i == 0 || input.Messages[i - 1].Role != "tool") { Control("<|im_start|>"); Plain("user"); }
                Plain("\n"); Control("<tool_response>"); Plain("\n" + message.Text.Trim() + "\n"); Control("</tool_response>");
                if (i + 1 == input.Messages.Count || input.Messages[i + 1].Role != "tool") { Control("<|im_end|>"); Plain("\n"); }
                continue;
            }
            Control("<|im_start|>"); Plain(message.Role + "\n");
            if (message.Role == "assistant" && i > lastUser) { Control("<think>"); Plain("\n\n"); Control("</think>"); Plain("\n\n"); }
            string content = message.Text.Trim(); int endThink = content.LastIndexOf("</think>", StringComparison.Ordinal); if (endThink >= 0) content = content[(endThink + 8)..].TrimStart('\n');
            Plain(content);
            if (message.Calls is { Count: > 0 })
            {
                bool first = true;
                foreach (var call in message.Calls)
                {
                    Plain(first && content.Length > 0 ? "\n\n" : first ? "" : "\n"); first = false;
                    Control("<tool_call>"); Plain("\n<function=" + call.Name + ">\n");
                    foreach (var parameter in call.Arguments) Plain("<parameter=" + parameter.Key + ">\n" + (parameter.Value is JsonValue v && v.TryGetValue<string>(out var s) ? s : Json(parameter.Value)) + "\n</parameter>\n");
                    Plain("</function>\n"); Control("</tool_call>");
                }
            }
            Control("<|im_end|>"); Plain("\n");
        }
        Control("<|im_start|>"); Plain("assistant\n"); Control("<think>"); Plain("\n\n"); Control("</think>"); Plain("\n\n");
        return ids.ToArray();
    }
}
