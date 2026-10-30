using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Bootstrap.Engine;

// Only ambiguous markup suffixes and atomic numeric/boolean values are held back.
// String values are JSON-escaped as they arrive; object/array values retain their
// original JSON spelling so final wire arguments exactly match their deltas.
internal sealed class ToolStreamParser(JsonArray tools, Action<string> text, Action<int, string, string> begin,
    Action<int, string> delta, Action<int, FunctionCall> end)
{
    enum Phase { Text, Markup, FunctionName, Parameters, ParameterName, Value }
    Phase phase;
    string pending = "", callId = "", functionName = "", valueType = "";
    JsonNode? definition;
    readonly StringBuilder plain = new(), arguments = new(), value = new(), whitespace = new();
    readonly HashSet<string> keys = new(StringComparer.Ordinal);
    readonly List<FunctionCall> calls = new();
    bool valueStarted, firstParameter = true;
    bool HasCall => calls.Count > 0;
    const string OpenTool = "<tool_call>", OpenFunction = "<function=", OpenParameter = "<parameter=", CloseParameter = "</parameter>", CloseFunction = "</function>", CloseTool = "</tool_call>";

    static int SafeLength(string input, params string[] markers)
    {
        int hold = 0;
        foreach (string marker in markers)
            for (int n = 1; n < marker.Length && n <= input.Length; n++)
                if (input.EndsWith(marker[..n], StringComparison.Ordinal)) hold = Math.Max(hold, n);
        int length = input.Length - hold;
        if (length > 0 && char.IsHighSurrogate(input[length - 1])) length--;
        return length;
    }
    void EmitText(string part) { if (part.Length > 0) { plain.Append(part); text(part); } }
    void Argument(string part) { if (part.Length > 0) { arguments.Append(part); delta(calls.Count, part); } }
    static string Escape(string part)
    {
        var result = new StringBuilder();
        foreach (char c in part)
            if (c == '"') result.Append("\\\"");
            else if (c == '\\') result.Append("\\\\");
            else if (c < 32 || char.IsSurrogate(c)) result.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
            else result.Append(c);
        return result.ToString();
    }
    void ValuePart(string part)
    {
        value.Append(part);
        if (valueType is "integer" or "number" or "boolean") return;
        var ready = new StringBuilder();
        foreach (char c in part)
        {
            if (char.IsWhiteSpace(c)) { if (valueStarted) whitespace.Append(c); }
            else { ready.Append(whitespace); whitespace.Clear(); ready.Append(c); valueStarted = true; }
        }
        Argument(valueType is "object" or "array" ? ready.ToString() : Escape(ready.ToString()));
    }
    void EndValue()
    {
        string raw = value.ToString().Trim();
        switch (valueType)
        {
            case "integer": Argument(long.Parse(raw, CultureInfo.InvariantCulture).ToString(CultureInfo.InvariantCulture)); break;
            case "number": Argument(JsonSerializer.Serialize(double.Parse(raw, CultureInfo.InvariantCulture))); break;
            case "boolean": Argument(bool.Parse(raw) ? "true" : "false"); break;
            case "object": case "array": _ = JsonNode.Parse(raw); break;
            default: Argument("\""); break;
        }
        value.Clear(); whitespace.Clear();
    }
    public void Append(string part)
    {
        pending += part;
        while (pending.Length > 0)
        {
            if (phase is Phase.Text or Phase.Markup)
            {
                string[] markers = phase == Phase.Text ? [OpenTool, OpenFunction] : [OpenTool, OpenFunction, CloseTool];
                var found = markers.Select(m => (Marker: m, At: pending.IndexOf(m, StringComparison.Ordinal))).Where(x => x.At >= 0).OrderBy(x => x.At).FirstOrDefault();
                if (found.Marker is null)
                {
                    int safe = SafeLength(pending, markers);
                    if (phase == Phase.Text) EmitText(pending[..safe]);
                    else if (!HasCall && !string.IsNullOrWhiteSpace(pending[..safe])) throw new InvalidDataException("Expected a function inside tool_call");
                    pending = pending[safe..]; return;
                }
                if (phase == Phase.Text) EmitText(pending[..found.At]);
                else if (!HasCall && !string.IsNullOrWhiteSpace(pending[..found.At])) throw new InvalidDataException("Expected a function inside tool_call");
                pending = pending[(found.At + found.Marker.Length)..];
                phase = found.Marker == OpenFunction ? Phase.FunctionName : Phase.Markup;
            }
            else if (phase == Phase.FunctionName)
            {
                int at = pending.IndexOf('>'); if (at < 0) return;
                functionName = pending[..at].Trim(); pending = pending[(at + 1)..];
                definition = tools.FirstOrDefault(t => t?["function"]?["name"]?.GetValue<string>() == functionName)?["function"];
                if (definition is null) throw new InvalidDataException("Model emitted an undefined function");
                callId = "call_" + Guid.NewGuid().ToString("N"); arguments.Clear(); keys.Clear(); firstParameter = true;
                begin(calls.Count, callId, functionName); Argument("{"); phase = Phase.Parameters;
            }
            else if (phase == Phase.Parameters)
            {
                pending = pending.TrimStart(); if (pending.Length == 0) return;
                if (pending.StartsWith(CloseFunction, StringComparison.Ordinal))
                {
                    pending = pending[CloseFunction.Length..]; Argument("}");
                    var call = new FunctionCall(callId, functionName, JsonNode.Parse(arguments.ToString())!.AsObject(), arguments.ToString());
                    end(calls.Count, call); calls.Add(call); phase = Phase.Markup;
                }
                else if (pending.StartsWith(OpenParameter, StringComparison.Ordinal)) { pending = pending[OpenParameter.Length..]; phase = Phase.ParameterName; }
                else if (CloseFunction.StartsWith(pending, StringComparison.Ordinal) || OpenParameter.StartsWith(pending, StringComparison.Ordinal)) return;
                else throw new InvalidDataException("Invalid function parameter markup");
            }
            else if (phase == Phase.ParameterName)
            {
                int at = pending.IndexOf('>'); if (at < 0) return;
                string name = pending[..at].Trim(); pending = pending[(at + 1)..];
                if (name.Length == 0 || !keys.Add(name)) throw new InvalidDataException("Empty or duplicate function parameter");
                valueType = definition?["parameters"]?["properties"]?[name]?["type"]?.GetValue<string>() ?? "string";
                Argument((firstParameter ? "" : ",") + JsonSerializer.Serialize(name) + ":"); firstParameter = false;
                value.Clear(); whitespace.Clear(); valueStarted = false;
                if (valueType is not ("integer" or "number" or "boolean" or "object" or "array")) Argument("\"");
                phase = Phase.Value;
            }
            else
            {
                int at = pending.IndexOf(CloseParameter, StringComparison.Ordinal);
                if (at < 0)
                {
                    int safe = SafeLength(pending, CloseParameter); ValuePart(pending[..safe]); pending = pending[safe..]; return;
                }
                ValuePart(pending[..at]); pending = pending[(at + CloseParameter.Length)..]; EndValue(); phase = Phase.Parameters;
            }
        }
    }
    public (string Text, List<FunctionCall> Calls) Finish()
    {
        if (phase == Phase.Text) { EmitText(pending); pending = ""; }
        else if (phase != Phase.Markup || calls.Count == 0) throw new InvalidDataException("Incomplete function call at end of generation");
        return (plain.ToString(), calls);
    }
}

internal sealed class StopSequenceBuffer(string[] stops, Action<string> emit)
{
    readonly StringBuilder pending = new();
    readonly int keep = stops.Length == 0 ? 0 : stops.Max(s => s.Length) - 1;
    public string? Matched { get; private set; }
    public bool Append(string piece)
    {
        pending.Append(piece); string tail = pending.ToString(); int end = tail.Length;
        foreach (string stop in stops)
        {
            int at = tail.IndexOf(stop, StringComparison.Ordinal);
            if (at >= 0 && at < end) { end = at; Matched = stop; }
        }
        int safe = Matched is null ? Math.Max(0, end - keep) : end;
        if (safe > 0 && safe < tail.Length && char.IsHighSurrogate(tail[safe - 1]) && char.IsLowSurrogate(tail[safe])) safe--;
        if (safe > 0) { emit(tail[..safe]); pending.Remove(0, safe); }
        return Matched is null;
    }
    public void Finish() { if (Matched is null && pending.Length > 0) { emit(pending.ToString()); pending.Clear(); } }
}
