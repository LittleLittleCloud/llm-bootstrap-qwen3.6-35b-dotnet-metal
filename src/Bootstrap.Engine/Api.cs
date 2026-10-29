using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Bootstrap.Engine;

internal static class Api
{
    public static async Task Run(QwenEngine engine, int port)
    {
        var builder = WebApplication.CreateBuilder(Array.Empty<string>());
        builder.WebHost.ConfigureKestrel(options => { options.ListenLocalhost(port); options.Limits.MaxRequestBodySize = 8 * 1024 * 1024; });
        var app = builder.Build(); var serial = new SemaphoreSlim(1, 1);
        app.MapGet("/v1/models", () => Results.Json(new { @object = "list", data = new[] { new { id = ChatParsing.Model, @object = "model", created = 0, owned_by = "bootstrap" } } }));
        foreach (var (path, api) in new[] { ("/v1/chat/completions", "chat"), ("/v1/responses", "responses"), ("/v1/messages", "messages") })
            app.MapPost(path, async (HttpContext context) =>
            {
                bool acquired = false;
                try
                {
                    var body = await JsonNode.ParseAsync(context.Request.Body, cancellationToken: context.RequestAborted) as JsonObject ?? throw new ArgumentException("Expected a JSON object");
                    var input = ChatParsing.Parse(body, api); int[] prompt = ChatFormatting.Format(engine.Tokenizer, input);
                    if (prompt.Length + input.Maximum > engine.ContextWindow) throw new ArgumentException("Context window exceeded");
                    await serial.WaitAsync(context.RequestAborted); acquired = true;
                    var writer = new ApiWriter(context, input, body, prompt.Length);
                    await writer.Start();
                    var raw = new StringBuilder(); int emitted = 0; string? stopFound = null;
                    int keep = input.Stops.Length == 0 ? 0 : input.Stops.Max(s => s.Length) - 1;
                    var generation = await Task.Run(() => engine.Generate(prompt, input.Maximum, input.IgnoreEos, piece =>
                    {
                        raw.Append(piece); int end = raw.Length;
                        foreach (var stop in input.Stops)
                        {
                            int position = raw.ToString().IndexOf(stop, StringComparison.Ordinal);
                            if (position >= 0 && position < end) { end = position; stopFound = stop; }
                        }
                        int safe = stopFound is null ? Math.Max(emitted, end - keep) : end;
                        if (safe > emitted && input.Tools.Count == 0)
                        {
                            writer.Text(raw.ToString(emitted, safe - emitted)).GetAwaiter().GetResult(); emitted = safe;
                        }
                        return stopFound is null;
                    }, context.RequestAborted), context.RequestAborted);
                    string finalText = generation.Text;
                    if (stopFound is not null) finalText = finalText[..finalText.IndexOf(stopFound, StringComparison.Ordinal)];
                    if (input.Tools.Count == 0 && finalText.Length > emitted) await writer.Text(finalText[emitted..]);
                    var parsed = input.Tools.Count > 0 ? ChatParsing.ParseOutput(finalText, input.Tools) : (finalText, new List<FunctionCall>());
                    await writer.Complete(parsed.Item1, parsed.Item2, generation, stopFound);
                }
                catch (OperationCanceledException) when (context.RequestAborted.IsCancellationRequested) { }
                catch (Exception error) when (!context.Response.HasStarted)
                {
                    int status = error is ArgumentException or JsonException or InvalidOperationException or FormatException ? 400 : 500;
                    context.Response.StatusCode = status;
                    var failure = new Dictionary<string, object?> { ["error"] = new { type = status == 400 ? "invalid_request_error" : "api_error", message = status == 400 ? error.Message : "Inference failed; see local server diagnostics." } };
                    if (api == "messages") failure["type"] = "error";
                    if (status == 500) Console.Error.WriteLine(error);
                    await context.Response.WriteAsJsonAsync(failure);
                }
                catch (Exception error) when (context.Response.HasStarted)
                {
                    if (!context.RequestAborted.IsCancellationRequested) Console.Error.WriteLine("Stream failed: " + error.Message);
                    context.Abort();
                }
                finally { if (acquired) serial.Release(); }
            });
        Console.Error.WriteLine($"Candidate model ready; Metal buffer bytes={engine.AllocatedBytes}; context={engine.ContextWindow}; original .NET + Metal execution only.");
        await app.RunAsync();
    }
}

internal sealed class ApiWriter(HttpContext context, ChatInput input, JsonObject body, int promptTokens)
{
    readonly string id = (input.Api == "chat" ? "chatcmpl_" : input.Api == "responses" ? "resp_" : "msg_") + Guid.NewGuid().ToString("N");
    readonly string itemId = "msg_" + Guid.NewGuid().ToString("N");
    readonly long created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
    int sequence; bool textStarted; readonly StringBuilder streamedText = new();
    static Dictionary<string, object?> D(params (string, object?)[] fields) => fields.ToDictionary(x => x.Item1, x => x.Item2);
    object Usage(int output, bool chat = false) => chat ? new { prompt_tokens = promptTokens, completion_tokens = output, total_tokens = promptTokens + output, prompt_tokens_details = new { cached_tokens = 0 } } : (object)new { input_tokens = promptTokens, output_tokens = output, total_tokens = promptTokens + output };
    object Part(string text) => new { type = "output_text", text, annotations = Array.Empty<object>(), logprobs = Array.Empty<object>() };
    object Item(string text, string status, bool empty = false) => new { id = itemId, type = "message", role = "assistant", status, content = empty ? Array.Empty<object>() : new[] { Part(text) } };
    object Response(string status, object[] output, int tokens) => D(("id", id), ("object", "response"), ("created_at", created), ("status", status), ("model", ChatParsing.Model), ("output", output), ("error", null), ("incomplete_details", status == "incomplete" ? new { reason = "max_output_tokens" } : null), ("usage", status == "in_progress" ? null : Usage(tokens)), ("tools", input.Tools), ("tool_choice", body["tool_choice"] ?? JsonValue.Create("auto")), ("parallel_tool_calls", false), ("temperature", 0), ("top_p", 1), ("metadata", new { }), ("store", false), ("max_output_tokens", input.Maximum));
    object Message(string? reason, object[] content, int tokens, string? stop) => new { id, type = "message", role = "assistant", model = ChatParsing.Model, content, stop_reason = reason, stop_sequence = stop, usage = new { input_tokens = promptTokens, output_tokens = tokens } };
    object Chunk(object delta, string? finish = null) => new { id, @object = "chat.completion.chunk", created, model = ChatParsing.Model, choices = new[] { new { index = 0, delta, finish_reason = finish } } };
    async Task Event(string? name, object value)
    {
        if (!input.Stream) return;
        if (input.Api == "responses" && value is Dictionary<string, object?> fields) fields["sequence_number"] = sequence++;
        else if (input.Api == "responses")
        {
            var node = JsonSerializer.SerializeToNode(value)!.AsObject(); node["sequence_number"] = sequence++; value = node;
        }
        string data = (name is not null ? "event: " + name + "\n" : "") + "data: " + JsonSerializer.Serialize(value) + "\n\n";
        await context.Response.WriteAsync(data, context.RequestAborted); await context.Response.Body.FlushAsync(context.RequestAborted);
    }
    public async Task Start()
    {
        if (!input.Stream) return;
        context.Response.ContentType = "text/event-stream"; context.Response.Headers.CacheControl = "no-cache";
        if (input.Api == "chat") await Event(null, Chunk(new { role = "assistant", content = "" }));
        else if (input.Api == "responses")
        {
            await Event("response.created", new { type = "response.created", response = Response("in_progress", [], 0) });
            await Event("response.in_progress", new { type = "response.in_progress", response = Response("in_progress", [], 0) });
        }
        else await Event("message_start", new { type = "message_start", message = Message(null, [], 0, null) });
    }
    public async Task Text(string text)
    {
        if (text.Length == 0) return;
        if (!textStarted && input.Stream)
        {
            if (input.Api == "responses")
            {
                await Event("response.output_item.added", new { type = "response.output_item.added", output_index = 0, item = Item("", "in_progress", true) });
                await Event("response.content_part.added", new { type = "response.content_part.added", item_id = itemId, output_index = 0, content_index = 0, part = Part("") });
            }
            else if (input.Api == "messages") await Event("content_block_start", new { type = "content_block_start", index = 0, content_block = new { type = "text", text = "" } });
        }
        textStarted = true; streamedText.Append(text);
        if (input.Api == "chat") await Event(null, Chunk(new { content = text }));
        else if (input.Api == "responses") await Event("response.output_text.delta", new { type = "response.output_text.delta", item_id = itemId, output_index = 0, content_index = 0, delta = text });
        else await Event("content_block_delta", new { type = "content_block_delta", index = 0, delta = new { type = "text_delta", text } });
    }
    public async Task Complete(string text, List<FunctionCall> calls, Generation generation, string? stop)
    {
        int count = generation.Tokens.Length;
        string finish = calls.Count > 0 ? "tool_calls" : stop is not null ? "stop" : generation.FinishReason;
        string status = finish == "length" ? "incomplete" : "completed";
        string reason = calls.Count > 0 ? "tool_use" : stop is not null ? "stop_sequence" : finish == "length" ? "max_tokens" : "end_turn";
        object[] toolCalls = calls.Select(c => (object)new { id = c.Id, type = "function", function = new { name = c.Name, arguments = c.Arguments.ToJsonString() } }).ToArray();
        var output = new List<object>();
        if (text.Length > 0) output.Add(Item(text, status));
        output.AddRange(calls.Select(c => (object)new { id = "fc_" + c.Id, type = "function_call", status = "completed", call_id = c.Id, name = c.Name, arguments = c.Arguments.ToJsonString() }));
        var blocks = new List<object>(); if (text.Length > 0) blocks.Add(new { type = "text", text });
        blocks.AddRange(calls.Select(c => (object)new { type = "tool_use", id = c.Id, name = c.Name, input = c.Arguments }));
        if (!input.Stream)
        {
            object value = input.Api switch
            {
                "chat" => new { id, @object = "chat.completion", created, model = ChatParsing.Model, choices = new[] { new { index = 0, message = D(("role", "assistant"), ("content", text.Length == 0 && calls.Count > 0 ? null : text), ("tool_calls", calls.Count > 0 ? toolCalls : null)), finish_reason = finish } }, usage = Usage(count, true) },
                "responses" => Response(status, output.ToArray(), count),
                _ => Message(reason, blocks.ToArray(), count, stop)
            };
            await context.Response.WriteAsJsonAsync(value, context.RequestAborted); return;
        }
        if (input.Tools.Count > 0 && text.Length > 0) await Text(text);
        if (input.Api == "chat")
        {
            for (int i = 0; i < calls.Count; i++)
            {
                var call = calls[i]; await Event(null, Chunk(new { tool_calls = new[] { new { index = i, id = call.Id, type = "function", function = new { name = call.Name, arguments = call.Arguments.ToJsonString() } } } }));
            }
            await Event(null, Chunk(new { }, finish));
            if (body["stream_options"]?["include_usage"]?.GetValue<bool>() == true) await Event(null, new { id, @object = "chat.completion.chunk", created, model = ChatParsing.Model, choices = Array.Empty<object>(), usage = Usage(count, true) });
            await context.Response.WriteAsync("data: [DONE]\n\n", context.RequestAborted); await context.Response.Body.FlushAsync(context.RequestAborted);
        }
        else if (input.Api == "responses")
        {
            if (textStarted)
            {
                await Event("response.output_text.done", new { type = "response.output_text.done", item_id = itemId, output_index = 0, content_index = 0, text });
                await Event("response.content_part.done", new { type = "response.content_part.done", item_id = itemId, output_index = 0, content_index = 0, part = Part(text) });
                await Event("response.output_item.done", new { type = "response.output_item.done", output_index = 0, item = Item(text, status) });
            }
            int offset = text.Length > 0 ? 1 : 0;
            for (int i = 0; i < calls.Count; i++)
            {
                var call = calls[i]; string item = "fc_" + call.Id; int index = offset + i; string arguments = call.Arguments.ToJsonString();
                await Event("response.output_item.added", new { type = "response.output_item.added", output_index = index, item = new { id = item, type = "function_call", status = "in_progress", call_id = call.Id, name = call.Name, arguments = "" } });
                await Event("response.function_call_arguments.delta", new { type = "response.function_call_arguments.delta", item_id = item, output_index = index, delta = arguments });
                await Event("response.function_call_arguments.done", new { type = "response.function_call_arguments.done", item_id = item, output_index = index, arguments });
                await Event("response.output_item.done", new { type = "response.output_item.done", output_index = index, item = output[index] });
            }
            await Event("response." + status, new { type = "response." + status, response = Response(status, output.ToArray(), count) });
        }
        else
        {
            if (textStarted) await Event("content_block_stop", new { type = "content_block_stop", index = 0 });
            int offset = text.Length > 0 ? 1 : 0;
            for (int i = 0; i < calls.Count; i++)
            {
                var call = calls[i]; int index = offset + i;
                await Event("content_block_start", new { type = "content_block_start", index, content_block = new { type = "tool_use", id = call.Id, name = call.Name, input = new { } } });
                await Event("content_block_delta", new { type = "content_block_delta", index, delta = new { type = "input_json_delta", partial_json = call.Arguments.ToJsonString() } });
                await Event("content_block_stop", new { type = "content_block_stop", index });
            }
            await Event("message_delta", new { type = "message_delta", delta = new { stop_reason = reason, stop_sequence = stop }, usage = new { output_tokens = count } });
            await Event("message_stop", new { type = "message_stop" });
        }
    }
}
