using System.Diagnostics;
using System.Text.Json;
namespace Bootstrap.Engine;
internal sealed partial class QwenEngine
{
    // Public fixed-length prompt from endpoint-bench. This is a development A/B,
    // not a browser evaluation report; total generation time excludes model load.
    public void CompareDecodeVariants(int runs)
    {
        if (draft is null || runs < 1) throw new ArgumentException("Draft and positive run count required");
        var tokens = Tokenizer.SimpleChat("Generate a continuous stream of lowercase English words separated by single spaces. Do not use punctuation, numbers, headings, explanations, or formatting. Begin immediately and continue generating until stopped.");
        TensorOpsPrecision = "fp16"; WidePrefill = true; PrefillBatchSize = 2048;
        for (int run = 0; run <= runs; run++) foreach (bool optimized in run % 2 == 0 ? new[] { false, true } : new[] { true, false })
        {
            AttentionMode = optimized ? "fp16-device" : "fp16"; OptimizedProjections = optimized;
            PredecodedProjections = optimized; PredecodedExperts = false; PrefillBatchSize = optimized ? 2048 : 512; ExpertTileRows = optimized ? 64 : 16; ClearPrefixCache();
            var watch=Stopwatch.StartNew();double firstMs=-1,lastMs=0;
            var result=Generate(tokens,1000,true,piece => { lastMs=watch.Elapsed.TotalMilliseconds; if(firstMs<0)firstMs=lastMs;return true; });
            double elapsed=watch.Elapsed.TotalMilliseconds;
            if(result.Tokens.Length!=1000)throw new Exception("Fixed-length decode ended early");
            Console.WriteLine(JsonSerializer.Serialize(new {type="decode_variant",kv_cache_precision=KvCachePrecision,run,variant=optimized?"optimized_attention_small_verifier":"previous_fast_prefill",warm=run>0,prompt_tokens=tokens.Length,output_tokens=result.Tokens.Length,wall_ms=elapsed,first_delta_ms=firstMs,last_delta_ms=lastMs,decode_tokens_per_second=(result.Tokens.Length-1)*1000/(lastMs-firstMs),speculation=LastSpeculationStats}));
        }
    }
}
