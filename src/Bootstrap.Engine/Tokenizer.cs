using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Security.Cryptography;

namespace Bootstrap.Engine;

// Original byte-BPE implementation using the checkpoint's vocabulary, merges,
// and declared pretokenization expression. Ordinary input never injects specials.
internal sealed class ByteBpe
{
    readonly Dictionary<string, int> vocabulary;
    readonly Dictionary<(string, string), int> ranks = new();
    readonly Dictionary<string, int> specials = new(StringComparer.Ordinal);
    readonly byte[][] decoded;
    readonly char[] byteToChar = new char[256];
    readonly Regex split;
    readonly Dictionary<string, int[]> cache = new(StringComparer.Ordinal);
    public int End => specials["<|im_end|>"];
    public int EndOfText => specials["<|endoftext|>"];
    public ByteBpe(string path, string expectedHash)
    {
        byte[] data = File.ReadAllBytes(path);
        if (Convert.ToHexStringLower(SHA256.HashData(data)) != expectedHash) throw new InvalidDataException("Tokenizer hash mismatch");
        using var doc = JsonDocument.Parse(data); var root = doc.RootElement;
        vocabulary = root.GetProperty("model").GetProperty("vocab").EnumerateObject().ToDictionary(p => p.Name, p => p.Value.GetInt32(), StringComparer.Ordinal);
        var bytes = Enumerable.Range(33, 94).Concat(Enumerable.Range(161, 12)).Concat(Enumerable.Range(174, 82)).ToList();
        var characters = new List<int>(bytes); int extra = 0;
        for (int i = 0; i < 256; i++) if (!bytes.Contains(i)) { bytes.Add(i); characters.Add(256 + extra++); }
        var reverse = new Dictionary<char, byte>();
        for (int i = 0; i < 256; i++) { byteToChar[bytes[i]] = (char)characters[i]; reverse[(char)characters[i]] = (byte)bytes[i]; }
        int rank = 0;
        foreach (var item in root.GetProperty("model").GetProperty("merges").EnumerateArray())
        {
            string[] pair = item.ValueKind == JsonValueKind.Array ? item.EnumerateArray().Select(x => x.GetString()!).ToArray() : item.GetString()!.Split(' ');
            if (pair.Length != 2) throw new InvalidDataException("Invalid tokenizer merge");
            ranks.Add((pair[0], pair[1]), rank++);
        }
        foreach (var item in root.GetProperty("added_tokens").EnumerateArray()) specials[item.GetProperty("content").GetString()!] = item.GetProperty("id").GetInt32();
        decoded = new byte[Math.Max(vocabulary.Values.Max(), specials.Values.Max()) + 1][];
        foreach (var (text, id) in vocabulary) decoded[id] = text.Select(c => reverse[c]).ToArray();
        foreach (var (text, id) in specials) decoded[id] = Encoding.UTF8.GetBytes(text);
        string pattern = root.GetProperty("pre_tokenizer").GetProperty("pretokenizers")[0].GetProperty("pattern").GetProperty("Regex").GetString()!;
        split = new Regex(pattern, RegexOptions.Compiled | RegexOptions.CultureInvariant, TimeSpan.FromSeconds(5));
    }
    public int Special(string text) => specials[text];
    public byte[] TokenBytes(int id) => decoded[id];
    public string Decode(IEnumerable<int> ids) => Encoding.UTF8.GetString(ids.SelectMany(TokenBytes).ToArray());
    public int[] Encode(string text)
    {
        var output = new List<int>(); int end = 0;
        foreach (Match match in split.Matches(text))
        {
            if (match.Index != end) throw new InvalidDataException("Pretokenizer omitted input characters");
            end = match.Index + match.Length;
            string token = string.Concat(Encoding.UTF8.GetBytes(match.Value).Select(b => byteToChar[b]));
            if (!cache.TryGetValue(token, out var ids))
            {
                if (vocabulary.TryGetValue(token, out var direct)) ids = [direct];
                else
                {
                    var pieces = token.Select(c => c.ToString()).ToList();
                    while (pieces.Count > 1)
                    {
                        int best = int.MaxValue, at = -1;
                        for (int i = 0; i < pieces.Count - 1; i++) if (ranks.TryGetValue((pieces[i], pieces[i + 1]), out int r) && r < best) { best = r; at = i; }
                        if (at < 0) break;
                        pieces[at] += pieces[at + 1]; pieces.RemoveAt(at + 1);
                    }
                    ids = pieces.Select(p => vocabulary[p]).ToArray();
                }
                if (cache.Count < 100_000) cache[token] = ids;
            }
            output.AddRange(ids);
        }
        if (end != text.Length) throw new InvalidDataException("Pretokenizer omitted trailing input");
        return output.ToArray();
    }
    public int[] SimpleChat(string user)
    {
        var tokens = new List<int> { Special("<|im_start|>") };
        tokens.AddRange(Encode("user\n")); tokens.AddRange(Encode(user.Trim())); tokens.Add(Special("<|im_end|>"));
        tokens.AddRange(Encode("\n")); tokens.Add(Special("<|im_start|>")); tokens.AddRange(Encode("assistant\n"));
        tokens.Add(Special("<think>")); tokens.AddRange(Encode("\n\n")); tokens.Add(Special("</think>")); tokens.AddRange(Encode("\n\n"));
        return tokens.ToArray();
    }
}
