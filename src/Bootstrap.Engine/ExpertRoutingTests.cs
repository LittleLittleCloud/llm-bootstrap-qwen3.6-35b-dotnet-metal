using System.Diagnostics;
using System.Text.Json;
namespace Bootstrap.Engine;

internal static class ExpertRoutingTests
{
    public static void Run(string source, int runs)
    {
        using var gpu = new Metal(source);
        foreach (int m in new[] { 1, 7, 32, 257, 2048 })
        foreach (int tileRows in new[] { 16, 32, 64 })
        foreach (bool skew in new[] { false, true })
        {
            int routes = m * 8, maxTiles = routes / tileRows + 256;
            using var ids = gpu.Allocate((ulong)routes * 4);
            using var counts = gpu.Allocate(256 * 4);
            using var offsets = gpu.Allocate(257 * 4);
            using var tiles = gpu.Allocate((ulong)maxTiles * 16);
            using var tileCount = gpu.Allocate(4);
            using var tokens = gpu.Allocate((ulong)(routes + 1) * 4);
            using var inverse = gpu.Allocate((ulong)(routes + 1) * 4);
            for (int r = 0; r < routes; r++) ids.Values<int>(routes)[r] = skew ? r % 8 : (r * 31 + r / 8 * 7) % 256;
            var p = new KernelArgs { M = (uint)m, A = (uint)tileRows };
            foreach (bool parallel in new[] { false, true })
            for (int run = 0; run <= runs; run++)
            {
                tokens.Values<uint>(routes + 1).Fill(uint.MaxValue);
                inverse.Values<uint>(routes + 1).Fill(uint.MaxValue);
                gpu.ResetWallTiming(); var watch = Stopwatch.StartNew(); gpu.Begin();
                if (parallel) gpu.Dispatch("expert_group_parallel", p, 256, 256, ids, counts, offsets, tiles, tileCount, tokens, inverse);
                else {
                    gpu.Dispatch("expert_counts", p, 256, 128, ids, counts);
                    gpu.Dispatch("expert_tiles", p, 1, 1, counts, offsets, tiles, tileCount);
                    gpu.Dispatch("expert_group", p, 256, 128, ids, offsets, tokens, inverse);
                }
                gpu.End(); double wall = watch.Elapsed.TotalMilliseconds;
                // Validate every route, tile and output guard, including empty experts.
                int expectedFirst = 0, expectedTile = 0;
                bool[] seen = new bool[routes];
                for (int e = 0; e < 256; e++) {
                    int count = 0;
                    for (int r = 0; r < routes; r++) if (ids.Values<int>(routes)[r] == e) {
                        int at = checked((int)inverse.Values<uint>(routes)[r]);
                        if (at < expectedFirst || at >= routes || seen[at] || tokens.Values<uint>(routes)[at] != r / 8) throw new Exception("Invalid routed permutation");
                        seen[at] = true; count++;
                    }
                    if (counts.Values<uint>(256)[e] != count || offsets.Values<uint>(257)[e] != expectedFirst) throw new Exception("Invalid counts/offsets");
                    for (int r = 0; r < routes; r++) if (ids.Values<int>(routes)[r] == e && inverse.Values<uint>(routes)[r] >= expectedFirst + count) throw new Exception("Route assigned to wrong expert");
                    for (int r = 0; r < count; r += tileRows) {
                        var t = tiles.Values<uint>(maxTiles * 4).Slice(expectedTile++ * 4, 4);
                        if (t[0] != e || t[1] != expectedFirst + r || t[2] != Math.Min(tileRows, count - r)) throw new Exception("Invalid expert tile");
                    }
                    expectedFirst += count;
                }
                if (tileCount.Values<uint>(1)[0] != expectedTile || offsets.Values<uint>(257)[256] != routes || seen.Any(x => !x) || tokens.Values<uint>(routes+1)[routes] != uint.MaxValue || inverse.Values<uint>(routes+1)[routes] != uint.MaxValue) throw new Exception("Incomplete routing or guard overwritten");
                Console.WriteLine(JsonSerializer.Serialize(new { type="routing_benchmark", m, tileRows, skew, parallel, run, wall_ms=wall, timing=gpu.ReadWallTiming(), correctness="all routes and tiles checked" }));
            }
        }
    }
}
