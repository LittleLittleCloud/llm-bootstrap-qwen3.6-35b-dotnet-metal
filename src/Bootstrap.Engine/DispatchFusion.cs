namespace Bootstrap.Engine;

internal sealed partial class QwenEngine
{
    // Independent development ablations: resident small weights, staged dense,
    // multiple projections, shared-expert epilogue, attention preparation.
    int dispatchFusion;
    public int DispatchFusion {
        get => dispatchFusion;
        set { if (value < 0 || value > 31) throw new ArgumentOutOfRangeException(nameof(value));
            if (dispatchFusion != value) { ClearPrefixCache(); dispatchFusion = value; } }
    }
    bool FuseDispatch(int bit, int rows) => (DispatchFusion & bit) != 0 && rows > VerificationCapacity && PredecodedProjections && TensorOpsPrecision == "fp16";
    readonly Dictionary<string, GpuBuffer> residentProjections = new();
    GpuBuffer? multiWeights, sharedPackedActivation;

    void PrepareDispatchFusion(int rows)
    {
        if (rows <= VerificationCapacity || TensorOpsPrecision != "fp16" || !PredecodedProjections) return;
        if ((DispatchFusion & 12) != 0 && multiWeights is null) {
            // Grow the existing projection scratch instead of retaining a second
            // 36 MiB allocation. No command is active at this boundary.
            var previous = batch!.PackedWeights;
            multiWeights = Buffer(9216 * Hidden / 2); batch.PackedWeights = multiWeights;
            if (previous is not null) { owned.Remove(previous); previous.Dispose(); }
        }
        // Routed expert down has consumed this buffer before shared projections.
        // Its FP16 activation storage is dead until the next layer's expert path.
        if ((DispatchFusion & 8) != 0) sharedPackedActivation = batch!.ExpertActivation;
        if ((DispatchFusion & 1) == 0 || residentProjections.Count != 0) return;
        gpu.Begin();
        for (int layer = 0; layer < 40; layer++) {
            string prefix = Prefix(layer);
            string[] names = (layer + 1) % 4 != 0
                ? [prefix + ".linear_attn.in_proj_a", prefix + ".linear_attn.in_proj_b", prefix + ".mlp.shared_expert_gate"]
                : [prefix + ".mlp.shared_expert_gate"];
            foreach (string name in names) {
                var w = weights[name + ".weight"];
                if (w.Shape[0] > 32) throw new InvalidDataException("Unexpected small projection");
                var packed = Buffer(w.Shape[0] * w.Shape[1] / 2);
                var scale = w.Kind is 0 or 3 ? weights[name + ".weight_scale_inv"].Binding : (Binding)dummy;
                var p = new KernelArgs { N = (uint)w.Shape[0], K = (uint)w.Shape[1], Kind = w.Kind };
                gpu.Dispatch("projection_unpack_half", p, p.N * p.K / 4, 128, w.Binding, scale, packed);
                residentProjections.Add(name, packed);
            }
        }
        gpu.End();
    }

    void BatchProjectMany(string[] names, GpuBuffer[] outputs, GpuBuffer input, int rows, bool reuse, bool activate = false)
    {
        var b = batch!; b.PackedInputs ??= Buffer(b.Capacity * 8192 / 2);
        var ts = names.Select(n => weights[n + ".weight"]).ToArray();
        int k = ts[0].Shape[1];
        if (ts.Any(t => t.Shape[1] != k) || names.Length is < 2 or > 3) throw new ArgumentException("Invalid grouped projections");
        if (!reuse) gpu.Dispatch("projection_input_half", new() { M = (uint)rows, K = (uint)k }, (uint)(rows * k / 4), 128, input, b.PackedInputs);
        var p = new KernelArgs { M = (uint)rows, K = (uint)k, A = (uint)ts[0].Shape[0], B = (uint)ts[1].Shape[0], C = ts.Length == 3 ? (uint)ts[2].Shape[0] : 0,
            D = ts[0].Kind, E = ts[1].Kind, F = ts.Length == 3 ? ts[2].Kind : 0 };
        Binding[] packed = new Binding[3];
        bool resident = (DispatchFusion & 1) != 0 && names.All(residentProjections.ContainsKey);
        if (resident) { for (int i = 0; i < names.Length; i++) packed[i] = residentProjections[names[i]]; packed[2] = names.Length == 3 ? packed[2] : dummy; }
        else {
            var bindings = new List<Binding>();
            for (int i = 0; i < 3; i++) {
                bindings.Add(i < ts.Length ? ts[i].Binding : dummy);
                bindings.Add(i < ts.Length && ts[i].Kind is 0 or 3 ? weights[names[i] + ".weight_scale_inv"].Binding : dummy);
            }
            if ((ulong)(p.A + p.B + p.C) * p.K * 2 > multiWeights!.Length) throw new ArgumentException("Grouped projections exceed scratch workspace");
            bindings.Add(multiWeights);
            gpu.Dispatch("projection_unpack_many", p, (p.A + p.B + p.C) * p.K / 4, 128, bindings.ToArray());
            packed[0] = multiWeights!; packed[1] = new(multiWeights!, (ulong)p.A * p.K * 2); packed[2] = new(multiWeights!, (ulong)(p.A + p.B) * p.K * 2);
        }
        if (activate) {
            if (p.A != p.B) throw new ArgumentException("Gate and up shapes differ");
            p.N = p.A;
            gpu.Dispatch2D("shared_gate_up_pack", p, ((p.N + 63) / 64) * 128, ((uint)rows + 63) / 64, 128,
                b.PackedInputs, packed[0], packed[1], b.SharedActivation, sharedPackedActivation!);
        } else {
            uint groups = (p.A + 63) / 64 + (p.B + 63) / 64 + (p.C + 63) / 64;
            gpu.Dispatch2D("matrix_product_many", p, groups * 128, ((uint)rows + 63) / 64, 128,
                b.PackedInputs, packed[0], packed[1], packed[2], outputs[0], outputs[1], outputs.Length == 3 ? outputs[2] : dummy);
        }
    }
}
