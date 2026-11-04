# Current prefill gap to Splash — 2026-11-04

The next target is the **MoE output/combine path**, starting with a precision-preserving combine-to-next-normalization fusion and a separate investigation of expert-down output traffic. Current dense TensorOps is competitive. The native INT8-KV Splash reference is faster overall, but its BF16-KV control is slightly slower than ours at 30K. A smaller expert threadgroup-memory diagnostic did not improve performance and is not adopted.

This is a development profile, not a new endpoint evaluation, implementation speedup, or quality-equivalence result. Production kernels and serving defaults remain unchanged.

## Identity and timing boundary

- Machine: Apple M5 Pro, 20 GPU cores, 48 GiB.
- Ours: merged main `c3de96555f1097f6a21a9172c4ad06b011d2c2a2`, affine-8 group-64 v2 derived from official Qwen3.6 FP8, BF16 DFlash2 weights, FP16 KV/operands, FP32 accumulation/state/output, dispatch mask 29, staged8 recurrence, fused preparation, 2048-token chunks, 32768 context.
- Splash host source: `1482f6bbc2abf12375332725302019594150b6cb`, with the previously reviewed host-only counter/fixture patch; original installed 1.4.0 metallib SHA-256 `e30124e8bceba1a9de7afe75cf1a10439bcfa5c1b777490c829ba4da0d0cd1cd`. Its target is the independently quantized MLX affine-8 checkpoint; native activations/output are BF16, draft projections affine-4, recurrent state FP32. All experts are resident.
- Same 10,520 and 29,939 input token IDs, freshly regenerated with the current tokenizer and checked against every supplied Splash ID. Loading and tokenization are excluded. First-token production and draft context setup are included.
- Each case has one first-use run, three warm uncached requests, and a separate counter request. Ours runs before and after all four Splash controls. GPU workloads are serial; no OS/power settings are changed.
- Ours clears prefix reuse but still creates its ordinary prefix checkpoints. Splash's direct runtime harness does not materialize those checkpoints and captures only its final draft context window. These are native-system comparisons, not identical-weight/precision causal comparisons.

The chat's configured `llm-bootstrap-qwen3.8-35b` path did not exist. The actual current experiment and runtime are Qwen3.6; diagnostics use a new `codex/prefill-gap-profile-20261104` worktree without changing the existing serving checkout.

## Fresh uninstrumented results

| Input tokens | Ours before, ms / tok/s | Ours after, ms / tok/s | Splash INT8 KV, ms / tok/s | Splash BF16 KV, ms / tok/s |
|---:|---:|---:|---:|---:|
| 10,520 | 3869.64 / 2718.6 | 3892.21 / 2702.8 | 3469.34 / 3032.3 | 3597.60 / 2924.2 |
| 29,939 | 13024.28 / 2298.7 | 13233.60 / 2262.3 | 12096.72 / 2475.0 | 13407.11 / 2233.1 |

Against native INT8 KV, ours takes 11.5–12.2% more time at 10K and 7.7–9.4% more at 30K. These ranges are the two process medians, not confidence intervals. Against Splash BF16 KV, ours takes 7.6–8.2% more time at 10K but 1.3–2.9% less at 30K. FP16 and BF16 are both two-byte formats, but they do not have identical numerical behavior.

Within Splash, changing only KV from INT8 to BF16 adds 128.3 ms at 10K and 1310.4 ms at 30K. Its sampled attention/preparation group accounts for approximately the same difference. A large long-context attention share does not by itself demonstrate that our attention implementation is inferior.

Warm GPU time is 98.6–99.3% of ours' wall time; warm process CPU medians are 38.7–105.7 ms. Host work and GPU time overlap. This does not support CPU launch overhead as the principal remaining bottleneck.

## Shape-attributed model profile

The extra diagnostic build changes host labels only; the Metal shader hash matches the production baseline. Its warm medians are 3853.20 ms / 2730.2 tok/s at 10K and 13092.69 ms / 2286.7 tok/s at 30K, consistent with the earlier process range. Sampling overhead is 15.0% / 5.8%.

| Sampled encoder intervals, ms | Ours 10K | Splash INT8 10K | Ours 30K | Splash INT8 30K |
|---|---:|---:|---:|---:|
| MoE including shared expert, routing, gather and combine | 1743.6 | 1561.1 | 4729.4 | 4296.3 |
| Dense plus remaining norm/residual preparation | 1109.3 | 979.2 | 3091.8 | 2775.3 |
| DeltaNet including preparation and gated normalization | 485.4 | 411.0 | 1380.5 | 1166.1 |
| Attention plus preparation | 474.3 | 486.7 | 3559.4 | 3760.3 |
| Draft context work, excluding final normalization | 64.8 | 10.2 | 206.9 | 10.8 |
| Embedding/output head | 10.1 | 2.6 | 13.0 | 4.5 |

These are diagnostic interval sums, not mutually exclusive latency components. Remaining fused-epilogue and normalization boundaries differ; a row difference is not an isolated causal speedup estimate. Nevertheless, the MoE deficit appears in both model profiles and matched expert-down kernels. The isolated dense measurements distinguish a competitive matmul core from additional norm/packing/conversion work around it.

Current `batch_mix` sums to 144.3 / 412.1 ms at 10K / 30K; Splash `moe_combine` sums to 78.6 / 221.0 ms. Ours reads FP32 expert results and separately applies the shared gate; Splash consumes BF16 expert results and has already prepared shared routing weights. This motivates a down-plus-combine experiment, not a claim that either precision or fusion alone explains the difference.

Disabling only our prefix caching measures 3842.21 / 13021.15 ms versus the shape build's 3853.20 / 13092.69 ms: 0.29% / 0.55% apparent throughput improvement. Both controls have zero new warm-window swap-outs and the same 7218 / 16344 dispatches. This separate-process difference is small compared with observed timing drift and does not establish a reliable implementation gain. Source inspection confirms `PrefixBoundaries` still selects the same chunks when caching is disabled. Prefix snapshot copying does not explain most of the remaining gap; disabling useful cache behavior is not the recommended optimization.

## Matched current-kernel measurements

The kernel harness uses identical logical activations and exactly checked decoded dyadic weights, 256 experts, top-8 routing, and the original physical layouts/epilogues. Ours retains FP16 activations and FP32 down output; Splash retains BF16 activations/output and its gate/up rounding. Native target checkpoints are not involved in this isolated test. Every measured operation receives at least 250 ms of warming; two three-observation passes reverse initial order and alternate order internally.

| 2048-token operation | Routing | Ours, ms | Splash, ms | Ours additional time |
|---|---|---:|---:|---:|
| Expert gate/up, K=2048 N=512 | uniform | 3.658 | 3.508 | 4.3% |
| Expert gate/up, K=2048 N=512 | skewed | 3.875 | 3.531 | 9.8% |
| Expert down, K=512 N=2048 | uniform | 2.172 | 1.876 | 15.8% |
| Expert down, K=512 N=2048 | skewed | 2.234 | 1.955 | 14.3% |

Uniform routing has 64 rows/expert; skewed routing alternates 1/127. These are synthetic controls, not measured route distributions of the real model.

| Dense M=2048, K → N | Ours incl affine unpack, ms | Splash, ms |
|---|---:|---:|
| 2048 → 512 | 0.149 | 0.186 |
| 2048 → 2048 | 0.578 | 0.616 |
| 2048 → 8192 | 2.314 | 2.346 |
| 4096 → 2048 | 1.244 | 1.228 |
| 512 → 2048 | 0.144 | 0.159 |

The one 1.3% slower dense shape is not evidence of a broad dense-core deficit. Input packing is excluded from both sides of these kernel measurements. Weight unpack is included for ours; Splash unpacks inside its matmul.

The two normal passes retain 324 timing observations and 60 checked cases (2996 CPU reference points per side, zero measured error on the deliberately dyadic inputs). A separate production-only Metal shader-validation pass completes all 18 cases with finite outputs and intact guards. This validates the measured current-kernel fixtures, not the rejected compact-stage variant or general whole-model numerical equivalence.

## Rejected compact-stage diagnostic

The current affine expert loop uses one staging buffer per SIMD group but inherits an 8 KiB allocation from generic double-buffer variants. A generated diagnostic reduces it to 4 KiB. For gate/up, an additional threadgroup barrier is required before FP32 gate scratch reuses the up group's stage. The intended control preserves values, rounding, live-row extents, arithmetic order and dispatch grid; the validation failure below prevents treating that intention as established correctness. The production shader is unchanged.

| 2048-token operation | Original ms | Compact-stage ms | Outcome |
|---|---:|---:|---|
| Gate/up uniform | 3.658 | 3.777 | 3.3% slower |
| Gate/up skewed | 3.875 | 3.953 | 2.0% slower |
| Down uniform | 2.172 | 2.170 | effectively unchanged |
| Down skewed | 2.234 | 2.247 | 0.6% slower |

Smaller threadgroup allocation alone is not the next optimization. It also **fails full-output bitwise comparison with Metal shader validation enabled**, at the down/uniform/M=512 case; validation stops there with exit 1. Both ordinary timing passes passed their complete-output bitwise comparisons, CPU samples, finite-output and guard checks, but that does not override the validation failure. The compact diagnostic is rejected; its timings are retained only as negative research evidence, never as a valid candidate implementation. The unmodified production kernels are validated separately.

## Next implementation experiment

1. **Fuse MoE combine with the following layer's input RMSNorm/FP16 packing.** Retain the FP32 residual output for draft capture and caching, preserve the ordered eight-expert sum, and reuse each combined value for the RMS reduction and packed projection input. This targets repeated reads, scalar-gate work and materialization surrounding MoE. Compare the whole `combine → norm → pack` sequence, then alternating whole-model runs. The existing vector-only `batch_mix_vec4` diagnostic was already ineffective; do not repeat it as the main hypothesis.
2. **Isolate the expert-down output representation.** The current 2048-token routed output contains 8 × 2048 × 2048 FP32 values: 128 MiB written and subsequently consumed by combine per layer/chunk. A two-byte temporary would halve that payload, but it changes numerical behavior. First separate storage/layout effects from matmul by measuring down plus combine together; if narrower storage is tried, retain FP32 accumulation and perform complete-output, cache/speculation and task-quality checks before any promotion. The measured 14–16% down deficit is not proof that output stores explain all of it.
3. **Consider bounded draft feature work after the MoE experiment.** Our draft context still processes every prompt chunk whereas Splash captures the required final window. Preserve the context needed by every saved prefix checkpoint; simply skipping early draft work would break cache-restoration behavior.

Do not prioritize another blind dense tile sweep, generic dispatch-count reduction, or blanket attention rewrite on the basis of these results. INT8 KV is a separate numerical/storage experiment, not a precision-preserving kernel optimization. No whole-model speedup is predicted by summing sampled encoder intervals.

## Instrumentation and evidence

Counter mode changes encoder boundaries and can perturb memory residency. Initial ours counter overhead was 27.1% at 10K and 63.7% at 30K; the later process observed 11.1% and 6.1%. Splash overhead was approximately 0–0.6%. Counter wall time is never used as throughput evidence.

The first ours lifecycle observed 2227.9 MiB of system-wide new swap-outs, including loading/first-use/counters; the warm 10K window had none, and the first warm 30K window had 56.6 MiB. The later ours warm windows had zero new swap-outs. No thermal warnings were recorded. These are whole-machine observations, not allocations or swap attributed solely to the engine.

Shape attribution is opt-in with `BOOTSTRAP_PROFILE_SHAPES=1`; it appends KernelArgs dimensions to sampled kernel labels without changing any dispatched parameters, pipelines or command boundaries. This is necessary because the same dense kernel implements target, shared-expert and draft FC work. Splash already includes shared-expert projections inside its MoE kernels; comparing its MoE group directly with only our routed experts would be misleading. Some norm/epilogue fusion boundaries still differ.

Local raw evidence is under `artifacts/current-gap/`: commands and binary/source hashes in `provenance.json` and per-case `identity.json`, full stdout/stderr, memory samples, Splash interval traces, `summary.json`, `kernels.csv`, and generated isolated-kernel source/results. All first-token repeat checks are retained; they are not full-answer or quality-equivalence tests. Official reports and leaderboard files are untouched.

All 60 model requests returned the same first token (22); this is a repeatability check, not a complete-answer score. Production shader files in both executable packages match the pinned baseline byte-for-byte. The 72-file local evidence manifest is `artifacts/current-gap/evidence-manifest.json`, SHA-256 `65c7195faf712c2e3d78656eeb6278eb45804b9a6e6bf23ce605efd7c018e130`.

Reproduction:

```sh
dotnet build src/Bootstrap.Engine -c Release
python3 scripts/profile-current-gap.py --help
python3 scripts/profile-current-gap-kernels.py --splash /path/to/clean/pinned/splash --out artifacts/current-gap/kernels
dotnet build src/Bootstrap.Engine -c Release -o artifacts/shape-bin
python3 scripts/profile-current-gap-followup.py artifacts/current-gap --dll artifacts/shape-bin/Bootstrap.Engine.dll
python3 scripts/summarize-current-gap.py artifacts/current-gap
```

Use the exact prior fixture/model/metallib locations recorded by the primary driver's arguments. Output directories must be new so observations cannot be silently overwritten.
