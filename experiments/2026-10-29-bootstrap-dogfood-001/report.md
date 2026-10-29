# Experiment: 2026-10-29-bootstrap-dogfood-001

Status: Frozen; candidate build not started. Acceptance: Unverified. Baseline preparation/preflight completed before freeze; no candidate implementation or formal candidate run has started.

## Question and frozen contract

- Hypothesis and platform-specific goal: Bring up a self-implemented .NET + Metal inference engine for official Qwen3.6-35B-A3B-FP8 on the local M5 Pro. User-supplied acceptance targets: prefill >= 500 tokens/s and decode >= 50 tokens/s.
- Contract file, revision, and freeze timestamp: `experiments/2026-10-29-bootstrap-dogfood-001/contract.json`; not frozen, no freeze commit or timestamp.
- Skill repository full SHA and instruction hashes: Updated installed skills to `9b71eb126c862a8c185272e5b68a26a55e953cb9`. All 17 skill/support files match a fresh detached checkout. Plugin semantic version remains 0.1.0. Full builder instruction hashes remain to be recorded before freeze. This supersedes the old planning treatment; the previous draft is preserved in ignored local evidence.
- Starting engine/scaffold SHA and permitted reuse/dependencies: No engine implementation exists. User-selected C#/.NET host plus independently implemented Metal compute kernels. Existing LLM inference backends are disallowed as the candidate backend. Scaffold/dependency pins remain unresolved.
- Hardware, OS, driver, compiler, power and memory limits: Apple M5 Pro; 20 GPU cores; 48 GiB unified memory; macOS 26.6.2 (25G83); Darwin 25.6.0; Metal 4. Observed: .NET SDK 10.0.100 and 11.0.100-rc.1.26425.128 (osx-arm64); Apple clang 21.0.0 (clang-2100.3.34.2); Metal 4; Node.js v26.7.0 for planning checks. No project global.json exists yet. xcrun could not locate the offline Metal compiler; runtime Metal shader compilation is to be checked before freeze. AC power observed; memory limits, thermal observations, and GPU contention still need preflight.
- Target model/revision; weight/tokenizer hashes; all precisions: Official `Qwen/Qwen3.6-35B-A3B-FP8`, revision `95a723d08a9490559dae23d0cff1d9466213d989`. FP8 E4M3 with 128x128 block scaling and excluded BF16 tensors. Contract contains 42 expected upstream shard hashes; weight blobs are not downloaded or locally verified. Tokenizer hash `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` and chat-template hash `e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259` were locally verified. Activations, KV, recurrent-state precision, and speculation settings remain unresolved.
- Builder model/provider/identifier, initial inference location, and tool access: Unresolved. Record actual exposed identity; mark unavailable provider details instead of fabricating them.
- Development harness version/hash: Unresolved. Local Codex CLI 0.157.0 was observed; desktop/harness identity must be pinned separately.
- Evaluation reference (id, repository revision, definition hash), warm-up, and formal repetitions: `engine-throughput`, revision `9b71eb126c862a8c185272e5b68a26a55e953cb9`, definition SHA-256 `b5857229e97b2d2451e0d89f9c43b4e95269cb33bc3fed6221c21de54308c0a5`. One warm-up and three formal repetitions; initial selected workloads are `prefill.in2k.out1` and `decode.in128.out256`. Endpoint conformance runs first at the same revision, definition SHA-256 `d710a08a4c2ce736652c3a96cb911bed4e1fe34683cda37935e736938b473081`. All scripts for the three released evaluations were hash-verified in the fresh checkout. No evaluation was executed.
- Baseline engines/revisions, selection date/method, and comparison tracks: Matched-configuration track; baseline candidates/revisions and baseline preflight remain unresolved.
- Primary metric, quality/resource gates: Not obtained; no formal run has occurred.
- Budget and stopping rule: No fixed time, builder-token, API-cost, or hardware-hour budget; stop on explicit user request and preserve partial progress. This does not waive required stage gates.
- Scope authorization and unresolved fields: The user selected new experiment, local M5 Pro, official FP8, .NET + Metal, targets 500/50 tokens/s, no fixed budget, and requested the bootstrap skill update. Evaluation availability and mandatory numerical-equivalence blockers are resolved by the new upstream version. Execution configuration and baseline preflight still need completion before freeze.

## Execution and evidence

- Run IDs and timestamps/timezone: No runs. Skill update recorded at 2026-10-29T14:32:04-07:00 (America/Los_Angeles).
- Actual runtime SHA and executable/kernel hashes, separate from checkout: Not obtained; no formal run has occurred.
- Reproduction commands and configuration hashes: Not obtained; no formal run has occurred.
- Warm-up, repetitions, resource sampling, and contention checks: Not obtained; no formal run has occurred.
- Public raw evidence links or local-only locations with hashes: Ignored `evidence/` contains source/model discovery, prior contract/report snapshots, check logs, and skill-update verification. Weight hashes are expected upstream identities; tokenizer/template hashes were locally verified.
- Human interventions and agent trajectory access/hash: User setup choices are summarized above. No manual implementation edits have occurred. Raw transcripts remain private.
- Builder token/cost, hardware hours, elapsed time, and attempts: Not obtained; no formal run has occurred.

## Results

Use “Not obtained” for unavailable values. Preserve all formal repetitions and failures.

| Metric / gate | Baseline | Candidate | Definition, coverage, and conditions |
| --- | --- | --- | --- |
| Official assertions / fixed task count | Not obtained | Not obtained | |
| Prefill tokens / seconds / tok/s | Not obtained | Not obtained | |
| Decode intervals / seconds / tok/s | Not obtained | Not obtained | N−1; native command metrics separate |
| Client TTFT median / variability | Not obtained | Not obtained | |
| Application duration and request/token totals | Not obtained | Not obtained | |
| Memory, pressure, swap-out and resource gates | Not obtained | Not obtained | |
| Errors / timeouts / incomplete tasks | Not obtained | Not obtained | |

## Interpretation and disposition

- Comparable conditions, differences, and limits of attribution: Not obtained; no formal run has occurred.
- Variability/repetitions and unresolved correctness/resource concerns: Not obtained; no formal run has occurred.
- Gap to each frozen target; bounded baseline/SOTA conclusion: Planned 500/50 tokens/s targets are unmeasured and not frozen. No baseline/SOTA claim.
- Self-hosting transition and subsequent development evidence, or not demonstrated: Not in scope; no self-hosting claim.
- Failures, amendments, recovery, and next steps: No engine failure. Complete remaining execution configuration and baseline preflight, then commit the frozen contract before engine work.
- Implementation PR, independent results PR, and adoption decision: No implementation, results, pull requests, or adoption decision. No project remote is configured.

## Feedback for future experiments

Reviewed text only. Refer to private artifacts by hash and access limitation; do not paste transcripts, weights, credentials, or private prompts. A suggestion here becomes a change only through a separate pull request.

- Failures, interruptions, and their causes: Earlier planning under revision 744d5caa2545852f9c43d82863e059a24d78344f was blocked by missing released performance evaluation and mandatory numerical-equivalence checks. Upstream revision 9b71eb126c862a8c185272e5b68a26a55e953cb9 resolves these: engine-throughput and humaneval-20-pi are released, and numerical equivalence is no longer required. No measured failure or success is inferred from this update.
- Preflight checks that would have caught them earlier: Resolve and pin the intended repository version, inspect released evaluation definitions, verify script hashes, and check protocol/schema consistency before freezing.
- Recurring human interventions and what would have avoided them: The user requested a skill refresh after clarifying goals. Automatically collect discoverable technical metadata to reduce intake burden.
- Suggested changes to skills or evaluations (cite the run and evidence hash): The previous numerical-equivalence/evaluation-release feedback is resolved upstream. The same 0.1.0 manifest version now represents different source commits; retain the full source SHA and installed-file verification for treatment identity. No formal run exists.
- Calibration notes for baselines, budgets, and thresholds (priors only; each contract fixes its own before tuning): Not obtained; no formal run has occurred.
- Limits on generalizing: preliminary cohorts, unmatched comparisons, hardware-specific effects: No model or benchmark has run. Installation, metadata hashes, and schema checks are not engine results.

## Skill update validation

Local-only `evidence/skill-update.json` SHA-256: `2662bf6585546e5804be82c292f3c190cf4bf00b10c17a330e56dcb339a53f5b`. The previous draft contract and report are preserved as hashed local-only evidence. The contract remains a draft; no freeze gate was bypassed.

Validation log `installed-check-skill-update.txt` SHA-256: `ab6f4601abed36aa97fbfdc0d0bf9ff37c16aa2db6cc35921700957619035979`.

Validation log `schema-check-skill-update.txt` SHA-256: `77515f0715ba6787a4bfa04aa3b3cf6bd86b0e50400000a0795d8e1b8b389d35`.

## Contract freeze and baseline preflight

The contract was completed and frozen at `2026-10-29T14:48:23-07:00`, before writing candidate engine code. The candidate workspace is the empty source revision `5fe496f89f7bd68eeb6ffc2f2f2efef375771c5e`. The baseline protocol adapter is independent baseline preparation and will not be reused by the candidate.

Baseline preflight used MLX-LM with affine8/group64 weights and BF16 defaults, while the candidate is official FP8 with FP32 math/state. This is a disclosed best-permitted-configuration comparison; no matched-precision speedup is claimed. Complete conformance had 60 pass, one skip, zero fail/error, and a verified negative control. The one-pass pre-freeze throughput preflight passed both selected workloads, with prefill 1770.88 tokens/s and decode 55.04 tokens/s. These are baseline preliminary setup observations, not candidate or formal three-repetition results.

- Local-only preflight evidence `conformance-attempt-001.json`: SHA-256 `23477b65c7b5431829920f49ee6614f7e794c61c3c0b797d7f2999ff9bf7d4d1`.
- Local-only preflight evidence `throughput-attempt-001.json`: SHA-256 `f14ce51769ceddb6dc3d4c754e806e407001a89a026b1b89388b97f0c0d61946`.
- Local-only preflight evidence `throughput-attempt-001.requests.evidence`: SHA-256 `5518e59e06c7c9b1ea1e8ea887b4db5ac9584b76883fe92c05f2f79831f0eed7`.
- Local-only preflight evidence `baseline-runtime-identity.json`: SHA-256 `626749b727a0540e8d8015c31c910370814cdc41080fef2c34791f708a1d051e`.

Remaining work: implement the .NET + Metal candidate; run baseline/candidate formal evaluations serially using this frozen contract; preserve all attempts; report every acceptance criterion and submit reviewed result records in a separate PR.

## Formal baseline result

Frozen contract commit: `5f7f7099be4a7c261140d6fbc4f800d7065266e1`. Run `2026-10-29-baseline-001` completed after the freeze, with complete API conformance (60 pass, one skip, no fail/error) and one throughput warm-up plus three successful formal passes. Prefill: **1723.22 tokens/s**. Decode: **54.76 tokens/s**. All six workload requests passed their gates. These are independent baseline measurements, not candidate results.

The sampled CPU resident footprint was 0.102 GiB, with the evaluator explicitly noting that GPU memory may be excluded; total GPU/model footprint was not obtained. Whole-machine observed swap-out delta during formal passes was 0 bytes. No competing engine process was run during the baseline measurement; general GUI GPU exclusivity is not claimed. The owned baseline server and adapter were stopped after measurement to release GPU resources before candidate development tests.
