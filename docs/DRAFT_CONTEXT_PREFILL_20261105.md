# Draft context prefill: skip overwritten windows

Batched target prefill now omits draft feature capture and projection for whole
chunks that cannot contribute to the next saved prefix checkpoint or the final
generation window. `--full-draft-prefill` restores the previous full-prompt work
for comparison. Draft/target precision, target chunks and all Metal kernels are
unchanged. This implementation starts from engine commit
`3ce6a76195a92fcf4293b41ae2bdb4da41bb11ab`.

## Why the omitted work is unnecessary

The draft consumes the most recent 2048 projected target-feature rows. During
prompt prefill there are no draft proposals. A draft context becomes observable
when a prefix checkpoint is saved or when generation starts. For each target
chunk, use the next prefix boundary as the required context end when prefix
caching is enabled, or the prompt end when caching is disabled. If the whole
chunk ends at or before `required_end - 2048`, its draft rows would have been
overwritten before that consumer.

Skip only entire chunks. Any chunk that intersects a required window retains
all its rows and its original FC dispatch shape, including the tiny-projection
selection. This avoids numerical changes from slicing rows or moving FP16
conversions. Target prefill and checkpoint boundaries remain identical.
Sequential prefill, decode and speculative verification retain full capture;
verification explicitly rejects an attempt to disable it. A policy change clears
the reusable prefix cache.

The final draft window is not the only retained window: intermediate checkpoints
must still support branching and shorter requests. For the fixed native fixtures
with prefix caching enabled:

| Prompt tokens | Previous projected rows | Windowed projected rows | Removed rows | Dispatches before / after |
| ---: | ---: | ---: | ---: | ---: |
| 10,520 | 10,520 | 6,424 | 4,096 (38.94%) | 7,198 / 7,172 |
| 29,939 | 29,939 | 5,363 | 24,576 (82.09%) | 16,324 / 16,168 |

Each omitted large chunk removes eight feature captures and five FC/norm/store
dispatches. Keeping intersecting chunks means a partially overlapping chunk can
still perform work outside the exact 2048-row window. The policy preserves
checkpoint capability instead of matching Splash's final-window-only harness.

## Validation method

`compare-draft-prefill` alternates full/windowed modes in one loaded model. Each
fixture warms both modes, measures five pairs, then captures a separate counter
pair. A second process reverses the initial ordering. Warm timing includes
ordinary prefix snapshot creation, but excludes correctness hashing. All sampled
and warm observations are retained.

Every observation compares the first output token, complete live target state
and KV, logits, live draft ring, and every saved checkpoint against the other
policy with SHA-256 equality checks. Checkpoint hashes include recurrent/conv
state, logits and live draft rows; unused ring slots for short prefixes are
deliberately excluded. No tolerance is substituted for bitwise equality.

`check-draft-prefill` compares seven real-model request scenarios in both modes:
cold long input, exact reuse, edited suffix, shortening to an intermediate block,
append after shortening, disabled prefix caching, and recovery after
cancellation. Each request generates 64 tokens. It checks generated tokens,
target/live draft state, checkpoint contents, cached-token counts, and
speculative cycle/acceptance counts. The existing prefix-cache suite additionally
covers KV growth, shorter requests, cancellation/callback failures and sequential
prefill. Both INT8 and FP16 target KV are included in the new scenario checks.

The measurement uses the same 10,520/29,939-token fixtures as the preceding
INT8 Splash comparison, on Apple M5 Pro with 20 GPU cores and 48 GiB. Target
INT8 KV, FP16 operands/draft KV, BF16 expert output, FP32 accumulation/state,
staged8, dispatch mask 29, 2048-token chunks, context 32768 and prefix caching
remain enabled. Weight revisions and binary/source hashes are captured locally.
All GPU jobs run serially. This is native profiling, not a new Full endpoint
evaluation or a submitted experiment.

## Results

Both INT8 and FP16 KV passed all seven request scenarios, with exact generated-token, live-state, checkpoint, cache-accounting and speculation-count equality against full draft prefill. The existing INT8 prefix-cache suite passed 48 checks across 12 cases. Both performance processes passed state/checkpoint equality on every warmup, warm and counter observation. The Release build completed with zero warnings or errors.

| Process | Tokens | Full median | Windowed median | Saved | Latency reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| forward | 10520 | 3736.62 ms | 3709.70 ms | 26.92 ms | 0.72% |
| forward | 29939 | 12191.88 ms | 12077.54 ms | 114.34 ms | 0.94% |
| reverse | 10520 | 3753.61 ms | 3733.42 ms | 20.20 ms | 0.54% |
| reverse | 29939 | 12219.29 ms | 12097.33 ms | 121.96 ms | 1.00% |

There are ten warm pairs per fixture across the two processes. All ten 30K pairs favor windowed prefill; their savings range from 75.31 to 135.12 ms. The 10K result is smaller: one of ten individual pairs regresses by 22.91 ms, while both process medians improve. Paired-median savings at 30K are 118.83 and 125.75 ms; differences of process medians are shown above.

The sampled draft-context group falls from 186.21 to 96.07 ms in the first 30K process, and from 189.70 to 68.88 ms in the reversed process. This grouping includes capture, FC packing/weight conversion/matmul and context store; draft normalization remains within the general norm group. The reduction is smaller and more variable than the 82.09% row-count reduction. In particular, the two remaining 2048-row input-pack dispatches total 69.61 ms in the first candidate trace, versus 45.35 ms across fourteen baseline dispatches. The reversed trace also shows substantial retained packing cost. These counters do not establish the cause (memory/cache/residency or attribution effects were not isolated), and no linear projection from skipped rows to time savings is claimed.

Warm whole-prefill timings are the primary performance evidence. Counter intervals can overlap and must not be summed as exclusive wall-time components. All four processes recorded zero new whole-machine swapouts and zero thermal-warning samples. Existing swap and swap-ins are not claimed absent.

The optimization is enabled by default in this working branch. It saves approximately 0.94–1.00% of 30K native prefill latency with prefix caching enabled and leaves intermediate checkpoint functionality intact. The full-work fallback flag remains available. No new Full endpoint evaluation, PR or result submission was performed in this change.

## Reproduction

Build into an isolated output directory:

```sh
dotnet build src/Bootstrap.Engine -c Release -o artifacts/draft-window-20261105/bin
```

Use the ordinary `--model-dir`, `--draft-dir`, `--affine-cache`,
`--tuned-prefill`, `--bf16-expert-output`, `--delta-mode staged8`,
`--dispatch-fusion 29`, `--context-window 32768`, and
`--kv-cache-precision int8` arguments with:

```text
check-draft-prefill --prefix-checks
compare-draft-prefill --prompt-file PROMPT_10K --long-prompt-file PROMPT_30K --runs 5
compare-draft-prefill --prompt-file PROMPT_10K --long-prompt-file PROMPT_30K --runs 5 --reverse-order
```

Repeat `check-draft-prefill` with `--kv-cache-precision fp16`. The runner under
`artifacts/draft-window-20261105/run.py` saves exact local paths, commands, runtime
hashes, source snapshots, raw outputs, machine memory/thermal samples and exit
codes. It refuses to overwrite per-case directories. The summary script derives
both process-level medians and per-pair savings from all five pairs.
