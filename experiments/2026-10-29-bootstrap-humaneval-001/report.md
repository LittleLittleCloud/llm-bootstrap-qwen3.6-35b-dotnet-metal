# Experiment: 2026-10-29-bootstrap-humaneval-001

Status: Frozen; candidate context work not started. Acceptance: Unverified.

## Question and frozen contract

The user requested HumanEval and noted that the old 4096-token window may be insufficient. The pinned Pi evaluation advertises 32768 context and reserves up to 4096 output tokens per request. This supplementary quality goal is recorded independently of the previous 4k throughput experiment; its published records are unchanged. The model, weights, device, FP32 computation/state and greedy sampling remain the same. Context allocation and evaluation change together, so results cannot be attributed to only one factor.

Contract freeze timestamp: `2026-10-29T16:42:37-07:00`. Contract commit will be recorded after the successful check. Starting source is `f51d0ac885758e3b1735a7549e3a35497d472f1b`, equivalent to main merge `70ddcb30188bcfd2d12ca4e81f3bb18bbba80dfc`. Evaluation: `humaneval-20-pi` at `9b71eb126c862a8c185272e5b68a26a55e953cb9`, definition SHA-256 `933285b68d72f20497ce7cd6c5d8d53856f76aa50bb06b4c16bf187e34623ae9`. One two-task warm-up; three formal passes of all 20 tasks. No quality pass-rate floor was invented. The 60-task coverage threshold denotes measurement completeness only.

## Setup evidence

The previous independent MLX affine8/BF16 baseline ran serially with the candidate stopped. At 32768 context it passed complete API conformance (60 passes, one advisory skip, verified negative control), followed by a single HumanEval/156 preflight: 1/1 official assertions, clean pass, verified negative controls. This deliberately limited setup probe is not a full baseline quality estimate or controlled speed comparison. Raw setup artifacts remain local-only.

- Preflight result SHA-256: `6d7f17318d25b610499a0f0ab594b5d836931a9ca926b1daacbad02dbc46eaaa`.
- Preflight conformance SHA-256: `45e57662d79606a6035e15c24ddf000b2a19bf5a30aa171d43716d1b3a148280`.
- Builder-task SHA-256: `d290c4a57308741115d302e3941e7b70156aa68a76459245ab8c828423875303`.

## Results

Not obtained; candidate context/configuration work has not started. Do not inherit the old 500/50 synthetic-throughput targets as HumanEval thresholds.

## Feedback for future experiments

A 4096 total-token window cannot accept a Pi request that reserves 4096 output tokens plus a nonempty input. Evaluation context/output budgets must be checked before a quality campaign. The larger window also requires reviewing cached-state allocation and dispatch integer dimensions, not only deleting an API check. Source version, actual build hashes and raw evidence access limitations will be recorded with each attempt.
