#!/usr/bin/env python3
"""Summarize only the two explicit measurement passes, never smoke/validation."""
import collections
import csv
import hashlib
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
groups = collections.defaultdict(list)
checks = []
identities = []
for label in ("forward", "reverse"):
    identity = json.loads((root / (label + "-identity.json")).read_text())
    assert identity["shader_validation"] == "0", "Instrumented run is not a performance sample"
    assert not identity["git_status"].strip(), "Measurement checkout must be clean"
    identities.append(identity)
    rows = [json.loads(line) for line in (root / (label + ".jsonl")).read_text().splitlines()]
    assert len([r for r in rows if r["type"] == "check"]) == 14
    for row in rows:
        if row["type"] == "check":
            assert row["all_live_outputs_finite"] and row["guards"]
            checks.append(row)
        elif row["type"] == "timing":
            key = (row["family"], row["m"], row["k"], row["n"], row["routing"], row["variant"])
            groups[key].append(dict(row, pass_name=label))
assert identities[0]["bootstrap_commit"] == identities[1]["bootstrap_commit"]
assert identities[0]["harness_hashes"] == identities[1]["harness_hashes"]
summary = []
for key, samples in sorted(groups.items()):
    times = [r["gpu_ms"] for r in samples]
    assert len(times) == identities[0]["runs"] + identities[1]["runs"]
    row = dict(zip(("family", "m", "k", "n", "routing", "variant"), key))
    row.update(samples=len(times), gpu_median_ms=statistics.median(times), gpu_min_ms=min(times),
               gpu_max_ms=max(times), wall_median_ms=statistics.median(r["wall_ms"] for r in samples),
               forward_median_ms=statistics.median(r["gpu_ms"] for r in samples if r["pass_name"] == "forward"),
               reverse_median_ms=statistics.median(r["gpu_ms"] for r in samples if r["pass_name"] == "reverse"))
    summary.append(row)
result = {"bootstrap_commit": identities[0]["bootstrap_commit"], "splash_commit": identities[0]["splash_commit"],
          "successful_checks": len(checks), "cpu_reference_points": sum(c["cpu_reference_points"] for c in checks),
          "max_reference_errors": {side: max(c[side + "_max_abs"] for c in checks) for side in ("ours", "splash")},
          "rows": summary}
(root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
with (root / "summary.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)
index = {(r["family"], r["m"], r["k"], r["n"], r["routing"], r["variant"]): r for r in summary}
lines = ["| Operation | M | K → N | Routing | Ours GPU ms | Splash GPU ms | Ours / Splash |", "| --- | ---: | --- | --- | ---: | ---: | ---: |"]
for key, row in index.items():
    family, m, k, n, routing, variant = key
    if variant != ("ours_with_weight_decode" if family == "dense" else "ours"):
        continue
    other = index[(family, m, k, n, routing, "splash")]
    a, b = row["gpu_median_ms"], other["gpu_median_ms"]
    lines.append(f"| {family} | {m} | {k} → {n} | {routing} | {a:.3f} | {b:.3f} | {a/b:.2f}x |")
(root / "TABLE.md").write_text("\n".join(lines) + "\n")
manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.iterdir()) if p.is_file() and p.suffix in (".jsonl", ".json", ".stderr", ".csv", ".log")
            and p.name != "evidence-manifest.json"}
(root / "evidence-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("\n".join(lines))
print(f"Passed {len(checks)} checks, {result['cpu_reference_points']} CPU reference points.")
