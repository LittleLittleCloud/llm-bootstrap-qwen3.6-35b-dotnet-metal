"""Summarize local development JSONL; never reads/writes evaluator reports."""
import collections
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1] / "artifacts"


def summarize(path):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    groups = collections.defaultdict(list)
    checks = []
    for row in records:
        kind = row.get("type", "")
        if kind.endswith("_check") or kind in ("model_smoke", "delta_model_answer"):
            checks.append(row)
        if kind == "prefill_variant":
            if row["mode"] == "cold_full_answer":
                checks.append({k: row[k] for k in ("variant", "answer_correct", "first_token", "text")})
                continue
            key = row["variant"]
        elif kind in ("expert_benchmark", "delta_benchmark", "dense_benchmark", "fusion_benchmark"):
            # Fusion runs are all timed after explicit warming; other families
            # retain run zero as a cold/priming observation, excluded here.
            if kind != "fusion_benchmark" and row["run"] == 0:
                continue
            key = row.get("variant", row.get("name", ""))
            for name in ("m", "n", "k", "skew"):
                if name in row:
                    key += f"/{name}={row[name]}"
        else:
            continue
        groups[key].append(row)
    output = {"checks": checks, "timings": {}}
    for key, rows in groups.items():
        wall = [r["wall_ms"] for r in rows]
        gpu = [r["timing"]["gpu_ms"] for r in rows]
        item = {"runs": len(rows), "wall_ms_median": statistics.median(wall),
                "wall_ms_range": [min(wall), max(wall)], "gpu_ms_median": statistics.median(gpu),
                "gpu_ms_range": [min(gpu), max(gpu)]}
        if "prefill_tokens_per_second" in rows[0]:
            item["tokens_per_second"] = statistics.median(r["prefill_tokens_per_second"] for r in rows)
        output["timings"][key] = item
    return output


for family in ("layout", "delta", "dense"):
    folder = ROOT / f"prefill-{family}-research"
    summaries = {}
    for path in sorted(folder.glob("*.jsonl")):
        summaries[path.stem] = summarize(path)
    (folder / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"{family}: summarized {len(summaries)} local logs")
