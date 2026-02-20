"""
Select best dead-air candidate from sweep metrics and render summary outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Candidate:
    label: str
    metrics_path: str
    lora_path: str
    conditioning_midi: str
    primer_max_tokens: int
    output_dir: str
    avg_dead_air_ratio: float
    avg_repetition_4gram: float
    avg_note_density: float
    files: int

    @property
    def tie_distance(self) -> float:
        return abs(self.avg_note_density - 1.0)


def load_metrics(path: Path) -> Dict[str, float]:
    data = json.loads(path.read_text())
    summary = data.get("summary", data)
    return {
        "files": int(summary.get("files", 0)),
        "avg_dead_air_ratio": float(summary.get("avg_dead_air_ratio", 1.0)),
        "avg_repetition_4gram": float(summary.get("avg_repetition_4gram", 1.0)),
        "avg_note_density": float(summary.get("avg_note_density", 0.0)),
    }


def parse_manifest(manifest: Path) -> List[Candidate]:
    rows: List[Candidate] = []
    with manifest.open(newline="") as fp:
        reader = csv.DictReader(fp, delimiter="\t")
        for row in reader:
            metrics_path = Path(row["metrics_path"])
            if not metrics_path.exists():
                continue
            m = load_metrics(metrics_path)
            rows.append(
                Candidate(
                    label=row["label"],
                    metrics_path=row["metrics_path"],
                    lora_path=row["lora_path"],
                    conditioning_midi=row["conditioning_midi"],
                    primer_max_tokens=int(row["primer_max_tokens"]),
                    output_dir=row["output_dir"],
                    avg_dead_air_ratio=m["avg_dead_air_ratio"],
                    avg_repetition_4gram=m["avg_repetition_4gram"],
                    avg_note_density=m["avg_note_density"],
                    files=m["files"],
                )
            )
    return rows


def choose_best(candidates: List[Candidate]) -> Candidate:
    if not candidates:
        raise ValueError("No valid candidates with metrics were found.")
    return sorted(
        candidates,
        key=lambda c: (
            c.avg_dead_air_ratio,
            c.avg_repetition_4gram,
            c.tie_distance,
        ),
    )[0]


def write_markdown_table(
    path: Path,
    candidates: List[Candidate],
    best_label: str,
    baseline: Optional[Dict[str, float]] = None,
) -> None:
    lines = [
        "# Dead-Air Sweep Summary",
        "",
        "| Candidate | Files | Dead-air | Repetition | Note Density |",
        "|---|---:|---:|---:|---:|",
    ]
    for c in sorted(candidates, key=lambda x: x.avg_dead_air_ratio):
        mark = " **(BEST)**" if c.label == best_label else ""
        lines.append(
            f"| {c.label}{mark} | {c.files} | {c.avg_dead_air_ratio:.6f} | "
            f"{c.avg_repetition_4gram:.6f} | {c.avg_note_density:.6f} |"
        )

    if baseline is not None:
        lines.extend(
            [
                "",
                "## Baseline",
                "",
                f"- files: {int(baseline.get('files', 0))}",
                f"- avg_dead_air_ratio: {float(baseline.get('avg_dead_air_ratio', 1.0)):.6f}",
                f"- avg_repetition_4gram: {float(baseline.get('avg_repetition_4gram', 1.0)):.6f}",
                f"- avg_note_density: {float(baseline.get('avg_note_density', 0.0)):.6f}",
            ]
        )

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Select best dead-air sweep candidate")
    parser.add_argument("--manifest", required=True, help="TSV manifest path")
    parser.add_argument("--output_json", required=True, help="Summary JSON path")
    parser.add_argument("--output_md", required=True, help="Summary Markdown path")
    parser.add_argument("--baseline_metrics", default="", help="Optional baseline metrics.json")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    baseline_summary = None
    if args.baseline_metrics:
        bpath = Path(args.baseline_metrics)
        if bpath.exists():
            baseline_summary = load_metrics(bpath)

    candidates = parse_manifest(manifest_path)
    best = choose_best(candidates)

    report = {
        "manifest": str(manifest_path),
        "baseline_metrics": args.baseline_metrics if args.baseline_metrics else None,
        "baseline_summary": baseline_summary,
        "candidates": [asdict(c) for c in candidates],
        "best_candidate": asdict(best),
        "selection_rule": [
            "min avg_dead_air_ratio",
            "tie-breaker: min avg_repetition_4gram",
            "tie-breaker: min abs(avg_note_density - 1.0)",
        ],
    }
    output_json.write_text(json.dumps(report, ensure_ascii=True, indent=2))
    write_markdown_table(output_md, candidates, best.label, baseline_summary)

    print(json.dumps({"best_label": best.label, "dead_air": best.avg_dead_air_ratio}, ensure_ascii=True))


if __name__ == "__main__":
    main()

