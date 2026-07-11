"""
Compare Stage A tiny-overfit full-model training against random-base LoRA-only.

The comparison intentionally treats a LoRA-only gate failure as diagnostic
output instead of a script failure. A failed full-model tiny run is still a
failure, because the current Stage A path depends on that smoke being possible.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n")


def run_mode(args: argparse.Namespace, run_id: str, lora_only: bool) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/run_stage_a_tiny_overfit.py",
        "--sample_count",
        str(args.sample_count),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--max_sequence",
        str(args.max_sequence),
        "--primer_max_tokens",
        str(args.primer_max_tokens),
        "--num_samples",
        str(args.num_samples),
        "--output_root",
        str(args.output_root),
        "--run_id",
        run_id,
        "--seed",
        str(args.seed),
    ]
    if lora_only:
        cmd.append("--lora_only")

    completed = subprocess.run(
        cmd,
        cwd=str(ROOT_DIR),
        check=False,
        text=True,
        capture_output=True,
    )
    report_path = Path(args.output_root) / run_id / "report.json"
    report: dict[str, Any] | None = None
    if report_path.exists():
        report = json.loads(report_path.read_text())

    return {
        "run_id": run_id,
        "returncode": int(completed.returncode),
        "cmd": cmd,
        "report_path": str(report_path),
        "report": report,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def extract_summary(result: dict[str, Any]) -> dict[str, Any]:
    report = result.get("report") or {}
    summary = report.get("summary") or {}
    return {
        "run_id": result.get("run_id"),
        "returncode": result.get("returncode"),
        "report_path": result.get("report_path"),
        "training_mode": summary.get("training_mode"),
        "passed_mvp_gate": summary.get("passed_mvp_gate"),
        "best_validation_loss": summary.get("best_validation_loss"),
        "valid_raw_sample_count": summary.get("valid_raw_sample_count"),
        "raw_sample_count": summary.get("raw_sample_count"),
        "fallback_used": summary.get("fallback_used"),
        "model_failure_reason": summary.get("model_failure_reason"),
    }


def build_decision(full_summary: dict[str, Any], lora_summary: dict[str, Any]) -> str:
    full_passed = full_summary.get("passed_mvp_gate") is True
    lora_passed = lora_summary.get("passed_mvp_gate") is True
    if full_passed and not lora_passed:
        return (
            "Use full-checkpoint/from-scratch training or a real pretrained base; "
            "do not rely on random-base LoRA-only Stage A training."
        )
    if full_passed and lora_passed:
        return "Both modes passed; LoRA-only needs a broader validation run before being trusted."
    return "Full-model tiny smoke failed; inspect tokenization/generation before expanding Stage A."


def write_markdown(path: Path, comparison: dict[str, Any]) -> None:
    full_summary = comparison["full_model_tiny"]
    lora_summary = comparison["lora_only"]
    lines = [
        "# Stage A Tiny Mode Comparison",
        "",
        f"- Decision: {comparison['decision']}",
        "",
        "## Full Model Tiny",
        "",
        f"- passed_mvp_gate: {full_summary.get('passed_mvp_gate')}",
        f"- best_validation_loss: {full_summary.get('best_validation_loss')}",
        f"- valid_raw_samples: {full_summary.get('valid_raw_sample_count')}/{full_summary.get('raw_sample_count')}",
        f"- fallback_used: {full_summary.get('fallback_used')}",
        f"- report: `{full_summary.get('report_path')}`",
        "",
        "## LoRA Only",
        "",
        f"- passed_mvp_gate: {lora_summary.get('passed_mvp_gate')}",
        f"- best_validation_loss: {lora_summary.get('best_validation_loss')}",
        f"- valid_raw_samples: {lora_summary.get('valid_raw_sample_count')}/{lora_summary.get('raw_sample_count')}",
        f"- fallback_used: {lora_summary.get('fallback_used')}",
        f"- model_failure_reason: {lora_summary.get('model_failure_reason')}",
        f"- report: `{lora_summary.get('report_path')}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Stage A tiny training modes")
    parser.add_argument("--output_root", type=str, default=str(ROOT_DIR / "outputs" / "stage_a_tiny_compare"))
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--sample_count", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max_sequence", type=int, default=128)
    parser.add_argument("--primer_max_tokens", type=int, default=24)
    parser.add_argument("--num_samples", type=int, default=3)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output_root = Path(args.output_root)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    full_result = run_mode(args, f"{run_id}_full_model", lora_only=False)
    lora_result = run_mode(args, f"{run_id}_lora_only", lora_only=True)

    full_summary = extract_summary(full_result)
    lora_summary = extract_summary(lora_result)
    comparison = {
        "run_id": run_id,
        "output_root": str(args.output_root),
        "full_model_tiny": full_summary,
        "lora_only": lora_summary,
        "decision": build_decision(full_summary, lora_summary),
        "raw_results": {
            "full_model_tiny": {"returncode": full_result["returncode"]},
            "lora_only": {"returncode": lora_result["returncode"]},
        },
    }

    comparison_json = args.output_root / run_id / "comparison.json"
    comparison_md = args.output_root / run_id / "comparison.md"
    comparison["comparison_json"] = str(comparison_json)
    comparison["comparison_md"] = str(comparison_md)
    write_json(comparison_json, comparison)
    write_markdown(comparison_md, comparison)
    print(json.dumps(comparison, ensure_ascii=True, indent=2))

    full_returncode = int(full_result["returncode"])
    lora_returncode = int(lora_result["returncode"])
    if full_returncode not in (0, 2) or lora_returncode not in (0, 2):
        return 1
    return 0 if full_summary.get("passed_mvp_gate") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
