from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .fallback import generate_fallback_midi
from .metrics import compute_midi_metrics, validate_metrics
from .schemas import GenerationRequest, GenerationResult


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LORA_PATH = PROJECT_ROOT / "checkpoints" / "jazz_lora_stage_a"
DEFAULT_CONDITIONING_MIDI = PROJECT_ROOT / "data" / "roles" / "lead" / "000000" / "conditioning.mid"


def metrics_dir_for(output_dir: Path) -> Path:
    if output_dir.name == "generated":
        return output_dir.parent / "metrics"
    return output_dir / "metrics"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n")


def run_stage_a_model(
    request: GenerationRequest,
    output_dir: Path,
    lora_path: Path,
    conditioning_midi: Path,
    primer_max_tokens: int,
    max_sequence: int,
) -> tuple[Path | None, str | None]:
    if not (lora_path / "lora_weights.pt").exists():
        return None, f"missing LoRA weights: {lora_path / 'lora_weights.pt'}"
    if not conditioning_midi.exists():
        return None, f"missing conditioning MIDI: {conditioning_midi}"

    model_output_dir = output_dir / f"{request.job_id}_model_raw"
    model_output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "generate.py"),
        "--lora_path",
        str(lora_path),
        "--conditioning_midi",
        str(conditioning_midi),
        "--primer_max_tokens",
        str(primer_max_tokens),
        "--num_samples",
        "1",
        "--length",
        str(max_sequence),
        "--max_sequence",
        str(max_sequence),
        "--seed",
        str(request.seed),
        "--output",
        str(model_output_dir),
    ]
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            check=False,
            text=True,
            capture_output=True,
        )
    except Exception as exc:
        return None, f"model subprocess failed to start: {exc}"

    if completed.returncode != 0:
        stderr = completed.stderr.strip()[-1000:]
        stdout = completed.stdout.strip()[-1000:]
        return None, f"model generation failed with exit {completed.returncode}: {stderr or stdout}"

    candidate = model_output_dir / "jazz_sample_1.mid"
    if not candidate.exists():
        return None, "model generation finished but did not create jazz_sample_1.mid"
    return candidate, None


def generate_midi_phrase(
    request: GenerationRequest,
    output_dir: str | Path = PROJECT_ROOT / "outputs" / "generated",
    use_model: bool = True,
    lora_path: str | Path = DEFAULT_LORA_PATH,
    conditioning_midi: str | Path = DEFAULT_CONDITIONING_MIDI,
    primer_max_tokens: int = 64,
    max_sequence: int = 512,
) -> GenerationResult:
    request.validate()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = metrics_dir_for(output_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    final_midi_path = output_dir / f"{request.job_id}.mid"
    metrics_path = metrics_dir / f"{request.job_id}.json"
    start = time.perf_counter()

    model_failure_reason = None
    fallback_used = False

    if use_model:
        candidate, model_failure_reason = run_stage_a_model(
            request=request,
            output_dir=output_dir,
            lora_path=Path(lora_path),
            conditioning_midi=Path(conditioning_midi),
            primer_max_tokens=primer_max_tokens,
            max_sequence=max_sequence,
        )
        if candidate is not None:
            generation_time_ms = int((time.perf_counter() - start) * 1000)
            metrics = compute_midi_metrics(candidate, generation_time_ms, fallback_used=False)
            is_valid, reason = validate_metrics(metrics, request.density)
            if is_valid:
                shutil.copyfile(candidate, final_midi_path)
                result = GenerationResult(
                    job_id=request.job_id,
                    status="COMPLETED",
                    midi_path=str(final_midi_path),
                    metrics_path=str(metrics_path),
                    fallback_used=False,
                    metrics=metrics,
                )
                write_json(metrics_path, result.to_dict())
                return result
            model_failure_reason = reason

    fallback_used = True
    try:
        generate_fallback_midi(request, final_midi_path)
        generation_time_ms = int((time.perf_counter() - start) * 1000)
        metrics = compute_midi_metrics(final_midi_path, generation_time_ms, fallback_used=True)
        is_valid, reason = validate_metrics(metrics, request.density)
        if not is_valid:
            result = GenerationResult(
                job_id=request.job_id,
                status="FAILED",
                midi_path=str(final_midi_path) if final_midi_path.exists() else None,
                metrics_path=str(metrics_path),
                fallback_used=True,
                metrics=metrics,
                failure_reason=reason,
                model_failure_reason=model_failure_reason,
            )
            write_json(metrics_path, result.to_dict())
            return result
    except Exception as exc:
        result = GenerationResult(
            job_id=request.job_id,
            status="FAILED",
            midi_path=None,
            metrics_path=str(metrics_path),
            fallback_used=True,
            metrics=None,
            failure_reason=f"fallback generation failed: {exc}",
            model_failure_reason=model_failure_reason,
        )
        write_json(metrics_path, result.to_dict())
        return result

    result = GenerationResult(
        job_id=request.job_id,
        status="COMPLETED",
        midi_path=str(final_midi_path),
        metrics_path=str(metrics_path),
        fallback_used=fallback_used,
        metrics=metrics,
        model_failure_reason=model_failure_reason,
    )
    write_json(metrics_path, result.to_dict())
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a valid MVP MIDI phrase and metrics JSON")
    parser.add_argument("--bpm", type=int, required=True)
    parser.add_argument("--chords", type=str, required=True, help="Comma-separated chords, e.g. Cm7,Fm7,Bb7,Ebmaj7")
    parser.add_argument("--bars", type=int, default=2)
    parser.add_argument("--time_signature", type=str, default="4/4")
    parser.add_argument("--key", type=str, default=None)
    parser.add_argument("--section", type=str, default="drop")
    parser.add_argument("--energy", type=str, default="mid")
    parser.add_argument("--density", type=str, default="medium")
    parser.add_argument("--style", type=str, default="personal_jazz")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--job_id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "outputs" / "generated"))
    parser.add_argument("--no_model", action="store_true", help="Skip Stage A model and use fallback generator")
    parser.add_argument("--lora_path", type=str, default=str(DEFAULT_LORA_PATH))
    parser.add_argument("--conditioning_midi", type=str, default=str(DEFAULT_CONDITIONING_MIDI))
    parser.add_argument("--primer_max_tokens", type=int, default=64)
    parser.add_argument("--max_sequence", type=int, default=512)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    request = GenerationRequest.from_cli_args(args)
    result = generate_midi_phrase(
        request=request,
        output_dir=args.output_dir,
        use_model=not args.no_model,
        lora_path=args.lora_path,
        conditioning_midi=args.conditioning_midi,
        primer_max_tokens=args.primer_max_tokens,
        max_sequence=args.max_sequence,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=True, indent=2))
    return 0 if result.status == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
