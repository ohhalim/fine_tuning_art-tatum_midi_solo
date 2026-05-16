from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .conditioning import build_request_conditioning_midi
from .fallback import generate_fallback_midi
from .metrics import compute_midi_metrics, validate_metrics
from .postprocess import repair_model_midi
from .schemas import GenerationRequest, GenerationResult


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LORA_PATH = PROJECT_ROOT / "checkpoints" / "jazz_lora_stage_a"
DEFAULT_CONDITIONING_MIDI = PROJECT_ROOT / "data" / "roles" / "lead" / "000000" / "conditioning.mid"
TARGET_DENSITY = {
    "sparse": 1.2,
    "medium": 3.0,
    "dense": 6.0,
}
CHORD_TONE_TARGET = 0.55
CHORD_TONE_WEIGHT = 1.0


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
    model_candidates: int,
    model_runner: Any | None = None,
) -> tuple[list[Path], str | None]:
    if not (lora_path / "lora_weights.pt").exists():
        return [], f"missing LoRA weights: {lora_path / 'lora_weights.pt'}"
    if not conditioning_midi.exists():
        return [], f"missing conditioning MIDI: {conditioning_midi}"

    if model_runner is not None:
        try:
            return (
                model_runner.generate_candidates(
                    request=request,
                    output_dir=output_dir,
                    conditioning_midi=conditioning_midi,
                    primer_max_tokens=primer_max_tokens,
                    max_sequence=max_sequence,
                    model_candidates=model_candidates,
                ),
                None,
            )
        except Exception as exc:
            return [], f"model runner generation failed: {exc}"

    model_output_dir = output_dir / f"{request.job_id}_model_raw"
    model_output_dir.mkdir(parents=True, exist_ok=True)
    for stale_candidate in model_output_dir.glob("jazz_sample_*.mid"):
        stale_candidate.unlink()
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
        str(max(1, int(model_candidates))),
        "--length",
        str(max_sequence),
        "--max_sequence",
        str(max_sequence),
        "--seed",
        str(request.seed),
        "--output",
        str(model_output_dir),
    ]
    if request.temperature is not None:
        cmd.extend(["--temperature", str(request.temperature)])
    if request.top_k is not None:
        cmd.extend(["--top_k", str(request.top_k)])
    if request.top_p is not None:
        cmd.extend(["--top_p", str(request.top_p)])
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            check=False,
            text=True,
            capture_output=True,
        )
    except Exception as exc:
        return [], f"model subprocess failed to start: {exc}"

    if completed.returncode != 0:
        stderr = completed.stderr.strip()[-1000:]
        stdout = completed.stdout.strip()[-1000:]
        return [], f"model generation failed with exit {completed.returncode}: {stderr or stdout}"

    candidates = sorted(model_output_dir.glob("jazz_sample_*.mid"))
    if not candidates:
        return [], "model generation finished but did not create any jazz_sample_*.mid"
    return candidates, None


def candidate_quality_score(metrics, density: str) -> float:
    target_density = TARGET_DENSITY.get(density, TARGET_DENSITY["medium"])
    density_penalty = abs(metrics.note_density - target_density) / max(1e-6, target_density)
    chord_tone_ratio = getattr(metrics, "chord_tone_ratio", None)
    chord_tone_penalty = 0.0
    if chord_tone_ratio is not None:
        chord_tone_penalty = max(0.0, CHORD_TONE_TARGET - float(chord_tone_ratio)) * CHORD_TONE_WEIGHT
    return (
        (metrics.dead_air_ratio * 2.0)
        + (metrics.repetition_score * 2.0)
        + (density_penalty * 3.0)
        + chord_tone_penalty
    )


def generate_midi_phrase(
    request: GenerationRequest,
    output_dir: str | Path = PROJECT_ROOT / "outputs" / "generated",
    use_model: bool = True,
    lora_path: str | Path = DEFAULT_LORA_PATH,
    conditioning_midi: str | Path | None = None,
    primer_max_tokens: int = 64,
    max_sequence: int = 256,
    model_candidates: int = 2,
    model_runner: Any | None = None,
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
    resolved_conditioning_midi: Path | None = None

    if use_model:
        resolved_conditioning_midi = (
            Path(conditioning_midi)
            if conditioning_midi is not None
            else build_request_conditioning_midi(request, output_dir / "_conditioning")
        )
        candidates, model_failure_reason = run_stage_a_model(
            request=request,
            output_dir=output_dir,
            lora_path=Path(lora_path),
            conditioning_midi=resolved_conditioning_midi,
            primer_max_tokens=primer_max_tokens,
            max_sequence=max_sequence,
            model_candidates=model_candidates,
            model_runner=model_runner,
        )
        if candidates:
            best_valid = None
            invalid_reasons: list[str] = []
            generation_time_ms = int((time.perf_counter() - start) * 1000)

            for candidate_index, candidate in enumerate(candidates, start=1):
                repaired_candidate = output_dir / f"{request.job_id}_model_repaired_{candidate_index}.mid"
                try:
                    candidate_for_metrics = repair_model_midi(candidate, repaired_candidate, request)
                except Exception as exc:
                    invalid_reasons.append(f"{candidate.name}: model repair failed: {exc}")
                    continue

                metrics = compute_midi_metrics(
                    candidate_for_metrics,
                    generation_time_ms,
                    fallback_used=False,
                    request=request,
                )
                is_valid, reason = validate_metrics(metrics, request.density)
                if not is_valid:
                    invalid_reasons.append(f"{candidate.name}: {reason}")
                    continue

                score = candidate_quality_score(metrics, request.density)
                if best_valid is None or score < best_valid[0]:
                    best_valid = (score, candidate_for_metrics, metrics)

            if best_valid is not None:
                _, candidate_for_metrics, metrics = best_valid
                generation_time_ms = int((time.perf_counter() - start) * 1000)
                metrics.generation_time_ms = generation_time_ms
                shutil.copyfile(candidate_for_metrics, final_midi_path)
                result = GenerationResult(
                    job_id=request.job_id,
                    status="COMPLETED",
                    midi_path=str(final_midi_path),
                    metrics_path=str(metrics_path),
                    fallback_used=False,
                    model_repaired=True,
                    conditioning_midi_path=str(resolved_conditioning_midi),
                    metrics=metrics,
                )
                write_json(metrics_path, result.to_dict())
                return result

            model_failure_reason = "; ".join(invalid_reasons) if invalid_reasons else model_failure_reason

    fallback_used = True
    try:
        generate_fallback_midi(request, final_midi_path)
        generation_time_ms = int((time.perf_counter() - start) * 1000)
        metrics = compute_midi_metrics(
            final_midi_path,
            generation_time_ms,
            fallback_used=True,
            request=request,
        )
        is_valid, reason = validate_metrics(metrics, request.density)
        if not is_valid:
            result = GenerationResult(
                job_id=request.job_id,
                status="FAILED",
                midi_path=str(final_midi_path) if final_midi_path.exists() else None,
                metrics_path=str(metrics_path),
                fallback_used=True,
                model_repaired=False,
                conditioning_midi_path=str(resolved_conditioning_midi) if resolved_conditioning_midi else None,
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
            model_repaired=False,
            conditioning_midi_path=str(resolved_conditioning_midi) if resolved_conditioning_midi else None,
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
        model_repaired=False,
        conditioning_midi_path=str(resolved_conditioning_midi) if resolved_conditioning_midi else None,
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
    parser.add_argument(
        "--conditioning_midi",
        type=str,
        default=None,
        help="Optional explicit primer MIDI. Defaults to a request-derived chord conditioning MIDI.",
    )
    parser.add_argument("--primer_max_tokens", type=int, default=64)
    parser.add_argument("--max_sequence", type=int, default=256)
    parser.add_argument("--model_candidates", type=int, default=2)
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
        model_candidates=args.model_candidates,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=True, indent=2))
    return 0 if result.status == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
