from __future__ import annotations

import os
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .generator import PROJECT_ROOT, generate_midi_phrase
from .schemas import GenerationRequest, VALID_DENSITIES, VALID_ENERGIES, VALID_SECTIONS


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "generated"

app = FastAPI(
    title="Personalized Live MIDI Improviser Inference API",
    version="0.1.0",
)


class MidiInferenceRequest(BaseModel):
    model_config = ConfigDict(validate_by_name=True)

    job_id: str = Field(default_factory=lambda: str(uuid4()), alias="jobId")
    bpm: int
    chord_progression: list[str] = Field(alias="chordProgression")
    bars: int = 2
    time_signature: str = Field(default="4/4", alias="timeSignature")
    key: str | None = None
    section: str = "drop"
    energy: str = "mid"
    density: str = "medium"
    style: str = "personal_jazz"
    temperature: float | None = None
    top_k: int | None = Field(default=None, alias="topK")
    top_p: float | None = Field(default=None, alias="topP")
    seed: int = 42
    use_model: bool = Field(default=True, alias="useModel")

    def to_generation_request(self) -> GenerationRequest:
        return GenerationRequest(
            job_id=self.job_id,
            bpm=self.bpm,
            chord_progression=self.chord_progression,
            bars=self.bars,
            time_signature=self.time_signature,
            key=self.key,
            section=self.section,
            energy=self.energy,
            density=self.density,
            style=self.style,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            seed=self.seed,
        )


def metrics_to_camel(metrics: Any) -> dict[str, Any] | None:
    if metrics is None:
        return None
    return {
        "generationTimeMs": metrics.generation_time_ms,
        "noteCount": metrics.note_count,
        "durationSec": metrics.duration_sec,
        "noteDensity": metrics.note_density,
        "deadAirRatio": metrics.dead_air_ratio,
        "repetitionScore": metrics.repetition_score,
        "pitchMin": metrics.pitch_min,
        "pitchMax": metrics.pitch_max,
        "fallbackUsed": metrics.fallback_used,
    }


def result_to_response(result: Any) -> dict[str, Any]:
    payload = {
        "jobId": result.job_id,
        "status": result.status,
        "midiPath": result.midi_path,
        "metricsPath": result.metrics_path,
        "fallbackUsed": result.fallback_used,
        "modelRepaired": result.model_repaired,
        "metrics": metrics_to_camel(result.metrics),
    }
    if result.failure_reason:
        payload["failureReason"] = result.failure_reason
    if result.model_failure_reason:
        payload["modelFailureReason"] = result.model_failure_reason
    return payload


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    return {
        "validSections": sorted(VALID_SECTIONS),
        "validEnergies": sorted(VALID_ENERGIES),
        "validDensities": sorted(VALID_DENSITIES),
        "defaultOutputDir": str(DEFAULT_OUTPUT_DIR),
    }


@app.post("/infer/midi")
def infer_midi(request: MidiInferenceRequest) -> dict[str, Any]:
    output_dir = os.environ.get("MIDI_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))
    generation_request = request.to_generation_request()
    try:
        result = generate_midi_phrase(
            request=generation_request,
            output_dir=output_dir,
            use_model=request.use_model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"inference failed: {exc}") from exc

    return result_to_response(result)
