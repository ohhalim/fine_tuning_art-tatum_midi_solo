from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4


VALID_SECTIONS = {"intro", "build", "breakdown", "drop"}
VALID_ENERGIES = {"low", "mid", "high"}
VALID_DENSITIES = {"sparse", "medium", "dense"}


@dataclass
class GenerationRequest:
    bpm: int
    chord_progression: list[str]
    bars: int = 2
    time_signature: str = "4/4"
    key: str | None = None
    section: str = "drop"
    energy: str = "mid"
    density: str = "medium"
    style: str = "personal_jazz"
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    job_id: str = field(default_factory=lambda: str(uuid4()))
    seed: int = 42

    def validate(self) -> None:
        if not (40 <= int(self.bpm) <= 240):
            raise ValueError("bpm must be between 40 and 240")
        if not (1 <= int(self.bars) <= 4):
            raise ValueError("bars must be between 1 and 4 for MVP")
        if not self.chord_progression:
            raise ValueError("chord_progression must not be empty")
        if self.section not in VALID_SECTIONS:
            raise ValueError(f"section must be one of {sorted(VALID_SECTIONS)}")
        if self.energy not in VALID_ENERGIES:
            raise ValueError(f"energy must be one of {sorted(VALID_ENERGIES)}")
        if self.density not in VALID_DENSITIES:
            raise ValueError(f"density must be one of {sorted(VALID_DENSITIES)}")

    @classmethod
    def from_cli_args(cls, args: Any) -> "GenerationRequest":
        chords = [c.strip() for c in args.chords.split(",") if c.strip()]
        return cls(
            bpm=args.bpm,
            chord_progression=chords,
            bars=args.bars,
            time_signature=args.time_signature,
            key=args.key,
            section=args.section,
            energy=args.energy,
            density=args.density,
            style=args.style,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            job_id=args.job_id or str(uuid4()),
            seed=args.seed,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationMetrics:
    generation_time_ms: int
    note_count: int
    duration_sec: float
    note_density: float
    dead_air_ratio: float
    repetition_score: float
    pitch_min: int | None
    pitch_max: int | None
    fallback_used: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationResult:
    job_id: str
    status: str
    midi_path: str | None
    metrics_path: str | None
    fallback_used: bool
    model_repaired: bool = False
    metrics: GenerationMetrics | None = None
    failure_reason: str | None = None
    model_failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.metrics is not None:
            payload["metrics"] = self.metrics.to_dict()
        return payload
