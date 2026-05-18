from __future__ import annotations

from pathlib import Path

from .schemas import GenerationRequest


class StageAModelRunner:
    """Reusable in-process Stage A model runner.

    The subprocess path reloads the checkpoint for every request. This runner
    keeps the LoRA Music Transformer in memory and only rebuilds the MIDI primer
    per request.
    """

    def __init__(
        self,
        lora_path: str | Path,
        checkpoint_path: str | Path | None = None,
        max_sequence: int = 256,
        n_layers: int = 6,
        num_heads: int = 8,
        d_model: int = 512,
        dim_feedforward: int = 1024,
        lora_r: int = 16,
        lora_alpha: int = 32,
    ) -> None:
        from scripts.generate import load_model_with_lora

        self.lora_path = Path(lora_path)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.max_sequence = int(max_sequence)
        self.model = load_model_with_lora(
            lora_path=str(self.lora_path),
            checkpoint_path=str(self.checkpoint_path) if self.checkpoint_path is not None else None,
            prefer_full_checkpoint=True,
            n_layers=n_layers,
            num_heads=num_heads,
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            max_sequence=self.max_sequence,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )

    def generate_candidates(
        self,
        request: GenerationRequest,
        output_dir: str | Path,
        conditioning_midi: str | Path,
        primer_max_tokens: int,
        max_sequence: int,
        model_candidates: int,
    ) -> list[Path]:
        import torch
        from scripts.generate import build_primer, decode_midi, generate_once

        output_dir = Path(output_dir)
        conditioning_midi = Path(conditioning_midi)
        model_output_dir = output_dir / f"{request.job_id}_model_raw"
        model_output_dir.mkdir(parents=True, exist_ok=True)
        for stale_candidate in model_output_dir.glob("jazz_sample_*.mid"):
            stale_candidate.unlink()

        target_sequence = min(self.max_sequence, int(max_sequence))
        primer = build_primer(
            conditioning_midi=str(conditioning_midi),
            primer_max_tokens=primer_max_tokens,
            append_sep_token=True,
        )
        if len(primer) >= target_sequence:
            primer = primer[-(target_sequence - 1) :]
        if target_sequence <= len(primer):
            target_sequence = min(self.max_sequence, len(primer) + 128)

        torch.manual_seed(int(request.seed))
        candidates: list[Path] = []
        for index in range(max(1, int(model_candidates))):
            output_path = model_output_dir / f"jazz_sample_{index + 1}.mid"
            generated_tokens = generate_once(
                model=self.model,
                primer=primer,
                target_length=target_sequence,
                strip_primer=True,
                temperature=request.temperature or 1.0,
                top_k=request.top_k,
                top_p=request.top_p,
            )
            if not generated_tokens:
                continue
            decode_midi(generated_tokens, str(output_path))
            if output_path.exists():
                candidates.append(output_path)

        return candidates
