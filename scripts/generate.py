"""
Music Transformer conditioned MIDI generation script.

Stage A usage:
    python scripts/generate.py \
      --lora_path ./checkpoints/jazz_lora \
      --conditioning_midi ./data/roles/lead/000000/conditioning.mid \
      --num_samples 3 \
      --length 1024 \
      --output ./samples
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List

import pretty_midi
import torch

# Add music_transformer to path
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "music_transformer"))
sys.path.insert(0, str(SCRIPT_DIR / "music_transformer" / "third_party"))

from model.music_transformer import MusicTransformer
from utilities.constants import TOKEN_END, TOKEN_PAD
from utilities.device import get_device
from midi_processor.processor import decode_midi, RANGE_NOTE_ON, RANGE_NOTE_OFF, RANGE_TIME_SHIFT

# Import LoRA helper from train script
from train_qlora import add_lora_to_model


def encode_midi_simple(file_path: str) -> List[int]:
    start_idx = {
        "note_on": 0,
        "note_off": RANGE_NOTE_ON,
        "time_shift": RANGE_NOTE_ON + RANGE_NOTE_OFF,
        "velocity": RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT,
    }

    mid = pretty_midi.PrettyMIDI(midi_file=file_path)
    notes = []
    for inst in mid.instruments:
        if not inst.is_drum:
            notes.extend(inst.notes)

    if not notes:
        return []

    events = []
    cur_time = 0.0
    cur_vel = 0
    note_events = []
    for note in notes:
        note_events.append(("note_on", note.start, note.pitch, note.velocity))
        note_events.append(("note_off", note.end, note.pitch, 0))
    note_events.sort(key=lambda x: (x[1], x[2], 0 if x[0] == "note_off" else 1))

    for event_type, event_time, pitch, velocity in note_events:
        time_diff = max(0.0, event_time - cur_time)
        time_steps = int(round(time_diff * 100))
        while time_steps >= RANGE_TIME_SHIFT:
            events.append(start_idx["time_shift"] + RANGE_TIME_SHIFT - 1)
            time_steps -= RANGE_TIME_SHIFT
        if time_steps > 0:
            events.append(start_idx["time_shift"] + time_steps - 1)

        if event_type == "note_on":
            mod_vel = int(max(0, min(31, velocity // 4)))
            if mod_vel != cur_vel:
                events.append(start_idx["velocity"] + mod_vel)
                cur_vel = mod_vel
            events.append(start_idx["note_on"] + int(pitch))
        else:
            events.append(start_idx["note_off"] + int(pitch))

        cur_time = event_time

    return events


def load_model_with_lora(
    lora_path: str,
    n_layers: int = 6,
    num_heads: int = 8,
    d_model: int = 512,
    dim_feedforward: int = 1024,
    max_sequence: int = 512,
    rpr: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
):
    """Load Music Transformer with LoRA weights."""
    device = get_device()

    model = MusicTransformer(
        n_layers=n_layers,
        num_heads=num_heads,
        d_model=d_model,
        dim_feedforward=dim_feedforward,
        max_sequence=max_sequence,
        rpr=rpr,
    )

    model, _ = add_lora_to_model(model, r=lora_r, alpha=lora_alpha)

    lora_weights_path = Path(lora_path) / "lora_weights.pt"
    if lora_weights_path.exists():
        lora_state = torch.load(lora_weights_path, map_location=device)
        model.load_state_dict(lora_state, strict=False)
        print(f"Loaded LoRA weights from {lora_weights_path}")
    else:
        raise FileNotFoundError(f"LoRA weights not found: {lora_weights_path}")

    model = model.to(device)
    model.eval()
    return model


def build_primer(
    conditioning_midi: str | None,
    primer_max_tokens: int,
    append_sep_token: bool,
) -> torch.Tensor:
    if conditioning_midi:
        tokens = encode_midi_simple(conditioning_midi)
        if not tokens:
            raise ValueError(f"conditioning MIDI produced empty token list: {conditioning_midi}")

        if append_sep_token:
            tokens = tokens + [TOKEN_END]
        if primer_max_tokens > 0:
            tokens = tokens[-primer_max_tokens:]
        return torch.tensor(tokens, dtype=torch.long)

    # fallback primer: random short prefix
    return torch.randint(0, 100, (1,), dtype=torch.long)


def sanitize_for_decode(tokens: List[int]) -> List[int]:
    # decode_midi expects only event tokens in [0, TOKEN_END).
    return [int(t) for t in tokens if 0 <= int(t) < TOKEN_END and int(t) != TOKEN_PAD]


def generate_once(
    model: MusicTransformer,
    primer: torch.Tensor,
    target_length: int,
    strip_primer: bool,
) -> List[int]:
    device = get_device()
    primer = primer.to(device)

    with torch.no_grad():
        generated = model.generate(
            primer=primer,
            target_seq_length=target_length,
            beam=0,
            beam_chance=1.0,
        )

    sequence = generated[0].detach().cpu().tolist()
    if strip_primer:
        sequence = sequence[len(primer) :]
    return sanitize_for_decode(sequence)


def main():
    parser = argparse.ArgumentParser(description="Generate MIDI with LoRA-finetuned Music Transformer")

    # Model
    parser.add_argument("--lora_path", type=str, default="./checkpoints/jazz_lora")
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--max_sequence", type=int, default=512)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)

    # Generation
    parser.add_argument("--conditioning_midi", type=str, default=None)
    parser.add_argument("--primer_max_tokens", type=int, default=256)
    parser.add_argument("--append_sep_token", action="store_true", default=True)
    parser.add_argument("--no_append_sep_token", action="store_false", dest="append_sep_token")
    parser.add_argument("--strip_primer_output", action="store_true", default=True)
    parser.add_argument("--keep_primer_output", action="store_false", dest="strip_primer_output")
    parser.add_argument("--length", type=int, default=512, help="Full generated length including primer")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--output", type=str, default="./samples")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    print("\n=== Loading Model with LoRA ===")
    model = load_model_with_lora(
        lora_path=args.lora_path,
        n_layers=args.n_layers,
        num_heads=args.num_heads,
        d_model=args.d_model,
        dim_feedforward=args.dim_feedforward,
        max_sequence=args.max_sequence,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

    primer = build_primer(
        conditioning_midi=args.conditioning_midi,
        primer_max_tokens=args.primer_max_tokens,
        append_sep_token=args.append_sep_token,
    )
    if len(primer) >= args.max_sequence:
        # Keep room for at least one generated token.
        primer = primer[-(args.max_sequence - 1) :]
        print(f"Primer truncated to {len(primer)} tokens to fit max_sequence={args.max_sequence}")

    print(f"Primer tokens: {len(primer)}")
    target_length = min(args.max_sequence, int(args.length))
    if target_length <= len(primer):
        target_length = min(args.max_sequence, len(primer) + 128)
        print(
            f"Requested length {args.length} <= primer size; "
            f"auto-adjusted generation length to {target_length}"
        )

    print(f"\n=== Generating {args.num_samples} samples ===")
    for i in range(args.num_samples):
        output_path = os.path.join(args.output, f"jazz_sample_{i + 1}.mid")
        print(f"\n--- Sample {i + 1}/{args.num_samples} ---")

        generated_tokens = generate_once(
            model=model,
            primer=primer,
            target_length=target_length,
            strip_primer=args.strip_primer_output,
        )
        if not generated_tokens:
            print("No decodable tokens produced; skipping sample.")
            continue

        decode_midi(generated_tokens, output_path)
        print(f"Saved: {output_path} (tokens={len(generated_tokens)})")

    print("\n=== Done ===")
    print(f"Samples saved to: {args.output}/")


if __name__ == "__main__":
    main()
