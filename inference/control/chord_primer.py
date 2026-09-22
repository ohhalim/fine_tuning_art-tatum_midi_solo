"""Give the model harmonic context the only way it can read: as notes.

Why not a chord token
---------------------
The vocabulary does contain Stage B chord tokens (`TOKEN_STAGE_B_CHORD_ROOT_*`),
and the checkpoint's embedding is wide enough to address them. Neither fact
means the model learned them. Counting the tokenized training sets settles it:

    jazz_full (2,777 pieces, armB pretrain)   max token id 377, control tokens 0
    roles/lead/tokenized (armD LoRA)          max token id 388, control tokens 0

Not one control token, let alone a chord token. Putting a chord symbol in the
prefix would feed the model an embedding row it has never seen a gradient for,
and calling that "chord conditioning" would be a claim with nothing behind it.

So this module goes the other way: it states the harmony in the note vocabulary
the model did train on, as a low-register voicing under the melodic context,
and lets the model continue from it. Whether it actually responds is a question
for measurement, not for this docstring - see
``scripts/run_chord_primer_ab.py``.

This is not constrained decoding. No pitch is filtered, forbidden, or forced,
so non-chord tones, passing tones and resolutions remain reachable; if harmony
shows up in the output it is because the model put it there.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Sequence

import pretty_midi

from inference.app.conditioning import build_request_conditioning_midi
from inference.app.fallback import parse_chord
from inference.app.schemas import GenerationRequest

# Keep the guide below the solo register so it reads as accompaniment context
# rather than as melodic material to imitate.
GUIDE_VELOCITY = 58


def chord_guide_midi(
    chord_progression: Sequence[str],
    *,
    bpm: int,
    bars: int,
    output_dir: str | Path | None = None,
) -> Path:
    """Write the harmonic guide for one phrase, reusing the existing builder."""
    request = GenerationRequest(bpm=bpm, chord_progression=list(chord_progression), bars=bars)
    request.validate()
    directory = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="chord-guide-"))
    return build_request_conditioning_midi(request, directory)


def chord_guide_notes(
    chord_progression: Sequence[str], *, bpm: int, bars: int
) -> list[pretty_midi.Note]:
    """The guide as notes, so a primer can be built without touching disk."""
    # No progression is a real caller state (the "none" arm), not an error.
    if not list(chord_progression):
        return []
    with tempfile.TemporaryDirectory() as directory:
        path = chord_guide_midi(chord_progression, bpm=bpm, bars=bars, output_dir=directory)
        midi = pretty_midi.PrettyMIDI(str(path))
        return [note for instrument in midi.instruments for note in instrument.notes]


def build_chord_primer(
    chord_progression: Sequence[str],
    *,
    bpm: int,
    bars: int,
    melodic_notes: Sequence[pretty_midi.Note] = (),
    primer_max_tokens: int = 48,
):
    """Tokens for a primer that carries harmony, and optionally recent playing.

    Returns ``(tokens, used_chord)``. The guide is laid under the melodic notes
    on the same timeline rather than concatenated before them, because Stage A
    encodes one stream: a chord stated and then abandoned is just a past event,
    while a chord sounding underneath is context.

    Truncation goes through ``truncate_tokens_preserving_velocity`` for the
    reason documented there - a tail slice silently drops the carried velocity
    and every note before the next velocity token decodes to silence.
    """
    from scripts.generate import encode_notes_simple, truncate_tokens_preserving_velocity

    notes = list(chord_guide_notes(chord_progression, bpm=bpm, bars=bars))
    used_chord = bool(notes)
    notes.extend(melodic_notes)
    if not notes:
        return [], False
    notes.sort(key=lambda note: (note.start, note.pitch))
    tokens = encode_notes_simple(notes)
    return truncate_tokens_preserving_velocity(tokens, primer_max_tokens), used_chord


def chord_tone_pitch_classes(chord: str) -> set[int]:
    """Pitch classes of one chord, for scoring only - never for filtering."""
    root_pc, intervals = parse_chord(chord)
    return {(root_pc + interval) % 12 for interval in intervals}
