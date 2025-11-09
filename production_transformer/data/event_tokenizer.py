"""
Event-based MIDI Tokenizer

Industry-standard tokenization for symbolic music:
- Similar to Google Magenta's Music Transformer
- Compatible with HuggingFace tokenizers API
- Event-based representation (NOTE_ON, NOTE_OFF, TIME_SHIFT)
- Supports chord conditioning

Why event-based?
- More natural than piano roll
- Variable length (like language)
- Explicit timing information
- Easy to condition on chords
"""

import pretty_midi
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class EventType(Enum):
    """MIDI event types"""
    NOTE_ON = 0
    NOTE_OFF = 1
    TIME_SHIFT = 2
    SET_TEMPO = 3
    CHORD = 4
    BAR = 5


@dataclass
class MIDIEvent:
    """Single MIDI event"""
    event_type: EventType
    value: int
    time: float  # Absolute time in seconds

    def __repr__(self):
        if self.event_type == EventType.NOTE_ON:
            pitch = self.value & 0x7F
            velocity = (self.value >> 7) & 0x7F
            return f"NoteOn(pitch={pitch}, vel={velocity})"
        elif self.event_type == EventType.NOTE_OFF:
            return f"NoteOff(pitch={self.value})"
        elif self.event_type == EventType.TIME_SHIFT:
            return f"TimeShift({self.value}ms)"
        elif self.event_type == EventType.CHORD:
            return f"Chord({self.value})"
        else:
            return f"{self.event_type.name}({self.value})"


class EventVocabulary:
    """
    Vocabulary for event-based MIDI tokenization

    Token structure (total: ~700 tokens):
    - 0: PAD
    - 1: BOS (beginning of sequence)
    - 2: EOS (end of sequence)
    - 3-130: NOTE_ON (128 pitches)
    - 131-258: NOTE_OFF (128 pitches)
    - 259-358: TIME_SHIFT (100 bins, 0-1000ms)
    - 359-458: SET_TEMPO (100 bins, 40-200 BPM)
    - 459-558: CHORD (100 chord types)
    - 559-590: BAR (32 bars)
    """

    def __init__(
        self,
        num_time_shift_bins: int = 100,
        max_time_shift_ms: int = 1000,
        num_tempo_bins: int = 100,
        num_chord_types: int = 100,
        num_bars: int = 32
    ):
        self.num_time_shift_bins = num_time_shift_bins
        self.max_time_shift_ms = max_time_shift_ms
        self.num_tempo_bins = num_tempo_bins
        self.num_chord_types = num_chord_types
        self.num_bars = num_bars

        # Special tokens
        self.PAD = 0
        self.BOS = 1
        self.EOS = 2

        # Calculate offsets
        self.note_on_offset = 3
        self.note_off_offset = self.note_on_offset + 128
        self.time_shift_offset = self.note_off_offset + 128
        self.set_tempo_offset = self.time_shift_offset + num_time_shift_bins
        self.chord_offset = self.set_tempo_offset + num_tempo_bins
        self.bar_offset = self.chord_offset + num_chord_types

        self.vocab_size = self.bar_offset + num_bars

        # Chord vocabulary (jazz chords)
        self.chord_to_id = self._init_chord_vocab()
        self.id_to_chord = {v: k for k, v in self.chord_to_id.items()}

    def _init_chord_vocab(self) -> Dict[str, int]:
        """Initialize chord vocabulary"""
        chords = {}
        idx = 0

        # 12 root notes
        roots = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

        # Common jazz chord types
        types = [
            'maj', 'min', 'maj7', 'min7', '7', 'dim', 'aug',
            'm7b5', 'maj9', 'min9', '9', 'maj11', 'min11'
        ]

        for root in roots:
            for chord_type in types:
                if idx >= self.num_chord_types - 1:  # Reserve one for "N" (no chord)
                    break
                chords[f"{root}{chord_type}"] = idx
                idx += 1

        chords['N'] = idx  # No chord
        return chords

    def encode_event(self, event: MIDIEvent) -> int:
        """Convert MIDIEvent to token ID"""
        if event.event_type == EventType.NOTE_ON:
            pitch = event.value & 0x7F
            return self.note_on_offset + pitch

        elif event.event_type == EventType.NOTE_OFF:
            return self.note_off_offset + event.value

        elif event.event_type == EventType.TIME_SHIFT:
            # Bin time shift (0-1000ms)
            bin_idx = min(
                int(event.value / self.max_time_shift_ms * self.num_time_shift_bins),
                self.num_time_shift_bins - 1
            )
            return self.time_shift_offset + bin_idx

        elif event.event_type == EventType.SET_TEMPO:
            # Bin tempo (40-200 BPM)
            tempo = max(40, min(200, event.value))
            bin_idx = int((tempo - 40) / 160 * self.num_tempo_bins)
            return self.set_tempo_offset + bin_idx

        elif event.event_type == EventType.CHORD:
            return self.chord_offset + event.value

        elif event.event_type == EventType.BAR:
            return self.bar_offset + (event.value % self.num_bars)

        return self.PAD

    def decode_token(self, token: int) -> Optional[MIDIEvent]:
        """Convert token ID to MIDIEvent"""
        if token in [self.PAD, self.BOS, self.EOS]:
            return None

        if self.note_on_offset <= token < self.note_off_offset:
            pitch = token - self.note_on_offset
            return MIDIEvent(EventType.NOTE_ON, pitch, 0.0)

        elif self.note_off_offset <= token < self.time_shift_offset:
            pitch = token - self.note_off_offset
            return MIDIEvent(EventType.NOTE_OFF, pitch, 0.0)

        elif self.time_shift_offset <= token < self.set_tempo_offset:
            bin_idx = token - self.time_shift_offset
            ms = int(bin_idx / self.num_time_shift_bins * self.max_time_shift_ms)
            return MIDIEvent(EventType.TIME_SHIFT, ms, 0.0)

        elif self.set_tempo_offset <= token < self.chord_offset:
            bin_idx = token - self.set_tempo_offset
            tempo = int(40 + bin_idx / self.num_tempo_bins * 160)
            return MIDIEvent(EventType.SET_TEMPO, tempo, 0.0)

        elif self.chord_offset <= token < self.bar_offset:
            chord_idx = token - self.chord_offset
            return MIDIEvent(EventType.CHORD, chord_idx, 0.0)

        elif self.bar_offset <= token < self.vocab_size:
            bar_idx = token - self.bar_offset
            return MIDIEvent(EventType.BAR, bar_idx, 0.0)

        return None


class EventTokenizer:
    """
    MIDI to event sequence tokenizer

    Compatible with HuggingFace tokenizer API
    """

    def __init__(
        self,
        vocab: Optional[EventVocabulary] = None,
        max_time_shift_ms: int = 1000
    ):
        self.vocab = vocab or EventVocabulary()
        self.max_time_shift_ms = max_time_shift_ms

    @property
    def vocab_size(self) -> int:
        return self.vocab.vocab_size

    @property
    def pad_token_id(self) -> int:
        return self.vocab.PAD

    @property
    def bos_token_id(self) -> int:
        return self.vocab.BOS

    @property
    def eos_token_id(self) -> int:
        return self.vocab.EOS

    def midi_to_events(self, midi_path: str) -> List[MIDIEvent]:
        """
        Convert MIDI file to event sequence

        Args:
            midi_path: Path to MIDI file

        Returns:
            List of MIDIEvent objects
        """
        midi = pretty_midi.PrettyMIDI(midi_path)
        events = []

        # Get all note events from all instruments
        for instrument in midi.instruments:
            if instrument.is_drum:
                continue

            for note in instrument.notes:
                # Note ON
                events.append(MIDIEvent(
                    EventType.NOTE_ON,
                    note.pitch | (note.velocity << 7),  # Combine pitch and velocity
                    note.start
                ))

                # Note OFF
                events.append(MIDIEvent(
                    EventType.NOTE_OFF,
                    note.pitch,
                    note.end
                ))

        # Sort by time
        events.sort(key=lambda e: e.time)

        # Convert to relative time shifts
        relative_events = []
        current_time = 0.0

        for event in events:
            # Calculate time difference
            time_diff_ms = int((event.time - current_time) * 1000)

            # Add time shifts (may need multiple if > max_time_shift_ms)
            while time_diff_ms > 0:
                shift = min(time_diff_ms, self.max_time_shift_ms)
                relative_events.append(MIDIEvent(
                    EventType.TIME_SHIFT,
                    shift,
                    current_time
                ))
                time_diff_ms -= shift
                current_time += shift / 1000.0

            # Add note event
            relative_events.append(event)
            current_time = event.time

        return relative_events

    def encode(
        self,
        midi_path: str,
        add_special_tokens: bool = True
    ) -> List[int]:
        """
        Convert MIDI file to token IDs

        Args:
            midi_path: Path to MIDI file
            add_special_tokens: Add BOS/EOS tokens

        Returns:
            List of token IDs
        """
        # Convert to events
        events = self.midi_to_events(midi_path)

        # Convert to tokens
        tokens = []
        if add_special_tokens:
            tokens.append(self.vocab.BOS)

        for event in events:
            token = self.vocab.encode_event(event)
            tokens.append(token)

        if add_special_tokens:
            tokens.append(self.vocab.EOS)

        return tokens

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True
    ) -> List[MIDIEvent]:
        """
        Convert token IDs to MIDI events

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Skip PAD/BOS/EOS

        Returns:
            List of MIDIEvent objects
        """
        events = []
        for token in token_ids:
            if skip_special_tokens and token in [self.vocab.PAD, self.vocab.BOS, self.vocab.EOS]:
                continue
            event = self.vocab.decode_token(token)
            if event:
                events.append(event)
        return events

    def events_to_midi(
        self,
        events: List[MIDIEvent],
        output_path: Optional[str] = None,
        tempo: int = 120
    ) -> pretty_midi.PrettyMIDI:
        """
        Convert events to MIDI file

        Args:
            events: List of MIDIEvent
            output_path: Optional path to save MIDI
            tempo: Tempo in BPM

        Returns:
            pretty_midi.PrettyMIDI object
        """
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        piano = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

        current_time = 0.0
        active_notes = {}  # pitch -> (start_time, velocity)

        for event in events:
            if event.event_type == EventType.TIME_SHIFT:
                current_time += event.value / 1000.0

            elif event.event_type == EventType.NOTE_ON:
                pitch = event.value & 0x7F
                velocity = (event.value >> 7) & 0x7F
                if velocity == 0:
                    velocity = 80  # Default velocity
                active_notes[pitch] = (current_time, velocity)

            elif event.event_type == EventType.NOTE_OFF:
                pitch = event.value
                if pitch in active_notes:
                    start_time, velocity = active_notes.pop(pitch)
                    note = pretty_midi.Note(
                        velocity=velocity,
                        pitch=pitch,
                        start=start_time,
                        end=current_time
                    )
                    piano.notes.append(note)

        midi.instruments.append(piano)

        if output_path:
            midi.write(output_path)

        return midi

    def __call__(self, midi_path: str, **kwargs) -> List[int]:
        """Make tokenizer callable (HuggingFace style)"""
        return self.encode(midi_path, **kwargs)


if __name__ == "__main__":
    print("Testing Event Tokenizer...\n")

    # Create tokenizer
    tokenizer = EventTokenizer()

    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"PAD token: {tokenizer.pad_token_id}")
    print(f"BOS token: {tokenizer.bos_token_id}")
    print(f"EOS token: {tokenizer.eos_token_id}")

    # Create sample events
    sample_events = [
        MIDIEvent(EventType.NOTE_ON, 60 | (80 << 7), 0.0),  # C4
        MIDIEvent(EventType.TIME_SHIFT, 500, 0.5),
        MIDIEvent(EventType.NOTE_OFF, 60, 0.5),
        MIDIEvent(EventType.NOTE_ON, 64 | (75 << 7), 0.5),  # E4
        MIDIEvent(EventType.TIME_SHIFT, 500, 1.0),
        MIDIEvent(EventType.NOTE_OFF, 64, 1.0),
    ]

    print(f"\nSample events: {len(sample_events)}")
    for event in sample_events:
        print(f"  {event}")

    # Encode events
    tokens = [tokenizer.vocab.BOS]
    for event in sample_events:
        tokens.append(tokenizer.vocab.encode_event(event))
    tokens.append(tokenizer.vocab.EOS)

    print(f"\nTokens: {tokens}")

    # Decode tokens
    decoded_events = tokenizer.decode(tokens)
    print(f"\nDecoded events: {len(decoded_events)}")
    for event in decoded_events:
        print(f"  {event}")

    # Convert to MIDI
    midi = tokenizer.events_to_midi(decoded_events, "test_output.mid")
    print(f"\n✅ MIDI saved to test_output.mid")
    print(f"   Duration: {midi.get_end_time():.2f}s")
    print(f"   Notes: {len(midi.instruments[0].notes)}")

    print("\n✅ Event tokenizer tests passed!")
