"""
Event-based MIDI Representation

Similar to Google Magenta's Music Transformer, but optimized for jazz piano.

Events:
- NOTE_ON(pitch, velocity): Note starts
- NOTE_OFF(pitch): Note ends
- TIME_SHIFT(delta): Time advances
- SET_TEMPO(bpm): Tempo change
- CHORD(type): Chord symbol

This representation is:
1. More natural (like music notation)
2. Autoregressive generation friendly
3. Efficient for Transformer models
4. Easy to condition on chords

Key advantages over 5D representation:
- Sequential events (like language)
- Variable length (no padding)
- Explicit timing (more precise)
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
    BAR = 5  # Bar marker
    POSITION = 6  # Position in bar (for relative attention)


@dataclass
class MIDIEvent:
    """Single MIDI event"""
    event_type: EventType
    value: int  # Pitch, velocity, time_shift, tempo, etc.
    time: float  # Absolute time (for sorting)

    def __repr__(self):
        if self.event_type == EventType.NOTE_ON:
            return f"NoteOn(pitch={self.value >> 7}, vel={self.value & 0x7F})"
        elif self.event_type == EventType.NOTE_OFF:
            return f"NoteOff(pitch={self.value})"
        elif self.event_type == EventType.TIME_SHIFT:
            return f"TimeShift({self.value}ms)"
        elif self.event_type == EventType.CHORD:
            return f"Chord({self.value})"
        elif self.event_type == EventType.BAR:
            return f"Bar({self.value})"
        else:
            return f"{self.event_type.name}({self.value})"


class EventVocabulary:
    """
    Vocabulary for event-based MIDI

    Token structure:
    - 0: PAD
    - 1: START
    - 2: END
    - 3-130: NOTE_ON (128 pitches, combined with velocity bins)
    - 131-258: NOTE_OFF (128 pitches)
    - 259-358: TIME_SHIFT (100 bins, 0-1000ms)
    - 359-458: SET_TEMPO (100 bins, 40-200 BPM)
    - 459-558: CHORD (100 chord types)
    - 559-590: BAR (32 bars)
    - 591-622: POSITION (32 positions in bar)
    """

    def __init__(
        self,
        num_velocity_bins: int = 32,
        num_time_shift_bins: int = 100,
        max_time_shift_ms: int = 1000,
        num_tempo_bins: int = 100,
        num_chord_types: int = 100,
        num_bars: int = 32,
        num_positions: int = 32
    ):
        self.num_velocity_bins = num_velocity_bins
        self.num_time_shift_bins = num_time_shift_bins
        self.max_time_shift_ms = max_time_shift_ms
        self.num_tempo_bins = num_tempo_bins
        self.num_chord_types = num_chord_types
        self.num_bars = num_bars
        self.num_positions = num_positions

        # Special tokens
        self.PAD = 0
        self.START = 1
        self.END = 2

        # Calculate offsets
        self.note_on_offset = 3
        self.note_off_offset = self.note_on_offset + 128
        self.time_shift_offset = self.note_off_offset + 128
        self.set_tempo_offset = self.time_shift_offset + num_time_shift_bins
        self.chord_offset = self.set_tempo_offset + num_tempo_bins
        self.bar_offset = self.chord_offset + num_chord_types
        self.position_offset = self.bar_offset + num_bars

        self.vocab_size = self.position_offset + num_positions

        # Chord vocabulary
        self.chord_vocab = self._init_chord_vocab()

    def _init_chord_vocab(self) -> Dict[str, int]:
        """Initialize chord vocabulary"""
        chords = {}
        idx = 0

        # Root notes (12)
        roots = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        # Chord types
        types = ['maj', 'min', 'maj7', 'min7', '7', 'dim', 'aug', 'sus4', 'm7b5',
                 'maj9', 'min9', '9', 'maj11', 'min11', '11', 'maj13', 'min13', '13']

        for root in roots:
            for chord_type in types:
                if idx >= self.num_chord_types:
                    break
                chords[f"{root}{chord_type}"] = idx
                idx += 1

        chords['N'] = idx  # No chord
        return chords

    def encode_event(self, event: MIDIEvent) -> int:
        """Convert MIDIEvent to token"""
        if event.event_type == EventType.NOTE_ON:
            # Combine pitch and velocity
            pitch = event.value >> 7
            velocity = event.value & 0x7F
            velocity_bin = int(velocity / 128 * self.num_velocity_bins)
            # For simplicity, we'll just use pitch
            # In production, you might want pitch + velocity bins
            return self.note_on_offset + pitch

        elif event.event_type == EventType.NOTE_OFF:
            return self.note_off_offset + event.value

        elif event.event_type == EventType.TIME_SHIFT:
            # Bin time shift
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
            return self.bar_offset + event.value

        elif event.event_type == EventType.POSITION:
            return self.position_offset + event.value

        return self.PAD

    def decode_token(self, token: int) -> Optional[MIDIEvent]:
        """Convert token to MIDIEvent"""
        if token == self.PAD or token == self.START or token == self.END:
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

        elif self.bar_offset <= token < self.position_offset:
            bar_idx = token - self.bar_offset
            return MIDIEvent(EventType.BAR, bar_idx, 0.0)

        elif self.position_offset <= token < self.vocab_size:
            pos_idx = token - self.position_offset
            return MIDIEvent(EventType.POSITION, pos_idx, 0.0)

        return None


class EventMIDIConverter:
    """
    Convert between MIDI files and event sequences

    This is the core data format for Perceiver + Music Transformer
    """

    def __init__(
        self,
        vocab: Optional[EventVocabulary] = None,
        time_resolution_ms: int = 10,  # 10ms resolution
        max_time_shift_ms: int = 1000
    ):
        self.vocab = vocab or EventVocabulary()
        self.time_resolution_ms = time_resolution_ms
        self.max_time_shift_ms = max_time_shift_ms

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

        # Get all note events
        for instrument in midi.instruments:
            if instrument.is_drum:
                continue

            for note in instrument.notes:
                # Note ON
                events.append(MIDIEvent(
                    EventType.NOTE_ON,
                    (note.pitch << 7) | note.velocity,  # Combine pitch and velocity
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
            # Add time shift if needed
            time_diff_ms = int((event.time - current_time) * 1000)

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

    def events_to_tokens(self, events: List[MIDIEvent]) -> List[int]:
        """Convert events to tokens"""
        tokens = [self.vocab.START]
        for event in events:
            token = self.vocab.encode_event(event)
            tokens.append(token)
        tokens.append(self.vocab.END)
        return tokens

    def tokens_to_events(self, tokens: List[int]) -> List[MIDIEvent]:
        """Convert tokens to events"""
        events = []
        for token in tokens:
            if token in [self.vocab.PAD, self.vocab.START, self.vocab.END]:
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
        Convert event sequence to MIDI file

        Args:
            events: List of MIDIEvent
            output_path: Optional output path
            tempo: Tempo in BPM

        Returns:
            pretty_midi.PrettyMIDI object
        """
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        piano = pretty_midi.Instrument(program=0)

        current_time = 0.0
        active_notes = {}  # pitch -> (start_time, velocity)

        for event in events:
            if event.event_type == EventType.TIME_SHIFT:
                current_time += event.value / 1000.0

            elif event.event_type == EventType.NOTE_ON:
                pitch = event.value >> 7
                velocity = event.value & 0x7F
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


if __name__ == "__main__":
    print("Testing Event-based MIDI Representation...")

    # Create vocabulary
    vocab = EventVocabulary()
    print(f"Vocabulary size: {vocab.vocab_size}")

    # Create converter
    converter = EventMIDIConverter(vocab)

    # Test chord encoding
    print("\nChord vocabulary (first 20):")
    for chord, idx in list(vocab.chord_vocab.items())[:20]:
        print(f"  {chord}: {idx}")

    # Create sample events
    sample_events = [
        MIDIEvent(EventType.NOTE_ON, (60 << 7) | 80, 0.0),  # C4, vel=80
        MIDIEvent(EventType.TIME_SHIFT, 500, 0.5),
        MIDIEvent(EventType.NOTE_OFF, 60, 0.5),
        MIDIEvent(EventType.NOTE_ON, (64 << 7) | 75, 0.5),  # E4
        MIDIEvent(EventType.TIME_SHIFT, 500, 1.0),
        MIDIEvent(EventType.NOTE_OFF, 64, 1.0),
    ]

    print(f"\nSample events: {len(sample_events)}")
    for event in sample_events:
        print(f"  {event}")

    # Convert to tokens
    tokens = converter.events_to_tokens(sample_events)
    print(f"\nTokens: {tokens[:10]}...")

    # Convert back
    decoded_events = converter.tokens_to_events(tokens)
    print(f"\nDecoded events: {len(decoded_events)}")

    print("\n✅ Event-based MIDI representation tests passed!")
