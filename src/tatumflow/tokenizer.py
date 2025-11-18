"""
Enhanced MIDI Tokenizer for TatumFlow
Combines best practices from Aria and REMI with new innovations
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import mido
from mido import MidiFile, MidiTrack, Message


@dataclass
class TokenizerConfig:
    """Configuration for tokenizer"""
    # Timing
    time_quantization_ms: int = 10  # 10ms resolution like Aria
    max_duration_ms: int = 10000
    max_time_shift_ms: int = 5000  # 5 second chunks

    # Pitch
    min_pitch: int = 21  # A0
    max_pitch: int = 108  # C8

    # Velocity
    velocity_bins: int = 32

    # Special tokens
    pad_token: str = "<PAD>"
    sos_token: str = "<SOS>"
    eos_token: str = "<EOS>"
    mask_token: str = "<MASK>"
    time_shift_token: str = "<T>"

    # Sustain pedal
    use_sustain: bool = True


class TatumFlowTokenizer:
    """
    Multi-dimensional tokenizer for expressive piano performance

    Token types:
    - TIME_SHIFT: absolute time within chunk (0-5000ms in 10ms steps)
    - NOTE_ON: pitch (21-108)
    - NOTE_OFF: pitch (21-108)
    - VELOCITY: 32 bins (0-127)
    - DURATION: duration in 10ms steps
    - SUSTAIN_ON/SUSTAIN_OFF
    - Special tokens: PAD, SOS, EOS, MASK, TIME_CHUNK
    """

    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or TokenizerConfig()
        self._build_vocabulary()

    def _build_vocabulary(self):
        """Build token vocabulary"""
        self.token_to_id = {}
        self.id_to_token = {}
        current_id = 0

        # Special tokens
        for token in [
            self.config.pad_token,
            self.config.sos_token,
            self.config.eos_token,
            self.config.mask_token,
            self.config.time_shift_token
        ]:
            self.token_to_id[token] = current_id
            self.id_to_token[current_id] = token
            current_id += 1

        # Time shifts (0-5000ms in 10ms steps = 500 tokens)
        max_time_steps = self.config.max_time_shift_ms // self.config.time_quantization_ms
        for i in range(max_time_steps + 1):
            token = f"TIME_{i}"
            self.token_to_id[token] = current_id
            self.id_to_token[current_id] = token
            current_id += 1

        # Note ON (pitch 21-108)
        for pitch in range(self.config.min_pitch, self.config.max_pitch + 1):
            token = f"NOTE_ON_{pitch}"
            self.token_to_id[token] = current_id
            self.id_to_token[current_id] = token
            current_id += 1

        # Note OFF (pitch 21-108)
        for pitch in range(self.config.min_pitch, self.config.max_pitch + 1):
            token = f"NOTE_OFF_{pitch}"
            self.token_to_id[token] = current_id
            self.id_to_token[current_id] = token
            current_id += 1

        # Velocity (32 bins)
        for vel in range(self.config.velocity_bins):
            token = f"VEL_{vel}"
            self.token_to_id[token] = current_id
            self.id_to_token[current_id] = token
            current_id += 1

        # Duration (0-10000ms in 10ms steps = 1000 tokens)
        max_duration_steps = self.config.max_duration_ms // self.config.time_quantization_ms
        for dur in range(max_duration_steps + 1):
            token = f"DUR_{dur}"
            self.token_to_id[token] = current_id
            self.id_to_token[current_id] = token
            current_id += 1

        # Sustain pedal
        if self.config.use_sustain:
            self.token_to_id["SUSTAIN_ON"] = current_id
            self.id_to_token[current_id] = "SUSTAIN_ON"
            current_id += 1

            self.token_to_id["SUSTAIN_OFF"] = current_id
            self.id_to_token[current_id] = "SUSTAIN_OFF"
            current_id += 1

        self.vocab_size = current_id

        # Store special token IDs
        self.pad_id = self.token_to_id[self.config.pad_token]
        self.sos_id = self.token_to_id[self.config.sos_token]
        self.eos_id = self.token_to_id[self.config.eos_token]
        self.mask_id = self.token_to_id[self.config.mask_token]
        self.time_shift_id = self.token_to_id[self.config.time_shift_token]

    def quantize_time(self, time_ms: float) -> int:
        """Quantize time to nearest step"""
        return round(time_ms / self.config.time_quantization_ms)

    def dequantize_time(self, steps: int) -> float:
        """Convert time steps back to milliseconds"""
        return steps * self.config.time_quantization_ms

    def quantize_velocity(self, velocity: int) -> int:
        """Quantize MIDI velocity (0-127) to bins"""
        bin_size = 128 / self.config.velocity_bins
        return min(int(velocity / bin_size), self.config.velocity_bins - 1)

    def dequantize_velocity(self, vel_bin: int) -> int:
        """Convert velocity bin back to MIDI velocity"""
        bin_size = 128 / self.config.velocity_bins
        return int((vel_bin + 0.5) * bin_size)

    def encode_midi(self, midi_path: str) -> List[int]:
        """
        Encode MIDI file to token sequence

        Returns:
            List of token IDs
        """
        midi = MidiFile(midi_path)

        # Merge all tracks and convert to absolute time
        events = []
        for track in midi.tracks:
            current_time = 0
            for msg in track:
                current_time += msg.time
                if msg.type in ['note_on', 'note_off', 'control_change']:
                    events.append({
                        'time': current_time,
                        'type': msg.type,
                        'note': getattr(msg, 'note', None),
                        'velocity': getattr(msg, 'velocity', None),
                        'control': getattr(msg, 'control', None),
                        'value': getattr(msg, 'value', None)
                    })

        # Sort by time
        events.sort(key=lambda x: x['time'])

        # Convert ticks to milliseconds
        ticks_per_beat = midi.ticks_per_beat
        # Assume 120 BPM if no tempo is set
        tempo = 500000  # microseconds per beat
        ms_per_tick = (tempo / 1000) / ticks_per_beat

        # Tokenize
        tokens = [self.sos_id]
        active_notes = {}  # pitch -> (start_time, velocity)
        chunk_start = 0
        current_time = 0

        for event in events:
            event_time_ms = event['time'] * ms_per_tick

            # Check if we need a new chunk
            if event_time_ms - chunk_start >= self.config.max_time_shift_ms:
                tokens.append(self.time_shift_id)
                chunk_start = event_time_ms

            # Add time shift token
            time_since_chunk = event_time_ms - chunk_start
            time_steps = self.quantize_time(time_since_chunk)
            time_steps = min(time_steps, self.config.max_time_shift_ms // self.config.time_quantization_ms)
            time_token = f"TIME_{time_steps}"
            tokens.append(self.token_to_id[time_token])

            # Process event
            if event['type'] == 'note_on' and event['velocity'] > 0:
                pitch = event['note']
                if self.config.min_pitch <= pitch <= self.config.max_pitch:
                    # Add NOTE_ON
                    tokens.append(self.token_to_id[f"NOTE_ON_{pitch}"])

                    # Add VELOCITY
                    vel_bin = self.quantize_velocity(event['velocity'])
                    tokens.append(self.token_to_id[f"VEL_{vel_bin}"])

                    # Track active note
                    active_notes[pitch] = (event_time_ms, event['velocity'])

            elif event['type'] == 'note_off' or (event['type'] == 'note_on' and event['velocity'] == 0):
                pitch = event['note']
                if pitch in active_notes and self.config.min_pitch <= pitch <= self.config.max_pitch:
                    start_time, velocity = active_notes[pitch]
                    duration_ms = event_time_ms - start_time
                    duration_steps = self.quantize_time(duration_ms)
                    duration_steps = min(duration_steps, self.config.max_duration_ms // self.config.time_quantization_ms)

                    # Add DURATION
                    tokens.append(self.token_to_id[f"DUR_{duration_steps}"])

                    # Add NOTE_OFF
                    tokens.append(self.token_to_id[f"NOTE_OFF_{pitch}"])

                    del active_notes[pitch]

            elif event['type'] == 'control_change' and event['control'] == 64:  # Sustain pedal
                if self.config.use_sustain:
                    if event['value'] >= 64:
                        tokens.append(self.token_to_id["SUSTAIN_ON"])
                    else:
                        tokens.append(self.token_to_id["SUSTAIN_OFF"])

        tokens.append(self.eos_id)
        return tokens

    def decode_tokens(self, tokens: List[int]) -> MidiFile:
        """
        Decode token sequence back to MIDI

        Args:
            tokens: List of token IDs

        Returns:
            MidiFile object
        """
        midi = MidiFile()
        track = MidiTrack()
        midi.tracks.append(track)

        # Tempo: 120 BPM
        tempo = mido.bpm2tempo(120)
        track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))

        current_time_ms = 0
        chunk_start_ms = 0
        last_event_time = 0
        active_notes = {}  # pitch -> (start_time, velocity)

        ticks_per_beat = 480
        midi.ticks_per_beat = ticks_per_beat
        ms_per_tick = (tempo / 1000) / ticks_per_beat

        i = 0
        while i < len(tokens):
            token_id = tokens[i]
            token_str = self.id_to_token.get(token_id, "")

            if token_str == self.config.time_shift_token:
                chunk_start_ms = current_time_ms

            elif token_str.startswith("TIME_"):
                time_steps = int(token_str.split("_")[1])
                current_time_ms = chunk_start_ms + self.dequantize_time(time_steps)

            elif token_str.startswith("NOTE_ON_"):
                pitch = int(token_str.split("_")[2])

                # Next token should be velocity
                i += 1
                if i < len(tokens):
                    vel_token = self.id_to_token.get(tokens[i], "")
                    if vel_token.startswith("VEL_"):
                        vel_bin = int(vel_token.split("_")[1])
                        velocity = self.dequantize_velocity(vel_bin)

                        # Convert time to ticks
                        time_ticks = int(current_time_ms / ms_per_tick)
                        delta_ticks = time_ticks - last_event_time

                        track.append(Message('note_on', note=pitch, velocity=velocity, time=delta_ticks))
                        last_event_time = time_ticks

                        active_notes[pitch] = current_time_ms

            elif token_str.startswith("DUR_"):
                # Duration token followed by NOTE_OFF
                duration_steps = int(token_str.split("_")[1])
                duration_ms = self.dequantize_time(duration_steps)

                # Next token should be NOTE_OFF
                i += 1
                if i < len(tokens):
                    note_off_token = self.id_to_token.get(tokens[i], "")
                    if note_off_token.startswith("NOTE_OFF_"):
                        pitch = int(note_off_token.split("_")[2])

                        if pitch in active_notes:
                            note_on_time = active_notes[pitch]
                            note_off_time = note_on_time + duration_ms

                            time_ticks = int(note_off_time / ms_per_tick)
                            delta_ticks = time_ticks - last_event_time

                            track.append(Message('note_off', note=pitch, velocity=64, time=delta_ticks))
                            last_event_time = time_ticks

                            del active_notes[pitch]

            elif token_str == "SUSTAIN_ON":
                time_ticks = int(current_time_ms / ms_per_tick)
                delta_ticks = time_ticks - last_event_time
                track.append(Message('control_change', control=64, value=127, time=delta_ticks))
                last_event_time = time_ticks

            elif token_str == "SUSTAIN_OFF":
                time_ticks = int(current_time_ms / ms_per_tick)
                delta_ticks = time_ticks - last_event_time
                track.append(Message('control_change', control=64, value=0, time=delta_ticks))
                last_event_time = time_ticks

            i += 1

        return midi

    def save_vocabulary(self, path: str):
        """Save vocabulary to file"""
        import json
        with open(path, 'w') as f:
            json.dump({
                'token_to_id': self.token_to_id,
                'config': {
                    'time_quantization_ms': self.config.time_quantization_ms,
                    'max_duration_ms': self.config.max_duration_ms,
                    'max_time_shift_ms': self.config.max_time_shift_ms,
                    'min_pitch': self.config.min_pitch,
                    'max_pitch': self.config.max_pitch,
                    'velocity_bins': self.config.velocity_bins,
                    'use_sustain': self.config.use_sustain
                }
            }, f, indent=2)

    @classmethod
    def load_vocabulary(cls, path: str):
        """Load vocabulary from file"""
        import json
        with open(path, 'r') as f:
            data = json.load(f)

        config = TokenizerConfig(**data['config'])
        tokenizer = cls(config)
        return tokenizer


if __name__ == "__main__":
    # Test tokenizer
    tokenizer = TatumFlowTokenizer()
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Special tokens: PAD={tokenizer.pad_id}, SOS={tokenizer.sos_id}, EOS={tokenizer.eos_id}")

    # Test encoding/decoding
    print("\nToken examples:")
    for i in range(5):
        print(f"  {i}: {tokenizer.id_to_token[i]}")
    print(f"  ...")
    print(f"  {tokenizer.vocab_size-1}: {tokenizer.id_to_token[tokenizer.vocab_size-1]}")
