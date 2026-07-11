"""
Jazz MIDI 데이터 전처리 스크립트
midi_dataset/ 폴더의 MIDI 파일들을 Music Transformer 형식으로 변환

Usage:
    python scripts/preprocess_jazz.py --input_dir ./midi_dataset/midi --output_dir ./data/jazz_processed
"""

import os
import sys
import argparse
import random
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Add music_transformer to path
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "music_transformer"))
sys.path.insert(0, str(SCRIPT_DIR / "music_transformer" / "third_party"))

from midi_processor.processor import decode_midi


def encode_midi_simple(file_path: str) -> list:
    """
    Simplified MIDI encoding that works without sustain pedal info.
    Based on midi_processor but handles edge cases better.
    """
    import pretty_midi
    
    RANGE_NOTE_ON = 128
    RANGE_NOTE_OFF = 128
    RANGE_TIME_SHIFT = 100
    
    START_IDX = {
        'note_on': 0,
        'note_off': RANGE_NOTE_ON,
        'time_shift': RANGE_NOTE_ON + RANGE_NOTE_OFF,
        'velocity': RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT
    }
    
    mid = pretty_midi.PrettyMIDI(midi_file=file_path)
    
    # Collect all notes from all instruments
    all_notes = []
    for inst in mid.instruments:
        if not inst.is_drum:  # Skip drum tracks
            all_notes.extend(inst.notes)
    
    if not all_notes:
        return []
    
    # Sort by start time
    all_notes.sort(key=lambda x: (x.start, x.pitch))
    
    # Convert to events
    events = []
    cur_time = 0.0
    cur_vel = 0
    
    # Create note on/off pairs
    note_events = []
    for note in all_notes:
        note_events.append(('note_on', note.start, note.pitch, note.velocity))
        note_events.append(('note_off', note.end, note.pitch, 0))
    
    # Sort by time
    note_events.sort(key=lambda x: x[1])
    
    for event_type, event_time, pitch, velocity in note_events:
        # Add time shift events
        time_diff = event_time - cur_time
        if time_diff > 0:
            time_steps = int(round(time_diff * 100))
            while time_steps >= RANGE_TIME_SHIFT:
                events.append(START_IDX['time_shift'] + RANGE_TIME_SHIFT - 1)
                time_steps -= RANGE_TIME_SHIFT
            if time_steps > 0:
                events.append(START_IDX['time_shift'] + time_steps - 1)
        
        # Add velocity event (only for note_on)
        if event_type == 'note_on':
            mod_vel = velocity // 4
            if mod_vel != cur_vel:
                events.append(START_IDX['velocity'] + mod_vel)
                cur_vel = mod_vel
            events.append(START_IDX['note_on'] + pitch)
        else:
            events.append(START_IDX['note_off'] + pitch)
        
        cur_time = event_time
    
    return events


def find_midi_files(input_dir: str, extensions=(".mid", ".midi")):
    """Find all MIDI files recursively"""
    midi_files = []
    for ext in extensions:
        midi_files.extend(Path(input_dir).rglob(f"*{ext}"))
    return sorted(midi_files)


def process_midi_file(midi_path: Path, max_seq: int = 2048):
    """Process a single MIDI file and return token sequence"""
    try:
        # Encode MIDI to tokens using our simplified encoder
        tokens = encode_midi_simple(str(midi_path))
        
        if tokens is None or len(tokens) == 0:
            return None
        
        # Convert to numpy array
        tokens = np.array(tokens, dtype=np.int32)
        
        # Skip if too short
        if len(tokens) < 100:
            return None
        
        return tokens
    
    except Exception as e:
        # print(f"Error processing {midi_path}: {e}")
        return None


def split_data(files, train_ratio=0.9):
    """Split files into train and validation sets"""
    random.shuffle(files)
    split_idx = int(len(files) * train_ratio)
    return files[:split_idx], files[split_idx:]


def main():
    parser = argparse.ArgumentParser(description="Preprocess Jazz MIDI files")
    parser.add_argument("--input_dir", type=str, default="./midi_dataset/midi",
                        help="Directory containing MIDI files")
    parser.add_argument("--output_dir", type=str, default="./data/jazz_processed",
                        help="Output directory for processed data")
    parser.add_argument("--max_seq", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--train_ratio", type=float, default=0.9,
                        help="Train/validation split ratio")
    parser.add_argument("--sample", type=int, default=None,
                        help="Process only N files (for testing)")
    
    args = parser.parse_args()
    
    # Create output directories
    train_dir = Path(args.output_dir) / "train"
    val_dir = Path(args.output_dir) / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all MIDI files
    print(f"\n=== Finding MIDI files in {args.input_dir} ===")
    midi_files = find_midi_files(args.input_dir)
    print(f"Found {len(midi_files)} MIDI files")
    
    if args.sample:
        midi_files = random.sample(list(midi_files), min(args.sample, len(midi_files)))
        print(f"Sampling {len(midi_files)} files for testing")
    
    # Process files
    print(f"\n=== Processing MIDI files ===")
    processed_files = []
    
    for midi_path in tqdm(midi_files, desc="Processing"):
        tokens = process_midi_file(midi_path, args.max_seq)
        
        if tokens is not None:
            processed_files.append((midi_path, tokens))
    
    print(f"Successfully processed {len(processed_files)} / {len(midi_files)} files")
    
    if len(processed_files) == 0:
        print("Error: No files were successfully processed!")
        return
    
    # Split into train/val
    train_files, val_files = split_data(processed_files, args.train_ratio)
    print(f"\nTrain: {len(train_files)} files, Val: {len(val_files)} files")
    
    # Save processed data
    print(f"\n=== Saving processed data ===")
    
    for i, (midi_path, tokens) in enumerate(tqdm(train_files, desc="Saving train")):
        output_path = train_dir / f"{i:05d}.npy"
        np.save(output_path, tokens)
    
    for i, (midi_path, tokens) in enumerate(tqdm(val_files, desc="Saving val")):
        output_path = val_dir / f"{i:05d}.npy"
        np.save(output_path, tokens)
    
    # Print stats
    print(f"\n=== Statistics ===")
    all_tokens = [t for _, t in processed_files]
    total_tokens = sum(len(t) for t in all_tokens)
    avg_length = total_tokens / len(all_tokens)
    max_length = max(len(t) for t in all_tokens)
    min_length = min(len(t) for t in all_tokens)
    
    print(f"Total tokens: {total_tokens:,}")
    print(f"Average sequence length: {avg_length:.1f}")
    print(f"Max sequence length: {max_length}")
    print(f"Min sequence length: {min_length}")
    
    print(f"\n=== Done ===")
    print(f"Train data saved to: {train_dir}")
    print(f"Val data saved to: {val_dir}")


if __name__ == "__main__":
    main()
