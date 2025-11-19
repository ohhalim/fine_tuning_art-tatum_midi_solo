#!/usr/bin/env python3
"""
Phase 1: 데이터셋 전처리 및 분할

MIDI 파일을 검증하고 Train/Val/Test로 분할합니다.
"""

import argparse
import os
import shutil
import json
from pathlib import Path
from tqdm import tqdm

def validate_midi(filepath):
    """MIDI 파일 검증"""
    try:
        import pretty_midi
        midi = pretty_midi.PrettyMIDI(filepath)

        # 길이 체크
        duration = midi.get_end_time()
        if duration < 5:  # 최소 5초
            return False, "too_short"

        # 피아노 트랙 확인
        has_piano = False
        for instrument in midi.instruments:
            if not instrument.is_drum and instrument.program < 8:  # Piano family
                has_piano = True
                break

        if not has_piano:
            return False, "no_piano"

        return True, "ok"

    except Exception as e:
        return False, f"corrupt: {e}"

def main():
    parser = argparse.ArgumentParser(description='MIDI 데이터셋 전처리')
    parser.add_argument('--input_dir', required=True, help='입력 디렉토리 (raw MIDI)')
    parser.add_argument('--output_dir', required=True, help='출력 디렉토리')
    parser.add_argument('--min_duration', type=float, default=10, help='최소 길이 (초)')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)

    args = parser.parse_args()

    print("=" * 60)
    print("MIDI 데이터셋 전처리")
    print("=" * 60)
    print(f"입력: {args.input_dir}")
    print(f"출력: {args.output_dir}")
    print()

    # MIDI 파일 수집
    midi_files = list(Path(args.input_dir).rglob('*.mid')) + \
                 list(Path(args.input_dir).rglob('*.midi'))

    print(f"발견한 MIDI 파일: {len(midi_files)}개")
    print()

    # 검증
    print("검증 중...")
    valid_files = []
    reject_reasons = {}

    for filepath in tqdm(midi_files):
        is_valid, reason = validate_midi(str(filepath))
        if is_valid:
            valid_files.append(filepath)
        else:
            reject_reasons[reason] = reject_reasons.get(reason, 0) + 1

    print()
    print(f"✅ 통과: {len(valid_files)}개")
    print(f"❌ 제외: {len(midi_files) - len(valid_files)}개")
    for reason, count in reject_reasons.items():
        print(f"   - {reason}: {count}개")
    print()

    if len(valid_files) == 0:
        print("❌ 유효한 파일이 없습니다!")
        return

    # 분할
    import random
    random.shuffle(valid_files)

    n = len(valid_files)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)

    train_files = valid_files[:n_train]
    val_files = valid_files[n_train:n_train+n_val]
    test_files = valid_files[n_train+n_val:]

    print(f"분할:")
    print(f"  훈련: {len(train_files)}개")
    print(f"  검증: {len(val_files)}개")
    print(f"  테스트: {len(test_files)}개")
    print()

    # 복사
    for split_name, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        split_dir = Path(args.output_dir) / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        for filepath in tqdm(files, desc=f"{split_name} 복사"):
            shutil.copy2(filepath, split_dir / filepath.name)

    # 메타데이터 저장
    metadata = {
        'total_files': n,
        'valid_files': len(valid_files),
        'train': len(train_files),
        'val': len(val_files),
        'test': len(test_files),
        'reject_reasons': reject_reasons
    }

    metadata_path = Path(args.output_dir) / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print()
    print(f"✅ 완료! 메타데이터 저장: {metadata_path}")
    print()
    print("다음 단계:")
    print("  python scripts/phase1_analyze_data.py")

if __name__ == '__main__':
    main()
