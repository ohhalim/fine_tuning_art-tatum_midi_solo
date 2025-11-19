#!/usr/bin/env python3
"""
Phase 1: MIDI 데이터 수집 스크립트

다양한 소스에서 Art Tatum MIDI 파일을 수집합니다.
"""

import argparse
import os
import requests
from pathlib import Path
from tqdm import tqdm

def collect_from_bitmidi(artist: str, output_dir: str, min_files: int = 50):
    """
    Bitmidi에서 아티스트 MIDI 파일 검색 및 다운로드

    주의: 실제로는 웹 크롤링 필요. 여기서는 템플릿만 제공.
    """
    print(f"Bitmidi에서 '{artist}' 검색 중...")
    print("⚠️  주의: 이 스크립트는 템플릿입니다.")
    print("   실제 다운로드는 https://bitmidi.com 에서 수동으로 하세요.")
    print()
    print("검색 URL:")
    search_url = f"https://bitmidi.com/search?q={artist.replace(' ', '+')}"
    print(f"  {search_url}")
    print()
    print("다운로드한 파일을 다음 경로에 저장하세요:")
    print(f"  {output_dir}/")

def collect_from_lmd(artist: str, lmd_dir: str, output_dir: str):
    """
    Lakh MIDI Dataset에서 특정 아티스트 추출
    """
    print(f"Lakh MIDI Dataset에서 '{artist}' 추출 중...")

    if not os.path.exists(lmd_dir):
        print(f"❌ LMD 디렉토리가 없습니다: {lmd_dir}")
        print()
        print("다운로드 방법:")
        print("  wget http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz")
        print("  tar -xzvf lmd_full.tar.gz")
        return

    # LMD에서 아티스트명으로 검색 (파일명 기반)
    midi_files = []
    for root, dirs, files in os.walk(lmd_dir):
        for file in files:
            if file.endswith('.mid') or file.endswith('.midi'):
                # 파일명에 아티스트명 포함 여부 확인 (대소문자 무시)
                if artist.lower() in file.lower():
                    midi_files.append(os.path.join(root, file))

    print(f"찾은 파일: {len(midi_files)}개")

    if len(midi_files) == 0:
        print("⚠️  파일을 찾지 못했습니다.")
        print("   LMD에 해당 아티스트가 없을 수 있습니다.")
        return

    # 출력 디렉토리로 복사
    os.makedirs(output_dir, exist_ok=True)

    import shutil
    for src_path in tqdm(midi_files, desc="복사 중"):
        filename = os.path.basename(src_path)
        dst_path = os.path.join(output_dir, filename)
        shutil.copy2(src_path, dst_path)

    print(f"✅ {len(midi_files)}개 파일을 {output_dir}/ 에 저장했습니다.")

def main():
    parser = argparse.ArgumentParser(description='MIDI 데이터 수집')
    parser.add_argument('--source', choices=['bitmidi', 'lmd'], default='bitmidi',
                        help='데이터 소스')
    parser.add_argument('--artist', default='Art Tatum',
                        help='아티스트 이름')
    parser.add_argument('--output_dir', default='data/art_tatum_midi/raw',
                        help='출력 디렉토리')
    parser.add_argument('--lmd_dir', default='lmd_full',
                        help='Lakh MIDI Dataset 경로 (source=lmd일 때)')
    parser.add_argument('--min_files', type=int, default=50,
                        help='최소 파일 개수')

    args = parser.parse_args()

    print("=" * 60)
    print("MIDI 데이터 수집")
    print("=" * 60)
    print(f"소스: {args.source}")
    print(f"아티스트: {args.artist}")
    print(f"출력: {args.output_dir}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.source == 'bitmidi':
        collect_from_bitmidi(args.artist, args.output_dir, args.min_files)
    elif args.source == 'lmd':
        collect_from_lmd(args.artist, args.lmd_dir, args.output_dir)

    print()
    print("=" * 60)
    print("수집 완료!")
    print("=" * 60)
    print()
    print("다음 단계:")
    print(f"  python scripts/phase1_prepare_dataset.py --input_dir {args.output_dir}")

if __name__ == '__main__':
    main()
