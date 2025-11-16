"""
A/B Test 스크립트

베이스 모델 vs 파인튜닝 모델을 블라인드 테스트로 비교해.

사용법:
    python scripts/ab_test.py \
        --base_dir ./base_model_audio \
        --finetuned_dir ./finetuned_audio \
        --num_pairs 10

실행 순서:
1. 동일한 프롬프트로 양쪽 모델 생성
2. A/B로 랜덤 라벨링해서 저장
3. 듣고 투표
4. 결과 분석
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple
import json


def create_ab_pairs(base_dir: str, finetuned_dir: str, output_dir: str, num_pairs: int = 10) -> List[dict]:
    """
    A/B 테스트용 페어 생성

    Args:
        base_dir: 베이스 모델 오디오 폴더
        finetuned_dir: 파인튜닝 모델 오디오 폴더
        output_dir: A/B 테스트 폴더 (출력)
        num_pairs: 생성할 페어 개수

    Returns:
        페어 정보 리스트
    """
    print("=" * 80)
    print("🎵 A/B Test 페어 생성")
    print("=" * 80)

    base_path = Path(base_dir)
    finetuned_path = Path(finetuned_dir)
    output_path = Path(output_dir)

    output_path.mkdir(exist_ok=True)

    # 오디오 파일 목록
    base_files = []
    for ext in ['*.wav', '*.mp3', '*.flac']:
        base_files.extend(list(base_path.glob(ext)))

    finetuned_files = []
    for ext in ['*.wav', '*.mp3', '*.flac']:
        finetuned_files.extend(list(finetuned_path.glob(ext)))

    if len(base_files) == 0 or len(finetuned_files) == 0:
        print("❌ 오디오 파일이 없습니다!")
        return []

    print(f"\n📂 파일 확인:")
    print(f"   베이스 모델: {len(base_files)}개")
    print(f"   파인튜닝: {len(finetuned_files)}개")

    # 페어 생성
    pairs = []
    num_pairs = min(num_pairs, len(base_files), len(finetuned_files))

    print(f"\n🔄 {num_pairs}개 페어 생성 중...")

    for i in range(num_pairs):
        base_file = base_files[i]
        finetuned_file = finetuned_files[i]

        # 랜덤하게 A/B 할당
        if random.random() > 0.5:
            a_file = base_file
            b_file = finetuned_file
            a_model = "base"
            b_model = "finetuned"
        else:
            a_file = finetuned_file
            b_file = base_file
            a_model = "finetuned"
            b_model = "base"

        # 파일 복사
        a_dest = output_path / f"pair_{i+1:02d}_A{a_file.suffix}"
        b_dest = output_path / f"pair_{i+1:02d}_B{b_file.suffix}"

        shutil.copy(a_file, a_dest)
        shutil.copy(b_file, b_dest)

        pair_info = {
            'pair_id': i + 1,
            'A': {
                'file': str(a_dest.name),
                'model': a_model,
            },
            'B': {
                'file': str(b_dest.name),
                'model': b_model,
            }
        }

        pairs.append(pair_info)

        print(f"   ✅ Pair {i+1:02d}: {a_dest.name} vs {b_dest.name}")

    # 메타데이터 저장
    metadata_file = output_path / 'ab_test_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(pairs, f, indent=2)

    print(f"\n💾 메타데이터 저장: {metadata_file}")

    return pairs


def create_voting_sheet(pairs: List[dict], output_dir: str):
    """
    투표용 CSV 생성

    Args:
        pairs: 페어 정보 리스트
        output_dir: 출력 폴더
    """
    output_path = Path(output_dir)

    csv_file = output_path / 'voting_sheet.csv'

    with open(csv_file, 'w') as f:
        # 헤더
        f.write("Pair,File_A,File_B,Vote (A or B),Notes\n")

        # 각 페어
        for pair in pairs:
            f.write(f"{pair['pair_id']},{pair['A']['file']},{pair['B']['file']},,\n")

    print(f"📝 투표 시트 생성: {csv_file}")
    print(f"\n사용법:")
    print(f"   1. {csv_file} 파일을 열어")
    print(f"   2. 각 페어의 A, B 파일을 들어봐")
    print(f"   3. 'Vote' 열에 A 또는 B 입력")
    print(f"   4. 저장 후 analyze_votes() 실행")


def analyze_votes(voting_sheet: str, metadata_file: str):
    """
    투표 결과 분석

    Args:
        voting_sheet: 투표 시트 CSV 파일
        metadata_file: 메타데이터 JSON 파일
    """
    print("\n" + "=" * 80)
    print("📊 A/B Test 결과 분석")
    print("=" * 80)

    # 메타데이터 로드
    with open(metadata_file, 'r') as f:
        pairs = json.load(f)

    # 투표 시트 로드
    votes = {}
    with open(voting_sheet, 'r') as f:
        lines = f.readlines()[1:]  # 헤더 제외

        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 4 and parts[3]:
                pair_id = int(parts[0])
                vote = parts[3].strip().upper()
                votes[pair_id] = vote

    if not votes:
        print("❌ 투표 데이터가 없습니다!")
        return

    # 결과 집계
    base_wins = 0
    finetuned_wins = 0
    invalid_votes = 0

    for pair in pairs:
        pair_id = pair['pair_id']

        if pair_id not in votes:
            continue

        vote = votes[pair_id]

        if vote == 'A':
            winner = pair['A']['model']
        elif vote == 'B':
            winner = pair['B']['model']
        else:
            invalid_votes += 1
            continue

        if winner == 'base':
            base_wins += 1
        elif winner == 'finetuned':
            finetuned_wins += 1

    total_valid = base_wins + finetuned_wins

    print(f"\n📊 투표 결과:")
    print(f"   총 투표: {total_valid}표")
    print(f"   베이스 모델: {base_wins}표 ({base_wins/total_valid*100:.1f}%)")
    print(f"   파인튜닝 모델: {finetuned_wins}표 ({finetuned_wins/total_valid*100:.1f}%)")

    if invalid_votes > 0:
        print(f"   무효표: {invalid_votes}표")

    # 판정
    print(f"\n🏆 최종 판정:")

    if finetuned_wins > base_wins:
        win_rate = finetuned_wins / total_valid * 100
        margin = finetuned_wins - base_wins

        print(f"   ✅ 파인튜닝 모델 승리!")
        print(f"   승률: {win_rate:.1f}%")
        print(f"   격차: +{margin}표")

        if win_rate >= 70:
            print(f"   → 압도적 개선! 파인튜닝 대성공 🎉")
        elif win_rate >= 60:
            print(f"   → 명확한 개선. 파인튜닝 효과 확실함 ✅")
        else:
            print(f"   → 약간 개선. 더 학습하면 좋을 듯 🟡")

    elif base_wins > finetuned_wins:
        print(f"   ❌ 베이스 모델 승리")
        print(f"   → 파인튜닝 실패. 재학습 필요")

    else:
        print(f"   🟡 동점")
        print(f"   → 파인튜닝 효과 미미. 하이퍼파라미터 조정 필요")

    print("=" * 80)

    # 결과 저장
    result_file = Path(voting_sheet).parent / 'ab_test_results.txt'
    with open(result_file, 'w') as f:
        f.write(f"A/B Test 결과\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"총 투표: {total_valid}표\n")
        f.write(f"베이스 모델: {base_wins}표 ({base_wins/total_valid*100:.1f}%)\n")
        f.write(f"파인튜닝 모델: {finetuned_wins}표 ({finetuned_wins/total_valid*100:.1f}%)\n\n")

        if finetuned_wins > base_wins:
            f.write(f"판정: 파인튜닝 모델 승리\n")
        elif base_wins > finetuned_wins:
            f.write(f"판정: 베이스 모델 승리\n")
        else:
            f.write(f"판정: 동점\n")

    print(f"\n💾 결과 저장: {result_file}")


def run_interactive_test(output_dir: str):
    """
    인터랙티브 A/B 테스트 (터미널에서 직접 투표)

    Args:
        output_dir: A/B 테스트 폴더
    """
    output_path = Path(output_dir)

    # 메타데이터 로드
    metadata_file = output_path / 'ab_test_metadata.json'
    if not metadata_file.exists():
        print("❌ A/B 테스트 메타데이터가 없습니다!")
        return

    with open(metadata_file, 'r') as f:
        pairs = json.load(f)

    print("=" * 80)
    print("🎧 인터랙티브 A/B 테스트 시작")
    print("=" * 80)
    print("\n각 페어를 들어보고 A 또는 B를 선택하세요.")
    print("(종료: q, 건너뛰기: s)\n")

    votes = {}

    for pair in pairs:
        pair_id = pair['pair_id']

        print(f"\n--- Pair {pair_id} ---")
        print(f"A: {pair['A']['file']}")
        print(f"B: {pair['B']['file']}")
        print(f"\n파일을 들어본 후 선택하세요.")

        while True:
            vote = input(f"Vote (A/B/s/q): ").strip().upper()

            if vote == 'Q':
                print("테스트 종료.")
                break
            elif vote == 'S':
                print("건너뜀.")
                break
            elif vote in ['A', 'B']:
                votes[pair_id] = vote
                print(f"✅ {vote} 선택됨.")
                break
            else:
                print("❌ A, B, s, q 중 하나를 입력하세요.")

        if vote == 'Q':
            break

    # 투표 저장
    if votes:
        voting_file = output_path / 'interactive_votes.csv'

        with open(voting_file, 'w') as f:
            f.write("Pair,File_A,File_B,Vote,Notes\n")

            for pair in pairs:
                pair_id = pair['pair_id']
                vote = votes.get(pair_id, '')

                f.write(f"{pair_id},{pair['A']['file']},{pair['B']['file']},{vote},\n")

        print(f"\n💾 투표 저장: {voting_file}")

        # 결과 분석
        analyze_votes(str(voting_file), str(metadata_file))
    else:
        print("\n투표 데이터가 없습니다.")


def main():
    parser = argparse.ArgumentParser(description="A/B Test")

    subparsers = parser.add_subparsers(dest='command', help='명령어')

    # create 명령어
    create_parser = subparsers.add_parser('create', help='A/B 테스트 페어 생성')
    create_parser.add_argument('--base_dir', type=str, required=True, help='베이스 모델 오디오 폴더')
    create_parser.add_argument('--finetuned_dir', type=str, required=True, help='파인튜닝 오디오 폴더')
    create_parser.add_argument('--output_dir', type=str, default='./ab_test', help='출력 폴더')
    create_parser.add_argument('--num_pairs', type=int, default=10, help='페어 개수')

    # analyze 명령어
    analyze_parser = subparsers.add_parser('analyze', help='투표 결과 분석')
    analyze_parser.add_argument('--voting_sheet', type=str, required=True, help='투표 시트 CSV')
    analyze_parser.add_argument('--metadata', type=str, required=True, help='메타데이터 JSON')

    # interactive 명령어
    interactive_parser = subparsers.add_parser('interactive', help='인터랙티브 테스트')
    interactive_parser.add_argument('--output_dir', type=str, required=True, help='A/B 테스트 폴더')

    args = parser.parse_args()

    if args.command == 'create':
        pairs = create_ab_pairs(args.base_dir, args.finetuned_dir, args.output_dir, args.num_pairs)
        if pairs:
            create_voting_sheet(pairs, args.output_dir)

    elif args.command == 'analyze':
        analyze_votes(args.voting_sheet, args.metadata)

    elif args.command == 'interactive':
        run_interactive_test(args.output_dir)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
