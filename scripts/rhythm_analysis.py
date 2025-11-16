"""
Rhythm Analysis (리듬 분석) 스크립트

재즈의 핵심인 리듬 복잡도와 싱코페이션을 측정해.

측정 항목:
- Tempo (BPM)
- Beat Strength (비트 강도)
- Syncopation Score (싱코페이션 점수)
- Onset Density (음표 밀도)

사용법:
    python scripts/rhythm_analysis.py \
        --generated_dir ./generated_audio \
        --reference_dir ./reference_jazz
"""

import argparse
import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple


def analyze_rhythm(audio_path: str, sr: int = 22050) -> Dict[str, float]:
    """
    단일 오디오 파일의 리듬 특징 분석

    Args:
        audio_path: 오디오 파일 경로
        sr: 샘플링 레이트

    Returns:
        리듬 특징 딕셔너리
    """
    # 오디오 로드
    y, _ = librosa.load(audio_path, sr=sr)

    # 1. Tempo 추정
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    # 2. Onset Detection (음표 시작점)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        backtrack=True
    )

    # 3. Onset Density (음표 밀도: 초당 음표 수)
    duration = librosa.get_duration(y=y, sr=sr)
    onset_density = len(onset_frames) / duration if duration > 0 else 0

    # 4. Beat Strength (비트 강도)
    beat_strength = np.mean(onset_env)

    # 5. Syncopation Score (싱코페이션 점수)
    # 박자와 onset이 얼마나 어긋나는지 측정
    syncopation = calculate_syncopation(onset_frames, beat_frames, sr)

    # 6. Rhythmic Regularity (리듬 규칙성)
    # onset 간격의 표준편차 (낮을수록 규칙적)
    if len(onset_frames) > 1:
        onset_intervals = np.diff(onset_frames)
        rhythm_regularity = np.std(onset_intervals) / np.mean(onset_intervals) if np.mean(onset_intervals) > 0 else 0
    else:
        rhythm_regularity = 0

    return {
        'tempo': tempo,
        'onset_density': onset_density,
        'beat_strength': beat_strength,
        'syncopation': syncopation,
        'rhythm_regularity': rhythm_regularity,
        'total_onsets': len(onset_frames),
        'total_beats': len(beat_frames),
    }


def calculate_syncopation(onset_frames: np.ndarray, beat_frames: np.ndarray, sr: int) -> float:
    """
    Syncopation Score 계산

    재즈의 핵심: 박자(beat)와 음표 시작(onset)이 얼마나 어긋나는지

    Args:
        onset_frames: 음표 시작 프레임
        beat_frames: 비트 프레임
        sr: 샘플링 레이트

    Returns:
        Syncopation score (0~1, 높을수록 싱코페이션 많음)
    """
    if len(onset_frames) == 0 or len(beat_frames) == 0:
        return 0.0

    # 각 onset이 가장 가까운 beat로부터 얼마나 떨어져 있는지 측정
    offsets = []

    for onset in onset_frames:
        # 가장 가까운 beat 찾기
        distances = np.abs(beat_frames - onset)
        min_distance = np.min(distances)

        # 가장 가까운 beat 간격으로 정규화
        closest_beat_idx = np.argmin(distances)

        if closest_beat_idx < len(beat_frames) - 1:
            beat_interval = beat_frames[closest_beat_idx + 1] - beat_frames[closest_beat_idx]
        elif closest_beat_idx > 0:
            beat_interval = beat_frames[closest_beat_idx] - beat_frames[closest_beat_idx - 1]
        else:
            beat_interval = 1

        # 정규화된 offset (0 = beat와 정확히 일치, 0.5 = beat 중간)
        normalized_offset = min_distance / beat_interval if beat_interval > 0 else 0

        offsets.append(normalized_offset)

    # Syncopation score: offset의 평균
    # 0에 가까우면 = beat와 정확히 맞음 (클래식적)
    # 0.3-0.5에 가까우면 = beat 사이에 많이 침 (재즈적)
    syncopation_score = np.mean(offsets)

    return syncopation_score


def analyze_directory_rhythm(directory: str, label: str = "Audio") -> Dict[str, List[float]]:
    """
    폴더 내 모든 오디오 파일의 리듬 분석

    Args:
        directory: 오디오 폴더
        label: 라벨

    Returns:
        전체 리듬 특징 딕셔너리
    """
    print(f"\n📂 {label} 리듬 분석 중...")

    directory = Path(directory)
    all_features = {
        'tempo': [],
        'onset_density': [],
        'beat_strength': [],
        'syncopation': [],
        'rhythm_regularity': [],
    }

    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.flac']:
        audio_files.extend(directory.glob(ext))

    if not audio_files:
        print(f"❌ 오디오 파일을 찾을 수 없습니다: {directory}")
        return all_features

    for i, audio_path in enumerate(audio_files):
        try:
            features = analyze_rhythm(str(audio_path))

            for key in all_features.keys():
                all_features[key].append(features[key])

            print(f"   ✅ {i+1}/{len(audio_files)}: {audio_path.name}")

        except Exception as e:
            print(f"   ❌ 실패: {audio_path.name} - {e}")

    return all_features


def compare_rhythm_features(generated: Dict, reference: Dict) -> None:
    """
    리듬 특징 비교

    Args:
        generated: 생성 오디오 리듬 특징
        reference: 레퍼런스 오디오 리듬 특징
    """
    print(f"\n" + "=" * 80)
    print(f"🥁 리듬 특징 비교")
    print("=" * 80)

    feature_names = {
        'tempo': 'Tempo (BPM)',
        'onset_density': 'Onset Density (notes/sec)',
        'beat_strength': 'Beat Strength',
        'syncopation': 'Syncopation Score',
        'rhythm_regularity': 'Rhythm Regularity',
    }

    print(f"\n{'특징':<30} {'생성':<15} {'레퍼런스':<15} {'유사도':<10}")
    print("-" * 80)

    for key, name in feature_names.items():
        if key not in generated or not generated[key]:
            continue

        gen_mean = np.mean(generated[key])
        ref_mean = np.mean(reference[key]) if reference[key] else 0

        # 유사도 계산
        if ref_mean > 0:
            similarity = (1 - abs(gen_mean - ref_mean) / ref_mean) * 100
            similarity = max(0, min(100, similarity))
        else:
            similarity = 0

        # 상태 아이콘
        if similarity >= 90:
            status = "✅"
        elif similarity >= 70:
            status = "🟡"
        else:
            status = "❌"

        print(f"{name:<30} {gen_mean:<15.2f} {ref_mean:<15.2f} {similarity:>5.1f}% {status}")

    print("-" * 80)

    # 특별 분석: Syncopation
    if generated['syncopation'] and reference['syncopation']:
        gen_sync = np.mean(generated['syncopation'])
        ref_sync = np.mean(reference['syncopation'])

        print(f"\n🎵 Syncopation (싱코페이션) 분석:")
        print(f"   생성: {gen_sync:.3f}")
        print(f"   레퍼런스: {ref_sync:.3f}")

        if gen_sync >= 0.3 and gen_sync <= 0.5:
            print(f"   ✅ 재즈다운 싱코페이션! (0.3~0.5 범위)")
        elif gen_sync < 0.3:
            print(f"   ⚠️  싱코페이션 부족 (너무 규칙적)")
        else:
            print(f"   ⚠️  싱코페이션 과다 (너무 불규칙)")


def plot_rhythm_comparison(generated: Dict, reference: Dict, output_path: str):
    """
    리듬 특징 비교 그래프

    Args:
        generated: 생성 오디오 리듬 특징
        reference: 레퍼런스 오디오 리듬 특징
        output_path: 저장 경로
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    features_to_plot = [
        ('tempo', 'Tempo (BPM)', 'Speed'),
        ('onset_density', 'Onset Density', 'Notes per Second'),
        ('beat_strength', 'Beat Strength', 'Accent Intensity'),
        ('syncopation', 'Syncopation Score', 'Off-beat Playing'),
        ('rhythm_regularity', 'Rhythm Regularity', 'Timing Consistency'),
    ]

    for i, (key, title, subtitle) in enumerate(features_to_plot):
        ax = axes[i]

        gen_data = generated.get(key, [])
        ref_data = reference.get(key, [])

        if gen_data and ref_data:
            # 박스플롯
            bp = ax.boxplot(
                [gen_data, ref_data],
                labels=['Generated', 'Reference'],
                patch_artist=True,
                widths=0.6
            )

            bp['boxes'][0].set_facecolor('lightgreen')
            bp['boxes'][1].set_facecolor('lightyellow')

            ax.set_title(f"{title}\n({subtitle})", fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # 평균값 표시
            gen_mean = np.mean(gen_data)
            ref_mean = np.mean(ref_data)
            ax.axhline(gen_mean, color='green', linestyle='--', alpha=0.5, linewidth=1)
            ax.axhline(ref_mean, color='orange', linestyle='--', alpha=0.5, linewidth=1)

    # 마지막 subplot 제거
    fig.delaxes(axes[-1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\n📊 리듬 분석 그래프 저장됨: {output_path}")


def evaluate_rhythm(generated_dir: str, reference_dir: str, output_dir: str = './evaluation'):
    """전체 리듬 분석 실행"""
    print("=" * 80)
    print("🥁 Rhythm Analysis (리듬 분석) 시작")
    print("=" * 80)

    # 1. 분석
    generated_features = analyze_directory_rhythm(generated_dir, "생성 오디오")
    reference_features = analyze_directory_rhythm(reference_dir, "레퍼런스 재즈")

    if not generated_features['tempo'] or not reference_features['tempo']:
        print("❌ 분석할 파일이 없습니다!")
        return

    # 2. 비교
    compare_rhythm_features(generated_features, reference_features)

    # 3. 그래프 생성
    Path(output_dir).mkdir(exist_ok=True)
    plot_rhythm_comparison(
        generated_features,
        reference_features,
        f"{output_dir}/rhythm_comparison.png"
    )

    # 4. 최종 판정
    print(f"\n" + "=" * 80)
    print("🏆 최종 판정:")

    # Syncopation 기준 평가
    if generated_features['syncopation']:
        avg_sync = np.mean(generated_features['syncopation'])
        ref_sync = np.mean(reference_features['syncopation']) if reference_features['syncopation'] else 0

        print(f"   Syncopation Score: {avg_sync:.3f} (레퍼런스: {ref_sync:.3f})")

        if 0.3 <= avg_sync <= 0.5:
            print(f"   ✅ 재즈다운 리듬감! 싱코페이션 완벽")
        elif 0.2 <= avg_sync < 0.3:
            print(f"   🟡 약간 규칙적. 더 스윙감 필요")
        elif avg_sync < 0.2:
            print(f"   ❌ 너무 규칙적. 재즈 느낌 부족")
        else:
            print(f"   ⚠️  너무 불규칙. 일관성 필요")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="리듬 분석")
    parser.add_argument(
        '--generated_dir',
        type=str,
        required=True,
        help='생성된 오디오 폴더'
    )
    parser.add_argument(
        '--reference_dir',
        type=str,
        required=True,
        help='레퍼런스 재즈 오디오 폴더'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./evaluation',
        help='결과 저장 폴더'
    )

    args = parser.parse_args()

    evaluate_rhythm(args.generated_dir, args.reference_dir, args.output_dir)


if __name__ == '__main__':
    main()
