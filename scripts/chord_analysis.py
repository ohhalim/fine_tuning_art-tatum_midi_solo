"""
Chord Progression Analysis (코드 진행 분석) 스크립트

재즈 이론에 맞는 코드 진행을 사용하는지 확인해.

분석 항목:
- Chord Detection (코드 감지)
- Jazz Chord Patterns (재즈 코드 패턴: ii-V-I, 블루스 등)
- Chord Complexity (코드 복잡도: 7th, 9th 등)
- Harmonic Coherence (화성 응집력)

사용법:
    python scripts/chord_analysis.py \
        --generated_dir ./generated_audio \
        --reference_dir ./reference_jazz
"""

import argparse
import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple


def detect_chords(audio_path: str, sr: int = 22050) -> Tuple[List, List]:
    """
    오디오에서 코드 감지

    Args:
        audio_path: 오디오 파일 경로
        sr: 샘플링 레이트

    Returns:
        (chroma, chord_changes)
    """
    # 오디오 로드
    y, _ = librosa.load(audio_path, sr=sr)

    # Chromagram 추출 (12개 음계별 에너지)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

    # 코드 변화 감지 (chromagram의 급격한 변화)
    chroma_diff = np.sum(np.abs(np.diff(chroma, axis=1)), axis=0)
    threshold = np.mean(chroma_diff) + 1.5 * np.std(chroma_diff)

    chord_changes = np.where(chroma_diff > threshold)[0]

    return chroma, chord_changes


def analyze_harmonic_complexity(chroma: np.ndarray) -> Dict[str, float]:
    """
    화성 복잡도 분석

    Args:
        chroma: Chromagram (12, T)

    Returns:
        복잡도 지표 딕셔너리
    """
    # 1. Active Notes (동시에 울리는 음 개수)
    # 각 시간 프레임에서 임계값 이상의 에너지를 가진 음 개수
    threshold = 0.3
    active_notes = np.sum(chroma > threshold, axis=0)
    avg_active_notes = np.mean(active_notes)

    # 2. Harmonic Entropy (화성 엔트로피)
    # 높을수록 복잡한 코드 (재즈 7th, 9th 등)
    chroma_norm = chroma / (np.sum(chroma, axis=0, keepdims=True) + 1e-8)
    entropy = -np.sum(chroma_norm * np.log(chroma_norm + 1e-8), axis=0)
    avg_entropy = np.mean(entropy)

    # 3. Tonal Complexity (조성 복잡도)
    # chromagram의 표준편차 (높을수록 다양한 음 사용)
    tonal_complexity = np.mean(np.std(chroma, axis=1))

    return {
        'avg_active_notes': avg_active_notes,
        'harmonic_entropy': avg_entropy,
        'tonal_complexity': tonal_complexity,
    }


def detect_jazz_patterns(chroma: np.ndarray, chord_changes: np.ndarray) -> Dict[str, int]:
    """
    재즈 특유의 코드 패턴 감지

    주요 패턴:
    - ii-V-I progression
    - Blues progression
    - Modal interchange

    Args:
        chroma: Chromagram
        chord_changes: 코드 변화 지점

    Returns:
        감지된 패턴 개수
    """
    patterns_found = {
        'ii_V_I': 0,
        'blues_pattern': 0,
        'dominant_7th': 0,
        'modal_shifts': 0,
    }

    # 간단한 휴리스틱 감지
    # (실제로는 더 정교한 코드 인식 모델 필요)

    # Dominant 7th 감지 (특정 음정 간격)
    for i in range(chroma.shape[1]):
        frame = chroma[:, i]

        # Major 3rd + minor 7th 패턴 찾기
        # (0=C, 4=E, 10=Bb 같은 패턴)
        if frame[0] > 0.3 and frame[4] > 0.3 and frame[10] > 0.3:
            patterns_found['dominant_7th'] += 1

    # Modal shifts 감지 (chromagram의 급격한 변화)
    if len(chord_changes) > 0:
        patterns_found['modal_shifts'] = len(chord_changes)

    return patterns_found


def analyze_chord_progression(audio_path: str) -> Dict[str, float]:
    """
    전체 코드 진행 분석

    Args:
        audio_path: 오디오 파일 경로

    Returns:
        분석 결과 딕셔너리
    """
    # 코드 감지
    chroma, chord_changes = detect_chords(audio_path)

    # 화성 복잡도
    complexity = analyze_harmonic_complexity(chroma)

    # 재즈 패턴
    patterns = detect_jazz_patterns(chroma, chord_changes)

    # 결과 합치기
    result = {
        **complexity,
        'num_chord_changes': len(chord_changes),
        'chord_change_rate': len(chord_changes) / (chroma.shape[1] / 22050) if chroma.shape[1] > 0 else 0,
        'dominant_7th_count': patterns['dominant_7th'],
        'modal_shifts': patterns['modal_shifts'],
    }

    return result


def analyze_directory_chords(directory: str, label: str = "Audio") -> Dict[str, List[float]]:
    """
    폴더 내 모든 오디오의 코드 진행 분석

    Args:
        directory: 오디오 폴더
        label: 라벨

    Returns:
        전체 분석 결과
    """
    print(f"\n📂 {label} 코드 진행 분석 중...")

    directory = Path(directory)
    all_features = {
        'avg_active_notes': [],
        'harmonic_entropy': [],
        'tonal_complexity': [],
        'num_chord_changes': [],
        'chord_change_rate': [],
        'dominant_7th_count': [],
    }

    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.flac']:
        audio_files.extend(directory.glob(ext))

    if not audio_files:
        print(f"❌ 오디오 파일을 찾을 수 없습니다: {directory}")
        return all_features

    for i, audio_path in enumerate(audio_files):
        try:
            features = analyze_chord_progression(str(audio_path))

            for key in all_features.keys():
                if key in features:
                    all_features[key].append(features[key])

            print(f"   ✅ {i+1}/{len(audio_files)}: {audio_path.name}")

        except Exception as e:
            print(f"   ❌ 실패: {audio_path.name} - {e}")

    return all_features


def compare_chord_features(generated: Dict, reference: Dict) -> None:
    """
    코드 진행 특징 비교

    Args:
        generated: 생성 오디오 코드 특징
        reference: 레퍼런스 오디오 코드 특징
    """
    print(f"\n" + "=" * 80)
    print(f"🎹 코드 진행 분석")
    print("=" * 80)

    feature_names = {
        'avg_active_notes': 'Avg Active Notes (동시 발음)',
        'harmonic_entropy': 'Harmonic Entropy (화성 복잡도)',
        'tonal_complexity': 'Tonal Complexity (조성 다양성)',
        'chord_change_rate': 'Chord Change Rate (변화 속도)',
        'dominant_7th_count': 'Dominant 7th Count (재즈 코드)',
    }

    print(f"\n{'특징':<35} {'생성':<15} {'레퍼런스':<15} {'유사도':<10}")
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

        print(f"{name:<35} {gen_mean:<15.2f} {ref_mean:<15.2f} {similarity:>5.1f}% {status}")

    print("-" * 80)

    # 특별 분석
    if generated['harmonic_entropy']:
        gen_entropy = np.mean(generated['harmonic_entropy'])

        print(f"\n🎵 화성 복잡도 평가:")

        if gen_entropy > 2.0:
            print(f"   ✅ 복잡한 화성 (재즈 7th, 9th 많음)")
        elif gen_entropy > 1.5:
            print(f"   🟡 중간 복잡도 (기본 재즈 코드)")
        else:
            print(f"   ⚠️  단순한 화성 (단순 3화음)")


def plot_chord_comparison(generated: Dict, reference: Dict, output_path: str):
    """
    코드 진행 비교 그래프

    Args:
        generated: 생성 오디오 코드 특징
        reference: 레퍼런스 오디오 코드 특징
        output_path: 저장 경로
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    features_to_plot = [
        ('avg_active_notes', 'Avg Active Notes', 'Chord Density'),
        ('harmonic_entropy', 'Harmonic Entropy', 'Complexity'),
        ('tonal_complexity', 'Tonal Complexity', 'Diversity'),
        ('chord_change_rate', 'Chord Change Rate', 'Changes/sec'),
        ('dominant_7th_count', 'Dominant 7th Count', 'Jazz Chords'),
    ]

    for i, (key, title, subtitle) in enumerate(features_to_plot):
        ax = axes[i]

        gen_data = generated.get(key, [])
        ref_data = reference.get(key, [])

        if gen_data and ref_data:
            bp = ax.boxplot(
                [gen_data, ref_data],
                labels=['Generated', 'Reference'],
                patch_artist=True,
                widths=0.6
            )

            bp['boxes'][0].set_facecolor('plum')
            bp['boxes'][1].set_facecolor('peachpuff')

            ax.set_title(f"{title}\n({subtitle})", fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # 평균값 표시
            gen_mean = np.mean(gen_data)
            ref_mean = np.mean(ref_data)
            ax.axhline(gen_mean, color='purple', linestyle='--', alpha=0.5, linewidth=1)
            ax.axhline(ref_mean, color='orange', linestyle='--', alpha=0.5, linewidth=1)

    # 마지막 subplot 제거
    fig.delaxes(axes[-1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\n📊 코드 분석 그래프 저장됨: {output_path}")


def evaluate_chords(generated_dir: str, reference_dir: str, output_dir: str = './evaluation'):
    """전체 코드 진행 분석 실행"""
    print("=" * 80)
    print("🎹 Chord Progression Analysis (코드 진행 분석) 시작")
    print("=" * 80)

    # 1. 분석
    generated_features = analyze_directory_chords(generated_dir, "생성 오디오")
    reference_features = analyze_directory_chords(reference_dir, "레퍼런스 재즈")

    if not generated_features['avg_active_notes'] or not reference_features['avg_active_notes']:
        print("❌ 분석할 파일이 없습니다!")
        return

    # 2. 비교
    compare_chord_features(generated_features, reference_features)

    # 3. 그래프 생성
    Path(output_dir).mkdir(exist_ok=True)
    plot_chord_comparison(
        generated_features,
        reference_features,
        f"{output_dir}/chord_comparison.png"
    )

    # 4. 최종 판정
    print(f"\n" + "=" * 80)
    print("🏆 최종 판정:")

    # 화성 복잡도 기준 평가
    if generated_features['harmonic_entropy']:
        avg_entropy = np.mean(generated_features['harmonic_entropy'])
        ref_entropy = np.mean(reference_features['harmonic_entropy']) if reference_features['harmonic_entropy'] else 0

        print(f"   화성 복잡도: {avg_entropy:.2f} (레퍼런스: {ref_entropy:.2f})")

        if avg_entropy >= 1.8:
            print(f"   ✅ 재즈다운 화성! 복잡한 코드 사용")
        elif avg_entropy >= 1.5:
            print(f"   🟡 기본적인 재즈 화성")
        else:
            print(f"   ❌ 화성이 너무 단순함")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="코드 진행 분석")
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

    evaluate_chords(args.generated_dir, args.reference_dir, args.output_dir)


if __name__ == '__main__':
    main()
