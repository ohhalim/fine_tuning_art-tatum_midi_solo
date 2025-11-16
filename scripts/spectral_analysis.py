"""
Spectral Analysis (주파수 분석) 스크립트

생성된 재즈의 주파수 특성을 실제 재즈와 비교해.

측정 항목:
- Spectral Centroid: 소리의 밝기 (중심 주파수)
- Spectral Rolloff: 고주파 에너지 분포
- Zero Crossing Rate: 리듬 복잡도
- Spectral Bandwidth: 주파수 대역폭

사용법:
    python scripts/spectral_analysis.py \
        --generated_dir ./generated_audio \
        --reference_dir ./reference_jazz
"""

import argparse
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List


def analyze_audio_file(audio_path: str, sr: int = 22050) -> Dict[str, float]:
    """
    단일 오디오 파일의 스펙트럼 특징 분석

    Args:
        audio_path: 오디오 파일 경로
        sr: 샘플링 레이트

    Returns:
        특징 딕셔너리
    """
    # 오디오 로드
    y, _ = librosa.load(audio_path, sr=sr)

    # 특징 추출
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    # RMS Energy (음량)
    rms = librosa.feature.rms(y=y)[0]

    return {
        'spectral_centroid': np.mean(centroid),
        'spectral_rolloff': np.mean(rolloff),
        'zero_crossing_rate': np.mean(zcr),
        'spectral_bandwidth': np.mean(bandwidth),
        'spectral_contrast': np.mean(contrast),
        'rms_energy': np.mean(rms),
        'centroid_std': np.std(centroid),  # 변화량
        'rolloff_std': np.std(rolloff),
    }


def analyze_directory(directory: str, label: str = "Audio") -> Dict[str, List[float]]:
    """
    폴더 내 모든 오디오 파일 분석

    Args:
        directory: 오디오 폴더
        label: 라벨 (출력용)

    Returns:
        전체 특징 딕셔너리
    """
    print(f"\n📂 {label} 분석 중...")

    directory = Path(directory)
    all_features = {
        'spectral_centroid': [],
        'spectral_rolloff': [],
        'zero_crossing_rate': [],
        'spectral_bandwidth': [],
        'spectral_contrast': [],
        'rms_energy': [],
        'centroid_std': [],
        'rolloff_std': [],
    }

    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.flac']:
        audio_files.extend(directory.glob(ext))

    if not audio_files:
        print(f"❌ 오디오 파일을 찾을 수 없습니다: {directory}")
        return all_features

    for i, audio_path in enumerate(audio_files):
        try:
            features = analyze_audio_file(str(audio_path))

            for key, value in features.items():
                all_features[key].append(value)

            print(f"   ✅ {i+1}/{len(audio_files)}: {audio_path.name}")

        except Exception as e:
            print(f"   ❌ 실패: {audio_path.name} - {e}")

    return all_features


def compare_features(generated: Dict, reference: Dict) -> None:
    """
    특징 비교 및 출력

    Args:
        generated: 생성 오디오 특징
        reference: 레퍼런스 오디오 특징
    """
    print(f"\n" + "=" * 80)
    print(f"📊 스펙트럼 특징 비교")
    print("=" * 80)

    feature_names = {
        'spectral_centroid': 'Spectral Centroid (Hz)',
        'spectral_rolloff': 'Spectral Rolloff (Hz)',
        'zero_crossing_rate': 'Zero Crossing Rate',
        'spectral_bandwidth': 'Spectral Bandwidth (Hz)',
        'rms_energy': 'RMS Energy',
        'centroid_std': 'Centroid Variability',
    }

    print(f"\n{'특징':<30} {'생성':<15} {'레퍼런스':<15} {'유사도':<10}")
    print("-" * 80)

    for key, name in feature_names.items():
        if key not in generated or not generated[key]:
            continue

        gen_mean = np.mean(generated[key])
        ref_mean = np.mean(reference[key]) if reference[key] else 0

        # 유사도 계산 (상대 오차)
        if ref_mean > 0:
            similarity = (1 - abs(gen_mean - ref_mean) / ref_mean) * 100
            similarity = max(0, min(100, similarity))  # 0-100% 범위
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


def plot_spectral_comparison(generated: Dict, reference: Dict, output_path: str):
    """
    스펙트럼 특징 비교 그래프 생성

    Args:
        generated: 생성 오디오 특징
        reference: 레퍼런스 오디오 특징
        output_path: 저장 경로
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    features_to_plot = [
        ('spectral_centroid', 'Spectral Centroid (Hz)', 'Brightness'),
        ('spectral_rolloff', 'Spectral Rolloff (Hz)', 'High Freq Energy'),
        ('zero_crossing_rate', 'Zero Crossing Rate', 'Rhythm Complexity'),
        ('spectral_bandwidth', 'Spectral Bandwidth (Hz)', 'Frequency Range'),
        ('rms_energy', 'RMS Energy', 'Loudness'),
        ('centroid_std', 'Centroid Variability', 'Brightness Change'),
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

            # 색상
            bp['boxes'][0].set_facecolor('skyblue')
            bp['boxes'][1].set_facecolor('lightcoral')

            ax.set_title(f"{title}\n({subtitle})", fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # 평균값 표시
            gen_mean = np.mean(gen_data)
            ref_mean = np.mean(ref_data)
            ax.axhline(gen_mean, color='blue', linestyle='--', alpha=0.5, linewidth=1)
            ax.axhline(ref_mean, color='red', linestyle='--', alpha=0.5, linewidth=1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\n📊 그래프 저장됨: {output_path}")


def plot_spectrogram_comparison(generated_dir: str, reference_dir: str, output_path: str):
    """
    대표 샘플의 스펙트로그램 비교

    Args:
        generated_dir: 생성 오디오 폴더
        reference_dir: 레퍼런스 오디오 폴더
        output_path: 저장 경로
    """
    # 첫 번째 파일 선택
    gen_file = None
    ref_file = None

    for ext in ['*.wav', '*.mp3']:
        if not gen_file:
            gen_files = list(Path(generated_dir).glob(ext))
            if gen_files:
                gen_file = gen_files[0]

        if not ref_file:
            ref_files = list(Path(reference_dir).glob(ext))
            if ref_files:
                ref_file = ref_files[0]

    if not gen_file or not ref_file:
        print("⚠️  스펙트로그램 비교를 위한 파일을 찾을 수 없습니다.")
        return

    # 로드
    y_gen, sr = librosa.load(gen_file, sr=22050, duration=10)
    y_ref, _ = librosa.load(ref_file, sr=22050, duration=10)

    # 스펙트로그램 생성
    D_gen = librosa.amplitude_to_db(np.abs(librosa.stft(y_gen)), ref=np.max)
    D_ref = librosa.amplitude_to_db(np.abs(librosa.stft(y_ref)), ref=np.max)

    # 플롯
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    librosa.display.specshow(D_gen, sr=sr, x_axis='time', y_axis='hz', ax=axes[0], cmap='viridis')
    axes[0].set_title(f'Generated Audio: {gen_file.name}', fontsize=12, fontweight='bold')
    axes[0].set_ylim([0, 8000])

    librosa.display.specshow(D_ref, sr=sr, x_axis='time', y_axis='hz', ax=axes[1], cmap='viridis')
    axes[1].set_title(f'Reference Jazz: {ref_file.name}', fontsize=12, fontweight='bold')
    axes[1].set_ylim([0, 8000])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"📊 스펙트로그램 저장됨: {output_path}")


def evaluate_spectral(generated_dir: str, reference_dir: str, output_dir: str = './evaluation'):
    """전체 스펙트럼 분석 실행"""
    print("=" * 80)
    print("🎵 Spectral Analysis (주파수 분석) 시작")
    print("=" * 80)

    # 1. 분석
    generated_features = analyze_directory(generated_dir, "생성 오디오")
    reference_features = analyze_directory(reference_dir, "레퍼런스 재즈")

    if not generated_features['spectral_centroid'] or not reference_features['spectral_centroid']:
        print("❌ 분석할 파일이 없습니다!")
        return

    # 2. 비교
    compare_features(generated_features, reference_features)

    # 3. 그래프 생성
    Path(output_dir).mkdir(exist_ok=True)

    plot_spectral_comparison(
        generated_features,
        reference_features,
        f"{output_dir}/spectral_comparison.png"
    )

    plot_spectrogram_comparison(
        generated_dir,
        reference_dir,
        f"{output_dir}/spectrogram_comparison.png"
    )

    # 4. 최종 판정
    print(f"\n" + "=" * 80)
    print("🏆 최종 판정:")

    # 평균 유사도 계산
    similarities = []
    for key in ['spectral_centroid', 'spectral_rolloff', 'zero_crossing_rate']:
        if generated_features[key] and reference_features[key]:
            gen_mean = np.mean(generated_features[key])
            ref_mean = np.mean(reference_features[key])
            if ref_mean > 0:
                sim = (1 - abs(gen_mean - ref_mean) / ref_mean) * 100
                similarities.append(max(0, min(100, sim)))

    if similarities:
        avg_similarity = np.mean(similarities)
        print(f"   평균 유사도: {avg_similarity:.1f}%")

        if avg_similarity >= 85:
            print(f"   ✅ 매우 유사! 실제 재즈와 거의 동일한 주파수 특성")
        elif avg_similarity >= 70:
            print(f"   ✅ 유사함. 재즈 스타일 잘 학습됨")
        elif avg_similarity >= 50:
            print(f"   🟡 보통. 개선 여지 있음")
        else:
            print(f"   ❌ 차이 큼. 재학습 필요")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="스펙트럼 분석")
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

    evaluate_spectral(args.generated_dir, args.reference_dir, args.output_dir)


if __name__ == '__main__':
    main()
