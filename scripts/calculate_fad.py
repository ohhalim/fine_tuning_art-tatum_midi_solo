"""
FAD (Frechet Audio Distance) 계산 스크립트

생성된 오디오와 실제 재즈 오디오의 유사도를 측정해.
낮을수록 더 유사함 (0에 가까울수록 좋음).

사용법:
    python scripts/calculate_fad.py \
        --generated_dir ./generated_audio \
        --reference_dir ./reference_jazz

필요한 패키지:
    pip install frechet_audio_distance librosa numpy
"""

import argparse
import numpy as np
from pathlib import Path
import librosa
import torch
from typing import List, Tuple


def load_audio_files(directory: str, sr: int = 16000) -> List[np.ndarray]:
    """
    폴더에서 모든 오디오 파일 로드

    Args:
        directory: 오디오 파일이 있는 폴더
        sr: 샘플링 레이트

    Returns:
        오디오 배열 리스트
    """
    audio_files = []
    directory = Path(directory)

    for ext in ['*.wav', '*.mp3', '*.flac']:
        for audio_path in directory.glob(ext):
            try:
                audio, _ = librosa.load(audio_path, sr=sr, mono=True)
                audio_files.append(audio)
                print(f"✅ 로드: {audio_path.name}")
            except Exception as e:
                print(f"❌ 로드 실패: {audio_path.name} - {e}")

    return audio_files


def extract_vggish_features(audio_list: List[np.ndarray]) -> np.ndarray:
    """
    VGGish 모델로 오디오 특징 추출

    Args:
        audio_list: 오디오 배열 리스트

    Returns:
        특징 벡터 배열 (N, 128)
    """
    try:
        # VGGish 모델 로드 (사전학습된 오디오 임베딩 모델)
        import tensorflow as tf
        import tensorflow_hub as hub

        model = hub.load('https://tfhub.dev/google/vggish/1')

        features = []
        for i, audio in enumerate(audio_list):
            # VGGish는 16kHz 모노 오디오 필요
            # 0.96초 청크로 분할
            chunk_length = int(0.96 * 16000)

            audio_chunks = [
                audio[i:i+chunk_length]
                for i in range(0, len(audio) - chunk_length, chunk_length)
            ]

            chunk_features = []
            for chunk in audio_chunks:
                if len(chunk) == chunk_length:
                    # VGGish 입력 형식으로 변환
                    chunk_tensor = tf.constant(chunk, dtype=tf.float32)
                    embedding = model(chunk_tensor)
                    chunk_features.append(embedding.numpy())

            if chunk_features:
                # 평균 특징 벡터
                mean_feature = np.mean(chunk_features, axis=0)
                features.append(mean_feature)

            print(f"   특징 추출 중... {i+1}/{len(audio_list)}")

        return np.array(features)

    except ImportError:
        print("⚠️  TensorFlow Hub 없음. librosa 특징으로 대체합니다.")
        return extract_librosa_features(audio_list)


def extract_librosa_features(audio_list: List[np.ndarray], sr: int = 16000) -> np.ndarray:
    """
    librosa로 간단한 오디오 특징 추출 (VGGish 대안)

    Args:
        audio_list: 오디오 배열 리스트
        sr: 샘플링 레이트

    Returns:
        특징 벡터 배열 (N, feature_dim)
    """
    features = []

    for i, audio in enumerate(audio_list):
        # 여러 특징 추출
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        zero_crossing = librosa.feature.zero_crossing_rate(audio)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)

        # 평균값으로 요약
        feature_vector = np.concatenate([
            np.mean(mfcc, axis=1),           # 13
            np.mean(spectral_centroid),      # 1
            np.mean(spectral_rolloff),       # 1
            np.mean(zero_crossing),          # 1
            np.mean(chroma, axis=1),         # 12
        ])

        features.append(feature_vector)
        print(f"   특징 추출 중... {i+1}/{len(audio_list)}")

    return np.array(features)


def calculate_frechet_distance(mu1: np.ndarray, sigma1: np.ndarray,
                               mu2: np.ndarray, sigma2: np.ndarray) -> float:
    """
    Frechet Distance 계산

    FD = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))

    Args:
        mu1, mu2: 평균 벡터
        sigma1, sigma2: 공분산 행렬

    Returns:
        Frechet distance (낮을수록 유사)
    """
    # 평균 차이
    diff = mu1 - mu2
    mean_dist = np.sum(diff ** 2)

    # 공분산 행렬의 제곱근
    from scipy import linalg
    covmean = linalg.sqrtm(sigma1.dot(sigma2))

    # 수치 오류로 인한 복소수 제거
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Frechet distance
    fd = mean_dist + np.trace(sigma1 + sigma2 - 2 * covmean)

    return fd


def compute_fad(generated_features: np.ndarray,
                reference_features: np.ndarray) -> float:
    """
    FAD (Frechet Audio Distance) 계산

    Args:
        generated_features: 생성된 오디오 특징 (N1, D)
        reference_features: 레퍼런스 오디오 특징 (N2, D)

    Returns:
        FAD 점수 (낮을수록 좋음)
    """
    # 통계량 계산
    mu_gen = np.mean(generated_features, axis=0)
    sigma_gen = np.cov(generated_features, rowvar=False)

    mu_ref = np.mean(reference_features, axis=0)
    sigma_ref = np.cov(reference_features, rowvar=False)

    # FAD 계산
    fad_score = calculate_frechet_distance(mu_gen, sigma_gen, mu_ref, sigma_ref)

    return fad_score


def evaluate_fad(generated_dir: str, reference_dir: str, output_dir: str = './evaluation'):
    """전체 FAD 평가 실행"""
    print("=" * 60)
    print("🎵 FAD (Frechet Audio Distance) 계산 시작")
    print("=" * 60)

    # 1. 오디오 로드
    print("\n📂 생성된 오디오 로드...")
    generated_audio = load_audio_files(generated_dir)

    print(f"\n📂 레퍼런스 재즈 오디오 로드...")
    reference_audio = load_audio_files(reference_dir)

    if len(generated_audio) == 0 or len(reference_audio) == 0:
        print("❌ 오디오 파일이 없습니다!")
        return

    print(f"\n✅ 로드 완료:")
    print(f"   생성 오디오: {len(generated_audio)}개")
    print(f"   레퍼런스: {len(reference_audio)}개")

    # 2. 특징 추출
    print(f"\n🔍 생성 오디오 특징 추출...")
    generated_features = extract_librosa_features(generated_audio)

    print(f"\n🔍 레퍼런스 오디오 특징 추출...")
    reference_features = extract_librosa_features(reference_audio)

    # 3. FAD 계산
    print(f"\n📊 FAD 계산 중...")
    fad_score = compute_fad(generated_features, reference_features)

    # 4. 결과 출력
    print(f"\n" + "=" * 60)
    print(f"🎯 FAD 점수: {fad_score:.2f}")

    if fad_score < 5.0:
        print(f"   ✅ 매우 유사 (FAD < 5.0)")
        print(f"   → 실제 재즈와 거의 구분 불가")
    elif fad_score < 10.0:
        print(f"   ✅ 유사 (FAD < 10.0)")
        print(f"   → 재즈 스타일 잘 학습됨")
    elif fad_score < 20.0:
        print(f"   🟡 보통 (FAD < 20.0)")
        print(f"   → 어느 정도 재즈 느낌은 있음")
    else:
        print(f"   ❌ 차이 큼 (FAD >= 20.0)")
        print(f"   → 재즈 스타일 학습 부족")

    print("=" * 60)

    # 5. 결과 저장
    Path(output_dir).mkdir(exist_ok=True)
    result_file = f"{output_dir}/fad_score.txt"

    with open(result_file, 'w') as f:
        f.write(f"FAD Score: {fad_score:.2f}\n")
        f.write(f"Generated samples: {len(generated_audio)}\n")
        f.write(f"Reference samples: {len(reference_audio)}\n")

    print(f"\n💾 결과 저장: {result_file}")


def main():
    parser = argparse.ArgumentParser(description="FAD 계산")
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

    evaluate_fad(args.generated_dir, args.reference_dir, args.output_dir)


if __name__ == '__main__':
    main()
