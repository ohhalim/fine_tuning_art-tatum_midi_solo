"""
Dummy Data Generator

사전학습 없이 즉시 테스트할 수 있도록 더미 데이터 생성
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def generate_dummy_audio(
    duration: float = 2.0,
    sample_rate: int = 16000,
    num_samples: int = 10
):
    """
    더미 오디오 생성 (간단한 사인파 + 노이즈)

    Args:
        duration: 오디오 길이 (초)
        sample_rate: 샘플레이트
        num_samples: 생성할 샘플 수

    Returns:
        audios: (num_samples, 1, T) 텐서
    """
    samples = int(duration * sample_rate)
    audios = []

    for i in range(num_samples):
        # 여러 주파수의 사인파 믹스 (재즈 느낌)
        t = np.linspace(0, duration, samples)

        # 베이스 주파수 (랜덤 재즈 음계)
        freqs = np.random.choice([110, 146.83, 196, 220, 246.94], 3)
        audio = np.zeros(samples)

        for freq in freqs:
            audio += np.sin(2 * np.pi * freq * t) * np.random.uniform(0.2, 0.5)

        # 노이즈 추가
        audio += np.random.randn(samples) * 0.1

        # 정규화
        audio = audio / (np.abs(audio).max() + 1e-8)

        audios.append(audio)

    audios = torch.FloatTensor(audios).unsqueeze(1)  # (N, 1, T)

    print(f"✅ 더미 오디오 생성 완료: {audios.shape}")
    return audios


def generate_dummy_text_prompts(num_samples: int = 10):
    """
    더미 텍스트 프롬프트 생성

    Args:
        num_samples: 생성할 프롬프트 수

    Returns:
        prompts: 텍스트 프롬프트 리스트
    """
    styles = ["modal", "bebop", "blues", "swing", "ballad"]
    tempos = ["slow", "medium", "fast", "uptempo"]
    moods = ["melancholic", "energetic", "calm", "intense"]

    prompts = []
    for i in range(num_samples):
        style = np.random.choice(styles)
        tempo = np.random.choice(tempos)
        mood = np.random.choice(moods)
        prompts.append(f"{mood} {style} jazz piano, {tempo} tempo")

    print(f"✅ 더미 프롬프트 생성 완료: {len(prompts)}개")
    return prompts


def visualize_audio(audio: torch.Tensor, title: str = "Audio Waveform"):
    """
    오디오 파형 시각화

    Args:
        audio: (1, T) 오디오 텐서
        title: 그래프 제목
    """
    if audio.dim() == 3:
        audio = audio[0, 0]  # (1, 1, T) → (T,)
    elif audio.dim() == 2:
        audio = audio[0]  # (1, T) → (T,)

    plt.figure(figsize=(12, 4))
    plt.plot(audio.numpy())
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def save_dummy_dataset(output_dir: str = "./data/dummy", num_samples: int = 20):
    """
    더미 데이터셋을 디스크에 저장

    Args:
        output_dir: 저장 경로
        num_samples: 샘플 수
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 오디오 생성
    audios = generate_dummy_audio(duration=2.0, num_samples=num_samples)

    # 프롬프트 생성
    prompts = generate_dummy_text_prompts(num_samples=num_samples)

    # 저장
    torch.save({
        'audios': audios,
        'prompts': prompts,
        'sample_rate': 16000
    }, output_dir / "dummy_dataset.pt")

    print(f"✅ 더미 데이터셋 저장: {output_dir / 'dummy_dataset.pt'}")
    print(f"   - {num_samples}개 오디오 샘플")
    print(f"   - 각 2초, 16kHz")

    return audios, prompts


if __name__ == "__main__":
    print("=" * 60)
    print("PersonalJazz V2 - 더미 데이터 생성 테스트")
    print("=" * 60)

    # 테스트 1: 오디오 생성
    print("\n[테스트 1] 더미 오디오 생성")
    audios = generate_dummy_audio(duration=2.0, num_samples=5)
    print(f"Shape: {audios.shape}")  # (5, 1, 32000)

    # 테스트 2: 프롬프트 생성
    print("\n[테스트 2] 더미 프롬프트 생성")
    prompts = generate_dummy_text_prompts(num_samples=5)
    for i, prompt in enumerate(prompts):
        print(f"  {i+1}. {prompt}")

    # 테스트 3: 시각화
    print("\n[테스트 3] 오디오 시각화")
    print("(matplotlib 창이 열립니다...)")
    visualize_audio(audios[0], title="Dummy Jazz Audio Sample")

    # 테스트 4: 데이터셋 저장
    print("\n[테스트 4] 데이터셋 저장")
    save_dummy_dataset(num_samples=20)

    print("\n" + "=" * 60)
    print("✅ 모든 테스트 통과!")
    print("=" * 60)
