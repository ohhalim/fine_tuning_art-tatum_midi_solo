"""
Training/Validation Loss 평가 스크립트

파인튜닝 로그에서 Loss를 추출해서 시각화하고 과적합 여부를 판단해줘.

사용법:
    python scripts/evaluate_loss.py --log_file ./ohhalim-jazz-style/trainer_state.json
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_training_log(log_file):
    """학습 로그 파일 로드"""
    with open(log_file, 'r') as f:
        data = json.load(f)
    return data


def extract_losses(trainer_state):
    """로그에서 train_loss와 eval_loss 추출"""
    log_history = trainer_state.get('log_history', [])

    train_losses = []
    eval_losses = []
    epochs = []

    for entry in log_history:
        if 'loss' in entry:  # training loss
            train_losses.append(entry['loss'])
            epochs.append(entry.get('epoch', 0))
        if 'eval_loss' in entry:  # validation loss
            eval_losses.append(entry['eval_loss'])

    return epochs, train_losses, eval_losses


def detect_overfitting(train_loss, eval_loss, threshold=0.15):
    """
    과적합 감지

    Args:
        train_loss: 최종 training loss
        eval_loss: 최종 validation loss
        threshold: 허용 가능한 차이 (기본값 0.15)

    Returns:
        bool: 과적합 여부
        float: loss 차이
    """
    if not eval_loss:
        return False, 0.0

    final_train = train_loss[-1] if train_loss else 0
    final_eval = eval_loss[-1] if eval_loss else 0

    gap = final_eval - final_train

    is_overfitting = gap > threshold

    return is_overfitting, gap


def plot_losses(epochs, train_losses, eval_losses, output_path):
    """Loss 그래프 생성"""
    plt.figure(figsize=(12, 6))

    if train_losses:
        plt.plot(epochs[:len(train_losses)], train_losses,
                label='Training Loss', marker='o', linewidth=2)

    if eval_losses:
        eval_epochs = np.linspace(epochs[0], epochs[-1], len(eval_losses))
        plt.plot(eval_epochs, eval_losses,
                label='Validation Loss', marker='s', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training & Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"📊 그래프 저장됨: {output_path}")


def evaluate_training(log_file, output_dir='./evaluation'):
    """전체 평가 실행"""
    print("=" * 60)
    print("🔍 파인튜닝 Loss 평가 시작")
    print("=" * 60)

    # 로그 로드
    trainer_state = load_training_log(log_file)
    epochs, train_losses, eval_losses = extract_losses(trainer_state)

    if not train_losses:
        print("❌ Training loss를 찾을 수 없습니다.")
        return

    # 결과 출력
    print(f"\n📈 학습 진행 상황:")
    print(f"   총 에폭: {int(epochs[-1]) if epochs else 0}")
    print(f"   총 스텝: {len(train_losses)}")

    print(f"\n📊 Loss 값:")
    print(f"   최종 Training Loss:   {train_losses[-1]:.4f}")

    if eval_losses:
        print(f"   최종 Validation Loss: {eval_losses[-1]:.4f}")

        # 과적합 감지
        is_overfitting, gap = detect_overfitting(train_losses, eval_losses)

        print(f"\n🎯 과적합 분석:")
        print(f"   Loss 차이: {gap:.4f}")

        if is_overfitting:
            print(f"   ⚠️  과적합 감지! (차이 > 0.15)")
            print(f"   대응: 에폭 줄이기, Dropout 늘리기, 데이터 추가")
        else:
            print(f"   ✅ 정상 학습 (차이 < 0.15)")
    else:
        print(f"   ⚠️  Validation loss 없음 (validation split 확인)")

    # 학습 추세 분석
    if len(train_losses) >= 10:
        recent_losses = train_losses[-10:]
        loss_std = np.std(recent_losses)

        print(f"\n📉 학습 안정성:")
        print(f"   최근 10 스텝 표준편차: {loss_std:.4f}")

        if loss_std < 0.01:
            print(f"   ✅ 학습이 수렴했습니다 (안정적)")
        elif loss_std < 0.05:
            print(f"   🟡 학습 중 (약간 변동)")
        else:
            print(f"   ⚠️  학습이 불안정합니다 (변동 큼)")

    # 그래프 생성
    Path(output_dir).mkdir(exist_ok=True)
    output_path = f"{output_dir}/loss_curves.png"
    plot_losses(epochs, train_losses, eval_losses, output_path)

    # 판정
    print(f"\n" + "=" * 60)
    print("🏆 최종 판정:")

    final_train = train_losses[-1]
    final_eval = eval_losses[-1] if eval_losses else None

    if final_train < 0.3 and (not final_eval or final_eval < 0.4):
        print("   ✅ 파인튜닝 성공!")
        print("   → DJ 세트에 바로 사용 가능")
    elif final_train < 0.5:
        print("   🟡 파인튜닝 보통")
        print("   → 더 학습하거나 하이퍼파라미터 조정 권장")
    else:
        print("   ❌ 파인튜닝 실패")
        print("   → 학습 데이터, 설정 재검토 필요")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="파인튜닝 Loss 평가")
    parser.add_argument(
        '--log_file',
        type=str,
        default='./ohhalim-jazz-style/trainer_state.json',
        help='Trainer state JSON 파일 경로'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./evaluation',
        help='결과 저장 폴더'
    )

    args = parser.parse_args()

    evaluate_training(args.log_file, args.output_dir)


if __name__ == '__main__':
    main()
