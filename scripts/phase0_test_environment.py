#!/usr/bin/env python3
"""
Phase 0: 환경 테스트 스크립트

TatumFlow 실행에 필요한 모든 요구사항을 검증합니다.
"""

import sys
import os

def test_pytorch():
    """PyTorch 설치 및 CUDA 확인"""
    try:
        import torch
        print(f"✅ PyTorch 설치 확인: {torch.__version__}")

        if torch.cuda.is_available():
            print(f"✅ CUDA 사용 가능: True")
            print(f"✅ GPU 이름: {torch.cuda.get_device_name(0)}")
            print(f"✅ GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("⚠️  CUDA 사용 불가 - CPU 모드로 실행 (훈련이 매우 느립니다)")
        return True
    except ImportError as e:
        print(f"❌ PyTorch 설치 실패: {e}")
        return False

def test_dependencies():
    """필수 라이브러리 확인"""
    required = [
        'numpy',
        'scipy',
        'pretty_midi',
        'tqdm',
        'tensorboard',
        'yaml'
    ]

    all_ok = True
    for lib in required:
        try:
            __import__(lib)
            print(f"✅ {lib} 설치 확인")
        except ImportError:
            print(f"❌ {lib} 설치 필요")
            all_ok = False

    return all_ok

def test_tatumflow_import():
    """TatumFlow 모듈 import 테스트"""
    try:
        # 프로젝트 루트를 path에 추가
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, project_root)

        from src.tatumflow import TatumFlow, MIDITokenizer
        print("✅ TatumFlow 모듈 import 성공")
        return True
    except Exception as e:
        print(f"❌ TatumFlow import 실패: {e}")
        return False

def test_forward_pass():
    """간단한 forward pass 테스트"""
    try:
        import torch
        from src.tatumflow import TatumFlow

        # CPU에서 테스트 (빠르게)
        device = 'cpu'
        model = TatumFlow(
            vocab_size=2048,
            hidden_dim=128,  # 작게
            latent_dim=64,
            num_layers=2,  # 작게
            num_heads=4
        ).to(device)

        # Dummy input
        batch_size = 2
        seq_len = 32
        tokens = torch.randint(0, 2048, (batch_size, seq_len)).to(device)

        # Forward pass
        with torch.no_grad():
            output = model(tokens)

        print(f"✅ Forward pass 성공 (출력 shape: {output['logits'].shape})")
        return True
    except Exception as e:
        print(f"❌ Forward pass 실패: {e}")
        return False

def test_checkpoint():
    """체크포인트 저장/로드 테스트"""
    try:
        import torch
        from src.tatumflow import TatumFlow

        model = TatumFlow(
            vocab_size=2048,
            hidden_dim=64,
            latent_dim=32,
            num_layers=1,
            num_heads=2
        )

        # 임시 저장
        checkpoint_path = '/tmp/test_checkpoint.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': 0
        }, checkpoint_path)

        # 로드
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

        # 삭제
        os.remove(checkpoint_path)

        print("✅ 체크포인트 저장/로드 성공")
        return True
    except Exception as e:
        print(f"❌ 체크포인트 테스트 실패: {e}")
        return False

def test_directories():
    """필요한 디렉토리 존재 확인"""
    required_dirs = [
        'data',
        'checkpoints',
        'outputs',
        'logs'
    ]

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    all_ok = True
    for dir_name in required_dirs:
        dir_path = os.path.join(project_root, dir_name)
        if os.path.exists(dir_path):
            print(f"✅ {dir_name}/ 디렉토리 존재")
        else:
            print(f"⚠️  {dir_name}/ 디렉토리 없음 - 생성 필요")
            all_ok = False

    return all_ok

def main():
    print("=" * 60)
    print("TatumFlow 환경 테스트")
    print("=" * 60)
    print()

    tests = [
        ("PyTorch & CUDA", test_pytorch),
        ("필수 라이브러리", test_dependencies),
        ("TatumFlow 모듈", test_tatumflow_import),
        ("Forward Pass", test_forward_pass),
        ("체크포인트", test_checkpoint),
        ("디렉토리 구조", test_directories)
    ]

    results = []
    for name, test_func in tests:
        print(f"\n[{name} 테스트]")
        print("-" * 60)
        result = test_func()
        results.append((name, result))
        print()

    print("=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    for name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{status}: {name}")

    all_passed = all(result for _, result in results)

    print()
    if all_passed:
        print("🎉 모든 테스트 통과! Phase 1로 진행하세요!")
        print()
        print("다음 단계:")
        print("  cat docs/phase1_data.md")
        return 0
    else:
        print("⚠️  일부 테스트 실패. 위 오류를 해결하세요.")
        print()
        print("도움말:")
        print("  docs/phase0_setup.md의 '문제 해결' 섹션 참고")
        return 1

if __name__ == '__main__':
    sys.exit(main())
