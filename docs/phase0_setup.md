# Phase 0: 환경 설정 🛠️

**목표**: GPU 환경을 구축하고 TatumFlow를 실행할 준비를 완료합니다.

**예상 시간**: 1-2일
**난이도**: ⭐⭐☆☆☆

---

## 📋 체크리스트

- [ ] GPU 환경 선택 및 접속
- [ ] TatumFlow 리포지토리 클론
- [ ] 필수 라이브러리 설치
- [ ] 환경 테스트 실행
- [ ] 체크포인트 저장 경로 설정

---

## 1. GPU 환경 선택

### 옵션 A: Google Colab Pro (추천 - 초보자)

**장점**:
- 설정이 간단함 (브라우저만 있으면 됨)
- A100 GPU 사용 가능
- Jupyter 노트북 인터페이스

**단점**:
- 시간당 과금 (~$3/시간)
- 세션이 끊기면 재시작 필요

**시작 방법**:
```python
# 1. https://colab.research.google.com 접속
# 2. 새 노트북 생성
# 3. 런타임 > 런타임 유형 변경 > GPU (A100 선택)
# 4. 아래 코드 실행

# GPU 확인
!nvidia-smi

# 리포지토리 클론
!git clone https://github.com/YOUR_USERNAME/fine_tuning_art-tatum_midi_solo.git
%cd fine_tuning_art-tatum_midi_solo

# 라이브러리 설치
!pip install -r requirements.txt
```

**예상 비용**: $50-100/월 (Phase 3 훈련 포함)

---

### 옵션 B: Kaggle Notebooks (추천 - 무료)

**장점**:
- **완전 무료**
- 주 30시간 GPU 제공
- P100 또는 T4 GPU

**단점**:
- A100보다 느림 (하지만 무료!)
- 주 30시간 제한

**시작 방법**:
1. https://www.kaggle.com 가입
2. "Create" > "New Notebook"
3. Settings > Accelerator > GPU T4 x2 선택
4. 아래 코드 실행:

```python
# GPU 확인
!nvidia-smi

# 리포지토리 클론
!git clone https://github.com/YOUR_USERNAME/fine_tuning_art-tatum_midi_solo.git
%cd fine_tuning_art-tatum_midi_solo

# 라이브러리 설치
!pip install -r requirements.txt
```

**예상 비용**: **무료** 🎉

---

### 옵션 C: 로컬 GPU (고급)

**요구사항**:
- NVIDIA GPU (RTX 3060 이상 권장, VRAM 8GB+)
- CUDA 11.8 이상
- Ubuntu 20.04+ 또는 Windows 11

**시작 방법**:
```bash
# CUDA 설치 확인
nvidia-smi

# Python 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 리포지토리 클론
git clone https://github.com/YOUR_USERNAME/fine_tuning_art-tatum_midi_solo.git
cd fine_tuning_art-tatum_midi_solo

# 라이브러리 설치
pip install -r requirements.txt
```

---

## 2. 필수 라이브러리 설치

### requirements.txt 확인

프로젝트 루트에 `requirements.txt` 생성:

```txt
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
pretty_midi==0.2.10
tqdm>=4.65.0
tensorboard>=2.13.0
pyyaml>=6.0
```

### 설치 실행

```bash
pip install -r requirements.txt
```

### FluidSynth 설치 (MIDI → MP3 변환용)

**Ubuntu/Colab/Kaggle**:
```bash
apt-get install -y fluidsynth
pip install midi2audio
```

**macOS**:
```bash
brew install fluidsynth
pip install midi2audio
```

**Windows**:
```powershell
# https://github.com/FluidSynth/fluidsynth/releases 에서 다운로드
pip install midi2audio
```

---

## 3. 환경 테스트

### 테스트 스크립트 실행

```bash
python scripts/phase0_test_environment.py
```

**예상 출력**:
```
✅ PyTorch 설치 확인: 2.1.0
✅ CUDA 사용 가능: True
✅ GPU 이름: NVIDIA A100-SXM4-40GB
✅ TatumFlow 모듈 import 성공
✅ 간단한 forward pass 성공
✅ 체크포인트 저장/로드 성공

🎉 환경 설정 완료!
```

**만약 에러가 발생하면**:
- PyTorch CUDA 버전 확인: `python -c "import torch; print(torch.cuda.is_available())"`
- CUDA 버전 확인: `nvidia-smi`
- PyTorch 재설치: https://pytorch.org/get-started/locally/

---

## 4. 디렉토리 구조 확인

```
fine_tuning_art-tatum_midi_solo/
├── data/                       # 데이터 저장 (Phase 1에서 채움)
│   └── art_tatum_midi/
├── src/
│   └── tatumflow/              # TatumFlow 모델
├── scripts/                    # 실행 스크립트
├── checkpoints/                # 체크포인트 저장
├── outputs/                    # 생성 결과
├── logs/                       # TensorBoard 로그
├── docs/                       # 문서
├── config.yaml                 # 설정 파일
├── requirements.txt
└── ROADMAP.md
```

### 필요한 폴더 생성

```bash
mkdir -p data/art_tatum_midi
mkdir -p checkpoints
mkdir -p outputs/generation
mkdir -p logs/tensorboard
```

---

## 5. 체크포인트 저장 경로 설정

### Google Drive 연동 (Colab 사용 시)

```python
from google.colab import drive
drive.mount('/content/drive')

# 체크포인트를 Drive에 저장
CHECKPOINT_DIR = '/content/drive/MyDrive/tatumflow_checkpoints'
!mkdir -p $CHECKPOINT_DIR
```

### config.yaml 수정

```yaml
# 저장 경로 설정
checkpoint_dir: './checkpoints'  # 로컬
# checkpoint_dir: '/content/drive/MyDrive/tatumflow_checkpoints'  # Colab

output_dir: './outputs'
log_dir: './logs/tensorboard'
```

---

## 6. TensorBoard 설정

### 로컬에서 실행

```bash
tensorboard --logdir=./logs/tensorboard
```

브라우저에서 `http://localhost:6006` 접속

### Colab에서 실행

```python
%load_ext tensorboard
%tensorboard --logdir ./logs/tensorboard
```

---

## 🎓 학습 내용

### GPU란?

**CPU vs GPU**:
- CPU: 복잡한 연산을 순차적으로 (뇌의 전두엽)
- GPU: 간단한 연산을 병렬로 수천 개 (뇌의 시각 피질)

딥러닝은 행렬 곱셈의 반복이므로 GPU가 **100배 이상 빠릅니다**.

### CUDA란?

NVIDIA GPU를 프로그래밍하는 플랫폼입니다.
- PyTorch는 CUDA를 사용해 GPU 연산
- `torch.cuda.is_available()` = CUDA 설치 확인

### Mixed Precision (AMP)란?

- 기본: FP32 (32비트 부동소수점)
- AMP: FP16 (16비트) + FP32 혼합
- **장점**: 2배 빠름, 메모리 50% 절감
- **단점**: 정밀도 약간 감소 (음악 생성엔 무시 가능)

TatumFlow는 AMP를 기본 지원합니다!

---

## 🚨 문제 해결

### 문제 1: CUDA out of memory

**증상**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**해결**:
```yaml
# config.yaml에서 batch size 줄이기
batch_size: 4  # 기본 8에서 줄임
```

### 문제 2: PyTorch CUDA 버전 불일치

**증상**:
```
torch.cuda.is_available() returns False
```

**해결**:
```bash
# CUDA 버전 확인
nvidia-smi  # 예: CUDA 11.8

# 해당 버전 PyTorch 재설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 문제 3: pretty_midi 설치 실패

**증상**:
```
ERROR: Failed building wheel for python-rtmidi
```

**해결**:
```bash
# Ubuntu/Colab
apt-get install -y libasound2-dev libjack-dev

# macOS
brew install jack

# Windows
# Anaconda 사용 권장: conda install -c conda-forge pretty_midi
```

---

## ✅ Phase 0 완료 체크

다음 항목이 모두 ✅ 이면 Phase 1로 진행하세요:

- [ ] `nvidia-smi` 실행 시 GPU 정보 표시
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` 출력 `True`
- [ ] `python scripts/phase0_test_environment.py` 성공
- [ ] `checkpoints/`, `outputs/`, `logs/` 폴더 생성됨
- [ ] TensorBoard 접속 가능

---

## 다음 단계

**Phase 1: 데이터 준비**로 이동:
```bash
cat docs/phase1_data.md
```

**축하합니다! 환경 설정 완료! 🎉**

이제 재즈 AI를 만들 준비가 되었습니다! 🎹
