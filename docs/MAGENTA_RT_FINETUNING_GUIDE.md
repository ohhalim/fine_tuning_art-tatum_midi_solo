# Magenta RealTime 재즈 피아노 Fine-tuning 가이드

실시간 재즈 피아노 즉흥연주 AI를 만들기 위한 단계별 가이드

> 참고: 이 문서는 오디오 기반 Magenta RT 실험용입니다.
> 현재 프로젝트의 메인 MVP 라인은 심볼릭 MIDI 파이프라인이며,
> 기준 문서는 `docs/JAMBOT_MIDI_REFACTOR_PLAN.md` 입니다.

---

## 목표

```
[사용자 연주] → [Magenta RT] → [재즈 스타일 즉흥연주 응답]
```

---

## 1단계: 데이터 준비 (로컬)

### 1.1 MIDI → WAV 변환

Magenta RT는 **오디오 기반**이라 MIDI를 WAV로 변환해야 함

```bash
# FluidSynth 설치 (Mac)
brew install fluidsynth

# 피아노 사운드폰트 다운로드
# 추천: Salamander Grand Piano (무료, 고품질)
# https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html
```

### 1.2 변환 스크립트 실행

```bash
# scripts/midi_to_wav.py 실행
python scripts/midi_to_wav.py \
    --input_dir ./midi_dataset/midi \
    --output_dir ./audio_dataset \
    --soundfont ./soundfonts/piano.sf2
```

### 1.3 Google Drive 업로드

```bash
# 변환된 WAV 파일들을 Google Drive에 업로드
# 폴더 구조:
# My Drive/
#   └── magenta_rt_data/
#       └── jazz_piano/
#           ├── track_001.wav
#           ├── track_002.wav
#           └── ...
```

**권장 데이터량**: 최소 30분 이상의 재즈 피아노 오디오

---

## 2단계: Colab 테스트 (무료)

### 2.1 공식 노트북 열기

1. [Magenta RT Fine-tuning Colab](https://colab.research.google.com/github/magenta/magenta-realtime/blob/main/notebooks/Magenta_RT_Finetune.ipynb) 접속
2. "런타임 > 런타임 유형 변경 > TPU" 선택

### 2.2 환경 설정 (5분)

```python
# Colab 첫 번째 셀 실행
!pip install magenta-realtime
```

### 2.3 Google Drive 연결

```python
from google.colab import drive
drive.mount('/content/drive')

# 데이터 경로 설정
DATA_PATH = "/content/drive/MyDrive/magenta_rt_data/jazz_piano"
```

### 2.4 기본 생성 테스트

```python
from magenta_rt import system

# 모델 로드
magenta_rt = system.MagentaRT(tag='large', device='tpu')
magenta_rt.warm_start()  # ~30초

# 텍스트 프롬프트로 생성
style = magenta_rt.embed_style("jazz piano improvisation")
chunk, state = magenta_rt.generate_chunk(style=style)

# 저장
chunk.write("test_output.wav")
```

### 2.5 Fine-tuning 실행

```python
# Training Data 준비
training_audio = load_audio_files(DATA_PATH)

# Fine-tuning (약 1-2시간)
magenta_rt.finetune(
    training_audio,
    epochs=10,
    batch_size=4
)

# 모델 저장
magenta_rt.save("./finetuned_jazz_rt")
```

---

## 3단계: RunPod Fine-tuning (유료, 선택)

Colab 세션 제한(12시간)을 피하려면 RunPod 사용

### 3.1 인스턴스 생성

1. [RunPod](https://runpod.io) 접속
2. GPU 선택: **A40** ($0.20/hr Spot) - 48GB VRAM
3. Template: PyTorch 2.0

### 3.2 환경 설정

```bash
# SSH 접속 후
git clone https://github.com/ohhalim/fine_tuning_art-tatum_midi_solo.git
cd fine_tuning_art-tatum_midi_solo

# JAX GPU 설치
pip install jax[cuda12]
pip install magenta-realtime
```

### 3.3 데이터 업로드

```bash
# Google Drive에서 다운로드하거나 직접 업로드
# 또는 rclone 사용
rclone copy gdrive:magenta_rt_data/jazz_piano ./audio_dataset
```

### 3.4 Fine-tuning 실행

```bash
python scripts/finetune_magenta_rt.py \
    --data_dir ./audio_dataset \
    --output_dir ./checkpoints/jazz_rt \
    --epochs 20
```

### 3.5 예상 비용

| 작업 | 시간 | 비용 |
|------|------|------|
| Fine-tuning | 2-3시간 | $0.40-0.60 |
| 테스트/생성 | 1시간 | $0.20 |
| **총합** | - | **$0.60-0.80** |

---

## 4단계: 실시간 연동 (향후)

### 4.1 오디오 입력 설정

```python
import sounddevice as sd

# 마이크 입력 캡처
def audio_callback(indata, frames, time, status):
    # 실시간 오디오 처리
    style = magenta_rt.embed_style(indata)
    # ...
```

### 4.2 MIDI 키보드 연동

```python
import mido

# MIDI 입력 → 오디오 변환 → Magenta RT
for msg in mido.open_input():
    if msg.type == 'note_on':
        # 노트 처리
        pass
```

### 4.3 FL Studio 연동 (선택)

- VST 플러그인 형태로 래핑
- 또는 MIDI Loopback 사용

---

## 체크리스트

### Phase 1: 데이터 준비
- [ ] FluidSynth 설치
- [ ] 피아노 사운드폰트 다운로드
- [ ] MIDI → WAV 변환 스크립트 작성
- [ ] WAV 파일 생성 (최소 30분)
- [ ] Google Drive 업로드

### Phase 2: Colab 테스트
- [ ] 공식 Colab 노트북 열기
- [ ] TPU 런타임 설정
- [ ] 기본 생성 테스트
- [ ] Fine-tuning 실행
- [ ] 결과 평가

### Phase 3: RunPod (선택)
- [ ] A40 인스턴스 생성
- [ ] 환경 설정
- [ ] 데이터 업로드
- [ ] Fine-tuning 실행
- [ ] 모델 저장/다운로드

### Phase 4: 실시간 연동
- [ ] 오디오 입력 테스트
- [ ] MIDI 키보드 연동
- [ ] 실시간 생성 테스트

---

## 참고 자료

- [Magenta RealTime GitHub](https://github.com/magenta/magenta-realtime)
- [Magenta RT Hugging Face](https://huggingface.co/google/magenta-realtime)
- [Fine-tuning Colab 노트북](https://colab.research.google.com/github/magenta/magenta-realtime/blob/main/notebooks/Magenta_RT_Finetune.ipynb)
- [아키텍처 분석](./MAGENTA_RT_ARCHITECTURE.md)

---

## 문제 해결

### JAX GPU 인식 안됨
```bash
# CUDA 버전 확인
nvidia-smi

# JAX GPU 재설치
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### 메모리 부족
```python
# batch_size 줄이기
magenta_rt.finetune(data, batch_size=2)

# 또는 gradient accumulation
magenta_rt.finetune(data, batch_size=2, gradient_accumulation_steps=4)
```

### Colab 세션 끊김
- Google Drive에 체크포인트 자동 저장 설정
- 또는 RunPod으로 전환
