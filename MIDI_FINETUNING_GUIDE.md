# MIDI Fine-tuning 실전 가이드

**"어떻게 MIDI 데이터로 AI를 학습시켜서 내가 원하는 스타일의 음악을 만들까?"**

이론 논문은 건너뛰고, **실제로 어떻게 작동하는지** 중심으로 설명합니다.

---

## 🎯 핵심 질문

1. **MIDI 파일을 AI가 어떻게 이해하나?** → Tokenization
2. **Fine-tuning은 정확히 뭘 하는 거야?** → 모델이 스타일 학습
3. **Brad Mehldau 스타일을 만들려면?** → 그의 MIDI 데이터로 학습
4. **얼마나 데이터가 필요해?** → 최소 50개, 이상적으로는 200개+ MIDI
5. **내 컴퓨터로 가능해?** → RTX 3060 (8GB)이면 충분

---

## 📊 전체 워크플로우 (한눈에 보기)

```
1. MIDI 파일 수집
   ↓
2. Tokenization (MIDI → 숫자 토큰)
   ↓
3. Dataset 준비 (HuggingFace Dataset)
   ↓
4. Pre-trained 모델 불러오기 (선택사항)
   ↓
5. Fine-tuning (QLoRA 사용)
   ↓
6. Generate (새로운 MIDI 생성)
   ↓
7. MIDI → 음악 (FL Studio 등에서 재생)
```

**소요 시간**:
- 데이터 준비: 1-2시간
- Fine-tuning: 2-6시간 (GPU에 따라)
- 생성: 몇 초

---

## 1️⃣ MIDI를 AI가 이해하는 방법 (Tokenization)

### MIDI 파일이란?

MIDI는 **이벤트 시퀀스**입니다:

```
Time    Event
0.0s    Note ON:  C4 (pitch=60, velocity=80)
0.5s    Note OFF: C4
0.5s    Note ON:  E4 (pitch=64, velocity=75)
1.0s    Note OFF: E4
1.0s    Note ON:  G4 (pitch=67, velocity=80)
1.5s    Note OFF: G4
```

### AI는 숫자만 이해함

MIDI 이벤트를 **토큰(숫자)**으로 변환:

```
원본 MIDI:
  Note ON C4, velocity 80

Tokenization 후:
  [BOS, NOTE_ON_60, VELOCITY_80, TIME_SHIFT_500, NOTE_OFF_60, ...]
  ↓
  [1, 63, 338, 259, 191, ...]  (실제 숫자 토큰)
```

이제 AI가 이해 가능! (언어 모델이 단어를 숫자로 바꾸는 것과 동일)

### 인기있는 Tokenization 방법

#### 1. **REMI (우리 프로젝트 사용)**

```python
from miditok import REMI

# Tokenizer 생성
tokenizer = REMI()

# MIDI → 토큰
tokens = tokenizer("brad_mehldau_solo.mid")
# 결과: [1, 63, 338, 259, 191, 63, 340, ...]

# 토큰 → MIDI
midi = tokenizer.tokens_to_midi(tokens)
midi.write("output.mid")
```

**REMI 토큰 종류:**
- `BAR`: 마디 구분
- `POSITION`: 마디 내 위치 (1/16 단위)
- `NOTE_ON_X`: X 음높이 노트 시작
- `NOTE_OFF`: 노트 끝
- `VELOCITY_X`: 세기 (0-127)
- `TEMPO_X`: 템포 (BPM)

#### 2. **Event-based (간단함)**

```python
# 우리 프로젝트 구현
tokens = [
    BOS,              # 시작
    NOTE_ON_60,       # C4 켜기
    TIME_SHIFT_500,   # 500ms 대기
    NOTE_OFF_60,      # C4 끄기
    NOTE_ON_64,       # E4 켜기
    TIME_SHIFT_500,
    NOTE_OFF_64,
    EOS               # 끝
]
```

### 실제 코드 예시

```python
# Production Transformer 브랜치에서
from data.event_tokenizer import EventTokenizer

tokenizer = EventTokenizer()

# 1. MIDI 파일 → 토큰
tokens = tokenizer.encode("brad_mehldau_solo.mid")
print(tokens)  # [1, 63, 338, 259, ...]

# 2. 토큰 → MIDI 파일
events = tokenizer.decode(tokens)
midi = tokenizer.events_to_midi(events, "output.mid")
```

**핵심**: MIDI 이벤트를 AI가 이해할 수 있는 숫자 시퀀스로 변환!

---

## 2️⃣ Fine-tuning이 정확히 뭘 하는가?

### Pre-training vs Fine-tuning

#### Pre-training (사전 학습)
- **목적**: 일반적인 음악 패턴 학습
- **데이터**: 수만 개의 다양한 MIDI (클래식, 재즈, 팝 등)
- **결과**: "음악이란 이런 거구나" 이해

```
Input:  [C, E, G]
Output: [C, E, G, C]  (코드 진행 예측)
```

#### Fine-tuning (미세 조정)
- **목적**: 특정 스타일 학습 (Brad Mehldau 스타일)
- **데이터**: Brad Mehldau MIDI만 50-200개
- **결과**: "Brad Mehldau는 이렇게 연주하는구나" 학습

```
Input:  [Cmaj7, Am7]
Output: [complex_brad_mehldau_voicing, rhythmic_pattern, ...]
```

### 비유

**Pre-training**: 한국어 문법과 일반 지식 학습
- "나는 학생입니다", "오늘 날씨가 좋다" 등

**Fine-tuning**: 특정 작가 스타일 학습
- 한강 작가 스타일로 글쓰기
- 김훈 작가 스타일로 글쓰기

### Fine-tuning이 학습하는 것들

Brad Mehldau MIDI로 fine-tuning하면:

1. **Harmony (화음)**
   - 그가 자주 쓰는 voicing
   - 텐션 노트 사용법
   - 코드 진행 패턴

2. **Rhythm (리듬)**
   - 독특한 타이밍
   - Syncopation (싱코페이션)
   - Rubato (템포 변화)

3. **Melody (멜로디)**
   - 프레이징
   - 음역 사용
   - 음정 이동 패턴

4. **Dynamics (다이나믹)**
   - Velocity 패턴
   - Crescendo/Diminuendo

### QLoRA Fine-tuning

우리 프로젝트는 **QLoRA** 사용:

```python
# Base model의 99%는 freeze (고정)
# 1%만 학습 (LoRA adapters)

Base Model (150M params): ❄️ Frozen
   ↓
LoRA Adapters (2.8M params): 🔥 Training
   ↓
Brad Mehldau style learned!
```

**장점**:
- 메모리 75% 절약
- 학습 시간 50% 단축
- 성능은 거의 동일

---

## 3️⃣ 실제 코드로 보는 Fine-tuning

### Step 1: 데이터 준비

```python
# 1. Brad Mehldau MIDI 파일 모으기
data/brad_mehldau/
  ├── solo_1.mid
  ├── solo_2.mid
  ├── ...
  └── solo_50.mid

# 2. Tokenization & Dataset 생성
from data.midi_dataset import create_dataset_from_midi_files
from data.event_tokenizer import EventTokenizer

tokenizer = EventTokenizer()

dataset = create_dataset_from_midi_files(
    midi_dir="data/brad_mehldau",
    tokenizer=tokenizer,
    max_seq_len=2048,
    train_split=0.8  # 80% 학습, 20% 검증
)

# 결과:
# dataset['train']: 40개 MIDI → 40,000개 토큰 시퀀스
# dataset['validation']: 10개 MIDI
```

### Step 2: 모델 & QLoRA 설정

```python
from models import MusicTransformerForGeneration
from models.qlora import QLoRAConfig, apply_qlora_to_model

# 1. Base model 불러오기 (또는 새로 생성)
model = MusicTransformerForGeneration.from_pretrained(
    "pretrained_music_transformer",  # 선택사항
    quantization_config=bnb_config   # 4-bit quantization
)

# 2. QLoRA 적용
qlora_config = QLoRAConfig(
    lora_rank=8,        # 낮을수록 효율적, 높을수록 표현력
    lora_alpha=16,      # LoRA scaling
    lora_dropout=0.1    # Overfitting 방지
)

model = apply_qlora_to_model(model, qlora_config)

# 출력:
# Total parameters: 150,000,000
# Trainable parameters: 2,800,000 (1.9%)
# ✅ 99%는 frozen, 1%만 학습!
```

### Step 3: Fine-tuning 실행

```python
from transformers import Trainer, TrainingArguments

# Training 설정
training_args = TrainingArguments(
    output_dir="experiments/brad_mehldau_v1",

    # Batch & Epochs
    per_device_train_batch_size=4,    # GPU 메모리에 따라
    gradient_accumulation_steps=4,    # Effective batch = 16
    num_train_epochs=5,                # 보통 5-10 epochs

    # Learning rate
    learning_rate=3e-4,                # LoRA는 높은 LR 사용
    warmup_steps=100,

    # 효율성
    fp16=True,                         # Mixed precision

    # Logging
    logging_steps=10,
    eval_steps=100,
    save_steps=500,

    # W&B (Weights & Biases)
    report_to="wandb"                  # 실험 추적
)

# Trainer 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation']
)

# 🚀 학습 시작!
trainer.train()

# 진행 상황:
# Epoch 1/5: 100%|████████| 1000/1000 [10:23<00:00]
# Loss: 2.456 → 1.823 → 1.234 (좋아지고 있음!)
```

### Step 4: 생성 (Generate)

```python
from inference.generator import MusicGenerator

# 1. Fine-tuned 모델 불러오기
generator = MusicGenerator("experiments/brad_mehldau_v1/final_model")

# 2. 코드 진행 제공
prompt = "Cmaj7 Am7 Dm7 G7"  # ii-V-I in C

# 3. 생성!
generator.generate_and_save(
    prompt=prompt,
    output_path="brad_mehldau_solo.mid",
    max_length=512,      # 토큰 수 (약 30초 음악)
    temperature=0.9,     # 0.7-1.0: 안정적, 1.0-1.5: 창의적
    top_p=0.95          # Nucleus sampling
)

# 결과: brad_mehldau_solo.mid 생성!
```

### Step 5: FL Studio에서 재생

```
1. FL Studio 실행
2. File → Import → MIDI file
3. brad_mehldau_solo.mid 선택
4. 피아노 VST 선택 (예: Keyscape, Addictive Keys)
5. 재생 ▶️
```

---

## 4️⃣ 실제 무슨 일이 일어나는가?

### Training 중 모델 내부

```python
# Epoch 1, Step 1
Input:  [BOS, Cmaj7, NOTE_ON_60, ...]
Target: [Cmaj7, NOTE_ON_60, TIME_SHIFT_100, ...]

Model prediction: [Cmaj7, NOTE_ON_62, ...]  ❌ 틀림
Loss: 2.5 (높음)
→ LoRA weights 업데이트

# Epoch 1, Step 100
Input:  [BOS, Cmaj7, NOTE_ON_60, ...]
Target: [Cmaj7, NOTE_ON_60, TIME_SHIFT_100, ...]

Model prediction: [Cmaj7, NOTE_ON_60, ...]  ✅ 맞음!
Loss: 1.8 (낮아짐)

# Epoch 5, Step 5000
Loss: 0.8 (매우 낮음)
→ Brad Mehldau 스타일 학습 완료!
```

### Generation 중 모델 내부

```python
# 사용자 입력: "Cmaj7 Am7 Dm7 G7"

# Step 1: 시작 토큰
generated = [BOS, CHORD_Cmaj7]

# Step 2: 다음 토큰 예측
logits = model(generated)
# logits: [0.01, 0.05, 0.8, 0.1, ...]  (각 토큰의 확률)

# Temperature로 조정
probs = softmax(logits / temperature)

# Top-p sampling으로 선택
next_token = sample(probs, top_p=0.95)  # NOTE_ON_60

generated = [BOS, CHORD_Cmaj7, NOTE_ON_60]

# Step 3-512: 반복
# 최종: [BOS, CHORD_Cmaj7, NOTE_ON_60, TIME_SHIFT_100, ...]
```

---

## 5️⃣ 얼마나 데이터가 필요한가?

### 최소 요구사항

```
50개 MIDI 파일 (각 1-3분)
= 약 50-150분 음악
= 충분히 학습 가능
```

### 이상적인 양

```
200+ MIDI 파일
= 200-600분 음악
= 고품질 결과
```

### Data Augmentation으로 늘리기

원본: 50개 MIDI
↓
**Transposition** (12 keys):
50 × 12 = 600개

**Tempo variation** (3 speeds: 0.9x, 1.0x, 1.1x):
600 × 3 = 1,800개

**최종**: 1,800개 학습 샘플!

```python
# 자동 augmentation
dataset = create_dataset_from_midi_files(
    midi_dir="data/brad_mehldau",
    tokenizer=tokenizer,
    augment=True  # 자동으로 12×3 = 36배 증가!
)
```

---

## 6️⃣ 실제 사용 시나리오

### 시나리오 1: Brad Mehldau 스타일 솔로 생성

```bash
# 1. 데이터 준비 (50개 MIDI 수집)
# 2. Fine-tuning
python training/train.py \
    --data_dir data/brad_mehldau \
    --output_dir experiments/brad_v1 \
    --num_train_epochs 5 \
    --use_qlora

# 시간: 3-6시간 (RTX 3060)

# 3. 생성
python inference/generator.py \
    --checkpoint experiments/brad_v1/final_model \
    --prompt "Fmaj7 Dm7 Gm7 C7" \
    --output solo.mid

# 시간: 5초

# 4. FL Studio에서 재생!
```

### 시나리오 2: 여러 버전 생성 & 선택

```python
# 5개 버전 생성
for i in range(5):
    generator.generate_and_save(
        prompt="Cmaj7 Am7 Dm7 G7",
        output_path=f"solo_v{i+1}.mid",
        temperature=0.8 + i*0.1  # 다양성
    )

# 결과:
# solo_v1.mid (temperature=0.8, 안정적)
# solo_v2.mid (temperature=0.9, 균형)
# solo_v3.mid (temperature=1.0, 창의적)
# solo_v4.mid (temperature=1.1, 실험적)
# solo_v5.mid (temperature=1.2, 무작위)

# → 가장 마음에 드는 것 선택!
```

### 시나리오 3: 내 멜로디에 반주 추가

```python
# 1. 내가 만든 멜로디 MIDI
my_melody = "data/my_melody.mid"

# 2. 코드 진행 추출
chords = extract_chords(my_melody)  # ["Cmaj7", "Am7", ...]

# 3. Brad Mehldau 스타일 반주 생성
generator.generate_and_save(
    prompt=" ".join(chords),
    output_path="accompaniment.mid"
)

# 4. FL Studio에서 합치기
# Track 1: my_melody.mid
# Track 2: accompaniment.mid
```

---

## 7️⃣ 성능 최적화 & 팁

### Temperature 선택

```python
temperature = 0.7  # 매우 안정적, 반복적
temperature = 0.9  # 추천! 안정 + 창의성
temperature = 1.0  # 기본값
temperature = 1.2  # 실험적, 예측 불가능
temperature = 1.5  # 거의 무작위
```

**실험 방법**:
1. 0.8-1.2 사이에서 5개 생성
2. 가장 좋은 것 선택
3. 그 temperature 사용

### Fine-tuning Hyperparameters

```yaml
# 빠른 테스트 (1시간)
num_train_epochs: 3
batch_size: 8
learning_rate: 5e-4

# 균형 (3-6시간)
num_train_epochs: 5
batch_size: 4
learning_rate: 3e-4

# 최고 품질 (12+ 시간)
num_train_epochs: 10
batch_size: 2
learning_rate: 2e-4
```

### LoRA Rank 선택

```python
lora_rank = 4   # 빠름, 메모리 적음, 표현력 낮음
lora_rank = 8   # 추천! 균형잡힘
lora_rank = 16  # 느림, 메모리 많음, 표현력 높음
lora_rank = 32  # 거의 full fine-tuning
```

---

## 8️⃣ 문제 해결 (Troubleshooting)

### Q: 생성된 음악이 이상해요 (무작위)

**원인**: Under-training 또는 temperature 너무 높음

**해결**:
1. Training loss 확인: >1.5이면 더 학습 필요
2. Temperature 낮추기: 0.9 → 0.7
3. Epochs 늘리기: 5 → 10

### Q: 생성된 음악이 너무 반복적이에요

**원인**: Over-fitting 또는 temperature 너무 낮음

**해결**:
1. Temperature 높이기: 0.9 → 1.1
2. Top-p 낮추기: 0.95 → 0.9
3. 더 많은 데이터 추가
4. Dropout 늘리기: 0.1 → 0.2

### Q: GPU 메모리 부족 (OOM)

**해결**:
```python
# 1. Batch size 줄이기
per_device_train_batch_size = 2  # 4 → 2

# 2. Gradient accumulation 늘리기
gradient_accumulation_steps = 8  # 4 → 8

# 3. Sequence length 줄이기
max_seq_len = 1024  # 2048 → 1024

# 4. 4-bit quantization 확인
load_in_4bit = True  # 꼭 True!
```

### Q: Loss가 안 줄어들어요

**원인**: Learning rate 문제

**해결**:
```python
# Learning rate 조정
learning_rate = 1e-4  # 3e-4 → 1e-4 (더 낮게)

# 또는
learning_rate = 5e-4  # 3e-4 → 5e-4 (더 높게)

# Warmup 늘리기
warmup_steps = 200  # 100 → 200
```

---

## 9️⃣ 실전 체크리스트

### 데이터 준비
- [ ] 50+ Brad Mehldau MIDI 파일 수집
- [ ] 품질 확인 (깨진 파일 제거)
- [ ] `data/brad_mehldau/` 디렉토리에 배치

### 환경 설정
- [ ] CUDA & cuDNN 설치
- [ ] `pip install -r production_transformer/requirements.txt`
- [ ] Weights & Biases 계정 생성 & 로그인

### Fine-tuning
- [ ] Dataset 생성 & 확인
- [ ] Config 파일 설정 (`configs/qlora_default.yaml`)
- [ ] Training 시작
- [ ] W&B에서 loss 모니터링

### 평가
- [ ] Validation loss < 1.5 확인
- [ ] 테스트 생성 (3-5개)
- [ ] 음악적 품질 청취

### 사용
- [ ] 마음에 드는 checkpoint 선택
- [ ] 여러 temperature 실험
- [ ] FL Studio에서 최종 작업

---

## 🎵 마무리: 핵심 정리

### 1. Tokenization (MIDI → 숫자)
```
MIDI 이벤트 → 토큰 시퀀스 → AI가 이해 가능
```

### 2. Fine-tuning (스타일 학습)
```
Pre-trained (일반 음악) + Brad MIDI → Brad 스타일
99% frozen + 1% LoRA = 효율적!
```

### 3. Generation (새로운 음악)
```
코드 입력 → 모델 예측 → 토큰 시퀀스 → MIDI 파일
```

### 4. 실제 사용
```
50+ MIDI → 5 epochs (3-6시간) → Generate → FL Studio
```

---

## 📁 우리 프로젝트에서 사용법

```bash
# 1. 데이터 준비
mkdir -p data/brad_mehldau
# MIDI 파일들을 여기에 복사

# 2. Dataset 생성
cd production_transformer
python data/midi_dataset.py \
    --midi_dir ../data/brad_mehldau \
    --output_dir ../data/processed \
    --augment

# 3. Fine-tuning
python training/train.py \
    --data_dir ../data/processed \
    --output_dir ../experiments/brad_v1 \
    --use_qlora \
    --num_train_epochs 5 \
    --wandb_project "brad-mehldau"

# 4. 생성
python inference/generator.py \
    --checkpoint ../experiments/brad_v1/final_model \
    --prompt "Cmaj7 Am7 Dm7 G7" \
    --output ../output/solo.mid \
    --temperature 0.9 \
    --num_samples 5

# 5. Gradio 데모 실행
python inference/gradio_demo.py \
    --checkpoint ../experiments/brad_v1/final_model \
    --port 7860

# 브라우저에서 http://localhost:7860 접속
```

---

## 🚀 다음 단계

1. **Brad Mehldau MIDI 수집** (50개 목표)
   - YouTube 연주 → MIDI 변환 (AnthemScore, Melodyne)
   - MIDI 데이터베이스 검색

2. **첫 실험** (작게 시작)
   - 10개 MIDI로 테스트
   - 3 epochs만 학습
   - 결과 확인

3. **본격 학습** (만족스러우면)
   - 50+ MIDI로 확장
   - 5-10 epochs
   - 최적 hyperparameter 찾기

4. **프로덕션**
   - Gradio 데모 공유
   - FL Studio 워크플로우 확립
   - 친구들과 공유!

---

**이제 MIDI fine-tuning이 어떻게 작동하는지 완전히 이해하셨나요?** 🎹

질문 있으면 언제든 물어보세요!
