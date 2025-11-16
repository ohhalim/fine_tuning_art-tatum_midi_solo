# Magenta RealTime Fine-tuning Guide (QLoRA)

**목표**: 드랍용 재즈 클립 생성 (10-20초) with "ohhalim style"

**환경**:
- Colab (무료 TPU/GPU) → 테스트 & 작은 실험
- 런팟 RTX 3060 8GB ($10) → 본격 fine-tuning

---

## 📋 전체 워크플로우

```
Day 1: 기본 생성 테스트
  → Colab 데모로 첫 재즈 클립 10개 생성
  → 작동 확인 & 품질 체크

Day 2: 데이터 준비 & Audio Injection
  → Public dataset 다운로드 (Bill Evans 등)
  → Audio prompt 테스트
  → Fine-tuning 데이터 준비

Day 3: Fine-tuning 실행 (QLoRA)
  → 런팟 또는 Colab Pro
  → "ohhalim style" 학습
  → 생성 테스트

Day 4+: FL Studio 통합
  → 드랍에 재즈 클립 삽입
  → Export & Rekordbox
```

---

## 🚀 Day 1: 기본 생성 테스트

### Step 1: Colab 데모 실행

**1-1. Colab 열기**
```
https://colab.research.google.com/github/magenta/magenta-realtime/blob/main/notebooks/Magenta_RT_Demo.ipynb
```

**1-2. 런타임 설정**
```
Runtime → Change runtime type → TPU v2-8 선택
```

**1-3. 전체 실행**
```
Runtime → Run all
```

**예상 시간**: 5-10분 (첫 실행 시 모델 다운로드)

---

### Step 2: 드랍용 재즈 클립 10개 생성

노트북 마지막에 새 셀 추가:

```python
# ═══════════════════════════════════════
# 드랍용 재즈 클립 Batch 생성
# ═══════════════════════════════════════

from magenta_rt import audio, system
import numpy as np

# 모델 로드 (이미 로드되어 있음)
mrt = system.MagentaRT()

# 재즈 스타일 10가지
jazz_styles = [
    "Bill Evans modal jazz piano, slow tempo, impressionistic",
    "Bud Powell bebop piano, fast 140 BPM, energetic",
    "Herbie Hancock jazz funk piano, groovy, syncopated",
    "Keith Jarrett improvisation, lyrical, flowing",
    "McCoy Tyner quartal harmony, modal, powerful",
    "Oscar Peterson swing piano, uptempo, virtuosic",
    "Chick Corea latin jazz piano, rhythmic",
    "Brad Mehldau contemporary jazz piano, introspective",
    "Red Garland blues piano, soulful, bluesy",
    "Wynton Kelly hard bop piano, swinging, bright"
]

# 생성 설정
CLIP_DURATION = 16  # 초 (드랍용)
CHUNK_LENGTH = 2    # 초 (Magenta RT 기본)
NUM_CHUNKS = CLIP_DURATION // CHUNK_LENGTH

# 생성!
all_clips = []

for i, style_text in enumerate(jazz_styles):
    print(f"\n{'='*60}")
    print(f"🎹 Generating {i+1}/10: {style_text[:50]}...")
    print(f"{'='*60}")

    # 스타일 임베딩
    style = system.embed_style(style_text)

    # 청크 생성 (16초 = 8 chunks)
    chunks = []
    state = None

    for j in range(NUM_CHUNKS):
        state, chunk = mrt.generate_chunk(
            state=state,
            style=style,
            temperature=1.0,  # 다양성
            top_k=40
        )
        chunks.append(chunk)
        print(f"  ✓ Chunk {j+1}/{NUM_CHUNKS} generated")

    # 합치기
    generated = audio.concatenate(chunks)

    # 저장
    filename = f"drop_jazz_{i:03d}.wav"
    generated.save(filename)
    print(f"  ✅ Saved: {filename}")
    print(f"     Duration: {CLIP_DURATION}s")
    print(f"     Style: {style_text}")

    all_clips.append({
        'file': filename,
        'style': style_text,
        'duration': CLIP_DURATION
    })

    # 메모리 정리
    del chunks, generated, state
    import gc
    gc.collect()

print(f"\n{'='*60}")
print(f"🎉 Complete! Generated {len(all_clips)} clips")
print(f"{'='*60}")

# 요약
for i, clip in enumerate(all_clips):
    print(f"{i+1}. {clip['file']} - {clip['style'][:40]}...")
```

**실행 후:**

```python
# 다운로드
from google.colab import files
import zipfile

# ZIP으로 묶기
with zipfile.ZipFile('jazz_clips_day1.zip', 'w') as zipf:
    for clip in all_clips:
        zipf.write(clip['file'])

print("📦 Downloading zip file...")
files.download('jazz_clips_day1.zip')
print("✅ Download complete!")
```

**결과:**
- ✅ 10개 재즈 클립 (각 16초)
- ✅ 다양한 스타일 (Bill Evans ~ Wynton Kelly)
- ✅ ZIP 다운로드
- ✅ FL Studio에 바로 사용 가능!

---

## 📊 Day 2: 데이터 준비

### Step 1: Public Dataset 다운로드

**Option A: PiJAMA Dataset (추천)**

새 Colab 노트북:

```python
# PiJAMA: 200+ hours jazz piano MIDI
# https://github.com/CPJKU/pijama

# 다운로드 (5-10분)
!wget https://zenodo.org/record/5120004/files/pijama_dataset_audio.zip
!unzip pijama_dataset_audio.zip -d pijama/

# 구조 확인
!ls -lh pijama/

# Bill Evans 스타일만 추출 (예시)
import glob

all_files = glob.glob("pijama/**/*.wav", recursive=True)
print(f"Total files: {len(all_files)}")

# 필터링 (파일명에 'evans' 포함)
bill_evans_files = [f for f in all_files if 'evans' in f.lower()]
print(f"Bill Evans files: {len(bill_evans_files)}")

# 첫 20개만 사용
training_files = bill_evans_files[:20]
```

**Option B: YouTube → MIDI 변환**

```python
# 당신의 재즈 연주가 있다면:
# 1. MIDI 파일 준비
# 2. 또는 Audio → MIDI 변환

# Colab에 업로드
from google.colab import files
uploaded = files.upload()  # MIDI/Audio 파일 선택

# 확인
import os
uploaded_files = list(uploaded.keys())
print(f"Uploaded: {uploaded_files}")
```

**Option C: 직접 녹음 (나중에)**

```
1. FL Studio에서 MIDI 녹음
2. 20-50개 즉흥연주
3. Export as MIDI
4. Fine-tuning 데이터로 사용
```

---

### Step 2: Audio Injection 테스트

**Audio Injection Colab 열기:**
```
https://colab.research.google.com/github/magenta/magenta-realtime/blob/main/notebooks/Magenta_RT_Audio_Injection.ipynb
```

**실행:**

```python
# 노트북 전체 실행 후...

from magenta_rt import audio, musiccoca
import numpy as np

# 스타일 모델
style_model = musiccoca.MusicCoCa()

# 레퍼런스 오디오 (당신의 연주 또는 Bill Evans)
reference_audio = audio.Waveform.from_file('/content/my_jazz.wav')

# Text + Audio blending
weighted_styles = [
    (3.0, reference_audio),  # Audio가 가장 강함!
    (1.0, "modal jazz piano"),
    (0.5, "bebop improvisation")
]

# 임베딩
weights = np.array([w for w, _ in weighted_styles])
styles = style_model.embed([s for _, s in weighted_styles])
weights_norm = weights / weights.sum()
blended_style = (weights_norm[:, np.newaxis] * styles).mean(axis=0)

# 생성 테스트
mrt = system.MagentaRT()
chunks = []
state = None

for i in range(8):  # 16초
    state, chunk = mrt.generate_chunk(
        state=state,
        style=blended_style,
        temperature=1.0
    )
    chunks.append(chunk)
    print(f"Chunk {i+1}/8")

generated = audio.concatenate(chunks)
generated.save("my_style_test.wav")

# 다운로드
from google.colab import files
files.download("my_style_test.wav")
```

**확인:**
- 생성된 음악이 레퍼런스와 유사한가?
- 스타일이 반영되었는가?
- → Fine-tuning하면 더 정확해짐!

---

## 🔥 Day 3: Fine-tuning (QLoRA)

### Step 1: Fine-tuning Colab 열기

```
https://colab.research.google.com/github/magenta/magenta-realtime/blob/main/notebooks/Magenta_RT_Finetune.ipynb
```

**런타임 설정:**
```
Runtime → Change runtime type
→ GPU: T4 (무료) 또는 V100 (Colab Pro)
```

---

### Step 2: 데이터 준비

노트북에서 실행:

```python
# ═══════════════════════════════════════
# Step 2-1: 데이터 업로드/다운로드
# ═══════════════════════════════════════

# Option A: Colab에 업로드
from google.colab import files
import os

print("Upload your jazz MIDI/Audio files:")
uploaded = files.upload()

training_files = list(uploaded.keys())
print(f"\n✅ Uploaded {len(training_files)} files")
for f in training_files:
    print(f"  - {f}")

# Option B: Public dataset
# (위의 PiJAMA 다운로드 코드 사용)

# ═══════════════════════════════════════
# Step 2-2: Audio → Tokens 변환
# ═══════════════════════════════════════

from magenta_rt import audio, spectrostream

# SpectroStream codec
codec = spectrostream.SpectroStream()

# 각 파일 처리
tokenized_data = []

for i, file in enumerate(training_files):
    print(f"\n[{i+1}/{len(training_files)}] Processing: {file}")

    # 오디오 로드
    waveform = audio.Waveform.from_file(file)

    # Tokenize (2초 청크)
    # SpectroStream: 48kHz stereo → discrete tokens
    tokens = codec.encode(waveform)

    print(f"  Shape: {tokens.shape}")
    print(f"  Duration: {waveform.duration:.1f}s")

    tokenized_data.append({
        'file': file,
        'tokens': tokens,
        'duration': waveform.duration
    })

print(f"\n✅ Tokenized {len(tokenized_data)} files")

# ═══════════════════════════════════════
# Step 2-3: Data Augmentation
# ═══════════════════════════════════════

# Augmentation으로 데이터 늘리기
# 1. Pitch shifting (±2 semitones)
# 2. Time stretching (0.9x, 1.0x, 1.1x)

augmented_data = []

for data in tokenized_data:
    original_tokens = data['tokens']

    # Original
    augmented_data.append(original_tokens)

    # Pitch shifts (±1, ±2 semitones)
    # Note: SpectroStream tokens는 직접 pitch shift 어려움
    # → Audio 단계에서 augmentation 권장

    # 간단 버전: 원본 데이터만 사용
    # (본격 버전은 MIDI에서 augmentation 후 audio 변환)

print(f"Training samples: {len(augmented_data)}")
```

---

### Step 3: QLoRA 설정

```python
# ═══════════════════════════════════════
# QLoRA Fine-tuning Setup
# ═══════════════════════════════════════

import torch
from transformers import AutoModelForCausalLM
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from transformers import BitsAndBytesConfig

# ═══════════════════════════════════════
# Step 3-1: 4-bit Quantization Config
# ═══════════════════════════════════════

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # 4-bit quantization
    bnb_4bit_quant_type="nf4",              # NormalFloat4
    bnb_4bit_compute_dtype=torch.float16,   # Compute in FP16
    bnb_4bit_use_double_quant=True          # Double quantization
)

print("✅ Quantization config ready")
print(f"  - 4-bit: {bnb_config.load_in_4bit}")
print(f"  - Type: {bnb_config.bnb_4bit_quant_type}")
print(f"  - Compute: {bnb_config.bnb_4bit_compute_dtype}")

# ═══════════════════════════════════════
# Step 3-2: Base Model 로드 (4-bit)
# ═══════════════════════════════════════

print("\n📥 Loading base model (4-bit)...")

# Magenta RT의 실제 모델 체크포인트
# (노트북에 경로가 있을 것)
model_checkpoint = "path/to/magenta-rt-checkpoint"  # 노트북에서 확인!

# 로드
model = AutoModelForCausalLM.from_pretrained(
    model_checkpoint,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

print("✅ Base model loaded")
print(f"  Model: {type(model).__name__}")
print(f"  Device: {next(model.parameters()).device}")

# GPU 메모리 확인
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    print(f"  GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

# ═══════════════════════════════════════
# Step 3-3: LoRA Preparation
# ═══════════════════════════════════════

print("\n🔧 Preparing model for k-bit training...")
model = prepare_model_for_kbit_training(model)
print("✅ Model prepared")

# ═══════════════════════════════════════
# Step 3-4: LoRA Config
# ═══════════════════════════════════════

lora_config = LoraConfig(
    r=8,                        # Rank (핵심 파라미터!)
    lora_alpha=16,              # Scaling factor (보통 r*2)
    target_modules=[            # Attention layers에 적용
        "q_proj",               # Query projection
        "v_proj",               # Value projection
        "k_proj",               # Key projection (optional)
        "o_proj",               # Output projection (optional)
    ],
    lora_dropout=0.1,           # Dropout for regularization
    bias="none",                # Bias 학습 안 함
    task_type="CAUSAL_LM",      # Causal Language Modeling
    inference_mode=False        # Training mode
)

print("✅ LoRA config ready")
print(f"  Rank (r): {lora_config.r}")
print(f"  Alpha: {lora_config.lora_alpha}")
print(f"  Target modules: {lora_config.target_modules}")
print(f"  Dropout: {lora_config.lora_dropout}")

# ═══════════════════════════════════════
# Step 3-5: Apply LoRA
# ═══════════════════════════════════════

print("\n🎯 Applying LoRA to model...")
model = get_peft_model(model, lora_config)
print("✅ LoRA applied")

# 학습 가능한 파라미터 확인
model.print_trainable_parameters()

# Expected output:
# trainable params: 2,097,152 / 760,000,000 = 0.28%
# → 99.7% 파라미터는 freeze!
```

---

### Step 4: Training 실행

```python
# ═══════════════════════════════════════
# Training Configuration
# ═══════════════════════════════════════

from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from transformers.trainer_callback import ProgressCallback
import os

# 출력 디렉토리
output_dir = "./ohhalim-jazz-style"
os.makedirs(output_dir, exist_ok=True)

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,

    # Epochs & Batch
    num_train_epochs=50,                    # 50 epochs (조정 가능)
    per_device_train_batch_size=1,          # GPU 메모리 고려
    gradient_accumulation_steps=4,          # Effective batch = 4

    # Learning rate
    learning_rate=1e-4,                     # QLoRA 권장
    warmup_steps=100,                       # Warmup
    lr_scheduler_type="cosine",             # Cosine decay

    # Optimization
    optim="paged_adamw_8bit",               # 8-bit AdamW (QLoRA)
    weight_decay=0.01,                      # L2 regularization
    max_grad_norm=1.0,                      # Gradient clipping

    # Mixed precision
    fp16=True,                              # FP16 training

    # Logging
    logging_steps=10,
    logging_dir=f"{output_dir}/logs",
    report_to="tensorboard",                # TensorBoard

    # Saving
    save_steps=500,
    save_total_limit=3,                     # 최근 3개 checkpoint만
    save_strategy="steps",

    # Evaluation (optional)
    # evaluation_strategy="steps",
    # eval_steps=500,

    # Hardware
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
)

print("✅ Training arguments configured")
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"  Effective batch: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Optimizer: {training_args.optim}")

# ═══════════════════════════════════════
# Dataset Preparation
# ═══════════════════════════════════════

# tokenized_data를 HuggingFace Dataset으로 변환
# (노트북에 예제 코드가 있을 것)

from datasets import Dataset

# 간단한 예제 (실제는 노트북 코드 사용)
train_dataset = Dataset.from_dict({
    'input_ids': [data['tokens'] for data in tokenized_data],
    # ... 기타 필요한 컬럼
})

print(f"✅ Dataset prepared: {len(train_dataset)} samples")

# ═══════════════════════════════════════
# Trainer 생성
# ═══════════════════════════════════════

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    # eval_dataset=val_dataset,  # Optional
    # data_collator=data_collator,
    callbacks=[ProgressCallback()]
)

print("✅ Trainer ready")

# ═══════════════════════════════════════
# Training 시작!
# ═══════════════════════════════════════

print("\n" + "="*60)
print("🔥 Starting fine-tuning...")
print("="*60)
print(f"Training samples: {len(train_dataset)}")
print(f"Epochs: {training_args.num_train_epochs}")
print(f"Steps per epoch: {len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)}")
print(f"Total steps: ~{len(train_dataset) * training_args.num_train_epochs // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)}")
print("="*60 + "\n")

# 시작!
trainer.train()

print("\n" + "="*60)
print("✅ Training complete!")
print("="*60)

# ═══════════════════════════════════════
# 모델 저장
# ═══════════════════════════════════════

print("\n💾 Saving model...")

# LoRA weights 저장 (작음! ~10MB)
model.save_pretrained(output_dir)
print(f"✅ Model saved to: {output_dir}")

# Tokenizer도 저장 (있다면)
# tokenizer.save_pretrained(output_dir)

# 파일 크기 확인
import subprocess
size = subprocess.check_output(['du', '-sh', output_dir]).split()[0].decode('utf-8')
print(f"  Model size: {size}")

# ═══════════════════════════════════════
# 다운로드 (Colab)
# ═══════════════════════════════════════

print("\n📦 Creating zip for download...")

!zip -r ohhalim-jazz-style.zip {output_dir}

from google.colab import files
files.download("ohhalim-jazz-style.zip")

print("✅ Download complete!")
print("\nNext steps:")
print("1. Extract zip file locally")
print("2. Upload to 런팟 or use locally")
print("3. Generate with your fine-tuned model!")
```

**예상 시간:**
- Colab 무료 GPU (T4): 3-6시간
- Colab Pro GPU (V100): 1-3시간
- 런팟 RTX 3060: 2-4시간

---

## 🎵 Day 4: Fine-tuned Model로 생성

### Step 1: Model 로드

```python
# ═══════════════════════════════════════
# Fine-tuned Model 로드
# ═══════════════════════════════════════

from peft import PeftModel
from transformers import AutoModelForCausalLM
import torch

# Base model 로드
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "path/to/magenta-rt-checkpoint",
    torch_dtype=torch.float16,
    device_map="auto"
)

# LoRA weights 적용
print("Loading LoRA weights...")
model_finetuned = PeftModel.from_pretrained(
    base_model,
    "./ohhalim-jazz-style"  # 다운로드한 폴더
)

print("✅ Fine-tuned model loaded!")

# ═══════════════════════════════════════
# Generation 함수
# ═══════════════════════════════════════

from magenta_rt import audio, system

def generate_with_finetuned_model(
    model,
    prompt="ohhalim jazz piano style",
    duration=16,
    temperature=1.0,
    output_file="ohhalim_jazz_001.wav"
):
    """
    Fine-tuned model로 재즈 클립 생성
    """

    # 스타일 임베딩
    style = system.embed_style(prompt)

    # 생성
    mrt = system.MagentaRT(model=model)  # Fine-tuned model 사용

    chunks = []
    state = None
    num_chunks = duration // 2

    for i in range(num_chunks):
        state, chunk = mrt.generate_chunk(
            state=state,
            style=style,
            temperature=temperature
        )
        chunks.append(chunk)
        print(f"Chunk {i+1}/{num_chunks}")

    # 합치기
    generated = audio.concatenate(chunks)

    # 저장
    generated.save(output_file)
    print(f"✅ Saved: {output_file}")

    return generated

# ═══════════════════════════════════════
# 생성 테스트!
# ═══════════════════════════════════════

# Test 1: 당신 스타일
jazz_1 = generate_with_finetuned_model(
    model_finetuned,
    prompt="ohhalim jazz piano improvisation, modal",
    duration=16,
    output_file="drop_jazz_ohhalim_001.wav"
)

# Test 2: Bill Evans 영향
jazz_2 = generate_with_finetuned_model(
    model_finetuned,
    prompt="ohhalim style, Bill Evans influence, introspective",
    duration=16,
    output_file="drop_jazz_ohhalim_002.wav"
)

# Test 3: Uptempo bebop
jazz_3 = generate_with_finetuned_model(
    model_finetuned,
    prompt="ohhalim jazz, fast bebop, 140 BPM",
    duration=16,
    output_file="drop_jazz_ohhalim_003.wav"
)

print("\n🎉 Generation complete!")
print("Listen and compare with base model!")
```

---

### Step 2: Batch 생성 (드랍용 10개)

```python
# 다양한 드랍 시나리오용 클립 생성

drop_scenarios = [
    {
        'prompt': "ohhalim jazz piano, energetic drop, 128 BPM",
        'duration': 16,
        'temperature': 1.0,
        'name': 'energetic_drop'
    },
    {
        'prompt': "ohhalim modal jazz, floating, ambient",
        'duration': 20,
        'temperature': 1.1,
        'name': 'ambient_drop'
    },
    {
        'prompt': "ohhalim bebop piano, fast lines, 140 BPM",
        'duration': 12,
        'temperature': 0.9,
        'name': 'fast_bebop'
    },
    {
        'prompt': "ohhalim jazz funk, groovy, syncopated",
        'duration': 16,
        'temperature': 1.0,
        'name': 'funk_drop'
    },
    {
        'prompt': "ohhalim blues jazz, soulful, slow",
        'duration': 20,
        'temperature': 1.2,
        'name': 'blues_drop'
    },
    # ... 5개 더 추가
]

# 생성!
for i, scenario in enumerate(drop_scenarios):
    print(f"\n[{i+1}/{len(drop_scenarios)}] {scenario['name']}")

    jazz = generate_with_finetuned_model(
        model_finetuned,
        prompt=scenario['prompt'],
        duration=scenario['duration'],
        temperature=scenario['temperature'],
        output_file=f"drop_ohhalim_{i:03d}_{scenario['name']}.wav"
    )

# ZIP으로 다운로드
!zip -r ohhalim_drop_clips.zip drop_ohhalim_*.wav
files.download("ohhalim_drop_clips.zip")
```

---

## 🎛️ FL Studio 통합

### 워크플로우

```
1. FL Studio 프로젝트 열기
   - 하우스/테크노 트랙 작업 중

2. 드랍 위치 파악
   - 보통: Bar 64, 128, 192 등

3. 재즈 클립 삽입
   - Playlist 오른쪽 클릭 → Insert → Audio clip
   - 다운로드한 재즈 클립 선택
   - 드랍 시작 위치에 배치

4. 이펙트 체인
   - EQ: High-pass 100Hz (킥과 분리)
   - Reverb: Wet 20-30% (공간감)
   - Sidechain: Kick에서 (펌핑 효과)

5. Export
   - File → Export → Wave file
   - 44.1kHz, 16/24-bit

6. Rekordbox로 Import
   - DJ용 최종 트랙!
```

---

### Python 자동화 (Advanced)

```python
# FL Studio Python API (FlStudioApi)
# https://github.com/demberto/PyFLP

from pyflp import Project

# 프로젝트 열기
project = Project.load("my_track.flp")

# 드랍 위치 찾기
drop_positions = [64, 128, 192]  # Bar 번호

# 재즈 클립 삽입
for i, pos in enumerate(drop_positions):
    # 오디오 클립 추가
    clip = project.add_audio_clip(
        file=f"drop_ohhalim_{i:03d}.wav",
        position=pos * 4 * 96  # Bar → Ticks 변환 (96 ticks/beat)
    )

    # 이펙트 추가
    clip.add_effect("Fruity Reverb 2", wet=0.3)
    clip.add_effect("Fruity Parametric EQ 2")

# 저장
project.save("my_track_with_jazz.flp")
```

---

## 🐛 트러블슈팅

### 문제 1: Out of Memory (OOM)

**증상:**
```
RuntimeError: CUDA out of memory
```

**해결:**

```python
# 1. Batch size 줄이기
per_device_train_batch_size=1  # 이미 최소

# 2. Gradient accumulation 줄이기
gradient_accumulation_steps=2  # 4 → 2

# 3. LoRA rank 줄이기
lora_config = LoraConfig(
    r=4,  # 8 → 4
    lora_alpha=8,  # 16 → 8
)

# 4. 데이터 길이 줄이기
# 청크를 1초로 (원래 2초)

# 5. Gradient checkpointing
model.gradient_checkpointing_enable()
```

---

### 문제 2: Training Loss 안 떨어짐

**증상:**
```
Epoch 1: loss=2.5
Epoch 10: loss=2.4
Epoch 20: loss=2.4
...
```

**해결:**

```python
# 1. Learning rate 조정
learning_rate=5e-5  # 1e-4 → 5e-5 (더 작게)

# 2. Warmup 늘리기
warmup_steps=200  # 100 → 200

# 3. Epochs 늘리기
num_train_epochs=100  # 50 → 100

# 4. 데이터 확인
# - 너무 적은가? (최소 10개)
# - 품질이 좋은가?
# - Augmentation 필요?
```

---

### 문제 3: 생성 품질이 Base model과 차이 없음

**증상:**
- Fine-tuned model이 base와 똑같이 들림
- 당신 스타일이 반영 안 됨

**해결:**

```python
# 1. 더 많은 데이터
# 10개 → 20-50개

# 2. 더 강한 Fine-tuning
lora_config = LoraConfig(
    r=16,  # 8 → 16 (더 강력)
    lora_alpha=32,
)

# 3. Learning rate 높이기
learning_rate=2e-4  # 1e-4 → 2e-4

# 4. Epochs 늘리기
num_train_epochs=100

# 5. Audio prompt weight 높이기
weighted_styles = [
    (5.0, my_audio),  # 3.0 → 5.0
    (1.0, "jazz piano"),
]
```

---

### 문제 4: Colab 12시간 제한

**해결:**

```python
# 1. Checkpoint 저장 자동화
save_steps=100  # 자주 저장

# 2. Resume from checkpoint
trainer.train(resume_from_checkpoint=True)

# 3. 런팟 사용 ($10)
# - 시간 제한 없음
# - 언제든 재시작 가능
```

---

## 📊 성능 비교

### Base Model vs Fine-tuned

**테스트:**

```python
# Base model 생성
base_jazz = generate_with_base_model(
    prompt="Bill Evans modal jazz piano",
    duration=16
)

# Fine-tuned model 생성
finetuned_jazz = generate_with_finetuned_model(
    model_finetuned,
    prompt="ohhalim jazz piano, modal",
    duration=16
)

# 비교 청취!
# 차이점:
# - 화성 voicing
# - 리듬 패턴
# - 프레이즈 길이
# - "나다움"
```

---

## 💰 비용 & 시간 예상

### Colab 무료

```
✅ 장점:
- 비용: $0
- TPU v2-8 사용 가능
- 테스트에 충분

❌ 단점:
- 12시간 세션 제한
- 90분 idle timeout
- Fine-tuning 중단 위험
```

**예상 시간:**
- 데이터 10개: 1-2시간
- 데이터 20개: 3-4시간
- 데이터 50개: 6-8시간

---

### Colab Pro ($10/month)

```
✅ 장점:
- 24시간 세션
- V100 GPU (더 빠름)
- 안정적

❌ 단점:
- 월 $10
```

**예상 시간:**
- 데이터 10개: 30분-1시간
- 데이터 20개: 1-2시간
- 데이터 50개: 3-4시간

---

### 런팟 RTX 3060 ($10 credit)

```
✅ 장점:
- 시간 제한 없음
- 언제든 재시작
- QLoRA 충분

❌ 단점:
- 초기 설정 필요
```

**예상 시간:**
- 데이터 20개: 2-3시간
- 데이터 50개: 4-6시간

**비용:**
- RTX 3060: $0.20/hour
- 3시간 = $0.60
- 6시간 = $1.20
- → $10 크레딧으로 충분!

---

## 🎯 추천 플랜

### 당신의 상황:
- 런팟 $10 크레딧
- Colab 무료 사용 가능
- 3일 남음

### 최적 전략:

**Day 1: Colab 무료**
```
✅ 기본 생성 테스트
✅ 첫 재즈 클립 10개
✅ 작동 확인
→ 비용: $0
```

**Day 2: Colab 무료**
```
✅ Audio Injection 테스트
✅ 데이터 준비
✅ Fine-tuning 시작 (작은 데이터)
→ 비용: $0
```

**Day 3: 런팟 ($1-2)**
```
✅ 본격 Fine-tuning (큰 데이터)
✅ 3-6시간 학습
✅ "ohhalim style" 완성
→ 비용: $0.60 - $1.20
```

**남은 크레딧 ($8-9)**
```
→ 나중에 추가 실험
→ 더 큰 모델 시도
→ 더 많은 데이터
```

---

## 📝 체크리스트

### Day 1: 기본 생성
- [ ] Colab 데모 실행
- [ ] 재즈 클립 10개 생성
- [ ] ZIP 다운로드
- [ ] 품질 확인

### Day 2: 데이터 준비
- [ ] Public dataset 다운로드
- [ ] Audio Injection 테스트
- [ ] Fine-tuning 데이터 준비
- [ ] Tokenization 완료

### Day 3: Fine-tuning
- [ ] QLoRA 설정
- [ ] Training 실행
- [ ] Model 저장 & 다운로드
- [ ] 생성 테스트

### Day 4: 통합
- [ ] Fine-tuned model 로드
- [ ] 드랍용 클립 10개 생성
- [ ] FL Studio 통합
- [ ] Export & Rekordbox

---

## 🔗 참고 자료

### 공식 링크
- GitHub: https://github.com/magenta/magenta-realtime
- Paper: https://arxiv.org/abs/2508.04651
- Colab Demo: https://colab.research.google.com/github/magenta/magenta-realtime/blob/main/notebooks/Magenta_RT_Demo.ipynb
- Finetune: https://colab.research.google.com/github/magenta/magenta-realtime/blob/main/notebooks/Magenta_RT_Finetune.ipynb

### QLoRA 리소스
- QLoRA Paper: https://arxiv.org/abs/2305.14314
- PEFT Library: https://github.com/huggingface/peft
- BitsAndBytes: https://github.com/TimDettmers/bitsandbytes

### Datasets
- PiJAMA: https://github.com/CPJKU/pijama
- Jazznet: https://github.com/tosiron/jazznet

---

## 💪 다음 단계 (3일 후)

### 당신이 가질 것:
```
✅ 드랍용 재즈 클립 10-20개
✅ "ohhalim style" fine-tuned model
✅ Generation pipeline
✅ FL Studio 통합 워크플로우
```

### 추가 개선:
```
1. 더 많은 연주 녹음 (50-100개)
2. 더 정교한 Fine-tuning
3. Real-time generation 도전
4. 라이브 세션 테스트
```

---

## 🎉 최종 목표

**"나와 가상의 내가 JAM!"**

```
나: FL Studio에서 하우스 트랙 작곡
    ↓
AI: 드랍에서 "ohhalim style" 재즈 즉흥연주
    ↓
나: Export → Rekordbox → 라이브 디제잉!
    ↓
청중: "와, 이 드랍 미쳤다!" 🔥

→ 꿈 실현! 💯
```

---

**Let's go! 지금 바로 시작하세요!** 🚀

**첫 단계:** Colab 데모 열기
```
https://colab.research.google.com/github/magenta/magenta-realtime/blob/main/notebooks/Magenta_RT_Demo.ipynb
```
