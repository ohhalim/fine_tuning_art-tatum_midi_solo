# Phase 1: 데이터 준비 🎵

**목표**: Art Tatum MIDI 데이터를 수집하고 훈련에 적합한 형태로 전처리합니다.

**예상 시간**: 3-5일
**난이도**: ⭐⭐⭐☆☆

---

## 📋 체크리스트

- [ ] MIDI 데이터 수집 (Art Tatum 50+ 곡)
- [ ] 데이터 품질 검증
- [ ] Train/Val/Test 분할 (80/10/10)
- [ ] 토크나이저 테스트
- [ ] 데이터 통계 분석

---

## 1. MIDI 데이터 수집

### 🎹 추천 데이터 소스

#### A. The Lakh MIDI Dataset (무료)
- **URL**: https://colinraffel.com/projects/lmd/
- **설명**: 17만+ MIDI 파일 (다양한 아티스트)
- **Art Tatum 포함 여부**: 일부 포함 (검색 필요)

**다운로드 방법**:
```bash
# 전체 데이터셋 (25GB)
wget http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz
tar -xzvf lmd_full.tar.gz

# Art Tatum만 추출
python scripts/phase1_collect_data.py --source lmd --artist "Art Tatum"
```

#### B. Bitmidi (무료)
- **URL**: https://bitmidi.com
- **설명**: 10만+ MIDI 파일 (웹 크롤링 가능)

**검색 방법**:
```bash
# Art Tatum 검색 및 다운로드
python scripts/phase1_collect_data.py --source bitmidi --artist "Art Tatum" --min-files 50
```

#### C. 직접 수집 (MIDI 변환)
YouTube에서 Art Tatum 연주 → Audio → MIDI 변환

**도구**:
- **basic-pitch** (Spotify 개발): 오디오 → MIDI 변환
  ```bash
  pip install basic-pitch
  basic-pitch output_dir input_audio.mp3
  ```

**주의**: 자동 변환은 완벽하지 않습니다. 수동 MIDI가 최고 품질입니다.

#### D. Musescore (무료 악보 → MIDI)
- **URL**: https://musescore.com
- Art Tatum 악보 다운로드 → MIDI 내보내기

---

### 📁 데이터 구조

```
data/
├── art_tatum_midi/
│   ├── raw/                    # 원본 MIDI
│   │   ├── tiger_rag.mid
│   │   ├── tea_for_two.mid
│   │   └── ...
│   ├── train/                  # 훈련 데이터 (80%)
│   ├── val/                    # 검증 데이터 (10%)
│   ├── test/                   # 테스트 데이터 (10%)
│   └── metadata.json           # 데이터 통계
```

---

## 2. 데이터 품질 검증

### 품질 기준

✅ **좋은 MIDI**:
- 피아노 연주만 포함 (드럼/베이스 없음)
- 10초 이상 길이
- 템포 정보 포함
- 노트 벨로시티 다양함

❌ **나쁜 MIDI**:
- 너무 짧음 (< 5초)
- 여러 악기 섞임
- 손상된 파일 (읽기 실패)
- 단순 반복만 (음악성 없음)

### 검증 스크립트 실행

```bash
python scripts/phase1_prepare_dataset.py \
  --input_dir data/art_tatum_midi/raw \
  --output_dir data/art_tatum_midi \
  --min_duration 10 \
  --filter_piano_only
```

**출력 예시**:
```
처리 중: 120 파일
✅ 통과: 87 파일
❌ 제외: 33 파일
  - 너무 짧음: 15
  - 손상됨: 8
  - 악기 불일치: 10

훈련 세트: 70 파일
검증 세트: 9 파일
테스트 세트: 8 파일
```

---

## 3. 토크나이저 테스트

### TatumFlow 토크나이저 이해하기

**토큰 타입**:
- `TRACK_START` (ID: 0)
- `TRACK_END` (ID: 1)
- `PAD` (ID: 2)
- `MASK` (ID: 3)
- `CHUNK_START` (ID: 4)
- `TIME` (ID: 5-505): 10ms 단위 시간
- `NOTE_ON` (ID: 506-593): 88 piano keys
- `NOTE_OFF` (ID: 594-681): 88 piano keys
- `VEL` (ID: 682-713): 32 velocity bins

**전체 어휘 크기**: 2048

### 테스트 코드

```python
from src.tatumflow import MIDITokenizer

# 토크나이저 초기화
tokenizer = MIDITokenizer(vocab_size=2048)

# MIDI → Tokens
tokens = tokenizer.encode('data/art_tatum_midi/raw/tiger_rag.mid')
print(f"토큰 개수: {len(tokens)}")
print(f"첫 10개 토큰: {tokens[:10]}")

# Tokens → MIDI (복원)
tokenizer.decode(tokens, 'outputs/reconstructed.mid')
print("재구성 완료: outputs/reconstructed.mid")
```

**예상 출력**:
```
토큰 개수: 4523
첫 10개 토큰: [0, 4, 45, 506, 705, 45, 594, 58, 508, 710]
재구성 완료: outputs/reconstructed.mid
```

**원본 vs 재구성 비교**:
```bash
# FluidSynth로 들어보기
fluidsynth -a alsa -m alsa_seq -l -i soundfont.sf2 outputs/reconstructed.mid
```

---

## 4. 데이터 통계 분석

### 분석 스크립트 실행

```bash
python scripts/phase1_analyze_data.py \
  --data_dir data/art_tatum_midi/train \
  --output metadata.json
```

### 분석 항목

1. **파일 개수**
   - 훈련: 70개
   - 검증: 9개
   - 테스트: 8개

2. **총 길이**
   - 평균: 2분 34초
   - 최소: 45초
   - 최대: 6분 12초
   - 총합: 3시간 12분

3. **음역대**
   - 최저음: A0 (MIDI 21)
   - 최고음: C8 (MIDI 108)
   - 평균 음역: 4 옥타브

4. **템포**
   - 평균: 180 BPM (빠른 스윙)
   - 범위: 120-240 BPM

5. **노트 밀도**
   - 평균: 8.5 notes/second
   - Art Tatum는 매우 빠름!

6. **폴리포니**
   - 평균 동시 노트: 2.8개
   - 최대: 8개 (화음)

### 시각화

```python
import matplotlib.pyplot as plt
import json

with open('data/art_tatum_midi/metadata.json') as f:
    stats = json.load(f)

# 길이 분포
plt.figure(figsize=(10, 4))
plt.hist(stats['durations'], bins=20)
plt.xlabel('길이 (초)')
plt.ylabel('파일 개수')
plt.title('MIDI 파일 길이 분포')
plt.savefig('outputs/duration_distribution.png')

# 음역대 분포
plt.figure(figsize=(10, 4))
plt.hist(stats['pitch_range'], bins=30)
plt.xlabel('MIDI 노트 번호')
plt.ylabel('빈도')
plt.title('사용된 피치 분포')
plt.savefig('outputs/pitch_distribution.png')
```

---

## 5. 데이터 증강 (선택)

### 증강 기법

1. **Pitch Shifting** (이조)
   - 원본을 -2 ~ +2 semitones 이조
   - 5배 데이터 증강

2. **Tempo Scaling**
   - 90% ~ 110% 템포 변경
   - 리듬 패턴 다양화

3. **Velocity Randomization**
   - 벨로시티에 ±10% 노이즈
   - 다이나믹 다양화

**주의**: 너무 많은 증강은 overfitting을 유발할 수 있습니다.

### 증강 실행

```bash
python scripts/phase1_augment_data.py \
  --input_dir data/art_tatum_midi/train \
  --output_dir data/art_tatum_midi/train_augmented \
  --pitch_shift_range 2 \
  --tempo_scale_range 0.1
```

---

## 🎓 학습 내용

### MIDI 파일 구조

MIDI는 **악보** (음표 정보)를 디지털로 저장합니다.

**구성 요소**:
- **Track**: 악기별 트랙 (피아노, 드럼, etc.)
- **Note On/Off**: 음 시작/종료
- **Velocity**: 음의 세기 (0-127)
- **Timing**: 절대 시간 또는 상대 시간 (ticks)
- **Tempo**: BPM (Beats Per Minute)

**vs Audio**:
- Audio (MP3, WAV): 파형 (waveform) - 연속 신호
- MIDI: 이벤트 시퀀스 (discrete events) - 악보

딥러닝으로는 **MIDI가 더 다루기 쉽습니다**!

### 토크나이제이션

MIDI를 딥러닝 모델에 넣으려면 **토큰**으로 변환해야 합니다.

**TatumFlow 방식 (Aria 기반)**:
```
MIDI Events:
  t=0ms: Note On (pitch=60, vel=80)
  t=500ms: Note Off (pitch=60)
  t=500ms: Note On (pitch=64, vel=85)

Tokens:
  [TRACK_START, TIME(0), NOTE_ON(60), VEL(80),
   TIME(50), NOTE_OFF(60), NOTE_ON(64), VEL(85), ...]
```

**장점**:
- 10ms 정밀도 (매우 정확)
- 절대 타이밍 (상대 타이밍보다 간단)
- Chunk 기반 (긴 곡도 처리 가능)

### Train/Val/Test 분할

왜 3개로 나눌까요?

- **Train (80%)**: 모델이 학습하는 데이터
- **Val (10%)**: 하이퍼파라미터 튜닝, Early Stopping
- **Test (10%)**: 최종 평가 (절대 훈련에 사용 금지!)

**중요**: Test 데이터는 **한 번만** 사용합니다. 여러 번 보면 오버피팅!

---

## 🚨 문제 해결

### 문제 1: MIDI 파일이 부족함

**해결책**:
1. 다른 아티스트 추가 (Oscar Peterson, Bud Powell 등)
2. Audio → MIDI 변환 (basic-pitch 사용)
3. 데이터 증강 활성화

최소 권장: **30개 파일** (작지만 proof-of-concept 가능)

### 문제 2: pretty_midi 읽기 실패

**증상**:
```
ValueError: Unknown chunk type: xxxx
```

**해결**:
- MIDI 파일 손상 → 다시 다운로드
- Type 0 MIDI로 변환:
  ```bash
  python -m mido.ports -t 0 input.mid -o output.mid
  ```

### 문제 3: 데이터 불균형

**증상**:
- 어떤 곡은 30초, 어떤 곡은 10분

**해결**:
- 긴 곡을 chunk로 분할 (2048 tokens씩)
- 짧은 곡 제외 (`--min_duration` 설정)

---

## ✅ Phase 1 완료 체크

다음 항목이 모두 ✅ 이면 Phase 2로 진행하세요:

- [ ] 최소 30개 MIDI 파일 수집
- [ ] Train/Val/Test 분할 완료
- [ ] `metadata.json` 생성됨
- [ ] 토크나이저 테스트 성공 (재구성 MIDI 들어봄)
- [ ] 데이터 통계 확인 (평균 길이, 음역대 등)

---

## 다음 단계

**Phase 2: 소규모 실험**으로 이동:
```bash
cat docs/phase2_experiment.md
```

**잘 하셨습니다! 데이터 준비 완료! 🎵**

이제 작은 모델로 빠르게 실험해봅시다!
