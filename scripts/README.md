# 파인튜닝 평가 스크립트 모음

파인튜닝한 Magenta RealTime 모델을 평가하는 스크립트들이야.

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

### 1. Training Loss 평가

파인튜닝 로그에서 Loss를 분석해서 과적합 여부 확인

```bash
python scripts/evaluate_loss.py \
    --log_file ./ohhalim-jazz-style/trainer_state.json \
    --output_dir ./evaluation
```

**출력**:
- Loss 그래프 (`evaluation/loss_curves.png`)
- 과적합 여부 판정
- 학습 안정성 분석

---

### 2. FAD (Frechet Audio Distance) 계산

생성된 오디오가 실제 재즈와 얼마나 유사한지 측정

```bash
python scripts/calculate_fad.py \
    --generated_dir ./generated_audio \
    --reference_dir ./reference_jazz \
    --output_dir ./evaluation
```

**출력**:
- FAD 점수 (낮을수록 좋음)
- 유사도 판정

---

### 3. Spectral Analysis (주파수 분석)

주파수 특성 비교

```bash
python scripts/spectral_analysis.py \
    --generated_dir ./generated_audio \
    --reference_dir ./reference_jazz \
    --output_dir ./evaluation
```

**출력**:
- 스펙트럼 비교 그래프 (`evaluation/spectral_comparison.png`)
- 스펙트로그램 비교 (`evaluation/spectrogram_comparison.png`)
- Spectral Centroid, Rolloff, ZCR 비교

---

### 4. Rhythm Analysis (리듬 분석)

재즈 리듬 특성 분석

```bash
python scripts/rhythm_analysis.py \
    --generated_dir ./generated_audio \
    --reference_dir ./reference_jazz \
    --output_dir ./evaluation
```

**출력**:
- 리듬 비교 그래프 (`evaluation/rhythm_comparison.png`)
- Tempo, Syncopation, Beat Strength 비교
- 재즈다운 리듬인지 판정

---

### 5. Chord Analysis (코드 진행 분석)

재즈 화성 분석

```bash
python scripts/chord_analysis.py \
    --generated_dir ./generated_audio \
    --reference_dir ./reference_jazz \
    --output_dir ./evaluation
```

**출력**:
- 코드 진행 그래프 (`evaluation/chord_comparison.png`)
- 화성 복잡도, 재즈 코드 사용 여부
- ii-V-I, Dominant 7th 패턴 감지

---

### 6. A/B Test

베이스 모델 vs 파인튜닝 모델 블라인드 테스트

**6-1. 테스트 페어 생성**

```bash
python scripts/ab_test.py create \
    --base_dir ./base_model_audio \
    --finetuned_dir ./finetuned_audio \
    --output_dir ./ab_test \
    --num_pairs 10
```

**6-2. 투표하기**

생성된 `ab_test/voting_sheet.csv` 파일을 열어:
1. 각 페어의 A, B 파일을 들어봐
2. "Vote" 열에 A 또는 B 입력
3. 저장

**6-3. 결과 분석**

```bash
python scripts/ab_test.py analyze \
    --voting_sheet ./ab_test/voting_sheet.csv \
    --metadata ./ab_test/ab_test_metadata.json
```

**6-4. 인터랙티브 테스트 (터미널에서 바로 투표)**

```bash
python scripts/ab_test.py interactive \
    --output_dir ./ab_test
```

---

## 전체 평가 워크플로우

### Step 1: 파인튜닝 완료 확인

```bash
# Loss 체크
python scripts/evaluate_loss.py \
    --log_file ./ohhalim-jazz-style/trainer_state.json
```

**통과 기준**: Validation Loss < 0.4

---

### Step 2: 오디오 생성

```python
# 베이스 모델로 10개 생성 → ./base_audio/
# 파인튜닝 모델로 10개 생성 → ./finetuned_audio/
# 레퍼런스 재즈 준비 → ./reference_jazz/
```

---

### Step 3: 정량적 평가

```bash
# FAD 계산
python scripts/calculate_fad.py \
    --generated_dir ./finetuned_audio \
    --reference_dir ./reference_jazz

# 스펙트럼 분석
python scripts/spectral_analysis.py \
    --generated_dir ./finetuned_audio \
    --reference_dir ./reference_jazz

# 리듬 분석
python scripts/rhythm_analysis.py \
    --generated_dir ./finetuned_audio \
    --reference_dir ./reference_jazz

# 코드 분석
python scripts/chord_analysis.py \
    --generated_dir ./finetuned_audio \
    --reference_dir ./reference_jazz
```

**통과 기준**:
- FAD < 15.0
- Spectral 유사도 > 70%
- Syncopation > 0.3

---

### Step 4: A/B 테스트

```bash
# 페어 생성
python scripts/ab_test.py create \
    --base_dir ./base_audio \
    --finetuned_dir ./finetuned_audio \
    --num_pairs 10

# 인터랙티브 투표
python scripts/ab_test.py interactive \
    --output_dir ./ab_test
```

**통과 기준**: 파인튜닝 승률 > 60%

---

### Step 5: 실전 테스트

FL Studio에서 드랍 섹션에 넣어보고 DJ 세트에서 사용해봐!

---

## 빠른 평가 (5분)

시간 없으면 이것만 해:

```bash
# 1. Loss 체크
python scripts/evaluate_loss.py --log_file ./ohhalim-jazz-style/trainer_state.json

# 2. A/B 테스트 (5개만)
python scripts/ab_test.py create --base_dir ./base_audio --finetuned_dir ./finetuned_audio --num_pairs 5
python scripts/ab_test.py interactive --output_dir ./ab_test
```

---

## 출력 파일 정리

모든 결과는 `./evaluation/` 폴더에 저장돼:

```
evaluation/
├── loss_curves.png              # Loss 그래프
├── fad_score.txt                # FAD 점수
├── spectral_comparison.png      # 스펙트럼 비교
├── spectrogram_comparison.png   # 스펙트로그램
├── rhythm_comparison.png        # 리듬 비교
├── chord_comparison.png         # 코드 비교
└── ab_test_results.txt          # A/B 테스트 결과
```

---

## 문제 해결

### "No module named 'librosa'"

```bash
pip install librosa
```

### "FAD 계산이 너무 오래 걸려요"

TensorFlow 없으면 librosa 특징으로 대체돼서 좀 느려. 괜찮으면 그냥 기다려.

### "그래프가 안 보여요"

터미널 환경이면 PNG 파일로 저장되니까 `evaluation/` 폴더 확인해봐.

---

## 다음 단계

평가 완료했으면:

✅ **성공** → FL Studio 통합, DJ 세트에서 사용!

🟡 **보통** → 하이퍼파라미터 조정 후 재학습

❌ **실패** → 학습 데이터 점검, 설정 재검토

---

행운을 빌어! 🎹✨
