# Phase 4: 평가 및 개선 📊

**목표**: 훈련된 모델의 품질을 객관적/주관적으로 평가하고 개선합니다.

**예상 시간**: 3-5일
**난이도**: ⭐⭐⭐☆☆

---

## 📋 체크리스트

- [ ] 객관적 메트릭 계산
- [ ] 주관적 품질 평가 (직접 듣기)
- [ ] 체크포인트 비교
- [ ] 개선 방향 도출
- [ ] 필요시 재훈련

---

## 1. 객관적 메트릭

### TatumFlow 메트릭 실행

```bash
python scripts/phase4_evaluate_model.py \
  --checkpoint checkpoints/production/best.pt \
  --test_dir data/art_tatum_midi/test \
  --output results/metrics.json
```

### 평가 메트릭

#### 1. Pitch Class KL Divergence
**측정**: 화성적 유사도

- **낮을수록 좋음** (<0.3 우수)
- 생성 vs 원본의 피치 클래스 분포 비교

#### 2. PCTM Cosine Similarity
**측정**: 화성 전환 패턴

- **높을수록 좋음** (>0.7 우수)
- Pitch Class Transition Matrix 비교

#### 3. Note Density
**측정**: 음표 밀도 (notes/second)

- Art Tatum: ~8-12 notes/sec
- 생성: 비슷해야 함

#### 4. Average IOI (Inter-Onset Interval)
**측정**: 리듬 패턴

- Art Tatum: ~100-150ms
- 너무 짧으면 불가능, 너무 길면 지루

#### 5. Unique Pitches
**측정**: 음역 다양성

- Art Tatum: 50-70 unique pitches
- 너무 적으면 단조로움

#### 6. Polyphony Rate
**측정**: 화음 비율

- Art Tatum: ~60-80% (자주 화음)
- Solo 재즈는 높아야 함

#### 7. Rhythmic Entropy
**측정**: 리듬 복잡도

- 높을수록 다양함
- Art Tatum: 높음 (즉흥적)

### 결과 예시

```json
{
  "pitch_class_kl": 0.234,
  "pctm_similarity": 0.782,
  "note_density": 9.2,
  "avg_ioi_ms": 125.3,
  "unique_pitches": 58,
  "polyphony_rate": 0.68,
  "rhythmic_entropy": 3.45
}
```

**해석**:
- ✅ 화성 유사도 우수 (KL=0.234)
- ✅ 리듬 패턴 유사 (IOI=125ms)
- ⚠️  폴리포니 약간 낮음 (68% vs 75% 목표)

---

## 2. 주관적 평가

### 직접 듣기

```bash
# 10개 샘플 생성
python scripts/generate_music.py \
  --checkpoint checkpoints/production/best.pt \
  --num_samples 10 \
  --output_dir outputs/evaluation

# MIDI → MP3 일괄 변환
for f in outputs/evaluation/*.mid; do
  python scripts/phase5_midi_to_mp3.py --input $f --output ${f%.mid}.mp3
done

# 재생
mpg123 outputs/evaluation/*.mp3
```

### 평가 기준

#### 1. Musical Coherence (음악적 일관성)
- [ ] 5초 이상 coherent?
- [ ] 갑작스런 단절 없음?
- [ ] 음악적 흐름이 자연스러운가?

#### 2. Jazz Idioms (재즈 어법)
- [ ] 스윙 리듬이 있나?
- [ ] 즉흥적 느낌?
- [ ] Blues scale 사용?

#### 3. Art Tatum Style (아트 테이텀 스타일)
- [ ] 빠른 패시지?
- [ ] 화려한 아르페지오?
- [ ] Stride piano 느낌?

#### 4. Technical Quality (기술적 품질)
- [ ] 잘못된 음 없음?
- [ ] 리듬 정확함?
- [ ] 템포 일정함?

### 점수표

| 항목 | 점수 (1-5) | 비고 |
|------|-----------|------|
| Coherence | 4 | 대부분 자연스러움 |
| Jazz Idioms | 3 | 스윙은 있으나 단조로움 |
| Tatum Style | 3 | 속도는 비슷, 화려함 부족 |
| Technical | 4 | 기술적 오류 거의 없음 |
| **Overall** | **3.5** | **양호, 개선 여지 있음** |

---

## 3. 체크포인트 비교

### 여러 체크포인트 테스트

```bash
for ckpt in checkpoints/production/epoch_*.pt; do
  python scripts/phase4_evaluate_model.py \
    --checkpoint $ckpt \
    --test_dir data/art_tatum_midi/test \
    --output results/$(basename $ckpt .pt)_metrics.json
done

# 결과 비교
python scripts/compare_checkpoints.py --results_dir results/
```

**발견**:
- Epoch 50이 best validation loss
- Epoch 60이 주관적으로 더 좋음
- Epoch 70부터 오버피팅 시작

**결론**: Epoch 60 선택!

---

## 4. 개선 방향 도출

### A. 메트릭 기반

**문제**: Polyphony rate 낮음 (60% vs 75%)
**원인**: 화음 생성 부족
**해결**: Theory loss 가중치 증가

```yaml
training:
  theory_loss_weight: 0.2  # 0.1 → 0.2
```

### B. 청취 기반

**문제**: 단조로운 리듬 패턴
**원인**: 데이터 부족 or 모델 크기
**해결**: 데이터 증강 또는 더 긴 훈련

### C. 스타일 기반

**문제**: Art Tatum의 화려함 부족
**원인**: 빠른 패시지 학습 부족
**해결**: Note density 높은 샘플 추가

---

## 5. 재훈련 (필요시)

### 개선 실험

```yaml
# config_v2.yaml
model:
  num_layers: 16  # 12 → 16 (더 크게)

training:
  epochs: 150  # 100 → 150 (더 길게)
  theory_loss_weight: 0.2

data:
  augment: true  # 데이터 증강 활성화
```

```bash
python scripts/phase3_train_production.py --config config_v2.yaml
```

**비교**:
- V1 (기본): PCTM=0.78, Overall=3.5
- V2 (개선): PCTM=0.85, Overall=4.2 ← 향상!

---

## 🎓 학습 내용

### 객관적 vs 주관적 평가

**객관적** (숫자):
- ✅ 재현 가능
- ✅ 비교 쉬움
- ❌ 음악성 완벽히 측정 못함

**주관적** (듣기):
- ✅ 음악성 직접 판단
- ❌ 개인차 있음
- ❌ 비교 어려움

**최선**: 둘 다 사용!

### Validation Loss vs 실제 품질

**주의**: 낮은 Loss ≠ 좋은 음악

**예**:
- Model A: Val Loss 1.5, 지루함
- Model B: Val Loss 1.8, 음악적으로 훌륭

**이유**: Loss는 통계적 유사도만 측정

**해결**: 주관적 평가 필수!

### 오버피팅 감지

**신호**:
- Train Loss ↓, Val Loss ↑
- Validation loss 5 epochs 정체
- 생성 샘플이 훈련 데이터와 너무 유사

**대응**:
- Early stopping
- Best checkpoint 사용
- Regularization 증가

---

## ✅ Phase 4 완료 체크

- [ ] 객관적 메트릭 모두 계산
- [ ] 10+ 샘플 직접 들어봄
- [ ] 최고 체크포인트 선정
- [ ] 개선 방향 문서화
- [ ] 필요시 재훈련 완료

---

## 다음 단계

**Phase 5: 생성 및 배포**로 이동:
```bash
cat docs/phase5_generation.md
```

**잘 하셨습니다! 이제 멋진 음악을 생성해봅시다! 🎼**
