# D1 실험 설계: 2단계 커리큘럼 — 넓은 사전학습이 스타일 FT의 다양성 붕괴를 막는가

> 상태: **설계 (사전 등록)** — 실행 전에 판정 기준을 고정한다.
> 선행: D0 (#1444, 결과 `docs/d0_experiment/D0_RESULTS.md`)
> 작성: 2026-07-12

---

## 1. 배경과 질문

D0의 판정: 스타일 파인튜닝의 다양성 붕괴 원인은 base 표현력이 아니라 **데이터 레짐**이었다.
16조각 직접 학습(Arm A)은 자기 데이터 상한의 93%에 도달하고도 val ppl ≈157에 머물렀고,
전체 2,777조각 학습(Arm B)은 ppl ≈24 + 생성 다양성 회복을 보였다.

그러나 **원래 목표는 스타일 서브셋(Brad Mehldau lead, 16조각) 스타일의 생성**이다.
D0는 "좁은 데이터를 직접 학습하면 붕괴한다"까지만 증명했다. D1의 질문:

> **넓은 코퍼스로 사전학습한 모델을 좁은 스타일 서브셋으로 파인튜닝하면,
> from-scratch 스타일 학습(Arm A)에서 잃었던 다양성을 유지하면서 스타일 적응이 되는가?**

부가 질문 2개:

- **FT 방식**: full FT vs LoRA-only(adapter) 중 어느 쪽이 다양성 보존↔스타일 적응 트레이드오프가 좋은가?
- **증강**: 전조(transposition) 증강(16 → ~192 시퀀스)이 FT 붕괴를 추가로 완화하는가? (선택 팔)

## 2. 가설 (사전 등록)

- **H-D1 (주)**: 사전학습→FT 팔(C, D)은 생성 풀링 고유 보이싱에서 from-scratch 팔(A=81)을
  유의미하게 상회하고, 사전학습 팔(B=145)의 상당 부분을 유지한다.
- **H-D1a**: LoRA-only FT(D)는 base가 동결되므로 full FT(C)보다 다양성 보존이 좋다.
  대신 스타일 적응(probe val loss 하강 폭)은 C가 클 수 있다.
- **H-D1b (선택 팔)**: 전조 증강 FT(E)는 비증강 full FT(C)보다 다양성 보존이 좋다.

## 3. 팔 구성

| 팔 | 초기화 | 학습 데이터 | 모드 | 상태 |
|---|---|---|---|---|
| **A** | random | probe 16 | full (from-scratch) | ✅ D0 완료 — 붕괴 대조군 |
| **B** | random | full 2,777 | full (from-scratch) | ✅ D0 완료 — 사전학습 참조(FT 없음) |
| **C** | **B epoch8 ckpt** | probe 16 | **full FT** (lr↓) | 신규 |
| **D** | **B epoch8 ckpt** | probe 16 | **LoRA-only** (base 동결) | 신규 |
| **E** (선택) | **B epoch8 ckpt** | probe 16 ×12키 전조 (~192) | full FT (lr↓) | 신규 |

고정 요소 (모든 신규 팔 공통, D0와 동일):

- 아키텍처: 체크포인트의 `model_config` 자동 적용 (6L/8H/512d/1024ff, max_seq 1024, rpr, LoRA r16/α32)
- batch 4, grad_accum(D0 값), seed 42, device mps
- 생성: primer `data/roles/lead/000000/conditioning.mid`, 24샘플, length 1024, temperature 1.0, seed 42
- 측정: `scripts/diversity_metrics.py`, D0의 A/B 수치를 고정 참조로 사용

팔별 차등 (사전 고정):

| | C (full FT) | D (LoRA) | E (증강 full FT) |
|---|---|---|---|
| lr | **3e-5** (D0의 1/10 — FT 관례) | **3e-4** (어댑터는 높게) | 3e-5 |
| epochs | 4 | 8 | 4 |
| 근거 | 좁은 데이터 과적합이 빠르므로(D0에서 6ep 반등) 짧게 | 학습 파라미터가 적어 여유 | C와 동일 조건, 데이터만 증강 |

## 4. 실행 절차 (검증된 커맨드)

사전 확인 완료 사항:

- `train_qlora.py --checkpoint`는 D0 Arm B 체크포인트(LoRA-래퍼 포함 full state, `model_config` 포함)를
  그대로 로드한다 (train_qlora.py:522-563에서 확인).
- `--checkpoint` + `--train_full_model` = full FT / `--checkpoint`만 = adapter(LoRA-only) 모드.
- `generate.py`는 `--lora_path` 디렉터리에서 최신 `checkpoint_epoch*.pt`를 자동 탐지, seed 기본 42.
- 전조 증강은 `archive/scripts/prepare_role_dataset.py --transpose_all_keys`로 지원됨(토큰화 포함).

공통 프리앰블:

```bash
cd /Users/ohhalim/git_box/fine_tuning_art-tatum_midi_solo
export PYTHONPATH="music_transformer:music_transformer/third_party:scripts"
PY=.venv/bin/python
CKPT_B=outputs/d0_experiment/armB_full2777/ckpt/checkpoint_epoch8.pt
```

### 4.1 Arm C — full FT

```bash
$PY scripts/train_qlora.py \
  --checkpoint "$CKPT_B" --train_full_model \
  --data_dir ./data/roles/lead/tokenized \
  --epochs 4 --batch_size 4 --lr 3e-5 --seed 42 --device mps \
  --output_dir outputs/d1_experiment/armC_fullft/ckpt

$PY scripts/generate.py \
  --lora_path outputs/d1_experiment/armC_fullft/ckpt \
  --conditioning_midi ./data/roles/lead/000000/conditioning.mid \
  --num_samples 24 --length 1024 --temperature 1.0 \
  --output outputs/d1_experiment/armC_fullft/samples
```

### 4.2 Arm D — LoRA-only FT (`--train_full_model` 없음)

```bash
$PY scripts/train_qlora.py \
  --checkpoint "$CKPT_B" \
  --data_dir ./data/roles/lead/tokenized \
  --epochs 8 --batch_size 4 --lr 3e-4 --seed 42 --device mps \
  --output_dir outputs/d1_experiment/armD_lora/ckpt

$PY scripts/generate.py \
  --lora_path outputs/d1_experiment/armD_lora/ckpt \
  --conditioning_midi ./data/roles/lead/000000/conditioning.mid \
  --num_samples 24 --length 1024 --temperature 1.0 \
  --output outputs/d1_experiment/armD_lora/samples
```

### 4.3 Arm E (선택) — 전조 증강 데이터 생성 후 full FT

```bash
# 증강 데이터셋 생성 (16 → ~192 train 시퀀스, val도 12배)
$PY archive/scripts/prepare_role_dataset.py \
  --input_dir "./midi_dataset/midi/studio/Brad Mehldau" \
  --output_dir ./data/roles_aug12 --role lead \
  --transpose_all_keys --seed 42

# 학습·생성은 Arm C와 동일하되 data_dir/출력 경로만 교체
$PY scripts/train_qlora.py \
  --checkpoint "$CKPT_B" --train_full_model \
  --data_dir ./data/roles_aug12/lead/tokenized \
  --epochs 4 --batch_size 4 --lr 3e-5 --seed 42 --device mps \
  --output_dir outputs/d1_experiment/armE_aug12/ckpt

$PY scripts/generate.py \
  --lora_path outputs/d1_experiment/armE_aug12/ckpt \
  --conditioning_midi ./data/roles/lead/000000/conditioning.mid \
  --num_samples 24 --length 1024 --temperature 1.0 \
  --output outputs/d1_experiment/armE_aug12/samples
```

### 4.4 측정

```bash
for ARM in armC_fullft armD_lora armE_aug12; do
  [ -d "outputs/d1_experiment/$ARM/samples" ] || continue
  $PY scripts/diversity_metrics.py \
    --npy_dir "outputs/d1_experiment/$ARM/samples" \
    --compare_dir outputs/d0_experiment/armA_probe16/samples \
    --output "outputs/d1_experiment/d1_${ARM}_vs_armA.json"
done
```

예상 소요(MPS, D0 실측 기준): 학습은 probe 데이터라 epoch당 ~4스텝(분 단위),
생성이 지배적 — 24샘플 × 팔당 ≈ 12–15분. C+D 기준 총 1시간 이내, E 포함 시 +30분.

## 5. 판정 기준 (사전 등록 — 실행 전 고정)

### 주 판정: 생성 다양성 (24샘플 풀링, D0 고정 참조 A=81 / B=145)

| 결과 | pooled_unique_voicings | 해석 |
|---|---|---|
| **강한 성공** | ≥ 116 (B의 80%) | 커리큘럼이 붕괴를 실질적으로 방지 |
| **부분 성공** | 105–115 (A의 1.3배 이상) | 완화하지만 손실 존재 → 증강/짧은 FT 검토 |
| **실패(귀무)** | < 105 | FT 자체가 붕괴를 유발 → 데이터 확장이 유일한 길 |

보조 지표: `pooled_voicing_entropy_bits` ≥ 4.85 (A=4.61, B=5.00)이면 강한 성공 뒷받침.
노트 밀도(A=0.79, B=1.77/s)가 B 수준을 유지하는지도 함께 보고.

### 스타일 적응 게이트 (다양성만으로 성공 선언 금지)

Arm A/C/D/E는 **같은 probe val 2파일**로 val loss를 계산하므로 직접 비교 가능:

1. **전이 이득**: C/D의 best val loss < **5.0585** (Arm A의 best) — 사전학습이 스타일
   서브셋 자체의 예측도 개선함을 확인.
2. **적응 발생**: 학습 중 val loss가 epoch 1 대비 하강해야 FT가 실제로 일어난 것.
   (하강 없이 다양성만 높으면 "FT가 안 된 Arm B"와 구별 불가 → 판정 무효)

### 트레이드오프 판정 (C vs D)

C와 D 모두 주 판정 통과 시: `pooled_unique_voicings`가 높은 쪽이 다양성 우위,
best val loss가 낮은 쪽이 적응 우위. 둘이 갈리면 **다양성 우위를 채택** (프로젝트
목표가 붕괴 방지이므로).

## 6. 리스크와 한계 (미리 명시)

1. **probe val이 2파일** → val loss 노이즈 큼. 그래서 주 판정은 생성 다양성이고
   val loss는 게이트로만 사용.
2. **Arm B 체크포인트의 LoRA 가중치가 이미 학습된 상태** (full-model 모드는 LoRA 모듈도
   함께 학습함). Arm D는 이 LoRA에서 이어서 학습하게 됨 — "순수 fresh LoRA"가 아니라는
   점을 결과 해석 시 명시. (재초기화는 코드 수정 필요 → 범위 밖)
3. **removed-pitch 디코드 버그 의심** (D0에서 관측) — 모든 팔에 동일하게 적용되므로
   비교는 유효하나, 절대값 해석에 주의. 별도 이슈로 추적.
4. **단일 시드(42), 24샘플** — D0와 같은 한계. 효과 크기가 작으면(±10% 이내) 판정 유보하고
   시드 반복을 후속으로.
5. **lr/epoch은 관례값으로 고정** — 스윕은 범위 밖. 실패 시 "이 설정에서 실패"로만 기록.
6. **망각(forgetting) 측정 생략** — FT 후 full-corpus val loss 평가는 eval-only 경로가
   없어 범위 밖. 생성 다양성이 프록시 역할.

## 7. 산출물

- `outputs/d1_experiment/arm{C,D,E}_*/ckpt/` — epoch별 체크포인트 (train/val loss 포함)
- `outputs/d1_experiment/arm{C,D,E}_*/samples/` — 팔당 24 MIDI
- `outputs/d1_experiment/d1_arm*_vs_armA.json` — 다양성 비교
- 결과 문서: `docs/d1_experiment/D1_RESULTS.md` (판정 기준 표에 실측값 대입)

## 8. 실행 체크리스트

- [ ] 하니스 `scripts/run_d1_experiment.sh` 작성 (D0 하니스와 동일한 STAGE 구조: train/gen/measure)
- [ ] Arm C 학습 + 생성
- [ ] Arm D 학습 + 생성
- [ ] 측정 (C vs A, D vs A) + 판정 기준 표 대입
- [ ] (선택) Arm E: 증강 데이터셋 생성 → 학습 → 생성 → 측정
- [ ] `D1_RESULTS.md` 작성 + PR
