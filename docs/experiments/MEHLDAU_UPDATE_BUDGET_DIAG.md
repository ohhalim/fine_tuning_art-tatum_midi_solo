# 멜다우 어댑터 진단 — "학습 안 됨"이 아니라 "업데이트 8회"

작성 2026-09-27. 로컬 브랜치 `exp/mehldau-update-budget-diag` (push 안 함).
대상: `MEHLDAU_PERSONALIZATION.md`의 "뚜렷한 개인화 효과 미확인".

`mehldau_style_verified: false` · `musical_quality_verified: false` · 청취 리뷰 없음.

## 0. 결론

- 기존 run은 **optimizer update 8회**였다(문서의 64 step은 오기). 파이프라인은 학습한다. 다만 8회로는 거의 움직이지 않는다.
- 같은 run을 이어서 34회까지 돌리면 train 고정 crop CE가 −0.020(8회) → −0.051(34회)로 **업데이트 수에 따라 단조 감소**한다. 결함보다는 **학습량 부족**을 지지하는 관측이다. 다만 원인이라고 확정하지는 않는다(§5).
- 데이터 중복(18/18이 base 사전학습셋에 있음)은 **일반화 평가를 막는 제약**일 뿐, 효과가 작은 원인으로 확정할 근거는 없다.
- 기존 평가 descriptor(bar당 ~20토큰, 고정 primer)로는 이 크기의 분포 변화를 검출할 수 없다.

## 1. 읽기전용 진단 — 기존 checkpoint 직접 측정

스크립트 `scripts/diag_mehldau_adapter_readonly.py`, 원시값 `docs/experiments/mehldau_diag/readonly_diag.json`.
기존 checkpoint는 읽기만 했다.

| # | 사실 | 근거 |
|---|---|---|
| F1 | 실제 optimizer update = **8회** | 두 팔 모두 ckpt `optimizer_state_dict` step = epoch 번호(1..8). 16곡/batch 4 = 4배치, `gradient_accumulation` 기본 4 (`scripts/train_qlora.py` 476행 기본값, 408–416행 step 조건) → epoch당 1회 |
| F2 | optimizer state는 상속되지 않았다 | `train_qlora.py`는 optimizer state를 저장만 하고(650행) 로드하지 않는다. epoch1 ckpt step=1 |
| F3 | max_sequence **1024** (문서 512 오기) | ckpt `model_config.max_sequence=1024`. `apply_checkpoint_model_config`(262행)가 시작 ckpt 값을 상속 |
| F4 | 바뀐 텐서는 out_proj `lora_A/B` 12개뿐. 베이스는 동결 | start ↔ ep8 state_dict 비교 |
| F5 | 업데이트 크기 ‖ΔW‖/‖W_eff‖ ≈ **0.9–1.4%**/layer (ep8) | 양 팔 동일 규모 |
| F6 | 저장 → `load_model_with_lora` 재로드 max_abs_diff **0.0** | from_tatum ep8 |
| F7 | from_base는 새 어댑터가 아니다 | armB ckpt의 LoRA delta 노름이 이미 1.0–2.2. 기존 어댑터를 이어서 학습했다 |
| F8 | cosine `T_max` = 배치 수×epoch(32)인데 scheduler는 update마다(8회) 호출 | lr 3e-4 → 2.56e-4. 거의 감쇠하지 않았다 (`train_qlora.py` 594–595행, scheduler.step 416행) |

**logits 비교** (label smoothing 없음, dropout 없음, 고정 512 crop, ep8 vs 시작점):

| 팔 | 셋 | 토큰 | CE 시작 → ep8 | ΔCE | KL (nats/token) | top-1 일치 |
|---|---|---|---|---|---|---|
| from_base | train 16곡 | 55,808 | 2.6250 → 2.6088 | −0.016 | 0.0025 | 95.3% |
| from_base | val 2곡 | 6,144 | 2.4408 → 2.4312 | −0.010 | 0.0017 | 96.4% |
| from_tatum | train | 55,808 | 2.6193 → 2.6032 | −0.016 | 0.0033 | 94.6% |
| from_tatum | val | 6,144 | 2.4313 → 2.4232 | −0.008 | 0.0022 | 95.7% |

→ 학습은 됐지만 효과가 매우 작다. 부호는 train에서 일관되게 감소 방향이다.

### 학습 loss와 평가 loss는 서로 비교할 수 없다

| | 학습 로그 train/val (3.3 / 3.1) | eval 스크립트 val_loss (2.47) |
|---|---|---|
| label smoothing | 0.1 | 없음 |
| dropout | train 켜짐 / val 꺼짐 | 꺼짐 |
| crop | 곡당 1024 랜덤 crop 1개. **val도 `random.randint`, 시드 고정 없음** | 256 고정 crop, 꼬리 누락 (`run_mehldau_adapter_eval.py` 60행) |

val은 2곡에서 랜덤 crop을 하나씩 뽑으므로 epoch 간 잡음이 효과(0.01–0.02)보다 크다.
"train loss가 3.28 → 3.31로 올랐다"는 관측은 crop 잡음으로 설명할 수 있다. 추세의 근거로 쓸 수 없다.

## 2. E1 — 한 연속 run 안에서 update 수만 비교

`scripts/run_mehldau_update_budget_diag.py --mode budget`, 원시값 `mehldau_diag/update_budget_v1_report.json`.

- 시작점 armB ep8(= 기존 from_base 시작점). `train_qlora`의 `MidiDataset`, loss(LS 0.1), LoRA wrapper, batch 4, accumulation 4, lr 3e-4, seed 42를 그대로 쓴다
- **snapshot은 모두 같은 run에서 뜬다.** effective batch, crop 순서, seed가 같고 update 수만 다르다
- cosine 길이는 이 run의 계획 epoch(64) 기준이다. update 8의 lr(2.89e-4)이 기존 run(2.56e-4)과 다르므로 **기존 run의 재현이 아니다**
- 평가: LS·dropout 없는 CE. train 16곡 × 곡당 최대 2개 고정 512 crop(16,384토큰), val 2곡 고정 crop(6,144토큰)
- 사전 등록 기준(탐색적): 예산 안에서 train 고정 crop CE가 0.1 이상 떨어지는가. 도달하지 못해도 파이프라인 결함으로 확정하지 않는다

실측: 학습 wall **307.8 s** (5분 예산을 epoch 경계 확인 때문에 7.8 s 초과), optimizer update **34** (Adam state step 34), 학습 토큰 556,512, CPU.

| update | lr | wall s | train CE | Δ | val CE | Δ |
|---|---|---|---|---|---|---|
| 0 | 3.00e-4 | 0 | 2.6386 | — | 2.4408 | — |
| 8 | 2.89e-4 | 72.7 | 2.6189 | −0.020 | 2.4311 | −0.010 |
| 16 | 2.56e-4 | 144.5 | 2.6051 | −0.034 | 2.4254 | −0.015 |
| 24 | 2.08e-4 | 217.3 | 2.5959 | −0.043 | 2.4235 | −0.017 |
| 32 | 1.50e-4 | 289.7 | 2.5889 | −0.050 | 2.4221 | −0.019 |
| 34 | 1.36e-4 | 307.8 | 2.5875 | **−0.051** | 2.4218 | −0.019 |

- update 8의 Δ(−0.020)가 기존 ep8의 Δ(−0.016, 다른 crop 집합)와 같은 규모다
- train CE는 update 수에 따라 단조 감소했다. 기울기는 lr 감쇠와 함께 줄었다
- **탐색적 기준 0.1은 예산 안에서 달성하지 못했다** (−0.051)
- val은 base가 이미 본 곡이라 일반화 해석은 하지 않는다

## 3. 경로 검증 — 고정 배치 overfit

`--mode overfit`, 원시값 `mehldau_diag/fixed_batch_overfit_v1_report.json`.
train 4곡의 앞 1024토큰(4,092 target 토큰)을 고정 배치로 쓰고, accumulation 1로 24회 update했다. 57.6 s 소요.

| update | 0 | 4 | 8 | 12 | 16 | 20 | 24 |
|---|---|---|---|---|---|---|---|
| 고정 배치 CE | 2.6408 | 2.6210 | 2.6031 | 2.5856 | 2.5686 | 2.5524 | 2.5367 |

→ gradient가 LoRA까지 전달되고 loss가 update당 약 0.004씩 선형으로 내려간다. **경로 검증으로만** 해석한다.
같은 데이터를 매번 보는 조건이다. 스타일·일반화 결과가 아니다. 이후 val CE는 2.4238이었다.

## 4. E3 — 생성 출력이 바뀌는가

프라이머 `outputs/chord_ab/ii_V_I.mid` (sha1 `06531ce9…`, gitignore라 해시로 기록), 32토큰.
seed 42/100/200 × 4마디 = 12 take. take마다 같은 seed를 쓰고, snapshot 0 / 8 / 34를 비교했다.
MIDI: `outputs/mehldau_diag/update_budget_v1/gen_update{000,008,034}_seed42.mid`.

| 비교 | 토큰 동일률 | 완전 동일 take | 첫 분기 위치 (take별) |
|---|---|---|---|
| 0 vs 8 | 0.53 | 1/12 | 91, 29, 36, 73, 48, 없음, 8, 7, 57, 33, 17, 11 |
| 0 vs 34 | 0.30 | 0/12 | 6, 29, 17, 2, 27, 3, 8, 6, 31, 25, 17, 3 |

- update가 늘수록 샘플 경로가 **더 일찍 갈라진다**. 출력이 바뀐다는 것까지는 사실이다
- 자기회귀 샘플링은 한 토큰만 달라도 이후 전체가 달라진다. 그래서 동일률은 **변화의 크기나 방향(멜다우다움)을 재지 못한다**
- 들리는 멜다우 스타일 솔로는 **입증되지 않았다**
- 기존 평가(`run_mehldau_adapter_eval.py`)의 take는 평균 ~20토큰(40 take에 938토큰)이다. 매 마디를 같은 32토큰 primer에서 새로 생성한다. KL 0.003 규모의 변화를 이 descriptor 중앙값으로 검출하기는 어렵다. 그 report에는 primer 경로도 기록되지 않았다

## 5. 판단과 한계

| 원인 후보 | 상태 |
|---|---|
| 업데이트 예산 부족 (8회 × lr ~3e-4, Adam은 update당 원소를 ~lr만큼 움직임) | **지지.** 같은 run에서 update 수에 비례해 CE 감소 (§2) |
| 저장·로드 결함 | **기각.** 재로드 diff 0.0 |
| gradient 미전달 / 베이스 오염 | **기각.** LoRA 12개만 변경, 고정 배치에서 CE 하락 |
| 학습/평가 loss 불일치 | **측정 체계 차이로 설명됨.** LS, dropout, 랜덤 crop |
| 데이터 중복 | **원인 확정 불가.** 일반화 평가만 막는다 |
| out_proj 전용 r=16 LoRA의 용량 한계 | **미검증.** 34회로는 포화에 닿지 않았다 |

**독립 검증이 불가능한 범위**
- 새 곡 일반화: 멜다우 18곡 전부가 base 사전학습셋에 있다
- 스타일 성공: 블라인드 청취를 하지 않았다
- 34 update 이후의 추세: 선형 외삽은 하지 않는다

## 6. 다음 실험 (각각 한 질문)

1. ~~업데이트 예산 확장~~ → **완료** (`MEHLDAU_STYLE_SHIFT.md` V2). MPS 512 update에서 train CE −0.199로 포화했고, update 32 이후 우도 특화 기준을 충족했다
2. ~~train_qlora 기본동작 정정~~ → **완료** (커밋 `7db5e7df`). 기존 동작은 `--scheduler_steps legacy_batches --val_crop_seed -1`로 재현하고, D0/D1 스크립트에 명시했다
3. **평가 민감도** — 고정 primer로 긴 연속 생성(마디 이어붙이기)을 하고, 분포 지표(pitch-class·IOI 히스토그램 거리)를 base vs adapter로 잰다. 그다음 블라인드 청취
4. **일반화** — base가 보지 않은 자료가 필요하다(`MEHLDAU_PERSONALIZATION.md` §8). 본인 연주 녹음이 이 프로젝트 목표와 가장 잘 맞는다

## 7. 재현

```sh
PY=/Users/ohhalim/git_box/fine_tuning_art-tatum_midi_solo/.venv/bin/python
$PY scripts/diag_mehldau_adapter_readonly.py > readonly_diag.json   # 읽기전용, ~25 s
$PY scripts/run_mehldau_update_budget_diag.py --primer outputs/chord_ab/ii_V_I.mid \
    --output-dir outputs/mehldau_diag/update_budget_v1 --budget-seconds 300
$PY scripts/run_mehldau_update_budget_diag.py --mode overfit --overfit-updates 24 \
    --budget-seconds 120 --output-dir outputs/mehldau_diag/fixed_batch_overfit_v1
$PY -m unittest tests.test_mehldau_update_budget
```
`run_mehldau_update_budget_diag.py`는 비어 있지 않은 출력 디렉터리를 거부한다. 읽기전용 진단은 stdout으로만 쓴다. 기존 `outputs/mehldau_lora/`는 읽기만 한다.
