# 로컬 MPS 학습 가이드 (Apple Silicon)

당신 맥(arm64, macOS 26.4)의 GPU(Metal/MPS)로 Music Transformer를 학습하는 방법.
어시스턴트 샌드박스에서는 MPS가 가려져 보이지 않으므로, **아래 명령은 당신이
맥 터미널에서 직접** 실행합니다. 코드 경로는 어시스턴트가 CPU로 검증 완료.

---

## 0. 무엇이 바뀌었나 (이번에 수정한 것)

| 파일 | 변경 |
|---|---|
| `music_transformer/utilities/device.py` | **MPS 지원 추가.** `get_device()` 우선순위: CUDA → MPS → CPU. `mps_device()` 추가. `FORCE_CPU=1` 환경변수로 MPS 끄기 가능. CUDA 경로는 그대로(하위호환). |
| `archive/scripts/train_qlora.py` | **`--device {auto,cuda,mps,cpu}` 인자 추가.** 선택한 device를 모델 내부 `get_device()`와 자동 일치시킴(중요: forward의 causal mask가 내부에서 `get_device()`를 부름). 또 스크립트 이동으로 깨져 있던 import 경로(레포 루트 탐색, `checkpoint_utils`) 수정. |
| `scripts/mps_smoke_test.py` | **신규.** MPS에서 forward+backward+step이 되는지 검증하고 CPU 대비 속도 비교. |

`rpr.py`(커스텀 상대위치 어텐션)는 device-agnostic하게 작성돼 있어 MPS 호환
가능성이 높습니다. 단 **확실한 검증은 아래 1단계 스모크 테스트로** 하세요.

---

## 1. 먼저: MPS 호환성 스모크 테스트 (필수, 30초)

```bash
cd /Users/ohhalim/git_box/fine_tuning_art-tatum_midi_solo
./.venv/bin/python scripts/mps_smoke_test.py
```

**해석:**
- `VERDICT: MPS is COMPATIBLE for rpr=True` → MPS에서 크래시 없이 학습 가능. **호환성 통과.**
- `at least one op is not MPS-compatible` → 그 op이 MPS 미지원. 두 선택지:
  - (a) 그냥 CPU로 학습 (`--device cpu`) — 13.4M 모델이라 probe/소규모는 CPU도 견딜 만함.
  - (b) 폴백 켜고 재시도:
    ```bash
    PYTORCH_ENABLE_MPS_FALLBACK=1 ./.venv/bin/python scripts/mps_smoke_test.py
    ```
    미지원 op만 CPU로 떨어뜨림(느리지만 동작). 학습도 같은 env로 실행.

### ⚠️ 중요: MPS가 항상 빠른 건 아님 (속도 크로스오버)

실측 결과(2026-07, 당신 맥): **작은 shape에서는 CPU가 MPS보다 빠릅니다.**

| shape | MPS | CPU | 승자 |
|---|---|---|---|
| batch 2, seq 256 (스모크) | 1.23s | 0.48s | **CPU ~2.5배** |

이유: 13.4M은 작은 모델이라 GPU 커널 실행 오버헤드 + CPU↔GPU 전송 비용이
실제 연산량보다 큽니다. 배치·시퀀스가 커지면(batch×seq²) MPS가 이기기 시작합니다.

**그래서 device는 실험 규모별로 실측해서 정하세요:**
```bash
./.venv/bin/python scripts/mps_smoke_test.py --scale
```
이 명령이 batch/seq를 키워가며 MPS vs CPU steps/sec을 재고 각 shape의 승자를
출력합니다. **그 표에서 당신 학습 shape의 승자를 골라 `--device`에 쓰세요.**

- **D0-(a) 16조각 probe → `--device cpu`** (데이터 작아 CPU가 빠름)
- **D0-(b) 전체 코퍼스 (batch 8, seq 512) → `--scale` 결과 보고 결정**

---

## 2. 실험 D0-(a): 16조각 probe 재현 (baseline)

현재 상태를 그대로 재현해 baseline perplexity/loss를 확보합니다.

```bash
./.venv/bin/python archive/scripts/train_qlora.py \
  --data_dir ./data/roles/lead/tokenized \
  --output_dir ./checkpoints/d0a_probe_repro \
  --device cpu \
  --epochs 20 --batch_size 4 --max_sequence 512 \
  --gradient_accumulation 1 --num_workers 0 \
  --train_full_model
```
> probe는 16조각뿐이라 CPU가 MPS보다 빠릅니다(위 크로스오버 참고). `--device cpu` 사용.

- `--train_full_model`: from-scratch full 학습(랜덤 base + LoRA-only 아님 — 그건 안 배웠던 실패 모드).
- probe가 16조각뿐이라 몇 epoch 안에 train loss가 급락하면 **과적합**입니다 — 그게 정상이고, 요점은 "파이프라인이 학습 신호를 만드는가" 확인.
- 어시스턴트 CPU 검증값(1 epoch): Train 5.74 / Val 5.42 → 시작점 참고.

## 3. 실험 D0-(b): 전체 코퍼스 투입 (핵심 대조)

2,703개 non-Brad 코퍼스를 학습 포맷으로 토크나이즈한 뒤 동일 아키텍처로 학습.
이게 "데이터 부족 vs base 한계"를 가르는 실험입니다.

```bash
# 1) 전체 코퍼스 토크나이즈 (CPU 작업, GPU 무관)
./.venv/bin/python scripts/build_jazz_training_manifests.py   # 매니페스트 생성
#    (실제 토크나이즈 스크립트/인자는 레포 파이프라인에 맞춰 어시스턴트와 확정)

# 2) 학습 (MPS)
./.venv/bin/python archive/scripts/train_qlora.py \
  --data_dir ./data/jazz_processed \
  --output_dir ./checkpoints/d0b_full_corpus \
  --device mps \
  --epochs 5 --batch_size 8 --max_sequence 512 \
  --gradient_accumulation 4 --num_workers 0 \
  --train_full_model
```

> 전체 코퍼스 학습은 MPS로도 수십 분~수 시간 걸릴 수 있습니다. 오래 걸리면
> 이 잡만 RunPod으로 옮기는 게 합리적(어시스턴트가 스크립트 그대로 던짐).

---

## 4. 문제 해결

| 증상 | 원인 / 조치 |
|---|---|
| `--device mps requested but MPS is not available` | macOS 14+ / Apple Silicon / MPS torch 빌드 확인. 1단계 스모크로 진단. |
| `RuntimeError: ... not implemented for 'MPS'` | 그 op 미지원. `PYTORCH_ENABLE_MPS_FALLBACK=1` 붙여 재실행하거나 `--device cpu`. |
| device mismatch (`cpu` vs `mps`) 에러 | 이번 패치로 해결됨(모델 내부 `get_device()`와 학습 device를 일치시킴). 재발 시 어시스턴트에 알릴 것. |
| loss가 `nan` | lr 낮추기(`--lr 1e-4`), 또는 MPS의 fp 정밀도 이슈면 `--device cpu`로 교차확인. |
| 느림 | batch_size↑(메모리 허용선까지), max_sequence를 512로 유지, num_workers는 MPS에서 0 권장. |

---

## 5. 다양성 지표 (다음 단계)

loss/perplexity만으로는 "보이싱 다양성 붕괴"를 못 잡습니다. 학습 후 생성 샘플에
대해 아래를 재는 스크립트를 어시스턴트가 추가 예정:
- 고유 pitch-class-set(보이싱) 수 / 엔트로피
- 화성 진행 n-gram 반복률
- note density 분포

이 지표로 D0-(a) vs (b)를 비교하면 "데이터를 더 넣으면 다양성이 회복되는가"에
정량 답이 나옵니다.
