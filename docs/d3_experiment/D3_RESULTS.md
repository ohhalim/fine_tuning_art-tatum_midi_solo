# D3 결과: 단선율의 진짜 원인은 lead-only 프라이머의 텍스처 잠김

> Issue #1458 / 로드맵 L4. 실행: 2026-07-12~13, 로컬 MPS. 판정 기준은 각 하위 실험 전 사전 등록.

## 요약

4개 가설을 순차 검증했다. 데이터(H-D3a)와 디코딩(H-D3b)은 필요조건이지만 불충분했고,
샘플링 보수화(H-D3d)와 사전학습 연장(H-D3c)은 기각됐다. **범인은 생성 조건(conditioning
프라이머)이었다** — lead 역할용(오른손 단선율) conditioning 프라이머가 모델을 그 텍스처에
가두고 있었다. 무프라이머 생성은 원본 두손 소스의 화성 밀도를 거의 그대로 재현한다.

## 실험 순서와 판정

### 1차: H-D3a 측정 + Arm F 처방 (#1458, 별도 PR #1459)

두손 소스(1.657) → lead 분리 데이터(1.323) → 생성(1.305)의 증거 사슬로 H-D3a를 확인했으나,
두손 데이터로 재 FT(Arm F)해도 voicing 불변(1.290) — **실패**. 원인: 두손 Mehldau는
이미 사전학습 코퍼스에 포함돼 있어 FT가 사실상 no-op(val 3.08→3.03, 거의 안 움직임).

### 2차-a: H-D3d 샘플링 스윕 — 기각

D1 Arm D 체크포인트, `--grammar_mask`, 8샘플씩 3개 설정:

| 설정 | voicing_size | 밀도 /s | pooled 고유(8샘플) |
|---|---|---|---|
| baseline (temp 1.0) | 1.305 | 2.58 | — |
| temp 0.8 | 1.163 | 3.16 | 65 |
| temp 0.9 + top_p 0.95 | 1.233 | 4.98 | 80 |
| top_k 40 | 1.238 | 3.67 | 78 |

보수적 샘플링은 voicing을 **오히려 낮췄다** (분포를 뾰족하게 만들수록 안전한 단음 선택에
수렴). 기각.

### 2차-b: H-D3c 사전학습 연장 — 기각

Arm B 체크포인트에서 전체 코퍼스로 8ep 추가 학습(총 16ep 상당), val loss 3.15→**2.998**
(계속 하강 — 아직 수렴 안 함). 그 위에서 (lead 프라이머로) 24샘플 생성:

| | voicing_size | 밀도 /s | pooled 고유 |
|---|---|---|---|
| 8ep base (기준) | 1.305 | 2.58 | 149 |
| **16ep base** | **1.306** | 3.07 | 158 |

val loss는 유의미하게 떨어졌는데 voicing은 **완전히 불변**. 사전학습 연장은 화음 문제와
무관함이 확인됨 — 화음 생성 능력 자체는 이미 있었다는 뜻 (다음 실험이 이를 증명).

### 2차-c: H-D3e 프라이머 텍스처 잠김 — **입증**

같은 16ep-연장 base에서 조건만 바꿔 생성 (24 vs 24, `--grammar_mask` 동일):

| 조건 | voicing_size | 밀도 /s | pooled 고유 보이싱 | pooled 엔트로피 |
|---|---|---|---|---|
| lead conditioning 프라이머 (기준) | 1.305 | 2.58 | 149 | 4.981 |
| **무프라이머 (unconditional)** | **1.618** | **5.99** | **253** | **5.825** |
| (참고) 원본 두손 소스 상한 | 1.657 | 5.52 | 254\* | — |

\*소스 상한은 D0에서 측정한 값(18곡 pooled)으로 표본 구성이 다르나 규모가 같은 수준.

무프라이머 생성이 voicing·밀도·다양성 전 지표에서 소스 상한 근처까지 도달했다.
**lead 역할용 conditioning 프라이머(오른손 단선율, split_pitch=60으로 자른 conditioning.mid)가
모델을 그 텍스처에 고정시키고 있었다** — base는 화음을 칠 줄 알지만, 이 프라이머를 보면
"오른손 단선율 모드"로 들어간다.

## 결론

단선율 문제는 데이터도, 디코딩도, 학습 부족도 아니었다. **conditioning 프라이머의 설계
결함**이었다. lead-only 프라이머는 Stage B 파이프라인(사람이 코드/가이드를 넣고 솔로를
받는 워크플로)을 위해 만들어진 것인데, 그 텍스처 자체가 "단선율만 치라"는 강한 신호로
작동한 것.

## 다음 방향 (중요 — 단순히 무프라이머를 쓰는 게 답이 아님)

무프라이머는 진단 도구였지 처방이 아니다. 로드맵 L5(조건 제어)가 최종 목표(co-improvisation)에
필수이므로, "프라이머 없이 생성"은 채택 불가. 대신:

1. **D3-후속**: conditioning 프라이머 포맷 자체를 재설계 — 오른손만이 아니라 두손(또는 화음
   포함) conditioning으로 프라이머를 구성했을 때도 화성 밀도가 유지되는지 검증.
2. 대안: conditioning은 코드 심볼/구조 토큰으로만 주고 MIDI 텍스처 프라이머는 배제 —
   Stage B의 chord-aware 경로(로드맵 L5)와 자연스럽게 합류.
3. 이 발견은 **L5(조건 제어) 설계에 직접적인 제약**을 부과한다: 조건이 "무엇을 연주할지"뿐
   아니라 "어떤 텍스처로 연주할지"까지 암묵적으로 강하게 고정한다는 것을 확인했으므로,
   앞으로의 conditioning 설계는 이 잠김 효과를 의도적으로 통제해야 한다.

## 재현

```bash
# 연장 사전학습
python scripts/train_qlora.py --checkpoint outputs/d0_experiment/armB_full2777/ckpt/checkpoint_epoch8.pt \
  --train_full_model --data_dir ./data/jazz_full --epochs 8 --lr 3e-4 --seed 42 \
  --output_dir outputs/d3_experiment/armB_ext16/ckpt

# 무프라이머 생성 (conditioning_midi 생략)
python scripts/generate.py --lora_path outputs/d3_experiment/armB_ext16/ckpt \
  --num_samples 24 --length 1024 --temperature 1.0 --grammar_mask \
  --output outputs/d3_experiment/primer_none_24/samples
```

원자료: `outputs/d3_experiment/`, 비교 JSON 사본은 이 디렉터리.
