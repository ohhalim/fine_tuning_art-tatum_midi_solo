# q8 실시간 Aria 경로의 LoRA 어댑터 적용 (사전 등록)

작성 2026-10-19 03시대 KST. 이슈 #1834. 학습, 다운로드, 실시간 기본값 변경, 청취 없음. `quality_claimed: false`.

## 1. 질문

- q8 실시간 모델(bf16 + `model.model` 8비트 그룹 32) 위에 기존 Aria LoRA를 얹을 수 있는지, delta 방식과 merge 방식 각각의 출력 차이·실행 비용·교체 비용

## 2. 조사 (실행 전)

- 기존 Aria LoRA 파일 형식: torch dict `{층}_{대상}.A` [r, in], `.B` [out, r]. r16, alpha 16(배율 1), 대상 `mixed_qkv`(1536→4608)·`att_proj_linear`(1536→1536) 16층, 2,359,296개(#1724 방식, `scripts/aria_lora.py`의 forward hook: out + x Aᵀ Bᵀ · alpha/r)
- 파일: `mehldau_aria/run/lora_u128.pt`(#1726 Mehldau pilot, 개발 판정 미달: 생성 유효성 1/8), `mehldau_aria/control_pool/lora_{M1,M2,C0,C1,C2}{,_tok}.pt`(#1734 CE 비교용). `wjazzd/*`, `t3_aria/cond_*`의 `adapter_*.pt`는 조건 임베딩용 가중치 한 장(`weight` [1536, k])이라 LoRA 아님
- MLX `TransformerLM` 블록 층 이름이 같아(`mixed_qkv`, `att_proj_linear`, 둘 다 bias 없음) 대상 매핑 그대로. q8에서는 대상이 `QuantizedLinear`
- 어떤 어댑터도 음악 품질·실사용 승인 없음. 이 단위는 적용 기제와 비용만

## 3. 방법 (실행 전 고정)

- 어댑터: 주 `lora_u128.pt`(M0). 교체 상대 `control_pool/lora_C0.pt`(교체 시간만)
- 두 방식(`scripts/aria_mlx_lora.py`)
  - delta: 대상 모듈(Linear 또는 QuantizedLinear)을 감싸 `base(x) + (x Aᵀ) Bᵀ · 배율`을 모델 dtype(fp32 / bf16)으로 계산. 교체는 A·B 배열만 바꿈
  - merge: float32에서 W + 배율 · B A, 모델 dtype으로 변환, q8이면 그룹 32 8비트 양자화. 교체는 32개 행렬 재병합·재양자화
- 팔: F0 fp32 base, FA fp32 + delta(기준), FM fp32 + merge, Q0 q8 base, QD q8 + delta, QM q8 + merge
- teacher forcing 입력: `mehldau_aria/data.json` **val** 녹음 첫 창 12개(1,017–1,023토큰). #1726 개발 평가에 이미 쓴 분할. #1726 test 분할은 미개봉 유지, #1811의 test 창도 쓰지 않음
- 시간(q8 세 팔, 같은 프로세스): 문맥 512에서 디코드 32스텝, 이어서 48토큰 suffix, 반복 5회, 반복마다 팔 순서 회전(#1811과 같은 방식). 교체 뒤 KV 재prefill 비용으로 Q0·QD 전체 prefill L 512·896 각 5회
- 교체 시간: delta(파일 로드 + A·B 교체) 6회, merge(재병합·재양자화·모듈 교체) 3회
- 열 압력 시작·fp32 뒤·동등성 뒤·끝 기록. GPU 잠금 `mkdir /Users/ohhalim/git_box/.gpu_lock`

## 4. 판정 기준 (실행 전 고정)

- C0 무결성(미충족 시 측정 무효 또는 구현 결함)
  - B를 0으로 둔 delta는 fp32 base와 logits 최대 절대 차이 0
  - fp32 base 반복 최대 절대 차이 0
  - 모든 팔 logits 유한
  - FA 대 FM(fp32 delta 대 merge) 위치 평균 KL ≤ 1e-6
- E1 양자화 오차 비증폭(방식별): 위치 평균 KL(FA‖Q방식) ≤ 2 × 위치 평균 KL(F0‖Q0) + 1e-4
- E2 어댑터 효과 보존(방식별): 위치 평균 KL(Q0‖Q방식) ÷ 위치 평균 KL(F0‖FA)가 0.8~1.25. 단 평균 KL(F0‖FA) < 10 × 평균 KL(F0‖Q0)이면 "효과 크기가 양자화 오차와 구분 안 됨"으로 기록만
- T1 실행 비용: 문맥 512 디코드 중앙값 QD ≤ 1.15 × Q0, QM ≤ 1.05 × Q0. suffix 48 중앙값도 같은 비율로 기록
- S1 교체: delta 교체(로드 + 교체) 중앙값 ≤ 100 ms면 블록 사이 교체 후보. merge 교체 중앙값 기록. 교체 뒤 KV는 무효이므로 교체 총비용 = 교체 + 전체 재prefill(L 512·896)로 마감 1박(469 ms)과 비교해 기록
- 권고 규칙: E1·E2·T1을 충족한 방식 중, 둘 다 충족이면 delta 교체 총비용이 merge의 0.25배 이하일 때 delta, 아니면 merge(실행 비용 없음). 하나도 충족 못 하면 "q8 위 어댑터 적용 보류"
- NLL은 기록만(어댑터 품질 판단 아님)

## 5. 범위 밖

- 어댑터 품질·연주자 특화, 생성 유효성, 청취, 학습, 실시간 producer 연결(교체 시점 계약), q4

## 6. 재현

- `scripts/rt_adapter_q8_probe.py <aria repo> <ckpt> <data json> <adapter .pt> <partner .pt> <report json>`
- 시험 `tests/test_aria_mlx_lora.py`
- 보고 `docs/experiments/rt_adapter_q8/report.json`

## 7. 결과 (2026-10-19 08:05–08:06, 실행 코드 e3090181, 1회 실행)

열 압력 시작·fp32 뒤·동등성 뒤·끝 모두 0. 판정: C0 충족, E1·E2 두 방식 모두 충족, T1은 merge만 충족(delta 디코드 1.235배), S1 delta 교체 6 ms. 권고 규칙상 **merge**.

| 비교 | 위치 평균 KL | KL p95 | KL 최대 | TV 평균 |
|---|---|---|---|---|
| F0 ‖ FA (어댑터 효과, fp32) | 0.4320 | 1.417 | 5.58 | 0.273 |
| FA ‖ FM (fp32 delta 대 merge) | 3.1e-11 | — | 4.4e-9 | 2.4e-6 |
| F0 ‖ Q0 (양자화 오차, base) | 0.000987 | 0.00239 | 0.176 | 0.0147 |
| FA ‖ QD (q8 delta) | 0.000850 | 0.00221 | 0.071 | 0.0136 |
| FA ‖ QM (q8 merge) | 0.000758 | 0.00193 | 0.099 | 0.0130 |
| Q0 ‖ QD | 0.4318 | 1.421 | 5.42 | 0.273 |
| Q0 ‖ QM | 0.4334 | 1.421 | 5.52 | 0.274 |
| QD ‖ QM | 0.001007 | 0.00253 | 0.070 | 0.0144 |

- C0: B 0 delta = base(최대 차이 0), fp32 반복 차이 0, 모든 logits 유한, FA ‖ FM 3.1e-11 ≤ 1e-6
- E1: 기준 2 × 0.000987 + 1e-4 = 0.00207. QD 0.000850, QM 0.000758 모두 기준 안
- E2: 어댑터 효과 0.432 ≥ 10 × 양자화 오차 0.00987. 보존 비율 QD 0.9997, QM 1.0032(0.8~1.25 안)
- T1: 문맥 512 디코드 중앙값 Q0 5.575 ms, QD 6.884 ms(1.235배, 기준 1.15 초과), QM 5.592 ms(1.003배). suffix 48 중앙값 Q0 35.0, QD 41.7(1.19배, 최대 107.4 ms 1회), QM 35.2(1.007배)
- S1: delta 교체 중앙값 로드 4.4 + 교체 1.6 = 6.0 ms(최대 12.9 + 8.8). merge 교체 중앙값 117.2 ms(최대 224.7, 첫 병합 142.6)
- 교체 뒤 KV 무효 → 전체 재prefill(q8): L 512 210.4 ms(QD 214.6), L 896 368.5 ms(QD 378.9). 교체 총비용 추정: delta 216~375 ms, merge 328~486 ms. 마감 1박(469 ms) 기준 merge + L 896은 초과

관측

- val NLL(기록만): F0 2.636 → FA 2.350(−0.286), Q0 2.635 → QD 2.350·QM 2.350. #1726 torch fp32 val ΔCE(녹음 평균 −0.297)와 같은 방향·비슷한 크기. torch 훅과 MLX 적용의 대응을 간접적으로 보여 주는 관측이며 판정 항목 아님
- q8 위 어댑터의 양자화 오차(0.00076~0.00085)는 base의 양자화 오차(0.00099)와 같은 크기. 어댑터가 오차를 키우지 않음
- delta는 교체가 빠르지만 매 토큰 1.235배 비용(32개 저랭크 경로 추가 연산). #1818 루프 샘플당 q8 6.42 ms 기준이면 약 7.9 ms 수준 추정(미측정)
- 교체 시점의 실제 비용은 교체 자체보다 KV 재prefill(210~369 ms)이 지배. 블록 경계에서 교체하면 그 블록 마감 여유 대부분을 씀

결론(4절 규칙): E1·E2·T1을 모두 충족한 방식은 merge. 실시간 기본은 merge 방식 후보, 교체는 재prefill 포함 1블록 비용(약 330~490 ms)으로 별도 계약 필요. delta는 교체가 잦은 실험용 후보(매 토큰 1.24배)로 기록. 어댑터 품질·청취 판단 아님.
