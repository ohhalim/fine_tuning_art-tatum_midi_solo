# Aria 조건 adapter 단일 배치 smoke (사전 등록)

작성 2026-10-08. 아스트라 결정. 새 학습 성과, 코드 제어, 스타일 적응을 검증하는 실험이 아니다. 실행 가능성만 본다. `musically_verified: false`.

## 질문 (하나)
이 맥에서, 원본을 바꾸지 않은 Aria 체크포인트에 작은 코드 조건 경로를 붙여 배치 하나로 forward, backward, optimizer 1 step, 저장·재로드를 할 수 있는가.

## 한 줄 정리 (실행 전)
- **백엔드:** 저장소 `aria` venv의 torch 2.14.1, MPS 단일 장치, fp32(체크포인트 dtype). CPU 장시간 대체, 다운로드, GPU 비용 없음
- **수정 파일:** 저장소 안 새 파일만 만든다(`scripts/aria_cond_contract.py`, `scripts/aria_cond_smoke.py`, 시험). upstream `aria` 코드, 공식 `train.py`, 체크포인트는 바꾸지 않는다
- **조건 시각 계약:** 아래 표. 입력 위치 i는 tokens[0..i]의 prefix 시각에서 울리는 계획 코드를 받는다

## 확인한 사실 (실행 전)
- 로컬 `ckpt/model-gen.safetensors`: 키 148개, 전부 F32, **파라미터 658,538,496개**. 설정 `medium`: d_model 1536, 16층, 24헤드, vocab 17,727
  - 앞선 문서들은 Aria를 "LLaMA 3.2 1B 구조"라고만 적었다. 이 체크포인트의 실제 수는 6.6억 개다
- 설정의 `grad_checkpoint: true`는 `training` 모드에서만 켜진다. 이 smoke는 eval 모드로 돌린다(dropout 없음)
- 토큰 순서: `(piano, 음높이, velocity)` → `(onset, 5초 구간 안 ms)` → `(dur, ms)`. 5초 구간이 넘어가면 그 음 앞에 `<T>`가 온다. 토크나이저는 앞 무음을 지운다

## 조건 시각 계약 (`scripts/aria_cond_contract.py`)
코드 계획은 미리 알려진 외부 조건이다. 위치 i의 hidden이 토큰 i+1을 예측하므로, 위치 i는 tokens[0..i]만 보고 정한 시각의 코드를 받는다.

| 위치 i의 입력 토큰 | 드러내는 것 | 위치 i의 prefix 시각 |
|---|---|---|
| prefix, `<S>` | 없음 | 0 |
| `(piano, p, v)` | 다음 음의 음높이·velocity | 그대로 |
| `(onset, x)` | 이 음의 onset | 5000 × (지금까지 `<T>` 수) + x |
| `(dur, d)` | 이 음의 길이 | 그대로(이 음의 onset) |
| `<T>` | 5초 경계를 지남 | max(이전, 5000 × `<T>` 수) |
| `<E>` 등 | 시간 정보 없음 | 그대로 |

- 음의 onset은 그 음보다 앞 토큰의 조건이 되지 않는다. 코드가 두 음 사이에서 바뀌면, 새 코드는 경계를 넘은 음의 onset 토큰부터 들어간다
- **한계(경계 첫 음 지연):** 토큰 순서가 음높이 → onset이므로, 코드가 바뀐 뒤 첫 음의 **음높이를 예측하는 위치는 아직 이전 코드를 받는다.** 누출은 막지만, 실시간으로 바뀐 코드가 첫 음에 전달되는지는 검증하지 않는다. 전환 시험과 onset 기준 조건 설계는 후속이다
- 계획 밖 구간은 코드 없음(0 벡터)이다. 코드나 라벨을 목표 MIDI에서 뽑지 않는다
- 시험 4개(`tests/test_aria_cond_contract.py`): 표대로 시각 계산, 코드 변경 경계, `<T>` 경계, 미래 토큰을 바꿔도 앞 위치 조건이 그대로인지(누출 검사), 계획 밖 = 0

## 조건 경로 (probe, 최종 구조 아님)
- 계획 코드의 12차원 chroma(근음 + quality의 코드음, `chord_label.QUALITIES`)를 학습 가능한 `Linear(12 → 1536, bias 없음, 0 초기화)`로 투영한다. 그 결과를 **마지막(16번째) transformer block 입력 hidden에 더한다**(forward pre-hook)
- base 파라미터는 전부 `requires_grad=False`다. 전체를 `no_grad`로 감싸지 않는다. gradient는 마지막 block과 출력층을 지나 adapter로 흐른다
- 늦게 넣는 이유: 구조를 가장 적게 바꾸는 실행 검사다. 음악적 효과는 보장하지 않는다

## 자료 경계
- 실제 자료(악보 3곡)는 `pipeline_test` 자격만 있다. A_Foggy_Day 앞 128토큰으로 **배치 정렬과 forward dry-run만** 한다. backward와 optimizer는 돌리지 않는다
- optimizer 1 step은 새로 만든 짧은 합성 fixture로 한다
  - 120 BPM 8분음표, Cmaj7(0–2초) → Cm7(2–4초) → Cmaj7(4–6초)
  - 코드음 아르페지오 24음이며 실제 곡을 흉내 내지 않는다
  - `<T>` 경계를 하나 포함한다
- adapter는 smoke 산출물이다. 배포하거나 실험 모델로 다시 쓰지 않는다

## 고정 설정
- batch 1, 길이 최대 128토큰, fixture 1개, optimizer 정확히 1 step
- Adam lr 1e-3 하나만 쓴다. rank, hidden, 학습률은 탐색하지 않는다. trainable은 chord projection 하나뿐이다(LoRA, base 학습 병행 없음)
- seed 0, eval 모드

## 완료 확인 (모두 기록)
1. MPS 소규모 forward(16토큰): 값이 유한하고, 같은 입력의 CPU logits와 차이를 기록한다
2. adapter가 0일 때 hook 있는 logits = hook 없는 base logits(합성 fixture, 실제 자료 둘 다)
3. loss 유한, adapter gradient 유한·0 아님, base 파라미터 grad 없음, optimizer 안 base 파라미터 0개
4. 1 step 뒤 adapter가 바뀜, base는 그대로: 체크포인트 파일 sha256 전후, 파라미터 바이트 sha256 전후
5. 저장·재로드 뒤 logits가 같음(fp32 허용오차 1e-6)
6. update 뒤 조건을 바꾸면 출력이 바뀌는지는 **배선 확인으로만** 적는다. loss 감소와 조건 따름은 성공 조건이 아니다

## 자원 기록과 중단 기준
- 시작 전과 구간마다: `kern.memorystatus_vm_pressure_level`(1 정상, 2 경고, 4 위험), `memory_pressure`의 free %
- 구간(load, forward, backward, step, save/reload)마다: wall 시간, `torch.mps.current_allocated_memory`와 `driver_allocated_memory`, 프로세스 RSS(현재·최대)
  - 통합 메모리라 이 값들을 더하지 않는다. RSS 하나를 전체 사용량으로 보지 않는다
- 즉시 중단: OOM, 유한하지 않은 값, MPS 미지원 연산(`PYTORCH_ENABLE_MPS_FALLBACK` 끔), base 변경, 누출 검사 실패, pressure level 4, 단일 실행 wall 15분 초과
- 자동 반복이나 범위 확대는 없다. 128토큰이 메모리 때문에 실패하면 64토큰으로 한 번만 다시 하고 따로 기록한다

## 결과 형식
가능 또는 불가. 병목과 환경 증거를 붙인다. 가능하더라도 13곡이나 3곡 조건 학습으로 넓히지 않는다. 권리와 학습 자격 문제는 이 smoke로 풀리지 않는다.

## 실행 기록 (실패 포함, `aria_cond_smoke/`)
- **1회차 — 중단.** MPS 16토큰 forward에서 장치 불일치 오류가 났다(`apply_rotary_emb`, mps:0과 cpu)
  - 원인은 프로브 순서다. CPU forward를 먼저 돌려 rotary 표 `freqs_cis`가 CPU에 캐시됐다. 이 표는 buffer가 아니라 속성이라 `model.to(mps)`로 옮겨지지 않는다
  - MPS 미지원 연산이 아니다. 장치를 옮긴 뒤 이 캐시를 비우도록 프로브만 고쳤다. upstream 코드는 그대로다
- **2회차 — 중단.** 합성 fixture의 "adapter 0 = base" 검사에서 차이 1.5e-5가 나왔다
  - 같은 검사를 실제 자료로 했을 때는 0.0이었다. 합성 쪽만 base를 `no_grad`로, hook 쪽을 grad 켬 상태로 계산해 서로 다른 모드끼리 비교했다
  - 비교를 같은 모드(둘 다 `no_grad`)로 고쳤다. 모드 간 차이는 정보 항목으로 따로 남겼다
  - 0 adapter는 hidden에 정확히 0을 더한다. 남은 차이는 autograd 모드에 따른 MPS 연산 경로 차이로 보지만, 어느 kernel인지는 확인하지 않았다
- **3회차 — 전 항목 통과.** 아래가 이 실행의 값이다. 범위 확대와 64토큰 축소는 없었다

## 결과 (3회차, `aria_cond_smoke/result_run3.json`)
**판정: 가능.** 이 맥(32 GB, MPS)에서 원본 Aria(6.6억 파라미터, fp32) 고정 + 12→1536 조건 projection으로 단일 배치 forward, backward, optimizer 1 step, 저장·재로드가 돈다.

| 확인 | 값 |
|---|---|
| MPS 16토큰 forward | 유한. 같은 입력 CPU logits와 최대 차이 6.5e-5 |
| 악보 정렬(A_Foggy_Day 앞 128토큰) | 앞 무음 231 ms 제거 뒤 첫 onset 0, 128위치 모두 계획 코드 있음 |
| adapter 0 = base(악보, fixture) | 둘 다 최대 차이 0.0(같은 `no_grad` 모드) |
| grad 모드 대 `no_grad`(adapter 0) | 최대 차이 1.5e-5. 정보 항목 |
| loss | 1.0765, 유한 |
| adapter gradient | 유한, norm 2.55 |
| base grad / optimizer 안 base 파라미터 | 없음 / 0개(optimizer 파라미터 1개) |
| 1 step 뒤 adapter 변화 | 최대 0.001(Adam lr 1e-3의 첫 step 크기) |
| 저장·재로드 logits | 최대 차이 0.0 |
| base 불변 | 파라미터 바이트 sha256 전후 같음, 체크포인트 파일 sha256 전후 같음 |
| 배선(정보 항목) | update 뒤 계획을 maj7↔m7로 바꾸면 logits 최대 1.86 달라짐. 같은 fixture loss 1.0765 → 0.9227 |

- 배선 값과 loss 변화는 성공 조건이 아니다. 같은 fixture로 한 step 학습한 뒤 그 fixture의 loss가 내려간 것을 관측했을 뿐이다(한 step 뒤 loss가 오를 수도 있다). 일반화나 코드 따름의 증거가 아니다

### 자원
| 구간 | wall(초) | RSS 현재 / 최대(GiB) | MPS 할당 / driver(GiB) | pressure level | free % |
|---|---|---|---|---|---|
| 체크포인트 해시 | 1.438 | 0.23 / 0.23 | 0.00 / 0.00 | 1 | 90% |
| CPU 로드 | 2.238 | 2.68 / 5.13 | 0.00 / 0.00 | 1 | 88% |
| 16토큰 forward(CPU) | 0.128 | 2.69 / 5.13 | 0.00 / 0.00 | 1 | 88% |
| MPS로 이동 | 0.268 | 2.12 / 5.13 | 2.45 / 3.01 | 1 | 78% |
| 16토큰 forward(MPS) | 0.148 | 2.24 / 5.13 | 2.46 / 3.02 | 1 | 77% |
| 파라미터 해시(전) | 1.744 | 2.28 / 5.13 | 2.46 / 3.02 | 1 | 79% |
| 악보 128토큰 forward | 0.039 | 2.29 / 5.13 | 2.47 / 3.02 | 1 | 78% |
| fixture forward(76토큰) | 0.379 | 2.30 / 5.13 | 2.49 / 3.03 | 1 | 78% |
| fixture backward | 0.665 | 2.31 / 5.13 | 2.47 / 3.03 | 1 | 78% |
| optimizer 1 step | 0.024 | 2.42 / 5.13 | 2.47 / 3.02 | 1 | 89% |
| 저장·재로드 | 0.019 | 2.42 / 5.13 | 2.49 / 3.03 | 1 | 77% |
| 파라미터·파일 해시(후) | 3.352 | 2.42 / 5.13 | 2.49 / 3.03 | 1 | 88% |
- 통합 메모리라 RSS와 MPS 할당을 더하지 않는다
- RSS 최대 5.13 GiB는 CPU 로드 구간이다. state dict와 모델이 함께 CPU에 있던 때다
- MPS 할당 약 2.45–2.49 GiB는 구간이 끝난 시점의 할당치다. 보장된 최대치(peak)가 아니다. fp32 가중치(6.6억 × 4바이트 ≈ 2.45 GiB)와 거의 같다
- pressure level은 내내 1(정상)이었다. 측정 구간 wall 합계는 10.4초다(import 제외)

## 해석 범위
- 이것은 "이 경로가 이 맥에서 돈다"까지다. base를 고정했고 gradient는 마지막 block과 출력층만 지나므로 backward가 가볍다. 다음은 **재지 않았다**
  - 더 앞쪽에 넣는 주입, LoRA, base 학습, 긴 시퀀스, 배치 2 이상의 메모리
  - 여러 step을 돌렸을 때 안정성
- 코드 제어, 학습 성과, 스타일 적응은 검증하지 않았다. 사용자 개인화 학습도 하지 않았다
- adapter 파일은 저장소 밖(`/Users/ohhalim/git_box/t3_aria/cond_smoke/adapter_smoke_only.pt`)에 두었다. smoke 산출물이며 다시 쓰지 않는다
- 13곡 또는 3곡 조건 학습으로 넓히지 않는다. 권리와 학습 자격 문제는 그대로다
