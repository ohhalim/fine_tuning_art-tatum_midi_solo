# 왼손 조건 adapter 최소 학습 (#1760) — 사전 등록

작성 2026-10-16. 아스트라가 #1759 리뷰에서 제안한 A–G를 따른다. 이 문서는 단계별로 고정한다. 지금 고정하는 것은 **F(학습 전 KV 검증)**다. E(메모리·시간 확인)와 학습·비교 단계는 F를 통과한 뒤 수치를 채우고, 실행 전에 다시 고정한다.

## 목표

블록 요청 단위 입력(#1756)과 crop 계약(#1758)으로 작은 adapter를 32 step 학습한다. 그 뒤 같은 문맥에서 현재 요청의 조건을 실제 조건과 다른 곡 조건으로 바꿨을 때 예측 손실이 달라지는지 본다. 조건을 켠 상태가 끈 상태보다 나빠지지 않는지도 본다. 음악성이나 개인화가 좋아졌다는 판단은 하지 않는다.

## 경로 (A)

- Aria medium은 고정한다(eval, 가중치 해시 불변)
- 조건 입력은 13차원이다(pc12 multihot + available). bias 없는 `Linear(13 → 1536)`이 마지막 encoder 층의 입력에 더해진다. 기존 torch adapter(`aria_cond_smoke.py`의 pre-hook)와 같은 위치다
- 실시간 경로(MLX)에서는 `scripts/aria_mlx_cond.py`의 `cond_forward`가 upstream forward를 그대로 따라가고 이 주입만 더한다. upstream 코드는 바꾸지 않는다
- 이 경로를 다시 쓴다고 안전한 것은 아니다. 기존 late-injection은 손해와 문법 악화를 낸 기록이 있다

## F. 학습 전 KV 검증 — `scripts/aria_mlx_cond_kv_check.py` (지금 고정)

0 가중치로 비교하면 잘못된 주입도 통과한다. 그래서 **고정된 비0 가중치**(seed 1760, 정규분포 × 0.05)를 쓴다.

| 항목 | 값 |
|---|---|
| 곡 | #1756 3회차 내보내기의 train 곡 0번, 7번 (실행 전 고정) |
| 구간 | 곡마다 앞 500음, 그 뒤 `<E>` |
| crop 설정 | CAP 16, 용량 384, keep 192. 짧은 구간에서도 crop이 여러 번 일어나게 하려고 작게 잡았다 |
| 실시간 경로 | 세션 하나에 KV 캐시 하나를 쓴다. crop이면 캐시를 비우고 `[prefix, <S>]` + 남긴 문맥을 원래 조건으로 prefill한다. 그다음 p0를 현재 조건으로 다시 쓰고 continuation을 한 토큰씩 넣는다 |
| 기준 | 새 캐시에 같은 세션 토큰과 같은 위치별 조건을 한 번에 prefill한다 |
| 통과 | 비교한 모든 위치에서 top-1이 같고 최대 \|logit 차\| ≤ 1e-3이다(#1741 KV smoke와 같은 기준) |
| 함께 확인 | 주입이 실제로 작동한다(실제 조건과 0 조건의 logit 차 > 1e-2). 비교 구간에 crop ≥ 2, 조건 변경 > 0, cold > 0, warm-empty > 0이 들어 있다 |

- 결과를 보고 허용 오차나 곡을 바꾸지 않는다. 기준에 못 미치면 숫자를 그대로 적고 "동등성 미확인"으로 둔다
- 위치별 조건 배정(어느 위치가 어느 요청의 조건을 받는지)이 맞는지는 #1758에서 순차 재연으로 이미 확인했다. F는 그 배정이 주어졌을 때 KV 경로가 전체 재계산과 같은지만 본다
- 실시간 producer(`rt_producer.py`)에는 아직 조건 주입이 없다. F는 MLX 모델 위의 검증이고, 실시간 준비가 끝났다는 뜻이 아니다
- 학습 뒤에는 저장했다가 다시 불러온 가중치로 같은 검증을 한 번 더 한다

```
/Users/ohhalim/git_box/t3_aria/.venv/bin/python scripts/aria_mlx_cond_kv_check.py /Users/ohhalim/git_box/t3_aria/aria \
  /Users/ohhalim/git_box/t3_aria/ckpt/model-gen.safetensors /Users/ohhalim/git_box/lh_request_export_run3 \
  docs/experiments/cond_adapter_min/kv_check.json
```

### 실행 전에 마친 시험

- `tests/test_aria_mlx_cond.py`(Aria venv, 작은 무작위 모델): 4개 통과
  - 0 가중치면 upstream forward와 같다
  - 비0 가중치면 logit이 바뀌고, 한 토큰씩 넣어도 전체 계산과 같다
  - 위치를 새 조건으로 다시 쓰면 처음부터 계산한 것과 같다

### F 실행 기록과 결과

- 42a28bee에서 한 번 돌렸다. 128초가 걸렸고, 열 압력은 0, 이탈은 없었다. 아스트라 사용량 한도 중이라 Claude가 혼자 실행했고, 사후 리뷰 대상이다
- **통과.** 두 곡에서 비교한 3,668개 위치 모두 top-1이 같았다. 최대 |logit 차|는 9.3e-5(0번 곡)와 5.1e-5(7번 곡)로, 기준 1e-3 안이다
- 주입은 실제로 작동했다. 실제 조건과 0 조건의 최대 logit 차가 11.8과 10.5다
- 비교 구간에 crop 11번, 조건 변경 154번, cold 2번, warm-empty 67번이 들어 있었다
- 보고는 `docs/experiments/cond_adapter_min/kv_check.json`에 있다
- 이것은 작은 용량(384) 설정에서 MLX 모델로 한 검증이다. 실시간 producer에는 아직 조건 주입이 없다

## B. 곡 선정 (메타데이터만 보고 고정) — `scripts/cond_adapter_split.py`

- 규칙: train 189곡을 sha1 순으로 보면서, 이미 고른 제목은 건너뛴다. 앞 8곡은 학습용, 다음 4곡은 확인용이다. val/test는 열지 않는다
- 학습용: 129 misterioso(Kenny Barron), 26 tribute to duke(Barry Harris), 8 would you like to take a walk(Barry Harris), 149 i got it bad and that ain't good(Junior Mance), 79 don't blame me(Hank Jones), 115 my heart skips a beat(Duke Jordan), 15 blue monk(Barry Harris), 75 what did i do to be so black and blue(Hank Jones)
- 확인용: 44 con alma(Cedar Walton), 68 honeysuckle rose(Hank Jones), 185 announcement(Bill Charlap), 134 ballad for benny carter(Kenny Barron)
- **불확실한 점:** 185번 "Announcement"는 제목만 보면 연주가 아닐 수도 있다. 하지만 앨범이 "Oscar, With Love — The Songs of Oscar Peterson"이고, 음 742개(초당 1.2음)로 manifest 상태는 ok다. 정해 둔 규칙을 결과를 보고 바꾸지 않으려고 그대로 두었다. 아스트라 판단을 받는다
- 결과: `docs/experiments/cond_adapter_min/split.json`

## 이후 단계 (F 통과 뒤 수치를 채워 다시 고정)

- **E. 메모리·시간(`scripts/cond_adapter_smoke.py`, 지금 고정):** 위 12곡의 요청 중 가장 긴 허용 입력(세션 문맥 + window, CAP 초과 window는 빼고 수를 적는다)으로 한 번 돌린다. batch 1, fp32, MPS다. window CE → adapter만 backward → Adam 1 step(LR 1e-3, 기존 레시피). 손실값은 읽지 않는다
  - 통과 조건: 손실이 유한하고, adapter 가중치가 바뀌었고, base 해시가 그대로이고, 옵티마이저에 adapter만 있고, 15분 안에 끝난다
  - 조건이 마지막 층에만 들어가므로, 아래 15층은 기울기 없이 계산할 수 있다(Claude 판단, 검토 대상)
  - OOM이나 시간 상한에 걸리면 기록하고 설계를 바꾼다. 긴 예제를 몰래 지우지 않는다
- **데이터(B):** train 189곡 안에서 학습 8곡과 확인용 4곡을 고른다. 고르는 규칙은 출력을 보기 전에 정한다(같은 곡 제목은 한쪽에만 둔다). val/test는 열지 않는다. 빈 window와 CAP 초과 요청은 빼고 수를 적는다. boundary에는 loss를 걸지 않는다
- **학습:** window CE로 32 update를 1회 한다. LR은 실행 전에 적는다
- **비교(C):** (a) adapter off (b) 실제 조건 (c) 다른 곡 조건. 과거 문맥 조건은 같게 두고 현재 요청 조건만 바꾼다
  - wrong 조건은 available이 같고 PC 수가 같은 다른 집합으로 정한다. 정답이나 결과를 보고 고르지 않는다
  - pitch/onset/dur/`<T>`별 NLL과 곡 평균을 따로 낸다. cold와 warm-empty는 따로 본다
- **해석(D):** b < c인데 둘 다 a보다 나쁘면 "조건은 쓰지만 손해"로 기록한다. available = 1이 만드는 공통 효과를 음악 조건 효과로 보지 않는다
- **생성(G):** 확인용 곡에서 2 primer × 2 seed로 a/b/c 짧은 생성을 하고, 문법 오류와 cap 종료를 확인한다

## E 실행 기록 — 이탈 1건

- **1회차(a5f1c3f5):** 첫 forward에서 shape 오류로 멈췄다. 결과는 나오지 않았다
  - 모델 입력은 마지막 토큰을 뺀 3,988개인데, 조건 행은 세션 토큰 수만큼 3,989개를 만들었다. 스크립트 버그다
  - 조건 행을 입력 위치 수(세션 길이 − 1)에 맞추고, 길이가 다르면 forward 전에 멈추는 assert를 넣었다. 설정은 바꾸지 않았다
- **2회차(768ef8a9): 통과.** 12곡에서 가장 긴 허용 입력은 3,989토큰이다(확인용 134번 곡 요청 353, 문맥 3,962 + window 27). CAP 초과로 뺀 요청은 1개다
  - forward + backward + 1 step에 1.99초가 걸렸다. MPS 할당은 현재 2.72 GiB, 드라이버 9.33 GiB였고, RSS 최대는 5.11 GiB, 메모리 압력 1, 열 압력 0이다
  - 손실은 유한했고 adapter 가중치가 바뀌었다. base 해시는 그대로였고, 옵티마이저에는 adapter만 있었다. 손실값은 읽지 않았다
  - 그러므로 32 update는 1분 남짓이면 된다. 평가 비용은 확인용 요청 수 × 세 조건 × 두 시점이라, 학습 설계에서 요청 수를 정한다

## 학습·비교 단계 — `scripts/cond_adapter_train.py` (사전 등록, 아스트라 리뷰 뒤 실행)

| 항목 | 값 |
|---|---|
| 학습 예제 | 학습용 8곡의 적격 요청(빈 window, CAP 초과 제외) 중 seed 1760으로 섞은 앞 32개 |
| 학습 | adapter만 학습한다. 0 초기화, Adam, LR 1e-3, batch 1, fp32, 32 update 1회. window 토큰 CE만 쓰고 boundary는 쓰지 않는다 |
| 평가 시점 | step 0, step 32. 그 사이에는 평가하지 않는다 |
| 평가 예제 | 확인용 4곡의 적격 요청에서 곡마다 48개를 고르게 뽑는다 |
| 비교 | (a) adapter off (b) 실제 조건 (c) 현재 요청 조건만 wrong으로 바꾼 것(문맥 위치 조건은 그대로) |
| wrong 조건 | 현재 PC 집합을 tritone(6반음) 옮긴다. 옮겨도 같으면 1반음 옮긴다. PC 수와 available은 같다. payload만 보고 정한다 |
| 층 | wrong이 정의되는 요청(with_wrong), cold, warm-empty를 따로 낸다. cold와 warm-empty는 a/b만 비교한다 |
| 지표 | window 토큰당 평균 NLL. 전체, 토큰 종류별(pitch, onset, dur, `<T>`), 곡별 |
| 생성(G) | 확인용 1·2번째 곡의 10번째 평가 요청 문맥을 primer로 쓴다. seed 1·2, 48토큰을 softmax 그대로 샘플한다. a/b/c마다 문법 위반 수와 `<E>` 수를 센다 |

- **읽는 규칙(실행 전 고정)**
  - step 0에서는 a, b, c가 정확히 같아야 한다(0 초기화 확인). 다르면 배선 버그로 보고 멈춘다
  - **현재 조건 사용 관측:** step 32의 with_wrong 층에서 b < c(전체 평균)이고, 4곡 중 3곡 이상에서 b < c
  - **켠 상태 손해:** with_wrong 층 전체 평균에서 b > a
  - b < c인데 b > a면 "조건은 쓰지만 순손해"로 적는다
  - 통계적 유의성은 주장하지 않는다. 확인용 4곡의 개발 probe이고, 일반화 판정이 아니다
  - available = 1이 만드는 공통 효과가 있을 수 있으므로, b < a라도 그것을 음악 조건의 효과로 보지 않는다. b와 c의 차이만 현재 조건의 효과로 읽는다
- **학습 뒤 확인:** 저장한 adapter 가중치로 F(KV 검증)를 다시 돌린다(`aria_mlx_cond_kv_check.py … <adapter .npy>`)
- **비용:** 학습 약 1분, 평가는 시점마다 약 550번 forward, 생성은 약 600번 forward(KV 캐시 없이). 모두 1시간 안을 예상한다
- adapter 가중치는 repo 밖 `/Users/ohhalim/git_box/cond_adapter_min/`에 둔다
- 배선 점검: `COND_ADAPTER_DRYRUN=1`이면 작은 무작위 모델로 끝까지 돌린다. 이 출력은 결과가 아니다
- 배선 점검 결과(작은 무작위 모델, 37초): step 0 세 조건 동일, base 해시 불변, 옵티마이저 adapter만, 32 step 완료. 학습용 곡의 적격 요청은 예를 들어 129번 485개, 26번 198개, 8번 180개다(빈 window와 CAP 초과를 뺀 수). 생성 문법 위반 수는 위반한 토큰 수다. 위반 토큰은 문법 상태를 바꾸지 않는다
- **아스트라 사용량 한도 중이라 Claude가 이 사전 등록을 커밋한 뒤 혼자 실행한다.** 사후 리뷰에서 설계가 바뀌면 다시 돌리고 두 결과를 모두 적는다

## 학습·비교 실행 기록과 결과

- b07b0d69에서 한 번 돌렸다. 676초가 걸렸고 이탈은 없었다. 열 압력은 실행 전 0, 끝난 뒤 2였다. 아스트라 한도 중이라 사후 리뷰 대상이다
- 확인: step 0 세 조건 동일, base 해시 불변, 옵티마이저 adapter만, 32 step 완료. adapter 크기(Frobenius)는 0.84다
- 가중치는 `/Users/ohhalim/git_box/cond_adapter_min/adapter_step32.npy`(sha256 d4d16989…)에 있고, 보고는 `docs/experiments/cond_adapter_min/train.json`에 있다

**window 토큰당 평균 NLL (확인용 4곡, step 32)**

| 층 | 토큰 | a 끔 | b 실제 | c wrong | b − a | b − c |
|---|---|---|---|---|---|---|
| with_wrong 전체 | 1,732 | 2.77059 | 2.77425 | 2.76603 | +0.00366 | +0.00822 |
| with_wrong pitch | 566 | 2.87742 | 2.88315 | 2.89067 | +0.00573 | −0.00752 |
| with_wrong onset | 566 | 2.22910 | 2.23262 | 2.22829 | +0.00352 | +0.00433 |
| with_wrong dur | 566 | 3.33977 | 3.34188 | 3.31633 | +0.00211 | +0.02555 |
| with_wrong `<T>` | 34 | 0.53130 | 0.52866 | 0.48237 | −0.00264 | +0.04629 |
| warm-empty 전체 | 967 | 2.62352 | 2.61431 | – | −0.00921 | – |
| cold 전체 | 36 | 5.17864 | 5.17864 | – | 0 | – |

곡별(with_wrong 전체, a / b / c): 44번 2.8447 / 2.8523 / 2.8219, 68번 2.5500 / 2.5688 / 2.5782, 185번 2.8124 / 2.8048 / 2.7986, 134번 2.9590 / 2.9458 / 2.9385

- **판정(사전 등록 규칙)**
  - **현재 조건 사용: 관측되지 않음.** with_wrong 전체에서 b > c다(실제 조건이 wrong보다 0.008 나쁘다). b < c인 곡은 4곡 중 1곡(68번)뿐이다
  - **켠 상태: 작은 손해.** with_wrong 전체에서 b − a = +0.004다
  - warm-empty에서 b가 a보다 0.009 낮다. 이것은 available = 1의 공통 효과로 보고, 음악 조건의 효과로 읽지 않는다
  - cold는 payload가 모두 0이라 adapter 출력도 0이다. 그래서 a와 b가 같다(예상대로)
  - pitch 토큰만 보면 b < c(−0.0075)다. 하지만 사전에 정한 판정은 전체 기준이고, 같은 층에서 b > a다. 부분 지표에서 효과를 골라 말하지 않는다
- **생성(G):** a/b/c 각각 4개(2 primer × 2 seed), 48토큰씩 생성했다. 문법 위반 0, `<E>` 0이다
- **학습 손실:** 32개 값이 요청마다 크게 흔들린다(1.42–6.35). batch 1이라 추세를 말하지 않는다
- **한계:** 차이가 모두 0.01 nats 안팎으로 작다. 32 step, LR 1e-3에서 adapter가 거의 움직이지 않은 것으로 보인다(Claude 판단). 확인용 4곡의 개발 probe이고 통계 판정이 아니다
- **학습 뒤 KV 재검증(15fe1a03, 104초): 통과.** 학습된 가중치로 같은 검증을 돌렸다. 최대 |logit 차| 9.3e-5와 8.4e-5, top-1은 모두 같았다. 실제와 0 조건의 최대 logit 차는 6.6과 4.1이라 주입도 작동한다. 보고: `docs/experiments/cond_adapter_min/kv_check_trained.json`

## 진단 — 조건 PC와 window 오른손 PC의 겹침 (학습 없음, 사전 등록)

결과가 "효과 없음"이라, 다음 방향을 정하기 전에 조건에 쓸 정보가 있는지 데이터만으로 센다. 판정은 하지 않고 기술 통계만 낸다.

- 대상: #1756 3회차 내보내기 train 189곡의 적격 요청 중 wrong 조건이 정의되는 것(조건 PC가 1개 이상)
- 요청마다: window의 pitch 토큰 PC 중 실제 조건 PC 집합에 드는 비율, tritone wrong 집합에 드는 비율, 집합 크기로 본 우연 기대값(|집합|/12)
- 보고: 요청 평균과 토큰 합산 비율. 집합 크기(1, 2, 3+)별로 나눈다
- 스크립트: `scripts/cond_pc_overlap.py`, 보고 `docs/experiments/cond_adapter_min/pc_overlap.json`

### 진단 결과 (fefe2053)

| 조건 PC 수 | 요청 | 음 | 실제 조건 PC에 듦 | wrong 집합에 듦 | 우연 기대값 |
|---|---|---|---|---|---|
| 1 | 21,460 | 129,177 | 10.5% | 6.5% | 8.3% |
| 2 | 12,290 | 68,716 | 21.4% | 13.1% | 16.7% |
| 3+ | 2,500 | 13,831 | 34.9% | 20.5% | 26.5% |
| 전체 | 36,250 | 211,724 | 15.6% | 9.6% | 12.2% |

- 조건 PC는 window 오른손 음과 겹친다. 실제가 wrong의 약 1.6배, 우연의 약 1.3배다. 정보는 있지만 크지 않다
- 이것은 겹침 비율이다. 모델이 쓸 수 있는 손실 감소의 크기는 아래 추정으로 따로 본다

## 진단 2 — 조건 PC로 얻을 수 있는 음당 이득의 대략적 상한 (학습 없음, 사전 등록)

- PC 수준의 간단한 모델 둘을 비교한다
  - A: 학습용 8곡 window 음의 PC 분포
  - B: A에 "조건 PC면 α배"를 곱하고 다시 정규화한 것. α 하나만 학습용 8곡에서 최대우도로 맞춘다(격자 1.00–4.00, 0.01 간격)
- 확인용 4곡과 전체 189곡에서, with_wrong 요청의 window 음당 평균 로그우도 차(B − A)를 nats로 낸다. 같은 α로 wrong 집합을 넣은 값도 함께 낸다
- 이 모델은 octave, 리듬, 문맥을 전혀 보지 않는다. 그래서 정확한 상한은 아니다. 다만 Aria가 이미 문맥에서 아는 정보와 겹칠 수 있어서, Aria 위의 실제 이득은 이 값보다 작을 가능성이 크다(Claude 판단)
- 스크립트: `scripts/cond_pc_gain.py`, 보고 `docs/experiments/cond_adapter_min/pc_gain.json`

### 진단 2 결과 (4bc7b4cf)

- 학습용 8곡에서 맞춘 α는 1.14다(약하다). 학습용 곡의 with_wrong 요청은 1,297개다

| 평가 | 요청 | 음 | 실제 조건 이득 (nats/음) | wrong 이득 (nats/음) |
|---|---|---|---|---|
| 확인용 4곡 | 771 | 3,473 | +0.0022 | −0.0054 |
| 전체 189곡 | 36,250 | 211,724 | +0.0031 | −0.0034 |

- PC만 보는 모델로는 이 조건에서 얻을 수 있는 이득이 음당 약 0.003 nats다. 조건(대부분 저음 PC 하나)에 담긴 정보가 아주 적다
- 32 step 결과의 pitch 토큰 b − c(−0.0075)는 이 모델의 실제·wrong 차이(확인용 0.0076)와 크기가 비슷하다. 하지만 사전에 정한 판정은 전체 기준이고, 부분 지표로 효과를 말하지 않는다. 기록만 해 둔다
- 그러므로 "학습이 부족해서 효과가 안 보였다"보다 "조건에 담긴 정보가 원래 작다"는 쪽이 더 그럴듯하다(Claude 판단). 다음 방향(조건 표현을 바꿀지, NLL 대신 다른 지표로 볼지, 여기서 멈출지)은 아스트라 리뷰 뒤에 정한다

### 185번 "Announcement" 확인 (음 데이터만)

- 742음, 213초, 초당 3.5음(다른 곡 16개 표본 5.2–11.8)이다. 음역 31–94, 음높이 중앙 62, 30 ms 안에 같이 시작하는 음의 비율 0.44로 다른 곡과 비슷하다
- 음 길이 중앙값은 1,057 ms다(다른 곡 127–734). 느리고 길게 누르는 곡이다
- 말소리를 옮긴 것이라면 짧고 들쭉날쭉한 음이 나왔을 것이다. 화음과 넓은 음역이 있으므로 느린 루바토 피아노 곡으로 본다(Claude 판단, 청취 확인 아님). 선정은 그대로 둔다

## 진단 3 — recent 조건이 sounding보다 정보가 많은가 (학습 없음, 사전 등록)

- 진단 2와 같은 PC 모델(학습용 8곡 PC 분포 × 조건 PC α배, α는 학습용 8곡에서 맞춘다)을 쓴다. 조건만 바꾼다
  - sounding: 지금 payload(s_k에 울리는 g3 인간 음)
  - recent: s_k 앞 937.5 ms 안에 시작한 g3 인간 음(#1750의 recent 정의, 블록 snapshot 기준)
- 같은 요청 집합을 쓴다. 요청은 #1756 적격 요청 중 두 조건이 모두 비어 있지 않은 것이다
- 확인용 4곡과 전체 189곡에서 음당 이득(nats)을 낸다. 학습 없이 데이터만 보는 정보량 비교이고, 새 조건으로 학습하자는 결정이 아니다
- 스크립트: `scripts/cond_pc_gain_recent.py`, 보고 `docs/experiments/cond_adapter_min/pc_gain_recent.json`

### 진단 3 결과 (95e0de05)

| 평가 | 요청 | 음 | 평균 PC 수 (sounding / recent) | sounding 이득 | recent 이득 |
|---|---|---|---|---|---|
| 확인용 4곡 | 628 | 3,007 | 1.49 / 1.81 | +0.0024 | +0.0028 |
| 전체 189곡 | 31,336 | 187,703 | 1.49 / 2.04 | +0.0032 | +0.0025 |

- α는 sounding 1.15, recent 1.09다(학습용 8곡, 두 조건이 모두 비어 있지 않은 요청 1,050개)
- recent는 PC가 조금 더 많지만, 두 조건 모두 음당 약 0.003 nats다. PC 수준에서는 sounding을 recent로 바꿔도 나아질 여지가 거의 없다
- 한계: α를 학습용 8곡에서 맞췄다. 이 곡들이 전체를 대표하지 않으면 α가 작게 나왔을 수 있다. 다만 진단 1의 겹침 비율(실제 15.6% 대 우연 12.2%)로 보아, α를 전체에서 맞춰도 이득은 음당 0.01 nats를 크게 넘기 어렵다(Claude 판단)
