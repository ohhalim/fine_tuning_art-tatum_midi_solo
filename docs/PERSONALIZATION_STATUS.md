# 개인화 솔로 모델 현황 — Art Tatum · Brad Mehldau

작성 2026-09-29. `style_verified: false` · `musical_quality_verified: false` · 청취로 판정한 적 없음

이 문서는 두 연주자 개인화의 최종 상태를 한 곳에 모은 것이다. 근거 수치는 링크한 실험 문서에 있다. 판정 기준은 실험 전에 등록했다. 다만 1차 기준에 미달해 후속 실험을 새로 등록한 경우가 있다(#1531 시작 예산, #1545 블록 지표). 이 경우 1차 미달은 그대로 두고, 후속은 원인 분리 탐색으로만 표시한다.

2026-09-29 Astra 리뷰를 반영해 교정했다(#1558). 조건은 다음과 같다.
- 모든 실시간 수치는 **가상 CoreMIDI 포트, M1 Max CPU, 무부하(명시한 경우 제외)** 조건이다
- 기본값은 이 제한된 환경에서의 **잠정 선택**이다

## 1. 한눈에

| | **Tatum 완성** | **멜다우 완성** |
|---|---|---|
| 체크포인트 (gitignore) | `outputs/final_tatum/export/checkpoint_update518.pt` | `outputs/clean_base/c2_export/checkpoint_update128.pt` |
| base | 공통 base(Tatum·멜다우를 둘 다 뺀 2,637곡) | 멜다우만 뺀 clean base(2,759곡) |
| 어댑터 | LoRA r16, out_proj + QKV | LoRA r16, out_proj |
| 학습 자료 | Tatum 98곡 | 멜다우 16곡(자료를 더 늘릴 수 없음) |
| 선택 방법 | val12에서 update 선택(518) | 곡 단위 4-fold CV(128) |
| **처음 보는 곡 예측 (base 대비 ΔCE)** | **fresh12 −0.126** [−0.135, −0.117] | **4-fold held-out −0.046**(음수 fold 4/4), 재사용 holdout 2곡 −0.047 |
| 일반 재즈 100곡 ΔCE | +0.009 | −0.024 |
| 연주자 특이성 | 멜다우 곡 +0.052(나빠짐) | 특화도 −0.022 |
| 학습곡 exact 16-gram (검사한 생성 샘플) | 미검출 | 미검출 |
| 128 BPM 생성 p50 / p95 (CPU) | 307 / 463 ms | 약 213 / 350 ms |
| 성립 템포 (fallback·미스 0) | 128–240 BPM | 128–240 BPM |
| 근거 | [FINAL_TATUM](experiments/FINAL_TATUM.md) | [MEHLDAU_CLEAN_BASE](experiments/MEHLDAU_CLEAN_BASE.md), [FINAL_MEHLDAU](experiments/FINAL_MEHLDAU.md) |

**"개인화 완료"의 뜻(가능도 수준):** 해당 연주자를 한 번도 보지 않은 base에서 출발해, 그 연주자의 **처음 보는 곡**을 더 잘 예측하게 됐다. 일반 재즈 능력은 거의 잃지 않았고, 검사한 생성 샘플에서 학습곡 exact 16-gram이 나오지 않았으며, 가상 포트 조건에서 실시간으로 돈다. **들리는 스타일은 검증하지 않았다**(§5).

## 2. 쓰는 법

```bash
# 한 연주자
python scripts/play_personalized.py --preset tatum          # 또는 mehldau, base
# 한 세션에서 4마디마다 Tatum ↔ 멜다우
python scripts/play_personalized.py --preset swap
# 키보드 프리셋 버튼으로 전환 (PC 0 = Tatum, PC 1 = 멜다우)
python scripts/play_personalized.py --preset swap --live-select --input-port "<키보드 입력 포트>"
# 옵션: --bpm 200 --chords "F7,Bb7,F7,C7" --bars 12 --seed 7 --capture, 실제 명령만 보기 --dry-run
# 연주 중 블록 지표 보기(어댑터·음 수·음높이·코드톤·보이싱·입력): ... -- --live-metrics
# CPU를 많이 쓰는 환경(DAW 등)에서 fallback이 나면: ... -- --start-budget-bars off
```
- 코드 primer 없는 기본 모드를 직접 쓸 때는 `--generation-tokens 192 --max-sequence 256`을 준다. 96토큰이면 Tatum 마디의 37.5%가 fallback된다([PLAIN_BUDGET](experiments/PLAIN_BUDGET.md))
- 출력은 가상 MIDI 포트 `ContinuousJazz`로 나간다(DAW에서 받으면 된다). 연주 기록은 `outputs/play/<preset>_seed<seed>/`(`played.mid`, `continuous_report.json`)에 남는다
- 런타임 기본값
  - 프리셋: 코드 primer, 마디당 2 sub-block을 반 마디 스케줄러 블록으로(#1532)
  - KV 캐시(#1507)
  - LoRA 병합(#1512, #1520)
  - 늦은 fetch 50 ms(#1526)
  - 생성 시작 예산 0.5블록(#1530): 잠정 기본값. 1차 기준(미스 0)은 미달이었고, 후속 교대 12회는 원인 분리 탐색일 뿐 비열등 입증이 아니다
  - 루프 안 블록 지표 기록(#1544, #1546): 잠정 기본값, 위와 같은 단서. 폐기된 늦은 블록의 지표가 섞이는 결함은 수정 예정(Astra M2)
  - 코드 토큰 예약(#1553)은 **실험 옵션, 기본 off**(#1558). 음역 따라가기가 5/6에서 2/6로 떨어진 비용이 있다
  - 스케줄러 spin 5 ms, QoS user-interactive

## 3. 실시간 성능 (가상 포트, M1 Max CPU)

| 항목 | 값 | 근거 |
|---|---|---|
| 성립 템포 | 두 모델 모두 128/160/200/240 BPM, 24/24회 fallback 0·미스 0. 생성 p95는 마디의 최대 29% | [FINAL_TEMPO](experiments/FINAL_TEMPO.md) |
| Tatum 완성 생성 p50 | 690 ms(초기) → 625(KV 캐시) → **307 ms**(LoRA 병합) | [KV_CACHE](experiments/KV_CACHE.md), [LORA_MERGE](experiments/LORA_MERGE.md) |
| 어댑터 스왑 | 같은 base 0.8 ms, 다른 base 2–20 ms. 스왑 세션 마디가 단독 세션과 48/48 동일 | [ADAPTER_SWAP](experiments/ADAPTER_SWAP.md) |
| 입력 → 반영 (도착에서 반영 블록 시작까지, 128 BPM, PC 기준) | 약 7.5 s → 2.7 s(늦은 fetch) → 1.8 s(시작 예산) → **평균 0.9 s, 0.5–1.3 s**(반 마디 블록, 프리셋 기본) | [GENERATION_LEAD](experiments/GENERATION_LEAD.md), [START_BUDGET](experiments/START_BUDGET.md), [HALF_BAR_BLOCKS](experiments/HALF_BAR_BLOCKS.md) |
| 키보드 음 입력 | 반 마디 블록에서 입력이 든 블록만 달라지고(입력 없는 블록은 26/26 동일) 4초 창이 지나면 원래대로 돌아온다. 최신 입력 → 블록 시작 0.52–0.55 s | [NOTE_INPUT_HALF_BAR](experiments/NOTE_INPUT_HALF_BAR.md) |
| 입력 음역 따라가기 (탐색) | 높은/낮은 음역 구절을 입력하면 출력 평균 음높이가 6개 비교 중 5개에서 그 방향으로 이동(+2.8 ~ +8.1 / −1.5 ~ −8.2 반음) | [INPUT_REGISTER_FOLLOW](experiments/INPUT_REGISTER_FOLLOW.md) |
| 입력 밀도 따라가기 (탐색) | 16분음표 16개 입력 시 출력 음 수가 3/3 seed에서 **줄었다**(−15 ~ −30%). 따라간다는 근거 없음. 이때 코드 진술이 primer에서 잘려 나감을 확인했다(인과는 미검증). 코드 토큰 예약(#1553, **실험 옵션, 기본 off**)을 켜면 입력 블록 코드톤 비율이 3/3 올랐지만(0.39–0.47 → 0.50–0.61) 음역 따라가기가 5/6에서 2/6로 떨어졌다 | [INPUT_DENSITY_FOLLOW](experiments/INPUT_DENSITY_FOLLOW.md), [RESERVE_CHORD_TOKENS](experiments/RESERVE_CHORD_TOKENS.md) |
| 루프 안 지표 | 블록마다 어댑터·음 수·음높이·코드톤·입력·보이싱 수·어댑터별 누적 고유 보이싱을 기록(`block_metrics`, 기본 on). fallback 없는 실행에서 오프라인 계산과 일치(288/288, 96/96). 교대 12회에서 미스 차이는 관측되지 않았다(비열등 미입증). 늦게 폐기된 블록의 지표가 섞이는 결함이 있다(수정 예정) | [LIVE_METRICS](experiments/LIVE_METRICS.md), [LIVE_DIVERSITY](experiments/LIVE_DIVERSITY.md) |
| CPU 부하 여유 (Tatum 완성, 반 마디 블록) | 바쁜 프로세스 2·4개: 128/240 BPM 성립. 8개(성능 코어 전부): 생성이 2.5–3배 느려짐. fallback은 고정 예산 22%, 적응형 4%, 예산 off 1.6% | [LOAD_MARGIN](experiments/LOAD_MARGIN.md), [ADAPTIVE_BUDGET](experiments/ADAPTIVE_BUDGET.md) |
| 키보드 전환 | Program Change/CC, 모든 메시지 적용(9/9, 15/15, 15/15). 반 마디 블록에서 도착 → 적용 블록 시작 평균 0.9 s | [ADAPTER_LIVE_SELECT](experiments/ADAPTER_LIVE_SELECT.md), [HALF_BAR_BLOCKS](experiments/HALF_BAR_BLOCKS.md) |

주의
- 음 입력(#1538)과 음역 따라가기(#1542)는 입력 음과 코드 음을 시간순으로 섞는 primer로 쟀다. 코드 토큰 예약을 끈 현재 기본값과 같은 구성이다
- 키보드 전환 지연 수치는 PC가 매번 어댑터를 바꾸고 fallback이 0인 실행에서 쟀다. no-op 요청과 fallback을 적용으로 세는 probe 결함(Astra M3)이 있다. 이 조건이라 영향은 낮을 것으로 예상하지만 검산 전이다. probe 수정과 기존 artifact 검산은 예정이다
- 늦은 fetch는 이전 블록의 note_off가 마디 경계에 있으면 fetch가 −50 ms가 아니라 경계에서 일어나는 결함이 있다(Astra M1, 수정 예정)

## 4. 쇼케이스 MIDI

`bash scripts/make_showcase.sh`(PY, OUT 지정)로 만든다. 모든 변형에서 primer·코드·seed·길이가 같다. 현재 세트는 `outputs/showcase_v1/midi/`에 있고, 요약은 [showcase_v1_summary.json](experiments/showcase_v1_summary.json)이다.

| 진행 | 변형 | 음 수 | 마디당 | 음역 | 코드톤 비율 | fallback |
|---|---|---|---|---|---|---|
| ii–V–I C, 16마디, 128 BPM | base / Tatum / 멜다우 / 스왑 | 188 / 396 / 248 / 341 | 11.8 / 24.8 / 15.5 / 21.3 | 55 / 60 / 40 / 52 | 0.52 / 0.51 / 0.66 / 0.60 | 0 |
| F 블루스, 12마디, 120 BPM | 같음 | 246 / 362 / 224 / 331 | 20.5 / 30.2 / 18.7 / 27.6 | 49 / 55 / 50 / 52 | 0.49 / 0.50 / 0.53 / 0.49 | 0 |
| minor ii–V–i C, 16마디, 128 BPM | 같음 | 224 / 410 / 269 / 364 | 14.0 / 25.6 / 16.8 / 22.8 | 70 / 60 / 41 / 52 | 0.54 / 0.52 / 0.55 / 0.53 | 0 |

- 기술값일 뿐 품질 판정이 아니다
- 관측
  - Tatum은 세 진행 모두에서 음이 가장 많다(base의 1.5–2.1배)
  - 멜다우는 음역이 가장 좁다(40–50)
  - 스왑 세션은 두 모델 사이 값이다
- 여기서 base는 공통 base다. 멜다우 어댑터의 base는 clean base다

## 5. 검증된 것과 아닌 것

**검증됨(사전 기준, 객관 지표)**
- 처음 보는 곡 가능도 개선과 연주자 특이성(위 표)
- 검사한 생성 샘플의 학습곡 exact 16-gram 미검출, 문법 유효성
- 가상 포트·무부하 조건의 실시간 성립(128–240 BPM). 성능 코어를 모두 쓰는 부하에서는 미성립(#1534)
- 스왑·병합·KV 캐시가 출력을 바꾸지 않음(토큰 또는 연주 마디 동일)

**검증 안 됨**
- **들리는 스타일.** 멜다우 블라인드 청취 2회는 지지가 없었다(사용자: "사투리 구분처럼 모호하다"). Tatum은 청취 1쌍에서 "그나마 제일 Tatum스럽다, 빠른 속주" 한 건뿐이다
- 스타일 분류기: 멜다우 descriptor 분류기는 타당성 기준에 미달했다(균형 정확도 0.734 < 0.75, [MEHLDAU_STYLE_SHIFT](experiments/MEHLDAU_STYLE_SHIFT.md))
- 음악적 품질, 실물 키보드·DAW 환경, 사람과의 실제 협연

## 6. 알려진 한계와 기각된 시도

- **두 최선 모델의 base가 다르다.** 공통 base 멜다우는 절대 CE가 0.028 나빠(16/16곡) 통합하지 않았다([SHARED_BASE](experiments/SHARED_BASE.md)). 스왑 프리셋은 전체 모델을 복사한다
- **멜다우는 자료 한계다.** 학습량을 64–384로 넓혀도 held-out 개선은 64–128에서 포화했다([FINAL_MEHLDAU](experiments/FINAL_MEHLDAU.md)). 16곡 중 15곡이 한 앨범이다
- **화성 추종은 음표 기반 primer다.** 학습된 코드 조건이 아니다. 마디 간 문맥 연결은 3차례 모두 기준 미달이었다(경계는 매끄러워지지만 코드톤 비율이 −0.11 ~ −0.20). 시험을 닫았다([USAGE_PATH_16BAR](experiments/USAGE_PATH_16BAR.md) §6–8)
- **입력 반영 지연**
  - 프리셋 기준 PC 메시지는 평균 0.9초(128 BPM)다
  - 음 입력은 스냅샷의 최신 음에서 블록 시작까지 0.52–0.55초다(#1538)
  - jam_bot의 멜로디 조건 목표 800 ms 근처다. 반주 100 ms 목표와는 여전히 차원이 다르다
  - 체감 지연(귀로 듣는 반응)은 재지 않았다
- 2026-09-29 세션에서 내가 낸 오류(모두 해당 문서에 정정했다)
  1. LoRA 병합이 out_proj를 놓쳤다. 출력은 맞았지만 설명과 속도 해석이 틀렸다(#1520에서 수정)
  2. 문맥 연결 3차 사전 등록이 실제 primer 길이(10토큰)를 확인하지 않았다(#1513)
  3. 부하 측정 스크립트의 `seq 1 0` 때문에 "부하 0"이 실제로는 부하 2였다(#1534)
  4. 라이브 선택 사전 등록이 producer 선행을 1마디로 잘못 가정했다. 실제로는 3마디였고, 그 원인을 #1526에서 고쳤다(#1525)

## 7. 다음 후보 (각각 한 질문)
1. 실물 키보드 + DAW 부하에서 성립 템포·전환·입력 반영 재측정(지금까지는 가상 포트와 합성 부하만 썼다)
2. 촘촘한 입력에서 코드 진술이 잘리는 문제: '잘린 뒤 코드 guide 음 누락'일 때만 예약하는 방식으로 재정의한다. 공동 검증(안 잘리면 토큰 동일, 잘리면 코드 보존, 음역 따라가기, 지연·fallback)을 사전 등록한 뒤 재검토한다(#1557 보류). 그 뒤 동기 모방·화성 따라가기 지표를 더한다
3. 무거운 부하에서 fallback 0을 만드는 예산 규칙(적응형은 22% → 4%까지 줄였고 기준 3% 이하에는 미달)
4. 청취: 사용자가 원할 때만. 쇼케이스 세트가 준비돼 있다
