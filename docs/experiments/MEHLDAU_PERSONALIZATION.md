# 멜다우 개인화 시도 — 뚜렷한 스타일 적응 효과 미확인

작성 2026-09-22. 브랜치 `exp/mehldau-personalization`.
**구현 의존성: PR #1482의 chord-primer / 런타임 플래그.** #1482 이후 develop에 통합하는 순서.

`mehldau_style_verified: false` · `musical_quality_verified: false`
**초기 청취 의견은 기록했으나 멜다우 스타일 성공을 주장하지 않는다.**

## 1. 데이터 provenance

원본 18곡, 전부 `midi_dataset/midi/studio/Brad Mehldau/` 아래 두 앨범:

| 앨범 | 곡 수 |
|---|---|
| Marian McPartland's Piano Jazz with Brad Mehldau | 3 |
| Suite - April 2020 | 15 |

전처리 `data/mehldau_full` = **16 train / 2 val, 곡 단위 분리**.
토큰 길이로 원본에 1:1 매핑해 확인했다. val 2곡:

- `Suite - April 2020/I. waking up` (1,031 토큰)
- `Suite - April 2020/X. in the kitchen` (5,367 토큰)

18곡의 토큰 길이는 모두 상이. 부분 중복·다른 전사 버전 여부까지 배제한 것은 아님. 토크나이저 호환: 최대 토큰 ID 376,
제어 토큰 0개 → 학습 범위(≤388) 안. checkpoint vocab 547 와 호환.

## 2. ★ 누출 — 이것이 모든 수치를 규정한다

**멜다우 18곡 전부가 이미 `jazz_full` 안에 있다.** 베이스 사전학습셋이다.
토큰 시퀀스 SHA-1 완전 일치로 확인했다: **18/18.**

더 나쁜 것은 **val 2곡도 `jazz_full/train` 에 있다**
(`jazz_full/train/00927`, `00340`). 즉:

- **진짜 held-out 멜다우 데이터가 존재하지 않는다**
- val loss 는 모든 팔에서 낙관적으로 편향된다. 베이스 포함
- 새 곡 일반화는 **이 데이터로 주장할 수 없다**
- "멜다우에 적응했다" 는 **베이스가 이미 맞춘 곡을 다시 맞추는 것**이다

`jazz_full` 이 `./midi_dataset/midi` 전체에서 만들어졌고 그 안에 Brad Mehldau
폴더가 있으므로 구조적으로 그렇게 된다.

## 3. 학습 — D1 Arm D 레시피 재사용

두 출발점을 구분했다. 기존 checkpoint 는 **덮어쓰지 않았다**
(새 디렉터리 `outputs/mehldau_lora/`).

| 출발점 | 체크포인트 |
|---|---|
| generic base | `d0_experiment/armB_full2777/ckpt/checkpoint_epoch8.pt` |
| Tatum-adapted | `d1_experiment/armD_lora/ckpt/checkpoint_epoch8.pt` |

설정: LoRA r=16 alpha=32, lr 3e-4, 8 epoch, batch 4, seed 42, max_seq 512,
CPU. smoke 1 epoch 로 경로 확인 후 본 학습. best-val epoch 자동 저장.

### 학습 loss — 거의 평평하다

```
from_base    train 3.281 -> 3.309   val 3.084 ~ 3.171 (무추세, best 3.073 @epoch6)
from_tatum   train 3.274 -> 3.302   val 3.066 ~ 3.164 (무추세, best 3.066 @epoch6)
```

train loss 가 **오히려 조금 올랐다.** 8 epoch × 8 step = 64 step 이고
학습 파라미터는 98,304 (0.72%) 다. 그리고 두 출발점의 곡선이 거의 같다.

## 4. 평가 — 네 측정이 모두 같은 방향

동일 고정 primer, seed 42/100/200/300/400, 8마디, 같은 예산.

| 팔 | val_loss | 노트 | 노트/take | range | pc | IOI ms | vel | dur ms |
|---|---|---|---|---|---|---|---|---|
| base | 2.4780 | 228 | 5.70 | 26 | 5 | 60 | 60 | 640 |
| tatum | 2.4694 | 174 | 4.35 | 20.5 | 4.5 | 60 | 56 | 708 |
| **mehldau_from_base** | 2.4710 | 231 | **5.78** | **24** | **5** | **60** | **60** | 745 |
| **mehldau_from_tatum** | 2.4630 | 196 | **4.90** | **21** | **5** | 90 | **56** | 623 |

range / pc는 음표가 2개 이상인 take별 값의 중앙값, IOI는 take 내부 간격들을 모은 중앙값이다(총 40 take). 전 seed 노트를 먼저
합치면 IOI 중앙값이 0 이 되고 range 는 합집합이 되어 무의미해진다
(실제로 한 번 그렇게 계산했다가 고쳤다).

| 측정 | 결과 |
|---|---|
| 학습 loss | **평평.** 8 epoch 내내 움직이지 않음 |
| val_loss 변화 | base 대비 **−0.0071 / −0.0150**. 각 출발점 대비 약 **−0.007 / −0.006**. 작고, 데이터 중복으로 일반화 해석 불가 |
| 정확 일치 검사 (8-token) | **0.000** — 검사한 생성 토큰의 8-gram에서 train 16곡과 정확 일치 없음 |
| 생성 descriptor | **출발점과 거의 동일.** from_base ≈ base, from_tatum ≈ tatum |

## 5. 판정 — 이번 설정에서 뚜렷한 개인화 효과 미확인

이번 학습·생성 표본에서 loss 변화는 작고, 집계 descriptor만으로 스타일 적응을 입증하지 못했다.
베이스 데이터 중복은 독립 검증을 방해하지만 **효과가 작은 원인으로 확정할 수 없다**.
이미 학습한 곡도 재가중·적응 학습으로 분포가 바뀔 수 있다. 학습률, 업데이트 크기,
64 step의 학습량, 조건 primer, 평가 민감도는 추가 확인 대상이다.

8-token exact match 0은 이 표본의 해당 검사에서 일치가 없었다는 뜻이다.
이조·리듬 변형·긴 음악 프레이즈 복사까지 배제하지 않는다.
사용자 초기 청취 의견은 “느낌은 비슷하고 베이스 음 뒤 솔로 같은 선율이 반복”이다.
블라인드 청취나 멜다우 스타일 성공 판정은 아니다.

## 6. 런타임 smoke — 어댑터 명시 선택

```
FORCE_CPU=1 uv run python scripts/run_continuous_jazz.py \
    --checkpoint outputs/mehldau_lora/from_tatum/checkpoint_epoch6.pt \
    --conditioning-midi <primer> --bars 8 --capture \
    --chord-primer --chord-blocks-per-bar 2 --output-dir outputs/continuous/mehldau
```

결과: **완주, model 8/8, 오류 0, 생성 p50 397ms, 캡처 216/216 손실 0,
데드라인 미스 0.** 어댑터가 기존 런타임에서 load/generate 된다.

## 7. 내가 한 실수 두 개 (기록)

1. **첫 평가가 `lora_weights.pt` 만 랜덤 초기화 모델에 얹었다.**
   `prefer_full_checkpoint=False` 면 베이스 가중치를 아예 불러오지 않는다
   (함수 docstring 이 그렇게 경고하고 있었다). val_loss 6.53 이 나왔고
   그것을 결과로 보고할 뻔했다. `checkpoint_epoch*.pt` 를 쓰도록 고쳤고,
   스크립트가 `lora_weights.pt` 를 받으면 거부한다.
2. **descriptor 를 전 seed 노트를 합쳐 계산했다.** IOI 중앙값이 0, range 가
   합집합으로 나왔다. 마디별 take 단위로 고쳤다.

## 8. 막힌 지점과 필요한 입력

**새 곡 일반화 검증에는 base가 보지 않은 멜다우 자료가 필요하다.** 효과 부재의 원인 진단과 구분한다.

선택지:

1. **베이스를 다시 학습한다** — `jazz_full` 에서 멜다우 18곡을 빼고 armB 를
   재학습. 이후 적응 train 16곡과 분리한 val 2곡을 독립 검증에 사용할 수 있다. 비용은 2,777곡 재학습이고
   원격 GPU를 사용할 경우 별도 승인이 필요하다
2. **멜다우 자료를 더 구한다** — 현재 코퍼스 밖의 트랜스크립션. 데이터 출처와
   권리 확인이 필요하고, 이 세션 범위 밖이다
3. **개인화 대상을 바꾼다** — 사용자 본인 연주. 새로 직접 녹음하고 데이터 중복을 확인하면 jam_bot 의 Rudess 구성과 같다. `--input-port` 로 수집 경로가
   이미 있다

**3번이 이 프로젝트 목표(§6 "내가 실제로 연주에 쓴다")와 가장 잘 맞는다.**

## 9. 산출물 경로

| 경로 | 내용 |
|---|---|
| `outputs/mehldau_lora/{from_base,from_tatum}/` | checkpoint_epoch1..8.pt, lora_weights.pt |
| `outputs/mehldau_lora/smoke/` | 1 epoch smoke |
| `outputs/mehldau_eval/` | report.json, 팔별 `.mid` / `.wav` |
| `outputs/continuous/mehldau/` | 런타임 smoke 리포트, played.mid |

`outputs/` 는 gitignore 다. 원격에 올린 것은 소스와 문서뿐이다.

## 10. 남은 사용자 한 단계

`outputs/mehldau_eval/{base,tatum,mehldau_from_base,mehldau_from_tatum}.wav`
초기 청취에서는 비슷한 느낌과 베이스 음·선율 반복이 보고됐다. 다음 비교에서는
동일 코드·반주·음색 조건을 고정하고, 스타일 변화와 화성 적합성을 구분해 평가한다.
