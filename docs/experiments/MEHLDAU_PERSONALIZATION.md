# 멜다우 개인화 — 학습은 돌았고, 효과는 없었다

작성 2026-09-22. 브랜치 `exp/mehldau-personalization`.
**의존성: `exp/chord-primer` (PR #1482, 미병합) 위에 올렸다.** 평가가 그 브랜치의
`fit_window` / 런타임 플래그를 쓴다. #1482 가 먼저 정리돼야 한다.

`mehldau_style_verified: false` · `musical_quality_verified: false`
**사용자가 듣기 전까지 멜다우 스타일 성공을 주장하지 않는다.**

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

중복 없음(18곡 토큰 길이 전부 상이). 토크나이저 호환: 최대 토큰 ID 376,
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

동일 held-out primer, seed 42/100/200/300/400, 8마디, 같은 예산.

| 팔 | val_loss | 노트 | 노트/take | range | pc | IOI ms | vel | dur ms |
|---|---|---|---|---|---|---|---|---|
| base | 2.4780 | 228 | 5.70 | 26 | 5 | 60 | 60 | 640 |
| tatum | 2.4694 | 174 | 4.35 | 20.5 | 4.5 | 60 | 56 | 708 |
| **mehldau_from_base** | 2.4710 | 231 | **5.78** | **24** | **5** | **60** | **60** | 745 |
| **mehldau_from_tatum** | 2.4630 | 196 | **4.90** | **21** | **5** | 90 | **56** | 623 |

range / pc / IOI 는 **마디별 take 40개의 중앙값**이다. 전 seed 노트를 먼저
합치면 IOI 중앙값이 0 이 되고 range 는 합집합이 되어 무의미해진다
(실제로 한 번 그렇게 계산했다가 고쳤다).

| 측정 | 결과 |
|---|---|
| 학습 loss | **평평.** 8 epoch 내내 움직이지 않음 |
| val_loss 변화 | **−0.0071 / −0.0150.** 사실상 0, 게다가 누출로 편향 |
| 복사 위험 (8-gram) | **0.000** — 학습 16곡의 8-gram 이 생성에 한 번도 안 나옴 |
| 생성 descriptor | **출발점과 거의 동일.** from_base ≈ base, from_tatum ≈ tatum |

## 5. 판정 — 개인화가 걸리지 않았다

**네 측정이 일관되게 "변화 없음" 을 가리킨다.** 지배적 설명은 §2 의 누출이다:
베이스가 이미 18곡 전부를 학습했으므로 멜다우 전용 어댑터가 새로 배울 것이
남아 있지 않다.

복사 위험 0 은 좋은 신호지만, **어댑터가 거의 아무것도 바꾸지 않았다는 것과도
일치한다.** 외운 것이 없는 이유가 "잘 일반화해서" 인지 "학습이 안 걸려서" 인지
이 데이터로는 가릴 수 없다.

**53분 남았다는 사실이 학습 시간 충분을 뜻하지 않는다.** 실제로 학습은
CPU 에서 팔당 약 1.5분에 끝났다. 막힌 것은 시간이 아니라 데이터다.

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

**필요한 것은 GPU 도 시간도 아니다. 베이스가 보지 않은 멜다우 자료다.**

선택지:

1. **베이스를 다시 학습한다** — `jazz_full` 에서 멜다우 18곡을 빼고 armB 를
   재학습. 그러면 그 18곡이 진짜 held-out 이 된다. 비용은 2,777곡 재학습이고
   **원격 GPU 승인이 필요하다**
2. **멜다우 자료를 더 구한다** — 현재 코퍼스 밖의 트랜스크립션. 데이터 출처와
   권리 확인이 필요하고, 이 세션 범위 밖이다
3. **개인화 대상을 바꾼다** — 사용자 본인 연주. 베이스가 본 적 없음이
   보장되고 jam_bot 의 Rudess 구성과 같다. `--input-port` 로 수집 경로가
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
네 개를 듣고 **멜다우 어댑터가 실제로 다르게 들리는지** 판단. 측정으로는
"거의 같다" 가 나왔으므로, 귀로도 같다면 §8 의 데이터 문제를 먼저 풀어야 한다.
