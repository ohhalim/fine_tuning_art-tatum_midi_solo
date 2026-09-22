# 코드 진행에 반응하는 생성 — 최소 구현과 A/B

작성 2026-09-22. 브랜치 `exp/chord-primer` (로컬 전용, push 안 함).
`musical_quality_verified: false` — 청취 리뷰 없음.

## 0. 먼저 가설을 검증했다 (그리고 틀렸다)

`RESUME_HANDOFF.md` 는 다음 작업으로 "`control_prefix_tokens` 에 코드 심볼을
더하는 형태" 를 제안했다. **가설이었고, 확인해 보니 성립하지 않는다.**

vocab 에 Stage B 코드 토큰이 있고(`TOKEN_STAGE_B_CHORD_ROOT_*`), checkpoint
임베딩도 547 행이라 ID 가 범위 안에 있다. 둘 다 "학습됐다" 를 뜻하지 않는다.
토큰화된 학습셋을 직접 세면 끝난다.

| 데이터셋 | 용도 | 최대 토큰 ID | 제어 토큰 등장 |
|---|---|---|---|
| `jazz_full` (2,777곡) | armB 사전학습 = 우리 베이스 | 377 | **0** |
| `roles/lead/tokenized` | armD LoRA 적응 = 우리 checkpoint | 388 | **0** |
| `roles_aug12` | Arm E (D1 에서 기각된 증강 팔) | 395 | 864 |

**우리 checkpoint 는 제어 토큰을 하나도 본 적이 없다.** 코드 토큰뿐 아니라
`ROLE_LEAD`·`TEMPO_*`·`BAR`·`COND_SEP` 전부다. 미학습 임베딩 행을 prefix 에
넣고 "conditioning 구현" 이라고 부르는 것은 근거 없는 주장이 된다.

### 덤으로 드러난 기존 결함

현재 런타임의 `build_primer(control_format="control_v1")` 는 매 primer 앞에
그 미학습 토큰 4개를 붙이고 있다. **이번 범위에서 고치지 않았다** (기본 경로
보존 지시). 별도 과제로 남긴다.

## 1. 선택한 경로 — 학습된 음표 vocabulary 의 chord primer

후보 둘 중 하나만 고르라는 지시에 따라:

| 후보 | 판단 |
|---|---|
| chord-aware constrained decoding / 약한 chord bias | 규칙 제어다. 모델이 반응하는 게 아니라 내가 거르는 것이고, "모든 음 강제 chord tone" 위험이 구조적으로 존재한다 |
| **기존 음표 vocabulary 의 chord primer** | **선택.** 학습된 토큰만 쓰고, 필터가 없으므로 비화성음·해결이 원리적으로 막히지 않는다 |

기존 자산 `inference/app/conditioning.build_request_conditioning_midi` 를
재사용한다(저음역 화성 가이드 MIDI). 호환성 확인:

```
chord primer 토큰 44개  ID 범위 38..370  제어 토큰 0개  (학습셋 최대 388)
```

**이것은 학습 기반 conditioning 이 아니다.** 모델은 코드 심볼을 모른다.
화성을 아는 음표를 깔아 주고 이어서 치게 하는 것이다. 보고서에서 이 둘을
섞지 않는다.

## 2. A/B 설계

같은 checkpoint·같은 seed·8마디. **primer 뒤의 코드 진행만 바꾼다.**

팔: `none`(가이드 없음) · `ii_V_I` · `minor` · `blues` · `shuffled`(ii_V_I 를
섞은 순서). seed 42/100/200 3회.

채점은 **생성된 음을 고정하고 잣대만 옮긴다.** 필터가 아니라 측정이다.

- `own` — 자기 진행 기준 chord-tone 비율
- `cross` — 다른 진행들 기준 (평균)
- `tritone` — **자기 진행을 6반음 전조한 것 기준.** 깨끗한 대조다

### 1/3 을 합격선으로 쓰지 않는다

코드음 4개 / 12 pitch class = 1/3 은 **음높이가 독립·균등일 때만** 성립한다.
재즈 솔로는 조성 안에 있고 음을 반복하며 음계도에 기댄다. 1/3 을 넘었다고
화성 반응의 증거가 아니고, 밑돌았다고 실패의 증거가 아니다. 참고선일 뿐이다.

## 3. 결과 (3 seed 평균)

| 팔 | valid 마디 | own | crossΔ | tritone | **triΔ** |
|---|---|---|---|---|---|
| none | **0~1/8** | – | – | – | – |
| ii_V_I | 7~8/8 | 0.549 | +0.175 | 0.186 | **+0.364** |
| minor | 6~7/8 | 0.473 | +0.130 | 0.238 | **+0.235** |
| blues | 7~8/8 | 0.500 | +0.045 | 0.252 | **+0.249** |
| shuffled | 7~8/8 | 0.495 | +0.078 | 0.188 | **+0.308** |

- **트라이톤 대조 triΔ: 평균 +0.289, 12/12 전부 양수, 범위 +0.164~+0.457**
- cross 대조 crossΔ: 실제 진행 평균 +0.117 vs shuffled +0.078 — **구별 안 됨**
- 생성 지연 p50 317~486ms (마디 1875ms), 다양성 pitch class 11~12/12

## 4. 판단 — 두 층으로 나눈다

**확인된 것**: chord primer 가 출력의 음높이 분포를 primer 화성 쪽으로
움직인다. 트라이톤 대조에서 12/12 일관되게 나왔고, 팔 사이에 다른 것은
primer 뿐이다.

**확인되지 않은 것**: 마디별로 코드 변화를 따라가는지. shuffled 대조가
실제 진행과 구별되지 않았다(+0.078 vs +0.117, 범위 겹침). 트라이톤 대조는
**"조성 안에 있다"와 "진행을 따라간다"를 구분하지 못한다** — 한 조성에
머무르기만 해도 트라이톤보다는 잘 맞는다.

즉 **"코드를 준 만큼 음이 달라진다" 까지가 근거 있는 주장이고,
"코드 진행을 따라 솔로한다" 는 아직 아니다.**

**음악적 품질은 주장하지 않는다.** 청취 리뷰 없음.

### 대조군 해석 주의

`none` 팔이 0~1/8 로 무너진 것은 "화성이 없어서" 가 아닐 수 있다. 이 팔의
primer 는 `[60]` 단일 음이라 **런타임 기본 경로(고정 conditioning MIDI)가
아니다.** "primer 가 빈약하면 생성이 무너진다" 와 "화성이 필요하다" 가
섞여 있다. 기본 경로와의 비교는 하지 않았다.

## 5. 실행

```
python scripts/run_chord_primer_ab.py --dry-run     # 모델 없이 primer 만 확인

FORCE_CPU=1 uv run python scripts/run_chord_primer_ab.py \
    --checkpoint <armD_lora/ckpt/checkpoint_epoch8.pt> \
    --bars 8 --bpm 128 --seed 42 --output-dir outputs/chord_ab
```

산출물: `report.json`, 팔별 `.mid`, 그리고 청취용 `*_with_chords.wav`
(솔로 + 그 마디 화성 가이드를 같이 깔아 귀로 대조할 수 있게 했다).

### 연속 런타임 연결 (opt-in)

```
FORCE_CPU=1 uv run python scripts/run_continuous_jazz.py ... --chord-primer
```

플래그가 없으면 기존 경로 그대로다. 두 경로를 실제로 돌려 확인했다
(8마디, seed 42, 독립 캡처):

| 경로 | 완주 | model 마디 | 오류 | 생성 p50 | 캡처 |
|---|---|---|---|---|---|
| 기본 (플래그 없음) | ✅ | 8/8 | 0 | 76ms | 44/44 손실 0 |
| `--chord-primer` | ✅ | 8/8 | 0 | 333ms | 214/214 손실 0 |

리포트에 `chord_primer_enabled`, `chord_primer_bar_count`,
`learned_chord_conditioning: false`, `chord_following_verified: false` 를 남긴다.

**opt-in 경로는 제어 prefix 를 붙이지 않는다.** §0 에서 확인했듯 그 토큰들이
미학습이기 때문이다. 기본 경로는 여전히 붙인다. 두 경로를 비교할 때
**화성 유무와 prefix 유무가 함께 달라진다는 교란**을 기억할 것. 이번에는
그 둘을 분리하지 않았다.

## 6. 다음 한 작업

**진행 추종을 분리하는 대조.** 현재 트라이톤 대조는 조성과 진행을 못 가른다.
같은 조성 안에서 **코드 순서만 다른** 두 진행을 겹치지 않는 코드로 만들어
비교하면 갈린다. 추가 학습 없이 가능하다.

## 7. 미검증 — 사용자 조작이 필요한 것

자동으로 주장하지 않는다. 한 번에 정리한다.

1. **실물 하드웨어 키보드** — 외부 장비 연결 후 `--input-port` 로 실행 필요
2. **FL Studio / DAW 오디오** — DAW 를 띄우고 포트를 연결해야 확인 가능
3. **청취 평가** — 위 `*_with_chords.wav` 를 듣고 화성이 맞게 들리는지 판단

WAV 는 사운드폰트 없이 `pretty_midi` 사인 합성으로 렌더링했다.
