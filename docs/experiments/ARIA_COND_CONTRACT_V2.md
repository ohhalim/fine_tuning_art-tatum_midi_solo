# 조건 계약 v2: 현재 코드 + 다음 계획 코드 + 변경까지 시간

작성 2026-10-09. 아스트라 결정. 새 adapter 학습과 새 자료 수집은 없다. 기존 12차원 계약과 그 실험은 그대로 보존한다. `musically_verified: false`.

## 왜 v2인가
- v1은 위치 i에 prefix 시각 t의 코드 하나만 준다. 토큰 순서가 음높이 → onset이라, 코드가 바뀐 뒤 첫 음의 음높이를 예측하는 위치는 이전 코드를 받는다(경계 첫 음 지연)
- 고정 lookahead로 현재 코드를 바꾸는 방법은 쓰지 않는다. 계획은 미리 알려진 값이라 누출은 아니다. 하지만 다음 음이 언제 올지 모르는 채로, 경계 전 음까지 새 코드로 너무 일찍 조건화할 수 있다
- 전환 직후 위치를 평가에서 분리하는 방법만으로는 최종 해결이 아니다
- v2는 계획 두 개(현재, 다음)와 남은 시간을 함께 준다. 모델은 참고할 정보를 얻지만, 정답인 다음 onset을 몰래 쓰지는 않는다. 경계 첫 음 반응이 실제로 좋아지는지는 실제 생성 검증 전까지 미확인이다

## 계약 (`scripts/aria_cond_contract_v2.py`)
- t: 위치 i의 prefix 시각. v1 정의 그대로이며 tokens[0..i]만 읽는다
- 위치마다 29차원

| 칸 | 뜻 |
|---|---|
| 0–11 | 현재 코드 chroma C(t) |
| 12–23 | 다음 계획 코드 chroma C(s) |
| 24 `current_known` | 계획 안이면 1. 계획 밖이나 unknown 구간이면 0 |
| 25 `has_next` | t 뒤에 다른 코드가 있으면 1. 계획의 마지막 코드면 0 |
| 26 `next_known` | 다음 구간이 알려진 코드(N.C. 포함)면 1 |
| 27 `delta_norm` | min(s − t, 4초) / 4초. 다음 코드가 없으면 0 |
| 28 `delta_clamped` | s − t > 4초면 1 |

- 경계: 변경 시각 s ≤ t면 이미 현재 코드다. 다음 코드는 엄격히 s > t다
- 같은 코드가 연달아 나오는 구간은 먼저 합친다. "다음"은 실제로 다른 코드다
- 4초는 128 BPM 4/4 두 마디(3.75초)를 덮는 상한이다. clamp 표시가 있어서, 먼 변경(has_next 1, clamp 1)과 변경 없음(has_next 0)이 구별된다
- delta는 계획의 tempo map으로 박 → 초 변환한다. 미래 연주의 onset, duration, target은 읽지 않는다
- unknown(계획 밖, 비어 있는 구간)과 N.C.(알려진 무음 코드: chroma 0, known 1)는 다른 벡터다
- 기존 12차원(v1)과의 변경점: 다음 코드 12칸과 플래그·시간 5칸이 늘었다. 입력 차원이 바뀌므로 v1 adapter와 섞어 쓰지 않는다. v1 실험 결과는 그대로 둔다

## fixture 시험 (`tests/test_aria_cond_contract_v2.py`, 11개 통과)
- 변경 직전, 정확히 경계, 직후
- 경계를 넘는 음의 음높이 위치는 이전 시각을 유지한다
- 같은 시각 다성음
- 쉼 동안 여러 변경을 지나면 첫 변경을 가리킨다
- `<T>` 5초 경계
- 박자 중간 tempo 변화(120 → 60 BPM)
- 먼 변경 clamp와 "없음"의 구별
- unknown과 N.C.의 구별, 계획 끝
- 같은 코드 연속 구간 합치기
- 다음 onset이 다른 두 suffix에서 공통 prefix 위치의 특성이 같다(누출 검사)
- 계획에서 다음 코드만 바꾸면 다음 chroma만 바뀐다

## 악보 3곡 read-only dry-run (`scripts/aria_cond_v2_dryrun.py`, `aria_cond_contract_v2/dryrun.json`)
모델과 학습 없이 토크나이저와 특성 생성기만 돌렸다. 특성이 나왔다고 학습 자격이 올라가지는 않는다(세 곡은 계속 `pipeline_test`).

| 곡 | 토큰 | 앞 무음(ms) | current_known / unknown | N.C. 위치 | has_next | clamp | 최대 delta(초) | onset에서 본 코드 변경 | lossy 구간 / 전체 | lossy 구간 위의 위치 | 원문 검산 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A_Foggy_Day | 240 | 230.8 | 240 / 0 | 0 | 240 | 0 | 1.85 | 28 | 13 / 46 | 24 | 6/6, onset 3/3 |
| All_The_Things_You_Are | 275 | 0.0 | 275 / 0 | 0 | 275 | 0 | 3.20 | 29 | 2 / 33 | 3 | 6/6, onset 3/3 |
| Billies_Bounce | 388 | 0.0 | 388 / 0 | 0 | 388 | 0 | 2.66 | 26 | 4 / 30 | 38 | 6/6, onset 3/3 |

- 29차원, 모든 값 유한
- 세 곡 모두 unknown 0, N.C. 0, clamp 0이었다. 최대 delta는 1.85–3.20초로 4초 상한 아래다. 실제 전환 간격에서 상한이 걸리지 않았다
- has_next가 모든 위치에서 1이다. 선율이 끝난 뒤에도 계획 코드가 이어지기 때문이다
- **lossy annotation:** degree(텐션·변화음), family로 접은 확장 kind(dominant-ninth·13th → 7), slash bass를 12차원 family chroma가 담지 못한다. 그런 구간을 lossy로 표시했다(A Foggy Day 13/46, ATTYA 2/33, Billie's 4/30). 이 chroma를 반음계 코드의 완전한 정답으로 쓰지 않는다
- **원문 검산(`scripts/aria_cond_v2_xmlcheck.py`, music21, 18행)**
  - 표본은 실행 전에 고정했다: 곡마다 5·10·15번째 코드 변경에서, 그 음의 음높이 위치와 onset 위치
  - 현재 코드, 다음 코드, delta가 music21이 원 XML에서 계산한 값과 18/18 일치했다
  - onset 행 9개는 그 시각(10 ms 토큰 간격의 절반 이내)과 바로 앞 음높이 토큰의 음높이를 가진 음이 원 XML에 모두 있었다(9/9)
- 실제 자료에서도 확인된 것: A Foggy Day 5번째 변경의 음높이 위치(7마디 4.5박)는 G7을 현재, Gm7을 다음(0.23초 뒤)으로 받는다. 그 음의 실제 onset은 9마디(Fmaj7)다. 쉼 동안 여러 변경을 지나는 경우가 실제 자료에도 있다

## 함께 정리한 결정 (아스트라)
1. **자료 자격:** "사용자 8 take만이 유일한 자료"라고 단정하지 않는다. 정확한 표현은 "현재 승인 범위 안에서 확보된 자료 없음"이다
   - 출처 annotation(`score_annotation`)도 사용 범위와 검증이 확인되면 후보가 될 수 있다. label_basis를 planned·performer_confirmed로만 고정하지 않는다
   - training 자격을 자동으로 켜지 않는다. 사용 범위, 분할, annotation 검증 근거를 기록한다. 새 외부 자료나 비용 승인은 추정하지 않는다. 사용자 녹음을 재촉하지 않는다
2. 8 take, held-out C 계획은 유지한다. 규모 충분성은 주장하지 않는다
3. **`<E>` 정책 후보:** "실제 곡 끝 `<E>`는 학습, 인위적 crop 끝 `<E>`는 target 제외, prefix는 loss 제외"
   - C팔 결과는 인위적으로 짧은 fixture의 종료 효과다. 모든 `<E>`를 빼는 근거가 아니다
   - 샘플마다 `true_end` / `crop_end` 출처 표시가 필요하다. 사용자 8마디 take도 음악적으로 끝났는지, 녹음만 멈췄는지를 metadata로 나눈다
   - `paired_take_v1` 검증기에 `end_kind`(true_end / crop_end) 필수 필드를 추가했다(시험 포함)
4. C팔 문구 교정: "끝내는 법을 못 배움"이 아니라 "이 학습에서 `<E>` target을 뺐고 생성 8개가 24토큰 상한까지 갔다"로 고쳤다. base의 종료 능력이 사라진 증거가 아니다. onset 역행 1건은 유효성 한계로 계속 적는다
5. 실제 평가에서는 코드 내부 위치와 경계 첫 음높이를 나누고, 맞음·반대·0 조건을 비교한다. 코드음 비율을 품질의 정답으로 강제하지 않는다. A/B 재학습은 지금 하지 않는다

## 중간 결론 (2026-10-09)
**확인된 것**
- 표상: Aria 토큰 왕복은 이 티그랑 변환본(약 32 ms 격자)의 음을 858/858 보존한다. CMT 경로는 최고음 대리·120 BPM 가정 전처리에서 곡별 23–60%를 잃는다(`REP_AUDIT.md`)
- 실행: 고정된 Aria(6.6억, fp32) + 작은 조건 adapter가 이 맥(MPS)에서 forward·backward·저장 재로드된다. 측정 시점 할당은 약 2.5 GiB였다(`ARIA_COND_SMOKE.md`)
- 합성: 코드 하나로 고정된 합성 문제에서 adapter는 조건에 따라 구별 음의 상대 선호를 바꿨다. loss 위치(prefix, `<E>`)는 절대 질량과 짧은 생성을 크게 바꿨다(`ARIA_COND_32STEP.md`, `ARIA_COND_LOSS_ARMS.md`)
- 계약: 경계 첫 음 문제에 대한 v2 특성(현재 + 다음 + delta)과 누출·정렬 검사를 만들었다. 실제 악보에서 원문과 대응이 맞았다

**확인되지 않은 것**
- 실제 음악에서의 코드 제어, 전환 반응, 생성 품질, 티그랑 스타일, 청취 평가
- v2 특성으로 학습했을 때의 효과(학습하지 않음)

**다음 학습 결정에 필요한 자료**
- 학습 자격이 있는 코드–연주 쌍 자료가 승인 범위 안에 **0**이다. 다음 중 하나가 있어야 한다
  - (a) `paired_take_v1` 계약에 맞는 연주 take: 128 BPM, 진행 A–D, C held-out, `end_kind` 표시, 검증기 통과. 수집은 사용자 일정이며 이 단계의 완료 조건이 아니다
  - (b) 사용 범위·권리·라벨 검증·분할 근거가 기록된 출처 annotation 자료
- 둘 중 하나가 생기기 전에는 합성 진단을 더 늘리지 않는다. 학습 결정은 자료 확보 여부를 보고 아스트라와 정한다

## 재개 경계 (아스트라 후속 결정)
- 자료 자격·분할 확정 → 실제 자료 소규모 pilot 사전 등록 → 학습. 이 순서를 건너뛰지 않는다
- 로컬 BebopNet XML은 프로젝트의 사용 조건 확인 기준을 채우지 못했다(법적 사용 불가 확정은 아님, `SCORE_PAIR_ENTRY.md`). 다음 출처 후보는 WJazzD다(공식 페이지 기준 DB는 ODbL, 사용자 다운로드 승인 필요)
- v2는 prefix 기준 첫 다음 코드까지 올바르게 읽는 계약이다. 전환 첫 음 문제의 해결을 입증하지 않았다. 쉼이 여러 변경을 건너는 음(악보 32 family에서 2.2%)은 표현에 없다. 평가에서 별도 층으로 남긴다

## late-injection adapter 경로 중단 (2026-10-09, 아스트라 결정)
**판단:** "고정 Aria의 마지막 block 입력에 `Linear(29 → 1536)` 조건 adapter를 붙이고, WJazzD 24곡으로 128 update 학습"하는 경로를 제품 후보에서 뺀다. 프로젝트 전체를 포기하는 것이 아니다. 일반적인 Aria 조건 학습이나 다른 학습법이 불가능하다는 판정도 아니다.

| 실험 | 같은 validation 8곡: pitch NLL base / 맞음 / 틀림 | continuation NLL base / 맞음 / 틀림 | 맞음 < base 곡 | 맞음 < 틀림 곡 | 생성 평가 split과 유효 수 |
|---|---|---|---|---|---|
| pilot(전 위치 주입, `WJAZZD_PILOT.md`) | 1.954 / 1.973 / 2.216 | 2.525 / 2.569 / 2.633 | 4/8 | 8/8 | test 앞 4곡 × seed 2: base 8/8, 맞음 5/8, 틀림 4/8 |
| gate(음높이 예측 위치만, `WJAZZD_GATE_PILOT.md`) | 1.954 / 2.071 / 2.367 | 2.525 / 2.518 / 2.617 | 1/8 | 8/8 | validation 앞 4곡 × seed 2: base 8/8, 맞음 4/8, 틀림 8/8 |

- **관측된 것은 조건 구별까지다.** 두 실험 모두 맞는 코드 계획이 틀린(+1 반음) 계획보다 pitch NLL이 낮았다(teacher forcing)
- **채택하지 않는 이유:** 채택에 필요한 base 대비 pitch 예측 순효과와 생성 유효성이 두 실험 모두 미충족이다
  - gate 실험의 continuation NLL(2.518)은 base(2.525)보다 조금 낮았다. 그러니 "모든 예측 성능이 나빠졌다"로 요약하지 않는다
- 두 생성 평가는 split이 달라(test 대 validation) 서로 직접 비교하지 않는다
- **원인은 미해결이다**
  - 주입 위치 대조(`WJAZZD_PILOT.md` 후속 진단 2)는 기존 pilot checkpoint와 같은 prefix에서 현재 위치 주입을 빼면 허용 onset 질량이 국소적으로 회복된다는 관측이다. 이 관측은 여전히 유효하다
  - gate 실험은 새로 학습한 별도 checkpoint에서 오류가 남았다. 결론은 "그 위치 개입을 빼서 얻은 국소 회복이, 따로 학습한 gate 모델의 생성 안전성으로 일반화되지 않았다"까지다
  - 남은 후보: 과거 위치 주입의 attention 경로, 학습 신호 자체(train 24곡 쪽 공통 이동), 자료 편향. 모두 확인하지 않은 후보다
- 기존 base도 코드 제어, 음악 품질, 티그랑 목표를 달성했다고 주장하지 않는다

**보존(로컬, 기본 런타임에는 연결하지 않음).** `inference/` 아래에 이 adapter를 참조하는 코드가 없음을 확인했다.

| 파일 | sha256 앞 16자리 |
|---|---|
| `/Users/ohhalim/git_box/wjazzd/wjazzd.db` | af6a0d9debf042c3 |
| `/Users/ohhalim/git_box/wjazzd/PROVENANCE.json` | e517c80306248102 |
| `/Users/ohhalim/git_box/wjazzd/pilot/data.json`(분할·덩어리·조건) | fc4e8ea2e91c1098 |
| `/Users/ohhalim/git_box/wjazzd/pilot/adapter_wjazzd_pilot_diagnostic_only.pt` | dbfe8333b0e891be |
| `/Users/ohhalim/git_box/wjazzd/pilot/result.json` | a15497fe832aef96 |
| `/Users/ohhalim/git_box/wjazzd/pilot/generation_regen.json`(재생성, 원 집계 16/16 일치) | 9a162edd830fc758 |
| `/Users/ohhalim/git_box/wjazzd/pilot/generation_base.json` | 9ebc4f50bbc600c0 |
| `/Users/ohhalim/git_box/wjazzd/gate/adapter_wjazzd_gate_diagnostic_only.pt` | 53a7183f527ad9c4 |
| `/Users/ohhalim/git_box/wjazzd/gate/gate_result.json`(생성 토큰 포함) | 53240464da3f49ef |

**재사용할 것:** WJazzD 감사·분할·창 도구, 29차원 v2 계약과 fixture, 토큰 유효성 검사기, 주입 위치 진단, RNG 재현 도구(`WJAZZD_AUDIT.md`, `WJAZZD_PILOT.md`, `WJAZZD_GATE_PILOT.md`).

**재개 조건.** 예산을 늘리는 것만으로는 재개하지 않는다. 새 근거가 있을 때만 재개한다.
- 피아노 생성 능력 보존과 코드 제어를 함께 검증하는 학습 구조와 자료 계획
- 지금까지 반복 진단에 쓴 validation·test와 구별된 평가 세트
- 지금은 새 모델 탐색이나 학습을 자동으로 시작하지 않는다
