# WJazzD 화성 조건 학습 pilot (사전 등록)

작성 2026-10-09. 아스트라 결정.

- **이 pilot은 관악 단선율로 화성 조건 사용을 검증하는 첫 시험이다.** 피아노 양손 연주나 티그랑 스타일 적응의 자료가 아니다. 이 구분을 완료 기준에서도 유지한다
- 사용자 승인 범위는 "학습 전 점검까지"다. 실제 optimizer 실행은 이 계획을 사용자에게 한 번 보여주고 승인받은 뒤에만 한다. 새 다운로드·업로드는 없다
- `musically_verified: false`

## 질문 (하나)
고정된 Aria에 v2 조건(29차원)을 넣는 작은 adapter를 실제 WJazzD 솔로로 128 update 학습하면, 처음 보는 곡에서 맞는 코드 계획이 틀린 계획보다 음높이 예측을 낫게 하는가.

## 자료 분할 (`scripts/wjazzd_pilot_data.py`, 결과 전 고정)
- 후보 곡: 아래 조건을 모두 만족하는 곡(composition)이다. 211곡, 솔로 317개다
  - `template`이 Blues나 I Got Rhythm이 아니다. 거대한 두 진행 그룹이 작은 표본을 지배하지 않게 하려는 것이다
  - 박자 4/4
  - 피아노가 아니다. 피아노 6솔로는 이 pilot 밖에 보존한다. 다만 Aria 사전학습 자료에 없었다고 주장하지는 않는다
- 곡 순서를 `random.Random(20261009)`로 섞는다. 곡마다 솔로 하나를 같은 난수로 고른다
- 유효 창이 하나 이상인 곡만 순서대로 받는다. 앞 24곡은 train, 다음 8곡은 validation, 다음 8곡은 test다
  - 건너뛴 곡 수와 이유를 기록한다. 부족하면 기준을 완화하지 않고 가용 수를 보고한다
- 곡 단위로 먼저 나누고, 자르기는 그 뒤에 한다. 전조 증강은 없다
- 남은 한계: template 표시가 없는 곡 안의 contrafact는 확인하지 않았다
- test는 설계나 예산 선택에 쓰지 않는다. 예산은 128 update로 고정이라 고를 것이 없다

## 창과 마스크
- 코드 부착은 실제 시각 기준이다(v2와 같음). 0–30 ms 이른 음을 박에 맞추거나 새 코드를 정답으로 강제하지 않는다
- **유효 음:** 다음 셋이 모두 "정확 대응" 코드인 음이다. 손실 대응, 대응 없음, NC, 계획 밖(unknown)은 모두 유효하지 않다
  - 그 음의 음높이 토큰 위치에서 본 현재 코드
  - 같은 위치의 다음 코드(다음 코드가 없으면 그 칸은 따지지 않음)
  - 그 음의 실제 onset 시각의 코드
  - 제외 집계에서는 NC와 unknown을 나눈다
- **창:** 유효 음이 연속된 구간을 150음 이하 덩어리로 자른다. 32음보다 짧은 덩어리는 버린다
  - 덩어리마다 따로 토큰화한다(앞 무음 제거, 계획도 같은 만큼 이동)
  - 512토큰을 넘으면 뒤에서부터 음을 줄인다
  - 음을 chord tone인지에 따라 거르지 않는다
- **loss target:** 덩어리의 첫 16음은 prefix라 target에서 뺀다. 17번째 음의 음높이 토큰부터 pitch·onset·dur target에 CE를 건다
  - `<E>` target은 뺀다. 덩어리의 끝은 잘라낸 끝(`crop_end`)이다. 입력에서 지우거나 생성에서 막지는 않는다
- velocity는 80 고정이다. loudness를 velocity로 바꾸는 규칙은 검증하지 않았다(변환 한계)
- 제외 수를 곡별·품질별로 기록한다

## 모델과 학습
- 고정된 Aria(로컬 F32 6.6억)에 새 0 초기화 `Linear(29 → 1536, bias 없음)`를 붙인다. 주입점은 16번째 block 입력으로 기존과 같다
- 입력은 v2 특성 29차원이다(`aria_cond_contract_v2.py`)
- Adam lr 1e-3, batch 1, seed 0, **정확히 128 update**, sweep 없음
- 매 update마다 train 곡을 균등하게 고르고, 그 곡의 덩어리를 균등하게 고른다(`random.Random(0)`)
- 학습 전 검사(optimizer 없음)
  - 0 adapter일 때 출력 = base
  - 마스크 위치
  - 조건 시간축(표본 위치의 현재·다음 코드를 계획과 대조)
  - adapter 저장·재로드
  - 512토큰 덩어리 하나의 forward·backward 시간과 메모리
- 즉시 중단: OOM, 유한하지 않은 값, base 변경, MPS 미지원 연산, pressure level 4, wall 15분

## 평가 (test 8곡, 같은 prefix와 target)
- 조건
  - base(adapter 없음)
  - adapter + 맞는 계획
  - adapter + 틀린 계획
  - adapter 비활성(29차원 0 벡터). bias가 없으므로 구조상 base와 같다. 독립 대조로 과장하지 않는다
- **틀린 계획:** 현재·다음 chroma 24칸만 +1 반음 전조한다. 시각, 변경 수, known 플래그, delta는 그대로 둔다
  - +6(트라이톤)은 쓰지 않는다. 딸림7화음의 3도·7도가 그대로 남기 때문이다
  - family 10종 중 +1 전조로 같은 chroma가 되는 것은 없다. 그래도 경우마다 같은지 확인해 기록한다
- **주지표: pitch NLL.** continuation 음높이 위치에서 −log Σ_v p((piano, p, v))로, velocity를 합한 값이다
  - 함께 적는 값: (음높이, velocity 80) 결합 토큰 NLL, 전체 continuation NLL(pitch·onset·dur, `<E>` 제외)
  - 음높이와 세기가 한 토큰에 묶여 있다는 한계를 적는다
- 곡 평균을 우선한다(덩어리 평균 → 곡 평균). 층별로 따로 낸다: 현재, 경계 첫 음, 표현 없음, anticipation(주석 박 코드 ≠ 시각 코드). 표본 수도 함께 낸다
- **판정 기술(결과 전 고정)**
  - 조건 사용 관측: 곡 평균 pitch NLL이 맞는 계획 < 틀린 계획이고, test 8곡 중 7곡 이상이 같은 방향이다
  - 부분: 같은 방향이 5–6곡이다
  - 관측 안 됨: 4곡 이하다
  - 맞는 계획이 base보다 좋아도 틀린 계획과 차이가 없으면 조건 사용 성공이 아니다
- validation 8곡은 같은 지표를 정보로만 낸다

## 자유 생성 (정보 항목)
- test 중 앞 4곡(사전 고정) × seed 1, 2 × 맞는/틀린 계획
- prompt는 그 곡 첫 덩어리의 prefix 16음이다. 최대 96토큰, temperature 1, min_p 0, `<E>`에서 멈춘다. 채택 규칙과 예산은 같다
- 기록: 반복(가장 긴 음높이 n-gram 반복), onset 역행, `<E>` 여부, 길이, 코드 변경 전후 음의 코드음 비율(서술만)
- 코드음 비율만으로 품질을 판정하지 않는다. NLL 개선을 생성 제어 성공으로 올리지 않는다

## 자료 구성 결과 (`wjazzd_pilot/data_summary.json`, 로컬 `/Users/ohhalim/git_box/wjazzd/pilot/data.json`)
- 후보 곡 211개(솔로 317)를 섞은 순서대로 받아 40곡을 채웠다
  - 유효 덩어리가 없어 건너뛴 곡: 20
  - 32음보다 짧아 버린 덩어리: 169
  - **토큰화 뒤 모든 위치의 현재·다음 코드가 정확 대응인지 다시 확인해 버린 덩어리: 94**
    - 원인 후보: `<T>` 경계가 prefix 시각을 손실 코드 구간으로 옮긴다. 확인하지 않았다
    - 이 재확인은 위 "유효 음" 정의를 위치 단위로 적용한 것이다. 사전 등록 문장에는 따로 적지 않았던 단계라 여기에 밝힌다

| 분할 | 곡 | 덩어리 | target 토큰 | 음높이 target | 현재 / 경계 첫 음 / 표현 없음 | anticipation |
|---|---|---|---|---|---|---|
| train | 24 | 68 | 21965 | 7204 | 6655 / 535 / 14 | 105 |
| validation | 8 | 9 | 1971 | 645 | 550 / 91 / 4 | 19 |
| test | 8 | 26 | 9374 | 3071 | 2786 / 283 / 2 | 64 |

- 음 제외 사유(첫 사유 기준, 40곡 전체 음): current_lossy 4576, next_lossy 2575, current_unmapped 729, current_nc 344, next_unmapped 290, next_nc 90, onset_lossy 14, onset_unmapped 2
- 제외에 많이 걸린 코드 표기: Eb79# 808, NC 434, C79b 434, Ebsus79 362, F79# 297, G79b 264, C7911# 237, D79b 215
  - 딸림7화음의 텐션 표기(79#, 79b 등)가 손실 대응이라 많이 빠졌다. 그래서 남은 자료는 텐션 표기 없는 코드 쪽으로 치우친다(편향으로 기록)
- 악기: train ts 8, tp 5, as 4, cor 2, ss 2, bs·vib·cl 1 / validation as 3, cl 2, ts 2, cor 1 / test as 4, ts 2, tp 1, cor 1
- 한계 예: validation의 Ornithology는 흔히 How High The Moon의 contrafact로 알려져 있다(검증 안 함). template 표시가 없는 contrafact가 분할을 넘을 수 있다

## 학습 전 검사 결과 (`wjazzd_pilot/precheck.json`, optimizer 없음)
- 0 adapter = base: 최대 차이 0.0
- 마스크와 형태: 덩어리 103개 모두 통과
  - 첫 target이 17번째 음의 음높이 토큰이다
  - prefix target이 없다
  - `<E>` target이 없다
  - 조건 행이 토큰 수와 같고 29차원이다
- 조건 시간축: 무작위 표본 15위치의 현재 chroma가 DB에서 직접 찾은 코드와 15/15 같다
- 틀린 계획이 맞는 계획과 같은 chroma가 되는 위치: 0
- 저장·재로드: 일치
- 비용: 가장 긴 train 덩어리(464토큰)의 forward·backward가 0.405초다
  - 측정 시점 MPS 할당 2.51 GiB, pressure level 1
  - 128 update는 1분 안팎, 평가·생성을 합쳐 5분 안팎으로 예상한다(추정)
- base 파라미터 sha256 불변
- **다음 단계(실제 128 update 학습)는 사용자 승인 뒤에 한다**
