# 악보 헤드 3곡 변환 pilot: 원 주석 → 정렬된 선율·코드

작성 2026-10-08. 아스트라가 paired 계약 다음 단위로 정했다. 새 학습·다운로드·새 검색 없음. `musically_verified: false`.

## 질문 (하나)
로컬 MusicXML 악보에 있는 선율과 코드 기호를, 빠뜨리거나 바꾸지 않고 tick 단위 MIDI와 코드 구간으로 옮길 수 있는가.

- **용도는 "악보 기반 화성 조건 데이터의 변환 검증"에 한정한다.** 이 자료는 즉흥 연주, 피아노 연주, 티그랑 스타일 자료가 아니다
- 이 단계가 성공해도 스타일 문제는 풀리지 않는다

## 선택 (결과 전 고정)
- 원본: `bebopnet-code/resources/xmls`의 51개 파일
- 규칙: basename으로 정렬한다. `_short`, 끝의 `_숫자` 템포 변형, `DespacitoS`를 원곡 family 하나로 묶는다. 그러면 33 family가 된다. 앞에서 세 family를 고르고, short도 변형도 아닌 원 파일을 변환한다
- 결과: **A_Foggy_Day, All_The_Things_You_Are, Billies_Bounce**. 쉬운 곡이나 성공한 곡으로 다시 고르지 않았다

## 원칙
- 원 MusicXML이 정본이다. 파생물은 `derived.mid`(템포 트랙과 선율)와 `conversion.json`(변환 manifest)이다. DAW 원본(`raw.mid`)으로 부르지 않는다
- 라벨 출처: `label_basis: score_annotation`, `source_type: score_head`, `rights_status: unknown`. 출처 설명에 "테마 선율, 즉흥 아님, 피아노 연주 아님"을 남긴다
- 원본 해시, 변환 코드 해시, 마디 → 절대 박 매핑, 코드 위치(마디, offset division)를 보존한다
- 반복 기호는 **기보대로** 둔다(펼치지 않음). 한 원본에 두 정책을 섞지 않는다
- 반주는 생성하지 않았다. 코드는 악보 주석 구간으로만 있다
- velocity는 악보에 없어서 80으로 고정했다

## 문법 지원 (`scripts/score_pair_convert.py`)
| 항목 | 처리 | 세 곡에 있었나 |
|---|---|---|
| divisions | 마디마다 갱신, 박 = 4분음표, tick = 박 × 480 | 2, 6, 6. 480으로 나누어떨어져 tick이 정확하다 |
| 셋잇단(time-modification) | `<duration>`이 실제 길이이므로 그대로 쓴다 | ATTYA 8음, Billie's 3음 |
| backup / forward | 커서를 앞뒤로 옮긴다 | 없음 |
| 붙임줄(tie) | 같은 음높이에서 앞 음 끝 = 다음 음 시작이면 합친다. 짝 없는 stop은 따로 기록한다 | 23, 27, 13번 합침 |
| 못갖춘마디·불규칙 마디 | 실제 길이로 다음 마디 시작을 정하고 기록한다 | 없음 |
| 박자표·템포 변화 | 위치와 함께 기록한다 | 박자표·템포 각 1개, 변화 없음 |
| 코드 offset | 현재 위치 + offset으로 둔다 | A Foggy Day 7개 |
| 전위(slash bass) | bass를 기록한다 | Billie's 2개(F7/A) |
| 무코드(kind none) | `no_chord` 구간. 앞 코드로 채우지 않는다 | 없음 |
| 첫 코드 앞 구간 | `unlabeled` 구간. 채우지 않는다 | 없음 |
| 대응 없는 kind | quality를 비워 두고 위치와 함께 목록에 넣는다 | 없음 |
| degree(텐션·변화음) | 원문 그대로 보존한다. quality는 기본 family로 둔다 | D7(add b9) 2, C7(#5) 1, D7(add #9) 2 |
| 반복·엔딩·segno·coda | 기보대로 두고 목록에 넣는다 | 없음 |
| 꾸밈음 | 길이가 없어 건너뛰고 목록에 넣는다 | 없음 |

- kind → quality: dominant → 7, minor-seventh → m7, major-seventh → maj7, major → maj, half-diminished → m7b5, diminished-seventh → dim7, major-sixth → 6, minor-sixth → m6. dominant-ninth와 dominant-13th → 7이며 원 kind를 같이 남긴다

## 결과 (`docs/experiments/score_pair_pilot/summary.json`, 원자료 `/Users/ohhalim/git_box/paired_score_pilot/`)
| 곡 | 마디 | 박 | 템포 | 선율 음(붙임줄 합친 뒤) | MIDI 왕복 누락 / 추가 | tick 오차 최대 | 코드 구간 / quality 대응 | 지원 불가 |
|---|---|---|---|---|---|---|---|---|
| A_Foggy_Day | 34 | 136 | 130 | 75 | 0 / 0 | 0 | 46 / 46 | 0 |
| All_The_Things_You_Are | 36 | 144 | 150 | 87 | 0 / 0 | 0 | 33 / 33 | 0 |
| Billies_Bounce | 24 | 96 | 180 | 126 | 0 / 0 | 0 | 30 / 30 | 0 |

- 세 곡 모두 변환됐다. 실패하거나 건너뛴 곡은 없다
- MIDI 왕복 검사는 변환기가 쓴 `derived.mid`를 다시 읽어 변환기 자신의 음 목록과 맞춘 것이다. MIDI 쓰기만 확인한다. 해석이 맞는지는 아래 독립 검산으로 본다

### 독립 검산 (music21 10.5.0, 전 마디)
변환기 코드를 쓰지 않는 별도 파서(music21)로 원 XML을 읽었다. 음은 `stripTies` 뒤 음높이·시작·끝 tick을, 코드는 순서·시작 tick·근음·bass·quality를 맞췄다.

| 곡 | 음: music21 / derived | 차이 | 코드: music21 / 변환기 | 차이 행 |
|---|---|---|---|---|
| A_Foggy_Day | 75 / 75 | 없음 | 46 / 46 | 0 |
| All_The_Things_You_Are | 88 / 87 | 1곳 | 33 / 33 | 0 |
| Billies_Bounce | 126 / 126 | 없음 | 30 / 30 | 0 |

- **차이 1곳: ATTYA 18–20마디 B4.** music21은 두 음(71.5–72박, 72–77.5박)이고, 변환기는 한 음(71.5–77.5박, 붙임줄 3개 합침)이다
- 원 XML을 직접 읽었다
  - 18마디 마지막 B4(duration 3)에 `<tie type="start"/>`가 있다
  - 19마디 B4(duration 24)에 `<tie type="stop"/>`과 `<tie type="start"/>`가 있다
  - 20마디 첫 B4(duration 9)에 `<tie type="stop"/>`이 있다
  - 원문 표기는 세 음을 한 음으로 잇는다. **변환기 결과가 원문과 맞다.** music21이 이 연결의 첫 붙임줄을 합치지 않은 이유는 조사하지 않았다
- 코드는 세 곡 109개 모두 시작 tick, 근음, bass, quality가 일치했다. A Foggy Day의 offset 코드 7개도 포함이다
  - 일치한 것은 quality family까지다. degree와 변화음까지 같은지는 검산하지 않았다(dominant-ninth·13th → 7). degree 원문을 보존했다고 해서 반음계 코드의 정답이 완전하다고 쓰지 않는다
  - offset을 소리 위치에 적용하는 해석은 MusicXML의 harmony 기본값(`sound` = yes)에 대한 내 이해다. 두 파서가 같은 위치를 냈다

## 자격 (목적별, 이번은 pipeline 전용)
- `pipeline_test`: 참(세 곡). 변환·정렬 시험에 쓸 수 있다
- `training`: 거짓. 권리 미확인, 테마 선율, 분할·라벨 정확도 승인 없음
- `musical`: 거짓. 사람 검토 없음
- 스타일 자료나 일반화 벤치마크 corpus에 넣지 않는다. 원본과 파생 파일을 자동으로 업로드하지 않는다. 저장소에는 음 목록 없는 요약만 올렸다

## 함께 고친 것: 검증기의 자격 판정
- 아스트라 지적: `paired_validate.py`가 기술 검사 통과만으로 `corpus_eligible`을 참으로 냈다. 변환 시험 가능, 학습 승인, 음악 검증을 한 칸에 섞은 것이다
- 수정: `eligibility = {pipeline_test, training, musical}`로 나눴다
  - `training`은 검증기가 참으로 만들지 않는다. 권리, 분할, 라벨 정확도에 대한 별도 승인이 필요하다
  - 사람 검토 통과는 `musical`만 바꾼다. 시험을 추가했다

## 시험
- `tests/test_score_pair_convert.py` 3개: 3마디 붙임줄 연결, offset·전위·무코드·대응 없는 kind, 마디 매핑·셋잇단 길이
- `tests/test_paired_validate.py` 9개: 위 자격 분리 포함

## 다음 (아스트라 제안, 지금은 하지 않음)
- 이 실제 자료로 Aria 조건 주입 학습 경로를 배치 하나로 돌릴 수 있는지 검토한다(메모리, gradient, 체크포인트 보존). 학습을 미리 키우지 않는다
