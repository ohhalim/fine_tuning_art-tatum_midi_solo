# 악보 헤드 자료의 학습 진입 판정 (BebopNet XML)

작성 2026-10-09. 아스트라 결정. 로컬 원문·메타데이터만 읽었다. 다운로드, 학습, training 자격 자동 승격은 없다. `musically_verified: false`.

## 질문 (하나)
로컬 BebopNet XML(51개 파일)을 코드 조건 학습 자료로 쓸 수 있는가. 출처, 사용 조건, 라벨 손실, family 단위 분할 가능성으로 판정한다.

## 판정: 현재 근거로는 학습 부적격 → 학습 중단 상태 유지
- 사용 조건을 확인할 근거가 로컬에 없다. 아래 "출처와 사용 조건"을 보라
- 라벨 손실, 분할 가능성, v2 표현 한계는 학습 가능 여부와 별개로 기록했다. 기술적으로는 family 단위 분할이 가능하다
- 확보할 수 있는 다음 출처 하나: **WJazzD**(Weimar Jazz Database)
  - 즉흥 솔로와 코드 주석이 정렬돼 있다
  - 사용자 다운로드 승인이 필요하다(현재 미승인). 라이선스는 출처에서 직접 확인해야 하며, 이 저장소 문서에는 기록이 없다
  - 역할은 화성 제어 쌍 자료다. 기존 조사 문서 기준으로 456곡 중 피아노는 약 6곡이고 대부분 관악이다. 스타일 자료가 아니다
- 다른 길(사용자 결정): 사용자가 이 XML의 사용 범위(예: 로컬 연구 전용, 재배포 없음)를 명시하면 그 범위에서 다시 판정할 수 있다. 자동으로 추정하지 않는다

## 출처와 사용 조건 (확인한 사실)
- 저장소 `shunithaviv/bebopnet-code`의 `LICENSE`는 MIT다. 대상은 "the Software"다. 악보 내용의 권리를 말하지 않는다. 저장소 라이선스를 악보 권리로 대신하지 않는다
- README는 XML을 "BebopNet이 즉흥 연주를 생성할 때 쓰는 XML과 backing track"으로 소개한다. 학습 자료로 배포한다는 문구는 없다. 학습 자료는 별도로 `resources/dataset`에 두라고 안내한다
- git 기록: `resources/xmls`의 마지막 변경은 커밋 `9dfa800`(2020-11-09, "Update README.md") 하나다
- 파일 메타데이터
  - MuseScore 2.0–3.3에서 내보낸 파일이다(인코딩 날짜 2016–2019). 전사자·출처·권리 정보가 대부분 비어 있다
  - 출처 URL(musescore.com 사용자 업로드 악보)이 있는 family는 8개다. 그 악보들의 라이선스는 로컬에 기록이 없다
  - rights 문구가 있는 family는 4개다: "- Transcribed by Markus Schulze (2016) -", "1930", "QuarkMuse, Inc", "Copyright ©". 어느 것도 사용 허락을 말하지 않는다
  - 메타데이터 오류: `Giant_Steps.xml`과 `Giant_Steps_200.xml`의 제목·출처 URL이 "Fly me to the moon"으로 돼 있다. `simple_songs.xml`의 제목은 "GREEN DOLPHIN STREET"인데 credit은 "שירים פשוטים"(쉬운 노래들)이다. 둘 다 내용은 서로 다르다(아래 중복 검사)
- `ours_iiVI_F.xml`은 bebopnet-code에서 git이 추적하지 않는 파일이다. music21이 2026-10-03에 만들었다. 우리 T1 작업에서 만든 것으로 보이며 BebopNet 출처에서 뺐다. 그래서 33 family 중 BebopNet family는 32개다
- 곡 분류(제목 기준): 재즈 헤드 25, 팝 5(Despacito, Dance Monkey, Never Gonna Give You Up, Juice, my_love), 연습곡 1(simple_songs), 솔로 전사로 보이는 것 1(Confirmation, 제목 "Confirmation Bird Solo")

## family별 감사 (`scripts/score_pair_entry_audit.py`, `score_pair_entry/audit.json`)
대표 파일(short·템포 변형이 아닌 원 파일) 하나씩을 메모리에서 파싱했다. 변환 파일은 쓰지 않았다.

| family | 파일 | 분류(제목 기준) | 마디 | 음 | 코드 구간 | lossy | 대응 없는 kind | 기타 미지원 | 출처 URL | rights 문구 | 다음 음 코드 표현 없음 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A_Foggy_Day | 2 | jazz_head_by_title | 34 | 75 | 46 | 13 | — | — | — | — | 11/75 |
| All_The_Things_You_Are | 2 | jazz_head_by_title | 36 | 87 | 33 | 2 | — | — | — | — | 1/87 |
| Billies_Bounce | 1 | jazz_head_by_title | 24 | 126 | 30 | 4 | — | — | — | — | 0/126 |
| Black_Orpheus | 2 | jazz_head_by_title | 32 | 69 | 42 | 13 | — | — | — | — | 5/69 |
| Blue_Bossa | 2 | jazz_head_by_title | 32 | 74 | 26 | 6 | — | — | 있음 | - Transcribed by Markus Schulze (2016) - | 0/74 |
| Blue_Monk | 1 | jazz_head_by_title | 24 | 110 | 32 | 28 | — | — | — | — | 0/110 |
| Cheese_Cake | 2 | jazz_head_by_title | 56 | 170 | 62 | 29 | — | — | — | — | 1/170 |
| Confirmation | 1 | solo_transcription_by_title | 32 | 182 | 50 | 3 | — | — | — | — | 0/182 |
| Dance_Monkey | 1 | pop | 24 | 148 | 24 | 0 | — | — | — | — | 0/148 |
| Despacito | 2 | pop | 24 | 196 | 24 | 0 | — | — | 있음 | — | 2/196 |
| Donna_Lee | 2 | jazz_head_by_title | 32 | 208 | 33 | 1 | — | — | — | — | 0/208 |
| Fly_me_to_the_moon | 2 | jazz_head_by_title | 32 | 81 | 39 | 3 | — | coda | 있음 | — | 6/81 |
| Four_Brothers | 2 | jazz_head_by_title | 32 | 183 | 49 | 6 | — | — | — | — | 0/183 |
| Giant_Steps | 2 | jazz_head_by_title | 16 | 26 | 26 | 0 | — | — | 있음 | — | 7/26 |
| How_high | 2 | jazz_head_by_title | 32 | 84 | 32 | 0 | — | — | — | — | 0/84 |
| I_Got_Rhythm | 1 | jazz_head_by_title | 32 | 75 | 59 | 6 | augmented, diminished | — | — | 1930 | 8/75 |
| It_dont_mean_a_thing | 1 | jazz_head_by_title | 32 | 101 | 32 | 0 | — | — | 있음 | — | 3/101 |
| Juice | 1 | pop | 16 | 94 | 21 | 4 | suspended-fourth | — | — | — | 0/94 |
| Never_Gonna_Give_You_Up | 1 | pop | 20 | 111 | 26 | 0 | — | — | — | — | 0/111 |
| Over_The_Rainbow_with_chords | 1 | jazz_head_by_title | 32 | 109 | 59 | 6 | suspended-fourth | — | 있음 | QuarkMuse, Inc | 5/109 |
| Song_for_My_Father_-_Horace_Silver | 1 | jazz_head_by_title | 40 | 142 | 29 | 5 | — | repeat | 있음 | — | 1/142 |
| There_Will_Never_Be_Another_You | 2 | jazz_head_by_title | 32 | 91 | 32 | 9 | — | — | — | — | 6/91 |
| Well_You_Neednt | 1 | jazz_head_by_title | 32 | 149 | 31 | 0 | — | — | 있음 | — | 0/149 |
| chega | 2 | jazz_head_by_title | 68 | 184 | 75 | 24 | — | — | — | — | 3/184 |
| greendol | 2 | jazz_head_by_title | 32 | 60 | 35 | 11 | — | — | — | — | 2/60 |
| just_friends | 2 | jazz_head_by_title | 32 | 68 | 35 | 0 | — | — | — | Copyright © | 1/68 |
| moose | 1 | jazz_head_by_title | 32 | 170 | 43 | 4 | — | — | — | — | 0/170 |
| my_love | 1 | pop | 24 | 81 | 24 | 2 | augmented | — | — | — | 0/81 |
| recordame | 2 | jazz_head_by_title | 32 | 121 | 42 | 4 | — | — | — | — | 6/121 |
| simple_songs | 2 | exercise | 32 | 65 | 27 | 6 | — | — | — | — | 3/65 |
| summertime | 2 | jazz_head_by_title | 32 | 90 | 47 | 6 | — | two harmonies at one onset | — | — | 7/90 |
| things_you_see | 1 | jazz_head_by_title | 40 | 132 | 42 | 7 | — | — | — | — | 1/132 |

## 라벨 손실
- 코드 구간 1,207개 중 lossy 202개(16.7%): degree(텐션·변화음), family로 접은 확장 kind, slash bass. 12차원 family chroma가 담지 못한다
- 대응 없는 kind 11개(4 family): augmented, diminished, suspended-fourth. quality를 비워 두고 채우지 않았다
- 그 밖에 coda 1, repeat 1(기보대로 둠), 한 시각에 코드 둘 1, 첫 코드 앞 unlabeled 1
- N.C. 0, 불규칙 마디 0, 템포·박자 변화 0

## 분할 가능성
- 32 family를 family 단위로 train/validation/test에 나눌 수 있다. short·템포 변형·`DespacitoS`는 이미 같은 family로 묶었다
- family 사이에서 첫 24음이 같은 경우는 없었다
- 코드 진행 유사도: 이조와 무관한 4-코드 n-gram의 Jaccard가 최대 0.152(Donna Lee–Four Brothers)였다. 0.3 이상인 쌍은 0개다
  - 이 측정은 정확히 같은 코드열만 잡는다. 대리 코드가 섞인 같은 진행(contrafact)은 놓칠 수 있다. 예를 들어 I Got Rhythm과 Moose The Mooch는 흔히 같은 rhythm changes로 알려져 있지만, 상위 15쌍에 들지 않았다(이 관계는 검증하지 않음)
  - 분할할 때는 진행 family(블루스, rhythm changes 등)도 사람이 한 번 묶어 봐야 한다
- pilot 3곡(A Foggy Day, ATTYA, Billie's Bounce)은 이미 여러 번 봤다. 최종 미사용 test로 부르지 않는다

## v2 표현의 한계 층 (아스트라 4번)
v2는 prefix 시각 기준 현재 코드와 첫 다음 코드를 준다. 쉼 동안 코드가 두 번 이상 바뀌면, 다음 음이 시작될 때의 코드는 표현에 없다.

- 32 family 합계(음 3,662개, 악보 수준 근사: prefix 시각 = 이전 음 onset, `<T>` 경계 무시)
  - 다음 음의 코드 = 현재: 2,669(72.9%)
  - = 첫 다음 코드(경계 첫 음): 914(25.0%)
  - **표현 없음: 79(2.2%)**
- 많은 family: Giant Steps 7/26, A Foggy Day 11/75, I Got Rhythm 8/75, Summertime 7/90
- 토큰 기준 정확 집계(pilot 3곡, `aria_cond_contract_v2/dryrun.json`)
  - A Foggy Day {'same_as_current': 48, 'equals_first_next': 18, 'not_represented': 9}
  - ATTYA {'same_as_current': 62, 'equals_first_next': 24, 'not_represented': 1}
  - Billie's {'same_as_current': 102, 'equals_first_next': 24, 'not_represented': 0}
  - 근사(11 / 1 / 0)보다 조금 적다. `<T>` 경계가 prefix 시각을 앞당기기 때문이다
- 평가에서는 이 층을 빼지 않고 별도 층으로 남긴다. 경계 첫 음(25%)도 따로 본다. v3 구현은 지금 하지 않는다

## 재개 경계
1. 자료 자격과 분할을 확정한다: 사용 조건 근거, family·진행 분할, 라벨 검증 기록
2. 실제 자료 소규모 pilot을 사전 등록한다
3. 학습한다

- v2 학습 효과, 음악 품질, 개인화 성공 주장은 계속 거짓이다
- 이 문서는 새 원격 게시 승인이나 데이터 사용 승인 확대가 아니다
