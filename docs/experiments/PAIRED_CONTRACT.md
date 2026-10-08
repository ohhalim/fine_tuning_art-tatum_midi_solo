# 코드–연주 쌍 자료: 파일 계약, 검증기, 확보 계획

작성 2026-10-08. 아스트라가 표상 감사(REP_AUDIT) 다음 단위로 정했다. 새 학습·다운로드·새 모델 실험 없음. `musically_verified: false`.

## 목적과 역할 분리
- 목적은 **코드 제어 자료 파이프라인을 검증하는 것**이다. 학습할 만큼 자료가 충분하다고 주장하지 않는다
- 스타일 자료와 화성 제어 쌍 자료는 역할이 다르다
  - 스타일 자료: 티그랑 등 목표 연주. 코드 라벨이 없어도 된다
  - 화성 제어 쌍 자료: 연주와 따로 정한 코드 계획에 맞춘 연주
- 사용자 녹음은 사용자 스타일 자료다. 티그랑 스타일의 정답을 대신하지 않는다

## 표상 감사 결론의 범위 교정
`REP_AUDIT.md`의 세 문장을 잰 범위로 좁혔다.
- 라벨 0: 이 티그랑 자료로는 코드 조건 학습 자료가 없다는 뜻이다. 코드 조건 없는 스타일 적응까지 불가능하다는 뜻은 아니다
- CMT 손실: 최고음 대리와 120 BPM 가정 전처리에서 나온 결과다. 올바른 선율 추출이나 박 정렬을 해도 불가능하다는 증거가 아니다
- Aria: 현 구현에 시간 정렬 코드 입력이 없다. 조건 경로를 바꾸고 학습해야 한다. 새 코드 토큰은 후보 중 하나일 뿐이다

## 기존 로컬 자료 조사 (기존 프로젝트 자료만, 다운로드 없음)
질문: 독립 코드 주석과 선율이 이미 정렬된 자료가 있는가.

| 자료 | 독립 코드 | 정렬된 선율 | 판정 |
|---|---|---|---|
| `midi_dataset/midi`, `midi_kong` (피아니스트 녹음 변환본, studio 95·live 53 폴더) | 없음 | — | 해당 없음 |
| `data/bebop_rh`, `data/bebop_guide` (235곡) | 없음. guide는 같은 연주의 반주에서 스크립트로 뽑았다 | 있음 | 자동 추출이라 정답 아님 |
| `data/eval/stage_b_chord_labeled_tiny` | 있음 | 있음 | 자체 표기상 실제 라벨이 아닌 합성 fixture. 제외 |
| `llm_rag_midi_improv` (MIDI 261개, 두 폴더 중복) | 없음. 전부 트랙 1개·채널 1개 | — | 코드 트랙이 따로 없다 |
| `flstudio-mcp/midi_data` (3개) | 코드로 만든 예시 | — | 합성. 제외 |
| **`bebopnet-code/resources/xmls`** (MusicXML 51개) | **있음: 코드 기호 1,915개**, 박 격자 명시 | **있음: 단선율 6,527음, 1,625마디** | **부분 대체 후보** |

- BebopNet XML의 사실
  - 51개 중 15개는 `short/` 판이고, 템포·이름 변형(`Giant_Steps_200`, `simple_songs_120`, `DespacitoS`)이 있다. 고유 곡은 더 적다
  - 코드 종류: dominant 622, minor-seventh 402, minor 192, major-seventh 185, major 148, half-diminished 94 등. 베이스 지정 코드 51개
  - 선율은 **즉흥 솔로가 아니라 테마(헤드)** 다. 연주 타이밍·velocity 표현이 없다. 피아노 연주도 아니다
  - 저장소는 `shunithaviv/bebopnet-code`(코드 MIT). XML 악보의 출처와 권리 표기는 없다. 곡 자체에는 저작권이 있을 수 있다. 권리 상태는 미확인으로 둔다
- 결론: **즉흥 연주와 코드가 쌍을 이룬 기존 자료는 없다.** 테마 선율과 코드가 쌍을 이룬 자료(BebopNet XML)는 있다. 이것을 화성 제어 파이프라인 검증에 쓸지는 아스트라와 정한다. 이번에는 변환하지 않았다

## 파일 계약 `paired_take_v1`
take 하나 = 디렉터리 하나. 원본은 저장소 밖 영구 경로(예: `/Users/ohhalim/git_box/paired_takes/<take_id>/`)에 둔다. 원본을 자동으로 업로드하지 않는다. 저장소에는 검증 요약만 올린다.

### `raw.mid` — DAW 내보내기 원본, 수정하지 않음
- 같은 DAW transport와 같은 tick 원점(세션 시작 = tick 0)에서 내보낸다. 벽시계 타임스탬프를 섞지 않는다
- 템포 트랙(set_tempo, 박자표), 솔로 트랙, 반주(코드) 트랙을 넣는다. 카운트인을 포함한다
- 솔로는 quantize나 velocity 평탄화를 하지 않는다

### `take.json`
| 필드 | 뜻 |
|---|---|
| `schema` | `paired_take_v1` |
| `take_id`, `progression_id` | take 식별자, 진행 식별자(분할 단위) |
| `split` | `train_candidate` 또는 `heldout_candidate`. 진행 family 단위로 나눈다. 한 take를 잘라 양쪽에 두지 않는다 |
| `source`, `source_type`, `rights_status`, `performer` | 출처, 종류(`user_recording`, `synthetic_fixture` 등), 권리 상태, 연주자 |
| `label_basis` | `planned`(수집 전에 고정한 계획) 또는 `performer_confirmed`. voicing만 보고 자동으로 정한 코드는 받지 않는다 |
| `ppq`, `time_signature`, `tempo_map` | 파일과 같아야 한다. tempo_map은 세션 시작 기준 `{beat, bpm}` |
| `count_in_beats`, `bars` | 카운트인 길이, 본 연주 마디 수 |
| `tracks` | `{solo, comp}` 트랙 이름 |
| `chords` | 카운트인 뒤 박 기준 `{onset_beat, end_beat, root, quality, bass, voicing}`. voicing은 계획한 MIDI 음높이 |
| `raw_sha256` | 원본 해시 |
| `independent_review` | 선택. `{reviewer, verdict}`. 연주자 아닌 사람의 검토 |
| `audio_latency_ms` | 선택. 정렬에 쓰지 않는다. 오디오 지연과 MIDI 정렬은 다른 문제다 |

## 검증기 `scripts/paired_validate.py`
- 사용: `python scripts/paired_validate.py <take 디렉터리> [--write-processed]` → `validation.json`
- **실패**(계약 위반)
  - 필수 필드, 원본 해시, ppq, 박자표, 템포 맵 보존(파일과 take.json 일치)
  - `label_basis`가 계획 또는 연주자 확인이 아님
  - 코드 계획이 본 연주 구간을 빈틈·겹침 없이 덮지 않음, 알 수 없는 근음·종류, voicing 없음
  - 반주 정렬: 코드 변경마다 반주 onset이 계획 tick ±1 tick 안에 없음
  - 음 형식: end ≤ start, 닫히지 않은 음, velocity 범위 밖
  - 템포 변환 왕복(tick → 초 → tick)이 1 tick 넘게 어긋남
  - 카운트인 제거본을 되돌렸을 때 원본과 다름
- **표시**(기록 그대로 두고 사람이 본다)
  - 계획 voicing이 계획 종류의 필수음·허용 텐션과 맞지 않음
  - 실제 반주 음이 계획 voicing과 다름
  - 같은 음높이 겹침(원본 유지), 본 연주 구간 밖 음
  - 솔로 onset 중 코드음·텐션 밖의 수. 정보일 뿐이고 음을 고치지 않는다
- `musically_verified`는 연주자가 아닌 사람의 검토가 `pass`일 때만 참이다. 기술 통과는 화성 정답이 아니다
- `corpus_eligible`은 `source_type`이 `synthetic_fixture`이면 항상 거짓이다

## 합성 fixture
- `tests/test_paired_validate.py`가 임시 디렉터리에 2마디 take를 만든다: Dm7 → G7, 카운트인 4박, 본 연주 중 120 → 100 BPM 템포 변화
- 시험 8개: 정상 통과(검증·corpus 자격은 거짓), 코드 빈틈·겹침, 반주 1 tick 허용·2 tick 실패, 같은 음 겹침 표시, 추정 라벨·템포 불일치 실패, 길이 0 음 실패, 원본 수정 시 해시 실패, 템포 변화 구간 변환 왕복
- fixture는 파서·정렬 시험용이다. `data/` 아래에 저장하지 않고 학습·평가 corpus에 섞지 않는다

## 확보 계획 (첫 pilot, 실행은 후속)
- 8마디 take 8개: 계획 진행 4개 × 독립 take 2개. 수 분 분량이다. 일반화 평가에 충분한 자료가 아니다
- 수집 전에 고정할 것: 길이 8마디, 4/4, 템포, 카운트인 1마디, 진행 4개
- 진행 초안(아스트라 확인 전). 장조·단조와 코드 전환을 넣는다
  - A: `Dm7 | G7 | Cmaj7 | Cmaj7 | Dm7b5 | G7 | Cm7 | Cm7` (장조 ii–V–I → 단조 ii–V–i)
  - B: `Fmaj7 | Fm7 | Em7 | A7 | Dm7 | G7 | Cmaj7 | Cmaj7` (같은 근음에서 maj7 → m7 전환)
  - C: `Cm7 | Cm7 | Abmaj7 | Abmaj7 | Fm7 | G7 | Cm7 | Cm7` (단조 중심)
  - D: `Ebmaj7 | Ebm7 | Dm7 | G7 | Cmaj7 | Am7 | Dm7 | G7` (반음 하행 전환)
- 분할: 진행 3개 × 2 take = 6개를 train 후보, 나머지 진행 1개 × 2 take = 2개를 held-out 후보로 둔다. 어느 진행을 held-out으로 할지는 수집 전에 고정한다
- 기록 방법(후보): FL Studio 한 프로젝트에서 코드 트랙에 계획 voicing을 넣는다. 솔로는 별도 트랙에 실시간으로 녹음하고, 두 트랙을 MIDI로 함께 내보낸다
- 사용자 녹음은 이번 단위의 완료 조건이 아니다

## `midi_kong` 동시음 측정: 보류
- 다른 변환본의 시간 해상도가 더 촘촘하다고 더 정확하다는 뜻은 아니다. 지금 병목(쌍 자료)도 풀지 못한다
- 토크나이저로 실제 학습하거나 변환본을 고르기 직전에, 합성 fixture로 범위를 정해 검사한다
  - 서로 다른 음높이와 같은 음높이를 따로 본다
  - 간격은 1·5·9·10·11 ms다
  - 같은 음높이 음의 삭제와 다른 음높이 onset의 동시화를 구분한다
- 필요하면 그 판의 한 구간을 확인한다. 그때까지 미확인으로 둔다

## 결정이 필요한 것 (아스트라)
1. BebopNet XML(테마 선율 + 코드)을 화성 제어 파이프라인 검증에 쓸 것인가, 범위 밖으로 둘 것인가
2. 진행 A–D 초안, 템포, held-out 진행
