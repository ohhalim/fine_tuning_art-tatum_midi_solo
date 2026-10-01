# 컴퓨터 자판으로 치고 FL Studio로 듣기 (#1581)

작성 2026-10-01. `musical_quality_verified: false`. 사용자 실연주 전 문서다.

## 구성
```
컴퓨터 자판 ──(터미널 1: typing_keyboard.py)──▶ 가상 MIDI "TypingKeyboard"
                                                      │ --input-port
                                         (터미널 2: 런타임, Tatum/멜다우)
                                                      │ 가상 MIDI "ContinuousJazz"
                                                      ▼
                                               FL Studio 악기 → 소리
```

## 자판 배열 (`scripts/typing_keyboard.py`, 추가 패키지 없음)
| 줄 | 키 | 음 |
|---|---|---|
| 코드(왼손) | Z S X D C V G B H N J M | C3 C#3 D3 D#3 E3 F3 F#3 G3 G#3 A3 A#3 B3 |
| 멜로디(오른손) | Q 2 W 3 E R 5 T 6 Y 7 U I 9 O 0 P | C4부터 E5까지 |

- 코드: 0.15초 안에 같이 누른 키를 한 코드로 묶는다. 다음 코드를 누르거나 **스페이스**를 칠 때까지 계속 눌린 것으로 친다
  - 예: Dm7 = X V N(D F A) 또는 X V N + Z(C3, 근음 D가 베이스가 아니어도 인식)
  - 예: G7 = B M V(G B F), Cmaj7 = Z C B M(C E G B)
- 멜로디: 누르면 0.25초짜리 음이 나간다. `[` `]`로 옥타브를 옮긴다. Esc로 끝낸다
- 한계: 터미널은 키를 뗀 순간을 모른다. 그래서 멜로디 길이는 고정이고, 코드는 "다음 코드까지 유지"로 처리한다
- 노트북 자판은 동시에 4–5키 이상 누르면 일부가 안 잡힐 수 있다(자판 하드웨어 한계)

## 실행 순서
1. **터미널 1** (자판 도구를 먼저 켜야 런타임이 포트를 찾는다)
   ```
   cd /Users/ohhalim/orca/workspaces/fine_tuning_art-tatum_midi_solo/즉흥연주재-설계
   PY=/Users/ohhalim/git_box/fine_tuning_art-tatum_midi_solo/.venv/bin/python
   $PY scripts/typing_keyboard.py
   ```
   - 체크포인트와 최신 코드는 이 워크트리에 있다. 본체 저장소(`git_box/...`)는 오래된 main이고 체크포인트가 없다. 파이썬만 본체의 venv를 쓴다
2. **터미널 2** (런타임. 128마디 = 128 BPM에서 4분)
   ```
   cd /Users/ohhalim/orca/workspaces/fine_tuning_art-tatum_midi_solo/즉흥연주재-설계
   PY=/Users/ohhalim/git_box/fine_tuning_art-tatum_midi_solo/.venv/bin/python
   FORCE_CPU=1 $PY scripts/play_personalized.py --preset tatum --bars 128 --bpm 128 \
     --chords Dm7,G7,Cmaj7,Cmaj7 --input-port TypingKeyboard -- --live-chords follow --live-metrics
   ```
   - `--live-chords follow`: 왼손 코드를 다음 블록 코드로 쓴다(약 1–1.3초 뒤 반영). 빼면 `--chords` 진행 그대로다
   - 멜다우는 `--preset mehldau`
3. **FL Studio**
   - Options → MIDI settings → Input 목록에서 **ContinuousJazz**를 켠다(Enable). 런타임이 실행 중일 때만 보인다. 안 보이면 Rescan devices
   - 피아노 계열 채널을 선택해 두면 그 채널로 소리가 난다
   - 내가 치는 음도 듣고 싶으면 **TypingKeyboard**도 켠다
4. **터미널 1에 포커스를 두고** 친다(자판 입력은 포커스된 터미널로만 간다)

## 확인할 것 (청취 메모용)
- 왼손 코드를 바꾸면 1–2블록(약 1–2초) 안에 반주가 따라오는가
- 끊기거나 이상한 블록이 귀에 걸리는가. 터미널 2의 `block … chord-tone …` 줄과 끝난 뒤 `continuous_report.json`의 `fallback_bar_count`로 대조할 수 있다
- 지연이 연주에 방해가 되는가

---

## 긴 실행 확인 (사전 등록, 실행 전)
- 마디 상한을 16 → 256으로 늘렸다. 16은 첫 구현(ace0787f)의 범위 축소였고 기술적 근거가 기록돼 있지 않다
- 측정: Tatum 프리셋, 128마디(반 마디 블록 256개), 128 BPM, 입력 없음, seed 42, `--capture`, 1회
- 판정
  1. 128마디를 끝까지 연주하고 fallback 0(미스는 보고만 한다)
  2. 보고만: 실행 시작 → 첫 블록까지 시간, 블록 생성 시간 p50·최대, 끝까지 생성 시간이 늘어나는 추세가 있는지
- 미달이면 원인을 기록한다. 상한 확대는 유지하되 체크리스트에 주의를 적는다

## 긴 실행 결과 (2026-10-01): 끝까지 연주, 추세 없음. fallback 3/256으로 판정 1 미달
원시값: `outputs/long_run/tatum_128/`, 로그 `outputs/long_run_tatum_128.log`. 실행 중 부하 평균 6.3–7.8(다른 작업과 동시).
- 128마디(256블록)를 끝까지 연주했다. 실행 전체 245초(연주 240초 + 시작 약 5초)
- fallback 3블록(1.2%): 검증 실패 2, 준비 시간 초과 1 ✗. 입력이 없어도 생긴다. 미스 0
- 블록 생성 ms p50 184, 최대 504. 4분위 구간별 중앙값 185 / 180 / 180 / 184 → 길게 돌려도 느려지지 않는다
- 캡처: 보낸 음 이벤트 5,954개를 모두 같은 순서로 받았다(손실·중복 0)
- 결론: 상한 확대는 유지한다. 긴 연주에서는 1분에 1블록(0.9초) 정도 fallback 패턴이 끼어들 수 있다. 소리가 끊기지는 않는다

## 자판 도구 동작 확인
- 가상 터미널로 X V N(D F A) → Q → 스페이스 → Esc를 보냈다
- 받은 MIDI: D3 F3 A3 on(20 ms 간격), C4 on → 0.26초 뒤 off, 스페이스에서 D3 F3 A3 off, Esc로 정상 종료
- 첫 시도에서 키를 cbreak 전환 전에 보내서 버려졌다. 준비 문구를 전환 뒤에 출력하도록 고쳤다. 문구가 보인 뒤에 치면 된다
