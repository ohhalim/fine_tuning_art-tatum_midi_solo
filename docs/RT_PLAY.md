# 실시간 연주 실행기 rt_play 사용법

이슈 #1826. 키보드 왼손 → Aria(q8 기본) → FL Studio/Serum 솔로. 연구용 실행기이며 품질 승인 전(`quality_claimed: false`).

## 1. 준비 (한 번)

- macOS `Audio MIDI 설정` → 메뉴 `윈도우 > MIDI 스튜디오 보기` → `IAC 드라이버` 더블클릭 → `기기가 온라인 상태임` 체크, 버스 하나 이상(예: `AI In`)
- FL Studio `Options > MIDI settings`
  - Input에서 IAC 버스(`AI In`) 활성화, 솔로를 낼 채널(Serum 등)에 포트 번호 연결
  - 키보드 왼손 소리를 FL Studio에서 들으려면 키보드 입력도 활성화. 왼손을 rt_play로만 보내도 되고, FL Studio에서 같은 입력을 함께 써도 됨(포트 동시 열기는 장치 드라이버에 따라 다름)
- 클릭을 쓰면 FL Studio Input에 `rt_play click` 가상 포트가 실행 중에만 보임. 채널 10 드럼(우드 블록 76·77)

## 2. 실행

```
cd <저장소>
.venv/bin/python scripts/rt_play.py --list                         # 포트 번호 확인
.venv/bin/python scripts/rt_play.py --lh 3 --out "AI In" --minutes 3 --click
.venv/bin/python scripts/rt_play.py --dry-run                      # 키보드 없이 연결 확인(가상 포트, 가짜 왼손)
```

- `--lh`, `--out`: `--list`의 번호 또는 이름 일부. 생략하면 실행 중 물어봄. 한글 IAC 이름은 깨진 상태로 와도 표시용으로 복원
- `--out-virtual 이름`: IAC 대신 그 이름의 가상 출력 포트를 만듦(FL Studio Input에서 선택)
- `--minutes`(기본 3) 또는 `--blocks`, `--precision q8|fp32|bf16`(기본 q8), `--count-in`(기본 4박, `--click`일 때만)
- 순서: 모델 로딩(10~20초) → 시작 전 블록 생성 → 클릭 카운트인 → 시작. 모델은 시작 시각부터 바로 솔로, 왼손 반영은 다음 블록(1~3박 뒤)부터
- 정지: Ctrl-C. producer 중단 → sender가 아직 시작 안 한 note-on 취소, 울리는 음 끄기, 포트 닫기. 정해진 길이가 끝나면 남은 예약 음을 다 친 뒤 끝남

## 3. 기록 (`/Users/ohhalim/git_box/rt_play_sessions/<시각>/`)

| 파일 | 내용 |
|---|---|
| `summary.json` | 진행 블록 수, 마감 초과 수, 여유 최소·중앙(ms), 중단 사유, 보낸 note-on 수, 정지 시 예약·울림 음 수, 깨끗한 정지 여부 |
| `producer_progress.jsonl` | 블록마다 여유(`ready_margin_ms`), 음 수, 왼손 음 수, suffix 시간 |
| `producer.json`, `sender.json` | 전체 기록(정상 종료 시). Ctrl-C면 producer.json 없음, 요약은 진행 기록에서 |
| `config.json`, `*.stderr` | 설정, 오류 출력 |

- 여유가 음수인 블록은 마감 초과. 그 블록은 보내지 않고 세션 중단(`deadline_miss`)
- `rejected_late_on` > 0이면 늦게 도착해 버린 음 존재

## 4. 알려진 한계

- 템포 128 BPM 고정. 블록 계약(`aria_mlx_stream_loop.BEAT/BLOCK`, 2박 = 937.5 ms)이 상수라 실행기에서 못 바꿈
- q8은 실시간 smoke 기준 통과(#1818)했지만 기본값 승인·청취 승인 전. fp32는 정상 경로에서도 마감 초과 관측(#1818)
- 마감 초과로 세션이 중단될 수 있음. 촘촘한 블록(지평 전 샘플 40개 이상)과 큰 suffix 재계산이 원인 후보(#1819, #1820)
- 왼손 관측 창 7.5초(#1773 계약). 왼손 포트로 들어온 모든 음을 왼손으로 취급, 서스테인 페달(CC64) 무시
- 중단 시 예약 음 처리 결함은 #1821(PR #1824)에서 수정. 이 실행기는 그 수정에 의존(머지 전 브랜치에서는 capacity_failure 중단 때 예약 음이 남을 수 있음)
- 실행 중 GPU를 쓰는 다른 작업이 있으면 여유가 줄어듦

## 5. 확인 기록

- 2026-10-19 dry run(q8, 가짜 왼손, 가상 포트): 8블록 마감 초과 0, 최소 여유 231.6 ms, 보낸 메시지 114 = 캡처 114, 정지 시 예약·울림 0
- Ctrl-C 시험(30블록 중 4블록에서 중단): sender `eof` 종료, 보낸 note-on 32개 모두 note-off, 포트 닫힘, 남은 프로세스 없음
- 실제 키보드·FL Studio 경로는 미확인(사용자 확인 필요)

## 정지 동작 확인 (2026-10-19, develop 병합 뒤, dry run)

- 사용자 정지(Ctrl-C·SIGTERM): 클릭과 sender를 producer와 동시에 정지. 예약만 되고 아직 안 나간 음은 보내지 않고 기록(`cancelled_pending_on`), 울리는 음은 모두 note-off. 13블록 진행 중 SIGTERM: 정지 1초 이내, 예약 음 3개 취소 기록, 울리던 음 6개 off, 송출 210 = 캡처 210, 남은 프로세스 없음
- 정상 종료: 예약 음을 끝까지 연주한 뒤 정지. 11블록: 송출 190 = 캡처 190, 예약·울림 음 0
- 마감 초과 중단: 64블록 시도 중 블록 10에서 `deadline_miss`(직전 블록 8·9에 15음·12음). 예약 음 0, 울리던 음 1개 off. 촘촘한 블록에서는 q8에서도 세션이 끊길 수 있음(이 확인 당시에는 음 수 상한 미연결, 지금은 아래 `--note-cap` 절 참조)
- 요약의 `blocks_timed`는 마감 판정까지 간 블록 수, `blocks_committed`는 확정·송출까지 끝난 블록 수

## 블록 음 수 상한 (#1840, 선택)

- `--note-cap 12`: 2박 블록 창마다 모델 음을 최대 12개로 제한(#1819 producer 옵션 전달). 기본 0 = 끔
- 결정적 왼손 짝 비교(#1835)에서 상한 없음 팔의 마감 초과 3건이 상한 12에서는 없었음(짝 관측 3 대 0, 인과 효과 미확정). 촘촘한 구간에 쉼이 생기는 음악 변화가 있어 청취 판단 전에는 기본값 끔 유지
- 값은 세션 `config.json`과 producer 보고서에 남음
