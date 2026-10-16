# 기준 클립 결정 (#1762)

작성 2026-10-16. 연구 문서(`REALTIME_PARTNER_RESEARCH.md` 2절)가 사용자에게 정해 달라고 한 다섯 가지를 사용자가 정했습니다. 이 문서는 그 결정을 그대로 기록합니다. 판정 기준(C1–C9)을 다시 정하는 일은 따로 합니다.

## 기준 클립

후보 22개는 `scripts/make_reference_candidates.py`로 만든 30초 구간입니다(`listening/reference_candidates/clips.json`, 오디오는 로컬에만 있음). 코멘트는 사용자 말 그대로입니다.

| 번호 | 연주자 — 곡 | 구간(초) | 사용자 판단 | 사용자 코멘트 |
|---|---|---|---|---|
| 01 | Tigran Hamasyan — A Fable | 75.9–105.9 | 좋음 | |
| 02 | Tigran Hamasyan — A Memory That Became A Dream | 39.2–69.2 | 아님 | |
| 03 | Tigran Hamasyan — Illusion | 22.76–52.76 | 고르지 않음 | |
| 04 | Tigran Hamasyan — Kakavik (The Little Partridge) | 88.22–118.22 | 좋음 | |
| 05 | Tigran Hamasyan — Longing | 76.22–106.22 | 아님 | |
| 06 | Tigran Hamasyan — Looking For The Block | 74.82–104.82 | 좋음 | 응 이게 좋아. 베이스 솔로잉 또 하이톤 솔로잉이 좋아. 컴핑은 줄이면서 |
| 07 | Tigran Hamasyan — Mother, Where Are You | 36.17–66.17 | 아님 | |
| 08 | Tigran Hamasyan — Rain Shadow | 23.68–53.68 | 아님 | |
| 09 | Tigran Hamasyan — Samsara | 82.91–112.91 | 아님 | |
| 10 | Tigran Hamasyan — Someday My Prince Will Come | 54.52–84.52 | 아님 | |
| 11 | Tigran Hamasyan — The Legend Of The Moon | 82.52–112.52 | 아님 | |
| 12 | Tigran Hamasyan — The Spinners | 44.02–74.02 | 아님 | |
| 13 | Tigran Hamasyan — What The Waves Brought | 97.22–127.22 | 좋음 | 처음 부분의 베이스를 사용하는 방법이 좋아 |
| 14 | Brad Mehldau — Holland (Live) | 165.99–195.99 | 아님 | |
| 15 | Brad Mehldau — Jigsaw Falling into Place (Live) | 176.38–206.38 | 아님 | |
| 16 | Brad Mehldau — Knives Out (Live) | 171.06–201.06 | 좋음 | 살짝 불협 같으면서도 다시 인으로 돌아오는 게 좋아 |
| 17 | Brad Mehldau — Teardrop (Live) | 211.81–241.81 | 좋음 | |
| 18 | Barry Harris — All God's Chillun Got Rhythm (Live, Maybeck 1990) | 69.5–99.5 | 좋음 | 솔로잉이 비밥처럼 이어지는 릭이 좋아 |
| 19 | Tommy Flanagan — Con Alma | 45.9–75.9 | 고르지 않음 | |
| 20 | Bill Charlap — A Foggy Day (Live) | 33.64–63.64 | 고르지 않음 | |
| 21 | Art Tatum — Come Rain Or Come Shine | 76.05–106.05 | 고르지 않음 | |
| 22 | Art Tatum — Elegy | 53.65–83.65 | 좋음 | 긴 릭이 좋았던 것 같아. 스타일은 별로고 |

## 그 밖의 결정

| 항목 | 결정 |
|---|---|
| 합격 문장 | 없음 |
| 박자 | 4/4와 7/8 같은 홀수 박자 모두 |
| 템포 | 90–130 |
| 코드음 다시 치기 | 가끔은 괜찮음 |
| 코드 진행 | 정하지 않음 |

## 바로 드러나는 영향

- **박자와 템포:** 지금 블록 계약(`scripts/block_contract.py`)과 런타임은 4/4, 128 BPM의 반 마디(937.5 ms) 블록을 전제로 합니다. 7/8 같은 홀수 박자와 90–130 템포를 받으려면 블록 길이와 격자를 박자·템포에 맞게 바꿔야 합니다
- **코드음 다시 치기:** C1은 입력 보이싱과 같은 음(echo)을 뺀 착지율을 주 지표로 삼았습니다. "가끔은 괜찮음"이므로 echo를 완전히 빼기보다 비율에 상한을 두는 쪽이 맞을 수 있습니다(판정 기준을 다시 정할 때 함께 봅니다)
- **합격 문장이 없음:** C1의 "코드가 바뀌면 다음 반 마디 안에 코드톤으로 착지"는 연구 문서가 예로 든 문장이고, 사용자가 고른 문장이 아닙니다. 계속 임시 기준으로 둡니다
- 좋다고 고른 클립의 코멘트에서 공통으로 보이는 것은 베이스 음역 솔로, 높은 음역 솔로, 적은 컴핑, 불협 뒤 해결, 길게 이어지는 릭입니다. 수치로 옮겨 아니라고 한 클립과 비교하는 일은 다음 단계에서 합니다
