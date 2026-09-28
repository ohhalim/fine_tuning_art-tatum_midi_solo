# 개인화 모델 실제 사용 경로 점검 — 16마디

작성 2026-09-29. 로컬 브랜치 `exp/usage-path-16bar`(아스트라 지시로 원격 작업 없음).
`musical_quality_verified: false` · `style_verified: false` · `listening_done: false`

아스트라 결정: nested CV는 보류하고, 새 학습 없이 기존 Tatum·멜다우 export를 기존 CLI로 같은 입력·코드·seed에 돌려 실제 사용 경로를 점검한다. 강제 최소 음수, 반주 추가 같은 포장은 하지 않는다.

## 1. 코드 경로 확인과 누락점
- 실행 경로는 `scripts/run_continuous_jazz.py`(기존 CLI, 새 프레임워크 없음)다
- **마디 간 문맥이 이어지지 않는다(설계상 누락).** 매 마디의 primer는 다음으로만 만든다
  - 코드 primer 모드: 그 마디의 코드 음 + 연주자 입력(`build_chord_live_primer`, `generate_sub`)
  - 기본 모드: 조건 MIDI + 연주자 입력(`build_live_primer`)

  **모델이 앞 마디에 생성한 음은 다음 마디 입력에 들어가지 않는다.** 이번 실행은 연주자 입력이 없어서 마디마다 사실상 독립 생성이다
- 코드 primer 모드는 반 마디 블록이 검증에 실패하면 조용히 건너뛴다(`make_sub_block_builder`). 끊김은 재생 음의 무음 구간으로 잰다
- 기본 모드는 마디당 새 토큰이 `--generation-tokens`(기본 96)로 제한된다. 음이 촘촘하면 한 마디를 채우기 전에 예산이 끝나 블록이 무효가 된다(아래 fallback 원인)
- 계측 추가(동작 불변): 리포트에 `played_bars`(마디 시작 기준으로 정렬한 재생 음)를 넣었다. `played.mid`는 첫 음 기준이라 마디 경계를 알 수 없었다

## 2. 조건
- 모델: 공통 base(`outputs/tvm/common_base`) / Tatum(`outputs/tvm/export_tatum/checkpoint_update128.pt`) / 멜다우(`outputs/tvm/export_mehldau/checkpoint_update128.pt`). #1499의 같은 base 위 비교 쌍이다
- 입력 `outputs/chord_ab/ii_V_I.mid`, 코드 `Dm7,G7,Cmaj7,A7`, **16마디**, 128 BPM(마디 1,875 ms), seed 42/43/44, `--capture`, CPU, 스레드 QoS와 spin 기본값
- 모드 두 가지
  - **코드 primer**: `--chord-primer --chord-blocks-per-bar 2`. README의 권장 실행이다
  - **기본**: 코드 primer 없음
- 실행 18회(3 모델 × 2 모드 × 3 seed), 분석은 `scripts/analyze_played_bars.py`

## 3. 결과 (seed 3회 평균, 원시값 `docs/experiments/usage16/analysis.json`)
| 모드 | 모델 | fallback 비율 | 생성 p50 / p95 / 최대 ms | 반 마디 이상 무음 마디(16마디 중) | 빈 마디 | 완전·조옮김 반복 마디 | 마디 간 4-gram 재사용 | 재생 노트 |
|---|---|---|---|---|---|---|---|---|
| 코드 primer | base | **0** | 430 / 1,014 / 1,078 | 1.0 | 0 | 0 / 0 | 1.1% | 225 |
| 코드 primer | Tatum | **0** | 743 / 1,409 / **1,703** | 0.7 | 0 | 0 / 0 | 0.2% | 348 |
| 코드 primer | 멜다우 | **0** | 497 / 990 / 1,278 | 1.0 | 0 | 0 / 0 | 0.1% | 251 |
| 기본 | base | **18.8%** | 1,029 / 1,582 / 1,601 | 0 | 0 | 0 / 0 | 0% | 240 |
| 기본 | Tatum | **37.5%** | 1,318 / 1,533 / 1,549 | 0 | 0 | 0 / 0 | 0% | 236 |
| 기본 | 멜다우 | **12.5%** | 776 / 1,518 / 1,531 | 0 | 0 | 0 / 0 | 0% | 219 |

- 오류·미스(3회 합)
  - 코드 primer: base 미스 1, 나머지 0
  - 기본: 오류 base 9 / Tatum 16 / 멜다우 6, 미스는 멜다우 2
- **기본 모드 fallback의 원인:** 거의 전부 `invalid model block`이다. 음악 길이 890–1,800 ms로 마디(1,875 ms)를 채우지 못했다. 토큰 예산(96)이 부족한 것이다. Tatum seed 42에서는 지각으로 인한 fallback(`fallback_not_ready`)도 1건 있었다
- **마디 간 연속성(관측):** 마디 경계의 음 도약 중앙값이 마디 안의 연속 음 도약보다 약 2배 크다

| 모드 | 모델 | 마디 안 |Δpitch| 중앙값 | 마디 경계 |Δpitch| 중앙값 |
|---|---|---|---|
| 코드 primer | base / Tatum / 멜다우 | 6 / 6 / 7 | 11 / 14 / 15 |
| 기본 | base / Tatum / 멜다우 | 8 / 7 / 7 | 17 / 15 / 16 |

  마디 간 문맥이 전달되지 않는다는 코드 확인과 일치한다. 다만 화음 음을 onset·pitch 순으로 정렬하는 방식이 경계 측정에 영향을 줄 수 있어 참고값이다. 반복(완전·조옮김)은 없었고, 4-gram 재사용도 거의 0이다. 반복이 없다는 것은 동기(motif) 연속성이 없다는 것과도 같은 관측이다
- **지연 여유:** 코드 primer 모드에서 Tatum의 최대 생성 시간은 1,703 ms로 마디의 91%다. 128 BPM 16마디에서는 fallback 0이었지만 여유가 얇다

## 4. 산출물
- 모델별 16마디 재생 MIDI 18개: `outputs/usage16_v2/compare_midi/{chord,plain}_{base,tatum,mehldau}_seed{42,43,44}.mid`
- 실행별 리포트: `outputs/usage16_v2/{chord,plain}_mode/*/bpm128_seed*/continuous_report.json`(played_bars 포함)

## 5. 정리
- **실사용 권장 경로(코드 primer 모드)에서는 두 개인화 모델 모두 16마디를 fallback 없이 출력한다.** 빈 마디 없음, 반복 없음, 16마디당 반 마디 이상 무음 약 1마디
- **누락점 1 — 마디 간 문맥 없음:** 앞 마디 생성이 다음 마디 입력에 들어가지 않는다. 경계 도약이 커지는 관측과 일치한다. 솔로가 이어진 프레이즈가 되려면 이전 생성의 끝부분을 primer에 넣는 경로가 필요하다(후속, 이번 범위 밖)
- **누락점 2 — 기본 모드 토큰 예산:** `--generation-tokens 96`에서는 촘촘한 모델(Tatum)이 마디를 못 채워 37.5%가 fallback이다. 코드 primer 모드는 반 마디 단위라 해당하지 않는다
- **누락점 3 — Tatum 지연 여유:** 최대 생성 시간이 마디의 91%다. 더 빠른 템포에서는 fallback 위험이 있다(#1489는 8마디 측정이었다)
- 청취·스타일·음악 품질은 검증하지 않았다
