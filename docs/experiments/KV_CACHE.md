# 생성 KV 캐시 — 사전 등록

작성 2026-09-29. 이슈 #1507, 브랜치 `perf/issue-1507-kv-cache`. CLAUDE.md §5 층 1("베낀다": KV 캐싱, 양자화)에 해당한다.

**이 절은 구현·측정 전에 작성했다.**

## 문제
`MusicTransformer.generate`는 토큰 하나를 뽑을 때마다 `self.forward(gen_seq[:cur_i])`로 **전체 시퀀스를 다시 계산**한다. 16마디 실측에서 Tatum의 최대 생성 시간은 마디의 91%였다(#1503). 문맥 연결은 지연 증가로 기준에 미달했다(#1505).

## 설계
- 인과 마스크 때문에 각 층의 위치 j 출력은 입력 ≤ j에만 의존한다. 그래서 층별 K·V를 캐시해도 수학적으로 같다
- 상대 위치 항(skew RPR)은 풀면 `q_i · Er[len_e − 1 − (i − j)]`(j ≤ i)다. 새 토큰마다 `q · Erᵀ`를 거리 인덱스로 gather해서 계산한다
- 절대 위치 인코딩 pe[t], post-norm, encoder norm, LoRA property(`in_proj_weight`, `out_proj.weight`)와 FFN 모듈 호출을 기존과 같게 쓴다
- `generate(..., use_kv_cache=True)`는 샘플링 경로(beam 0, batch 1, rpr 모델)에서만 동작한다. 그 밖에는 기존 경로로 간다
- CLI: `generate_once(use_kv_cache=...)`, `run_continuous_jazz.py --kv-cache/--no-kv-cache`

## 채택 기준 (모두 충족해야 기본값을 켠다)
1. **logits 동등성:** 실제 체크포인트 3개(공통 base, Tatum·멜다우 export)에서 길이 64–192의 입력 여러 개에 대해, 캐시 경로와 전체 계산 logits의 최대 절대 차이가 **< 1e-4**(CPU, float32)
2. **생성 토큰 동일성:** 같은 3개 모델, 코드 primer 블록 조건(반 마디, 96토큰, grammar mask, 길이 목표), seed 1–8에서 **생성 토큰 열이 전부 동일**
3. **속도:** 런타임 16마디 코드 primer 모드(3 모델 × seed 42/43/44, 128 BPM)에서 fallback이 늘지 않고, 모델별 생성 p95가 §3 기준선(USAGE_PATH_16BAR.md)보다 **30% 이상 감소**

1·2를 통과하지 못하면 캐시는 opt-in으로도 합치지 않는다(원인 기록). 3에 미달하면 opt-in으로 합치고 기본값은 바꾸지 않는다.

## 결과 — 기준 3개 모두 충족, 런타임 기본값을 켬
### 1·2. 동등성 (`docs/experiments/kv_cache/verify.json`, CPU float32)
| 모델 | logits 최대 절대 차이 (입력 9개, 길이 64/128/192) | 생성 블록 토큰 동일 (seed 1–8 × 코드 4개) | 생성 시간 (블록 32개) |
|---|---|---|---|
| 공통 base | 1.0e-5 | **32/32** | 6.49 → 3.59 s (1.81배) |
| Tatum export | 1.1e-5 | **32/32** | 11.08 → 5.63 s (1.97배) |
| 멜다우 export | 1.3e-5 | **32/32** | 7.71 → 4.06 s (1.90배) |

단위 테스트(`tests/test_kv_cache.py`)
- 전체 forward vs 캐시(한 번에 / 토큰 단위) 차이 < 1e-5. LoRA out_proj+QKV+FFN 포함, 최대 길이까지
- 생성 토큰·정지 이유·forward 횟수 동일(seed 6개)
- 비-RPR 모델은 기존 경로로 간다

### 3. 런타임 16마디 (`docs/experiments/kv_cache/runtime16_analysis.json`)
코드 primer 모드, 3 모델 × seed 42/43/44, 128 BPM. 기준선은 캐시 없는 같은 조건(USAGE_PATH_16BAR.md §3)이다.

| 모델 | 생성 p50 ms | **생성 p95 ms** | 최대 ms (마디 대비) | fallback | 미스(3회 합) |
|---|---|---|---|---|---|
| base | 430 → 238 | 1,014 → 497 (**−51%**) | 1,078 → 548 (29%) | 0 → 0 | 1 → 1 |
| Tatum | 743 → 358 | 1,409 → 623 (**−56%**) | 1,703 → 732 (91% → **39%**) | 0 → 0 | 0 → 1 |
| 멜다우 | 497 → 272 | 990 → 475 (**−52%**) | 1,278 → 561 (30%) | 0 → 0 | 0 → 0 |

- **끝에서 끝까지 동일:** 16마디 재생 음 높이 열이 캐시 없는 실행과 **9/9 완전히 같다**(모델 3 × seed 3). 속도만 바뀌고 출력은 같다
- 미스 1건(Tatum seed 44)은 스케줄러 지각 꼬리다(20 ms 초과 1회). 생성과 무관한 기존 현상(RUNTIME_STALL_CAUSE.md)으로 본다

### 결정
- 사전 규칙대로 `run_continuous_jazz.py`의 **기본값을 `--kv-cache`(켜짐)**로 바꿨다. 기존 경로는 `--no-kv-cache`
- 라이브러리 함수 `generate_once`와 `MusicTransformer.generate`의 기본값은 그대로 꺼짐이다(평가·학습 스크립트 동작 불변)
- 이득: Tatum의 최악 생성 시간이 마디의 91%에서 39%로 줄었다. 더 빠른 템포, 문맥 연결(#1505에서 지연 때문에 미달) 재시험의 여지가 생겼다

## 추가 확인 — MPS 동등성 (#1512 작업 중)
한계로 남아 있던 "CPU에서만 검증"을 MPS에서 확인했다(`docs/experiments/kv_cache/verify_mps.json`). 모델은 Tatum 완성(#1509), Tatum16, 멜다우16이다.

| 모델 | logits 최대 차이 (MPS) | 생성 블록 토큰 동일 |
|---|---|---|
| Tatum 완성 | 1.05e-5 | 32/32 |
| Tatum16 | 1.14e-5 | 32/32 |
| 멜다우16 | 1.19e-5 | 32/32 |

동시에 다른 학습(멜다우 CV)이 GPU를 쓰는 상태에서 쟀으므로 시간 수치는 기록하지 않는다. 동등성만 확인했다.
