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
