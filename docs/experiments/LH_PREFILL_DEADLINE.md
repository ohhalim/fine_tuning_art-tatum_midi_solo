# 관측 왼손 문맥: 블록별 전체 prefill 마감 측정 (사전 등록)

작성 2026-10-18 10:4x KST. 이슈 #1791. 측정만 합니다. 청취·학습·새 자료 없음. 음악 품질과 무관합니다.

## 1. 왜

- Aria 실시간 경로는 생성 → 가상 MIDI 송출 → 독립 캡처까지 확인됐습니다(#1743, #1747). 빠진 것은 **키보드 입력**입니다
- #1773 계약은 관측한 사람 왼손을 오른손 문맥의 과거 위치에 끼워 넣습니다. 그러면 기존 KV가 무효가 되어, 블록마다 **전체 prefill**을 해야 합니다(계약 문서 "KV" 행)
- 블록 계약(#1770)상 새 입력을 반영한 블록은 시작 s_k에서 마감 s_k + 1박(128 BPM 468.75 ms) 안에 확정·예약해야 합니다. 전체 prefill + 그 블록 토큰 생성이 이 안에 드는지 잰 적이 없습니다

## 2. 질문 (하나)

#1773의 b 문맥(오른손 이력 + 관측 왼손)을 블록마다 전체 prefill하고 그 블록 토큰을 생성할 때, MLX에서 1박(468.75 ms) 안에 끝나는 요청은 얼마나 되는가?

## 3. 측정 (실행 전 고정)

- 모델: Aria medium `t3_aria/ckpt/model-gen.safetensors`, `aria.inference.model_mlx.TransformerLM`, float32, KV 캐시(#1743과 같은 로드 방식). M1 Max
- 토큰: dev 곡 export(`lh_request_export_run3/requests/NNN.json.gz`)의 실제 Aria 토큰열. 가장 긴 곡을 쓰고, 그보다 긴 길이는 두 번째로 긴 곡을 이어 붙입니다(계산 비용은 내용과 거의 무관하다고 보고, 한 번 다른 곡 토큰으로 같은 길이를 재서 확인합니다)
- 길이 격자 L: 128, 256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4000
- 길이마다: 준비 1회(버림) + 측정 5회. 1회 = 위치 0부터 L토큰 전체 prefill(마지막 logits까지 `mx.eval`) → 이어서 48토큰을 한 개씩 decode(KV 사용, 매 토큰 `mx.eval`)
- 기록: prefill ms, decode 토큰별 ms(첫 토큰 따로), 열 압력(길이마다 전후), RSS
- 내용 확인: L = 2048에서 다른 곡 토큰으로 prefill 5회를 따로 재서 중앙값 차이를 적습니다

## 4. 대입 (측정 뒤, 규칙은 지금 고정)

- dev 4곡 1,314 요청(`lh_context_run1/arms`, status ok)의 b 문맥 길이 L_b = `label_start.b`와 블록 생성 토큰 수 n = `win_labels + boundary_labels`
- 예측 시간 = prefill(L_b) + n × decode 중앙값(첫 토큰은 따로 더함). prefill(L_b)는 격자 중앙값의 선형 보간
- 보고: 예측 시간 중앙값·p90·최대, 468.75 ms 안 비율. 같은 값을 a 문맥 길이로도 냅니다(왼손이 늘린 몫을 보려고)
- 이것은 **예측**입니다. 실제 실시간 루프에서 잰 끝-끝 지연이 아닙니다. 샘플링·문법 마스크·송출 비용은 빠져 있습니다

## 5. 읽는 법

| 관측 | 말할 수 있는 것 | 말하지 않는 것 |
|---|---|---|
| 대부분 1박 안 | 전체 prefill 방식으로 입력 경로를 먼저 붙여 볼 수 있음 | 실시간에서 늘 마감을 지킨다 |
| 긴 문맥에서 넘침 | 문맥 상한(crop) 축소, 왼손 삽입 위치를 KV 뒤로 미루는 설계, 또는 마감 완화가 필요 | 실시간 입력이 불가능하다 |

## 6. 재현

- `scripts/lh_prefill_deadline.py <aria repo> <ckpt> <export dir> <arms dir> <report json>`(t3_aria venv)
- 시험 `tests/test_lh_prefill_deadline.py`(대입 계산만, 모델 없음)
- 보고 `docs/experiments/lh_prefill_deadline/report.json`
