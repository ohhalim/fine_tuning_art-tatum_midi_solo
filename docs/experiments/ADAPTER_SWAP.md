# 런타임 어댑터 스왑 — 한 세션에서 Tatum↔멜다우 (사전 등록)

작성 2026-09-29. 이슈 #1519. `style_verified: false` · `musical_quality_verified: false`

**이 절은 측정 전에 작성했다.** 구현은 측정 전에 끝냈다.

## 목적
재설계 층 2의 첫 항목인 "base 1개 + 런타임 LoRA 어댑터 스왑"을 연속 연주 런타임 안에서 작동시킨다. jam_bot은 곡마다 360M 모델을 따로 학습했다. 여기서는 base 하나에 어댑터를 바꿔 끼운다.

## 설계
- `scripts/adapter_bank.py` `AdapterBank`
  - 어댑터마다 체크포인트를 읽어 병합한다(#1512)
  - 병합 후 서로 다른 텐서가 LoRA가 건드리는 가중치(in_proj, out_proj, FFN linear)뿐인지 검사한다. 다른 텐서까지 다르면 **base가 다르다**고 보고 거부한다
  - 모델 인스턴스는 하나만 둔다. `select(name)`은 그 어댑터의 병합 가중치(다른 텐서만)를 제자리 복사한다
  - `allow_different_bases=True`(opt-in)이면 base가 달라도 받는다. 이때는 스왑할 때 다른 텐서 전부(사실상 모델 전체)를 복사한다
- `run_continuous_jazz.py`
  - `--swap-adapter NAME=CKPT`(여러 번 가능), `--adapter-name`(`--checkpoint`의 이름, 기본 primary), `--adapter-schedule "tatum:4,mehldau:4"`(마디 수로 순환), `--allow-different-bases`
  - 스왑은 **마디 시작에서만**, producer 스레드에서 생성과 생성 사이에 한다
  - 리포트에 `adapter_swap`(마디별 어댑터, 스왑 횟수, 스왑 ms)을 남긴다
- 문맥 연결이 없으면 각 마디의 primer는 코드 음뿐이다. seed는 `seed + bar×13 + sub×977`이다. 따라서 **스왑 세션의 마디 b는, 그 어댑터 단독 세션의 마디 b와 토큰이 같아야 한다.** 이것이 스왑이 정확하다는 가장 강한 확인이다

## 쌍
- **S(같은 base):** Tatum 완성(`outputs/final_tatum/export/checkpoint_update518.pt`, 공통 base, out_proj+QKV) + 공통 base 멜다우 u128(`outputs/tvm/export_mehldau/checkpoint_update128.pt`, out_proj)
- **D(다른 base, opt-in):** Tatum 완성 + 멜다우 완성 #1497(`outputs/clean_base/c2_export/checkpoint_update128.pt`, clean base). 공통 base 판정(#1517)에서 최선의 두 모델은 base가 다르다고 나왔으므로 함께 잰다

## 채택 기준
1. **동등성(CPU):** 스왑 20회 동안 매번 bank 모델의 logits가 따로 읽어 병합한 그 어댑터 모델과 같다(최대 차이 < 1e-5). 쌍 S와 D 모두. D는 opt-in 없이는 **거부**돼야 한다
2. **마디 동일성:** 128 BPM, 코드 primer 2 sub-block, 코드 Dm7,G7,Cmaj7,A7, 16마디, `--adapter-schedule tatum:4,mehldau:4`, seed 42/43/44. 스왑 세션에서 연주된 마디의 음높이 시퀀스가 같은 seed 단독 세션의 같은 마디와 **16/16 동일**하다(쌍마다 48마디)
   - 단독 세션 기준
     - Tatum 완성: `outputs/lora_merge/runtime_merge`
     - 공통 base 멜다우: `outputs/kv_cache/runtime16/mehldau`(병합 전이지만 out_proj는 병합해도 출력이 같다, #1512)
     - 멜다우 #1497: 이번에 같은 설정으로 새로 잰다
3. **실시간성:** 모든 스왑 세션에서 fallback 0이고, 스왑 시간 최대가 쌍 S는 **20 ms 이하**, 쌍 D는 **100 ms 이하**다(마디 1,875 ms, 생성 여유 안)

1–3을 모두 충족하면 기능으로 합치고 README에 사용법을 적는다. 쌍 D는 opt-in으로 남긴다. 미달한 항목은 원인을 기록하고 합치지 않는다.

---

## 결과 — 기준 1–3 모두 충족, **기능으로 합친다**
원시값: `docs/experiments/adapter_swap/`(`verify_S.json`, `verify_D.json`, `identity.json`, `played_bars.json`). 런타임 원본: `outputs/adapter_swap/`. 측정 중 이 브랜치에서 **병합 결함(#1520)을 발견해 먼저 고쳤다.** 모든 수치는 수정 후 코드로 쟀다.

### 기준 1 — 동등성 (CPU, 스왑 20회)
| 쌍 | base | 스왑 텐서 | 스왑 ms p50 / 최대 | logits 최대 차이 | 비고 |
|---|---|---|---|---|---|
| S | 같음 | 12 (6층 × in_proj, out_proj) | 0.83 / 0.97 | 0.0 | #1497 멜다우를 넣으면 **거부**("59 non-LoRA tensors differ") |
| D | 다름(opt-in) | 83 (사실상 전체) | 1.97 / 3.65 | 0.0 | |

### 기준 2 — 마디 동일성 (128 BPM, `tatum:4,mehldau:4`, seed 42/43/44)
- 쌍 S: 스왑 세션의 48마디가 단독 세션의 같은 마디와 **48/48 동일**하다
- 쌍 D: **48/48 동일**하다
- 마디 배정: T T T T M M M M T T T T M M M M(세션당 스왑 3회)

### 기준 3 — 실시간성
| 세션 | fallback | 데드라인 미스 | 스왑 ms 최대 (seed별) | 생성 p50 / p95 / 최대 ms | 코드톤 |
|---|---|---|---|---|---|
| 스왑 S | 0 | 0 | 9.8 / 3.3 / 8.5 (기준 ≤ 20) | 269 / 426 / 497 | 0.534 |
| 스왑 D | 0 | 0 | 19.6 / 18.4 / 18.5 (기준 ≤ 100) | 281 / 398 / 441 | 0.522 |
| 단독 멜다우 #1497 (참고) | 0 | 0 | — | 204 / 339 / 389 | 0.580 |

- 런타임 안의 스왑 시간(2–20 ms)은 단독 검증(1–4 ms)보다 길다. 스케줄러 스레드와 같은 프로세스에서 돌기 때문으로 보인다(원인 분리 안 함)
- 스왑 시간은 모두 다음 마디 생성 전에 끝나고, 마디 길이 1,875 ms에 비해 작다

### 판정과 사용법
기준을 모두 충족했으므로 합친다.

```bash
# 같은 base 쌍 (권장): Tatum 완성 + 공통 base 멜다우
FORCE_CPU=1 python scripts/run_continuous_jazz.py \
  --checkpoint outputs/final_tatum/export/checkpoint_update518.pt --adapter-name tatum \
  --swap-adapter mehldau=outputs/tvm/export_mehldau/checkpoint_update128.pt \
  --adapter-schedule tatum:4,mehldau:4 \
  --conditioning-midi outputs/chord_ab/ii_V_I.mid --chord-primer --chord-blocks-per-bar 2 \
  --bars 16 --bpm 128 --output-dir outputs/swap_demo
# 멜다우 완성 #1497로 바꾸려면(base가 다름):
#   --swap-adapter mehldau=outputs/clean_base/c2_export/checkpoint_update128.pt --allow-different-bases
```

한계
- 스왑 시점은 고정 스케줄(마디 수 순환)뿐이다. 연주 중 키보드/CC로 바꾸는 입력은 아직 없다(다음 후보)
- 쌍 S의 멜다우는 #1497보다 절대 CE가 0.028 나쁘다(#1517). 쌍 D는 두 최선 모델을 쓰지만 base가 달라 전체 모델을 복사한다
- 가상 포트, CPU, seed 3개로 쟀다. 스타일·음악 품질은 검증하지 않았다
