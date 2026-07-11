# D2 실험 결과: 문법 제약 디코딩 — 전 기준 통과, 채택

> Issue #1455 (설계·판정 기준 사전 등록) / 로드맵 L1 (docs/RESEARCH_ROADMAP.md)
> 실행: 2026-07-12. D1 Arm D 체크포인트 그대로, 동일 프라이머·temp 1.0·seed 42, 24샘플 전후 비교.

## 구현

샘플링 시 문법적으로 무효인 토큰을 로짓 -inf 마스킹 (`--grammar_mask`, 기본 off):

- **고아 note_off 차단**: 활성 pitch 집합에 없는 pitch의 note_off 토큰 마스킹
- **중복 note_on 차단**: 이미 활성인 pitch의 note_on 마스킹 (디코더의 조용한 노트 손실 방지)
- 활성 집합은 프라이머로 초기화, 생성 중 갱신
- 위치: `music_transformer/model/music_transformer.py::generate` + `scripts/generate.py` 플래그

## 결과 — 사전 등록 기준표 (#1455) 대입

| 지표 | 기존 (unmasked) | 마스크 (masked) | 기준 | 판정 |
|---|---|---|---|---|
| 고아 note_off /샘플 | ~32 | **0** | ~0 | ✅ |
| 노트 밀도 /s | 1.84 | **2.58** (+40%) | 상승 | ✅ |
| n_notes /샘플 | 68.6 | **89.3** (+30%) | 상승 | ✅ |
| pooled 고유 보이싱 | 142 | **149** | ≥135 | ✅ (사전학습 상한 145도 상회) |
| pooled 보이싱 엔트로피 | 5.03 | 4.98 | (비열화 참고) | ≈ 유지 |

**전 기준 통과.** 버려지던 토큰(노트 이벤트의 ~1/3)이 실제 노트로 전환되어 밀도가 40%
올랐고, 다양성은 유지·소폭 상승했다. 분포 왜곡의 징후(엔트로피 급락, n-gram 반복 급증)
없음 (voicing 4gram repeat 0.001→0.004, 무시 가능 수준).

## 부수 관찰

- **mean_voicing_size 1.34 → 1.31 (불변)** — 단선율 문제는 문법 오류 때문이 아니었다.
  D3의 가설 H-D3b(디코딩 원인)가 사실상 기각되고, H-D3a(데이터 원인: lead 역할 분리가
  화음을 갈라놓음)의 우선순위가 올라간다.
- duration 51→40초: 같은 448 토큰에서 노트가 많아진 만큼 time_shift 비중이 줄어든 효과.
  절대 시간이 필요한 용도에서는 length를 늘리면 된다.

## 채택

- 이후 모든 생성·평가에서 `--grammar_mask`를 표준으로 사용한다 (하니스 반영은 다음 실험부터).
- 플래그 기본값은 off로 유지 — 과거 실험(D0/D1)과의 재현 호환성 보존.

## 재현

```bash
PYTHONPATH="music_transformer:music_transformer/third_party:scripts" \
.venv/bin/python scripts/generate.py \
  --lora_path outputs/d1_experiment/armD_lora/ckpt \
  --conditioning_midi ./data/roles/lead/000000/conditioning.mid \
  --num_samples 24 --length 1024 --temperature 1.0 --grammar_mask \
  --output outputs/d2_experiment/armD_masked/samples
```

원자료: `outputs/d2_experiment/`, 비교 JSON 사본은 이 디렉터리.
