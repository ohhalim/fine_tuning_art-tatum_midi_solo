# guide 형식 비밥 어댑터 (A, 사전 등록)

작성 2026-10-03. 이슈 #1631. 선행: `HARMONY_CONTRACT.md`. U2에서 bebop의 Δshuf가 base 절반 미만이라 go가 나왔다(경계). 아스트라와 합의한 A 병행 조건을 따른다. 판정 규칙은 학습 전에 고정했다. `musical_quality_verified: false`.

## 가설
학습할 때 각 솔로 창 앞에 그 창의 화성 guide(계약 형식)를 두면, 모델이 guide 내용에 더 크게 반응한다(Δshuf가 커진다). 이것은 가설 실험이다. 통과해도 자유 생성에서 화성이 맞는다는 증명은 아니다.

## 데이터 (`scripts/build_guide_dataset.py` → `data/bebop_guide`)
- `data/bebop_rh`와 곡, split, 파일 이름이 같다(train 189 / val 28 / test 18)
- 곡마다 0.9375초 창을 이어 붙인 스트림 `[guide][솔로 창][guide][솔로 창]…`
  - 솔로: 전체 음의 50 ms 최고음 중 G3 이상(`bebop_rh`와 같은 규칙)
  - guide: 그 창의 accompaniment proxy. pc가 2개 미만이면 이전 guide를 이어 쓴다. 첫 guide가 나오기 전 창에는 guide가 없다
- 런타임과의 대응
  - history 없음: primer = `[guide]` → 스트림 시작과 같다
  - history 있음: `[과거][guide]` → 스트림 중간과 같다
- 한계
  - 손 분리가 아니다. 오른손 내성이 섞여 있다
  - guide가 과거 솔로 뒤에 시간상 0.85창만큼 끼어든다. 실제 연주에는 없는 시간이다
  - loss는 guide 토큰에도 걸린다(trainer에 타깃 마스크가 없다)

## 학습 (bebop 어댑터와 같은 레시피, 데이터 형식만 다르다)
- `run_mehldau_update_budget_diag.py --checkpoint outputs/tvm/common_base/checkpoint_epoch8.pt --data-dir data/bebop_guide --mode budget --budget-seconds 6000 --planned-epochs 43 --batch-size 4 --gradient-accumulation 4 --lr 3e-4 --label-smoothing 0.1 --seed 42 --device mps --lora-targets out_proj,qkv`
- 약 516 update. 마지막 스냅샷을 쓴다. val로 고르지 않는다
- 출력: `outputs/bebop_guide/{run,export}`. 기존 체크포인트는 건드리지 않는다
- 컴퓨트 상한: 6000초. 넘으면 중단하고 기록한다. 재시도는 한 번만 한다(같은 설정)

## 주평가 (학습 전 고정, 봉인 대상 아님)
- U2와 같은 스크립트·창·조건(val 28곡, 491창). 모델: base, bebop, **guide**
- **통과:** guide 어댑터가 둘 다 충족
  1. Δshuf CI 하한 > 0.024(bebop Δshuf CI 상한). bebop보다 분명히 크다는 뜻이다
  2. Δshuf 평균 ≥ 0.07(base 0.035의 2배)
- 보고만
  - NLL true(같은 val 창)
  - Δabs
  - 곡 macro, Δshuf > 0인 곡 수
- 미달이면 기록하고 끝낸다. 형식이나 하이퍼파라미터 sweep은 하지 않는다

## 봉인 (U3가 끝날 때까지)
- guide 어댑터의 자유 생성 평가, 청취, 런타임 통합은 하지 않는다
- 이 평가들은 U3 대조 반응표로 지표 타당성을 확인한 뒤에 별도로 등록한다
- U3 대조가 실패하면 A 산출물은 탐색용으로만 남긴다

## 노출 이력
- val 곡은 base 사전학습에 들어 있다(`val_songs_in_base_pretrain: true`, #1618 report)
- val과 test는 화음 적합 지표 탐색에서 이미 봤다
- 그래서 이번 결과는 모두 **탐색**으로 표시한다

## 생성 평가 (봉인 해제 후 등록: U3 결과 3f4008f9 이후, 학습 결과를 보기 전)
- U3 판정에 따라 rel_js(합산 분포)만 게이트로 쓴다. fit과 clash는 집단 수준 보고로만 쓴다
- 스크립트: `scripts/guide_gen_eval.py`
- 데이터: val 491창
- 생성 조건: 각 창의 true guide만 primer로 준다(history 없는 런타임 계약). 0.9375초, T 1.0, top-k 32, top-p 0.95, grammar mask, seed 2개
- 솔로: 생성 결과의 G3 이상 최고음 선율
- 비교: bebop, guide 어댑터, 실제(같은 창의 실제 솔로)
- 보고만
  - 같은 생성을 donor 화성으로 잰 fit. guide 화성을 따르는지 보는 값이다
  - 빈 창 수
- **통과:** guide 어댑터가 모두 충족
  1. rel_js < bebop
  2. rel_js ≤ 0.10. 실제 val은 .048, 화성 교환 대조는 .088이다
  3. fit ≥ bebop + 0.03(집단 수준)
  4. 초당 음 수가 실제 창의 0.67–1.5배
- 미달이면 기록한다. sweep은 하지 않는다. 통과해도 런타임 통합과 청취는 별도 단위다. "듣기 좋다"는 주장은 하지 않는다
