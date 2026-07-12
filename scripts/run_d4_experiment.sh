#!/usr/bin/env bash
# D4 [L5] — 조건 프라이머 텍스처 제어 실험
# D3가 입증한 "프라이머 텍스처 잠김"의 처방 검증. 재학습 없이 조건만 바꿔 생성→측정.
#
# 팔:
#   base   : lead 프라이머 (왼손 저음, voicing~1.23)  = D3 하한 재현
#   twohand: 두손 프라이머 (voicing~1.43)             = H-D4a
#   chordal: 화음강조 프라이머 (voicing~1.83)          = H-D4a 세부
#   none   : 무프라이머                               = D3 상한 재현
#
# 사용법:
#   bash scripts/run_d4_experiment.sh                  # 전체
#   STAGE=primers bash scripts/run_d4_experiment.sh    # 프라이머 생성만
#   ARMS="base twohand" bash scripts/run_d4_experiment.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PY="${PY:-./.venv/bin/python}"
DEVICE="${DEVICE:-mps}"
BASE_CKPT="${BASE_CKPT:-outputs/d3_experiment/armB_ext16/ckpt}"
PRIMER_SRC="${PRIMER_SRC:-./data/roles/lead}"
PRIMER_DIR="${PRIMER_DIR:-./data/d4_primers}"
OUT_ROOT="${OUT_ROOT:-outputs/d4_experiment}"
NUM_SAMPLES="${NUM_SAMPLES:-24}"
GEN_LENGTH="${GEN_LENGTH:-1024}"
GEN_TEMP="${GEN_TEMP:-1.0}"
SEED="${SEED:-42}"
STAGE="${STAGE:-all}"
ARMS="${ARMS:-base twohand chordal none}"
# 기준 conditioning (base 팔): 첫 lead 샘플
LEAD_PRIMER="${LEAD_PRIMER:-${PRIMER_SRC}/000000/conditioning.mid}"
export PYTHONPATH="music_transformer:music_transformer/third_party:${PYTHONPATH:-}"

mkdir -p "${OUT_ROOT}"

# 팔 -> conditioning MIDI 경로 매핑 (000000 샘플 기준)
primer_for () {
  case "$1" in
    base)    echo "${LEAD_PRIMER}" ;;
    twohand) echo "${PRIMER_DIR}/000000/twohand.mid" ;;
    chordal) echo "${PRIMER_DIR}/000000/chordal.mid" ;;
    none)    echo "" ;;
    *) echo "unknown arm: $1" >&2; exit 1 ;;
  esac
}

make_primers () {
  echo ">>> [PRIMERS] 두손/화음강조 프라이머 생성 -> ${PRIMER_DIR}"
  "${PY}" scripts/make_d4_primers.py \
    --roles_dir "${PRIMER_SRC}" --out_dir "${PRIMER_DIR}" --num "${NUM_SAMPLES}"
}

gen_arm () {
  local arm="$1"; local primer; primer="$(primer_for "${arm}")"
  local base="${OUT_ROOT}/${arm}"
  echo; echo ">>> [GEN] Arm ${arm} (primer='${primer:-<none>}') -> ${base}/samples"
  mkdir -p "${base}/samples"
  local cond_arg=()
  [ -n "${primer}" ] && cond_arg=(--conditioning_midi "${primer}")
  "${PY}" scripts/generate.py \
    --lora_path "${BASE_CKPT}" \
    ${cond_arg[@]+"${cond_arg[@]}"} \
    --num_samples "${NUM_SAMPLES}" \
    --length "${GEN_LENGTH}" \
    --temperature "${GEN_TEMP}" \
    --seed "${SEED}" \
    --grammar_mask \
    --output "${base}/samples"
}

measure_arm () {
  local arm="$1"; local base="${OUT_ROOT}/${arm}"
  local json="${OUT_ROOT}/d4_${arm}_vs_base.json"
  echo; echo ">>> [MEASURE] Arm ${arm} vs base -> ${json}"
  "${PY}" scripts/diversity_metrics.py \
    --npy_dir "${base}/samples" \
    --compare_dir "${OUT_ROOT}/base/samples" \
    --json_out "${json}"
}

case "${STAGE}" in
  primers) make_primers ;;
  gen)     for a in ${ARMS}; do gen_arm "$a"; done ;;
  measure) for a in ${ARMS}; do [ "$a" != base ] && measure_arm "$a"; done ;;
  all)
    make_primers
    for a in ${ARMS}; do gen_arm "$a"; done
    for a in ${ARMS}; do [ "$a" != base ] && measure_arm "$a"; done
    ;;
  *) echo "unknown STAGE: ${STAGE}" >&2; exit 1 ;;
esac
echo; echo ">>> D4 완료. 결과 JSON: ${OUT_ROOT}/d4_*_vs_base.json"
