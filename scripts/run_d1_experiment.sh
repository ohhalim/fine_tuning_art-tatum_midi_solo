#!/usr/bin/env bash
# =============================================================================
# D1 실험 하니스: 2단계 커리큘럼 — 사전학습→스타일 FT가 다양성 붕괴를 막는가
#
# 설계: docs/d1_experiment/D1_DESIGN.md (판정 기준 사전 등록됨)
# 선행: D0 실행 완료 (outputs/d0_experiment/ 에 Arm A 샘플 + Arm B 체크포인트)
#
# 사용:
#   bash scripts/run_d1_experiment.sh                 # C, D 팔 전체 (train→gen→measure)
#   ARMS="C D E" bash scripts/run_d1_experiment.sh    # 증강 팔 E 포함
#   STAGE=gen bash scripts/run_d1_experiment.sh       # 특정 단계만 재실행
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="music_transformer:music_transformer/third_party:scripts${PYTHONPATH:+:$PYTHONPATH}"
PY="${PY:-.venv/bin/python}"

STAGE="${STAGE:-all}"          # all | train | gen | measure
ARMS="${ARMS:-C D}"            # 실행할 팔: C(full FT) D(LoRA) E(증강 full FT)
DEVICE="${DEVICE:-mps}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SEED="${SEED:-42}"

# 팔별 하이퍼파라미터 (D1_DESIGN.md §3 — 사전 고정, 변경 금지)
LR_C="${LR_C:-3e-5}";  EPOCHS_C="${EPOCHS_C:-4}"
LR_D="${LR_D:-3e-4}";  EPOCHS_D="${EPOCHS_D:-8}"
LR_E="${LR_E:-3e-5}";  EPOCHS_E="${EPOCHS_E:-4}"

# 생성 파라미터 (D0와 동일 — 비교 가능성 유지, 변경 금지)
NUM_SAMPLES="${NUM_SAMPLES:-24}"
GEN_LENGTH="${GEN_LENGTH:-1024}"
GEN_TEMP="${GEN_TEMP:-1.0}"
PRIMER="${PRIMER:-./data/roles/lead/000000/conditioning.mid}"

# 경로
CKPT_B="${CKPT_B:-outputs/d0_experiment/armB_full2777/ckpt/checkpoint_epoch8.pt}"
DATA_PROBE="${DATA_PROBE:-./data/roles/lead/tokenized}"
DATA_AUG="${DATA_AUG:-./data/roles_aug12/lead/tokenized}"
AUG_INPUT="${AUG_INPUT:-./midi_dataset/midi/studio/Brad Mehldau}"
SAMP_A="${SAMP_A:-outputs/d0_experiment/armA_probe16/samples}"   # 대조군 (D0)
OUT_ROOT="${OUT_ROOT:-outputs/d1_experiment}"

arm_dir () {  # arm_dir C -> outputs/d1_experiment/armC_fullft
  case "$1" in
    C) echo "${OUT_ROOT}/armC_fullft" ;;
    D) echo "${OUT_ROOT}/armD_lora" ;;
    E) echo "${OUT_ROOT}/armE_aug12" ;;
    *) echo "!! 알 수 없는 팔: $1" >&2; exit 1 ;;
  esac
}

echo "== D1 하니스 =="
echo "   stage=${STAGE}  arms=${ARMS}  device=${DEVICE}  seed=${SEED}"
echo "   base ckpt: ${CKPT_B}"

# ---- 사전 점검 --------------------------------------------------------------
[ -f "${CKPT_B}" ] || { echo "!! Arm B 체크포인트 없음: ${CKPT_B} (D0 먼저 실행)"; exit 1; }
[ -d "${DATA_PROBE}/train" ] || { echo "!! probe 데이터 없음: ${DATA_PROBE}/train"; exit 1; }
if [ "${STAGE}" = "all" ] || [ "${STAGE}" = "measure" ]; then
  [ -d "${SAMP_A}" ] || { echo "!! Arm A 샘플 없음: ${SAMP_A} (D0 gen 필요)"; exit 1; }
fi

# ---- Arm E 증강 데이터 준비 ---------------------------------------------------
prepare_aug () {
  if [ -d "${DATA_AUG}/train" ] && [ "$(ls "${DATA_AUG}/train" | wc -l)" -gt 0 ]; then
    echo ">>> 증강 데이터 이미 존재: ${DATA_AUG}"
    return
  fi
  echo ">>> [AUG] 전조 증강 데이터셋 생성 (×12키)"
  "${PY}" archive/scripts/prepare_role_dataset.py \
    --input_dir "${AUG_INPUT}" \
    --output_dir ./data/roles_aug12 --role lead \
    --transpose_all_keys --seed "${SEED}"
}

# ---- 공통 학습 함수 -----------------------------------------------------------
train_arm () {
  local arm="$1" data_dir="$2" lr="$3" epochs="$4" full_flag="$5"
  local out; out="$(arm_dir "${arm}")/ckpt"
  echo
  echo ">>> [TRAIN] Arm ${arm}  (data=${data_dir} lr=${lr} epochs=${epochs} mode=${full_flag:-adapter})"
  mkdir -p "${out}"
  # shellcheck disable=SC2086
  "${PY}" scripts/train_qlora.py \
    --checkpoint "${CKPT_B}" ${full_flag} \
    --data_dir "${data_dir}" \
    --output_dir "${out}" \
    --device "${DEVICE}" \
    --epochs "${epochs}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${lr}" \
    --seed "${SEED}"
}

gen_arm () {
  local arm="$1"
  local base; base="$(arm_dir "${arm}")"
  echo
  echo ">>> [GEN] Arm ${arm}  -> ${base}/samples"
  mkdir -p "${base}/samples"
  "${PY}" scripts/generate.py \
    --lora_path "${base}/ckpt" \
    --conditioning_midi "${PRIMER}" \
    --num_samples "${NUM_SAMPLES}" \
    --length "${GEN_LENGTH}" \
    --temperature "${GEN_TEMP}" \
    --output "${base}/samples"
}

measure_arm () {
  local arm="$1"
  local base; base="$(arm_dir "${arm}")"
  local json="${OUT_ROOT}/d1_arm${arm}_vs_armA.json"
  echo
  echo ">>> [MEASURE] Arm ${arm} vs Arm A  -> ${json}"
  "${PY}" scripts/diversity_metrics.py \
    --npy_dir "${base}/samples" \
    --compare_dir "${SAMP_A}" \
    --json_out "${json}"
}

# ---- 실행 ---------------------------------------------------------------------
for arm in ${ARMS}; do
  case "${arm}" in
    C) data="${DATA_PROBE}"; lr="${LR_C}"; ep="${EPOCHS_C}"; flag="--train_full_model" ;;
    D) data="${DATA_PROBE}"; lr="${LR_D}"; ep="${EPOCHS_D}"; flag="" ;;
    E) prepare_aug; data="${DATA_AUG}"; lr="${LR_E}"; ep="${EPOCHS_E}"; flag="--train_full_model" ;;
    *) echo "!! 알 수 없는 팔: ${arm}"; exit 1 ;;
  esac
  if [ "${STAGE}" = "all" ] || [ "${STAGE}" = "train" ]; then
    train_arm "${arm}" "${data}" "${lr}" "${ep}" "${flag}"
  fi
  if [ "${STAGE}" = "all" ] || [ "${STAGE}" = "gen" ]; then
    gen_arm "${arm}"
  fi
  if [ "${STAGE}" = "all" ] || [ "${STAGE}" = "measure" ]; then
    measure_arm "${arm}"
  fi
done

echo
echo "== D1 완료 =="
echo "   판정 기준(사전 등록): pooled_unique_voicings ≥116 강한 성공 / 105–115 부분 / <105 실패"
echo "   게이트: best val loss < 5.0585(Arm A) + 학습 중 하강"
echo "   결과 JSON: ${OUT_ROOT}/d1_arm*_vs_armA.json"
