#!/usr/bin/env bash
# =============================================================================
# D0 진단 실험 하니스 — "데이터 부족이 다양성 붕괴의 원인인가?"
#
# 두 팔(arm)을 동일 설정으로 from-scratch 학습한 뒤, 각 모델의 생성 샘플
# 다양성을 측정해 비교한다.
#
#   Arm A (baseline)  : 16조각 probe 코퍼스로 학습  -> 붕괴 재현 여부
#   Arm B (treatment) : 전체 2,777 코퍼스로 학습     -> 다양성 회복 여부
#
# 두 팔은 데이터 크기만 다르고 모델/하이퍼파라미터/생성 설정은 동일하다.
# 따라서 생성 다양성의 차이는 데이터 레짐의 효과로 귀속된다.
#
# 사용법 (맥 터미널, MPS):
#   bash scripts/run_d0_experiment.sh
#
# 개별 단계만 다시 돌리고 싶으면 STAGE 환경변수:
#   STAGE=train   bash scripts/run_d0_experiment.sh   # 학습만
#   STAGE=gen     bash scripts/run_d0_experiment.sh   # 생성만
#   STAGE=measure bash scripts/run_d0_experiment.sh   # 다양성 측정만
#   STAGE=all     (기본)                              # 전체
#
# 주요 파라미터는 아래 환경변수로 덮어쓸 수 있다 (기본값은 MPS/32GB 기준).
# =============================================================================
set -euo pipefail

# ---- 레포 루트로 이동 (스크립트 위치 기준) --------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ---- 파이썬 / 경로 ----------------------------------------------------------
PY="${PY:-./.venv/bin/python}"
export PYTHONPATH="music_transformer:music_transformer/third_party:scripts${PYTHONPATH:+:$PYTHONPATH}"

# ---- 실험 파라미터 (환경변수로 override 가능) -------------------------------
DEVICE="${DEVICE:-mps}"              # auto|cuda|mps|cpu
EPOCHS="${EPOCHS:-8}"                # from-scratch 이므로 probe 대비 넉넉히
BATCH_SIZE="${BATCH_SIZE:-4}"        # MPS 메모리 보수적
MAX_SEQ="${MAX_SEQ:-1024}"           # 학습 시퀀스 길이 (crop)
LR="${LR:-3e-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"

NUM_SAMPLES="${NUM_SAMPLES:-24}"     # 팔당 생성 샘플 수 (다양성 통계 안정)
GEN_LENGTH="${GEN_LENGTH:-1024}"     # 생성 토큰 길이
GEN_TEMP="${GEN_TEMP:-1.0}"

# 데이터 디렉터리
DATA_PROBE="${DATA_PROBE:-./data/roles/lead/tokenized}"   # 16조각 (train/val 하위)
DATA_FULL="${DATA_FULL:-./data/jazz_full}"                # 전체 2,777

# 출력 경로
OUT_ROOT="${OUT_ROOT:-./outputs/d0_experiment}"
CKPT_A="${OUT_ROOT}/armA_probe16/ckpt"
CKPT_B="${OUT_ROOT}/armB_full2777/ckpt"
SAMP_A="${OUT_ROOT}/armA_probe16/samples"
SAMP_B="${OUT_ROOT}/armB_full2777/samples"

# 생성 프라이머 (조건 MIDI). 두 팔 모두 동일한 프라이머를 써서 공정 비교.
PRIMER="${PRIMER:-./data/roles/lead/000000/conditioning.mid}"

STAGE="${STAGE:-all}"

mkdir -p "${OUT_ROOT}" "${CKPT_A%/*}" "${CKPT_B%/*}" "${SAMP_A%/*}" "${SAMP_B%/*}"

echo "=================================================================="
echo " D0 실험 하니스"
echo "   device=${DEVICE}  epochs=${EPOCHS}  batch=${BATCH_SIZE}  max_seq=${MAX_SEQ}"
echo "   probe data : ${DATA_PROBE}"
echo "   full  data : ${DATA_FULL}"
echo "   out root   : ${OUT_ROOT}"
echo "   stage      : ${STAGE}"
echo "=================================================================="

# ---- 사전 점검: 데이터가 실제로 있는가 --------------------------------------
count_npy () { find "$1" -name '*.npy' 2>/dev/null | wc -l | tr -d ' '; }
if [ "${STAGE}" = "all" ] || [ "${STAGE}" = "train" ]; then
  PA_TRAIN=$(count_npy "${DATA_PROBE}/train")
  FB_TRAIN=$(count_npy "${DATA_FULL}/train")
  echo "[check] probe train npy=${PA_TRAIN}, full train npy=${FB_TRAIN}"
  if [ "${PA_TRAIN}" -eq 0 ]; then echo "!! probe train 데이터 없음: ${DATA_PROBE}/train"; exit 1; fi
  if [ "${FB_TRAIN}" -eq 0 ]; then
    echo "!! full train 데이터 없음: ${DATA_FULL}/train"
    echo "   먼저 전체 코퍼스를 토크나이즈하세요:"
    echo "   PYTHONPATH=\"music_transformer:music_transformer/third_party\" ${PY} archive/scripts/preprocess_jazz.py --input_dir ./midi_dataset/midi --output_dir ${DATA_FULL}"
    exit 1
  fi
fi

# ---- 공통 학습 함수 ---------------------------------------------------------
train_arm () {
  local name="$1" data_dir="$2" out_dir="$3"
  echo
  echo ">>> [TRAIN] ${name}  (data=${data_dir} -> ${out_dir})"
  mkdir -p "${out_dir}"
  "${PY}" scripts/train_qlora.py \
    --train_full_model \
    --data_dir "${data_dir}" \
    --output_dir "${out_dir}" \
    --device "${DEVICE}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --max_sequence "${MAX_SEQ}" \
    --lr "${LR}" \
    --gradient_accumulation "${GRAD_ACCUM}"
}

# ---- 공통 생성 함수 ---------------------------------------------------------
gen_arm () {
  local name="$1" ckpt_dir="$2" samp_dir="$3"
  echo
  echo ">>> [GEN] ${name}  (ckpt=${ckpt_dir} -> ${samp_dir})"
  mkdir -p "${samp_dir}"
  "${PY}" scripts/generate.py \
    --lora_path "${ckpt_dir}" \
    --conditioning_midi "${PRIMER}" \
    --num_samples "${NUM_SAMPLES}" \
    --length "${GEN_LENGTH}" \
    --temperature "${GEN_TEMP}" \
    --output "${samp_dir}"
}

# ---- STAGE: train -----------------------------------------------------------
if [ "${STAGE}" = "all" ] || [ "${STAGE}" = "train" ]; then
  train_arm "Arm A (probe16)"  "${DATA_PROBE}" "${CKPT_A}"
  train_arm "Arm B (full2777)" "${DATA_FULL}"  "${CKPT_B}"
fi

# ---- STAGE: gen -------------------------------------------------------------
if [ "${STAGE}" = "all" ] || [ "${STAGE}" = "gen" ]; then
  gen_arm "Arm A (probe16)"  "${CKPT_A}" "${SAMP_A}"
  gen_arm "Arm B (full2777)" "${CKPT_B}" "${SAMP_B}"
fi

# ---- STAGE: measure ---------------------------------------------------------
if [ "${STAGE}" = "all" ] || [ "${STAGE}" = "measure" ]; then
  echo
  echo ">>> [MEASURE] 생성 샘플 다양성 비교 (Arm A vs Arm B)"
  JSON_OUT="${OUT_ROOT}/d0_generated_armA_vs_armB.json"
  # diversity_metrics.py 는 --npy_dir 에 .npy 또는 .mid/.midi 디렉터리를
  # 넘기면 확장자를 자동 판별한다. generate.py 는 MIDI 를 출력하므로 그대로 사용.
  "${PY}" scripts/diversity_metrics.py \
    --npy_dir "${SAMP_A}" \
    --compare_dir "${SAMP_B}" \
    --labels armA_probe16 armB_full2777 \
    --json_out "${JSON_OUT}"
  echo
  echo "=================================================================="
  echo " 완료. 결과 JSON: ${JSON_OUT}"
  echo " 붕괴가 데이터 부족 탓이라면: Arm B 가 보이싱/PC 엔트로피 ↑,"
  echo " 고유 보이싱 ↑, n-gram 반복 ↓ 로 나와야 한다."
  echo "=================================================================="
fi
