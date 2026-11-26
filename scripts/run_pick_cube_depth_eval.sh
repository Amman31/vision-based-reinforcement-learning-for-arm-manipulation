#!/usr/bin/env bash
# Evaluate pick_cube_depth like move_cube testing (100 attempts → accuracy + time plots).
#
# Writes:
#   results/pick_cube_depth/testing_1/result.json
#   results/pick_cube_depth/testing_1/plots/01_accuracy.png
#   results/pick_cube_depth/testing_1/plots/02_completion_time.png
#   results/pick_cube_depth/testing_1/plots/summary.txt
#
# Usage (from repo root, Isaac / thesis venv active)::
#
#   bash scripts/run_pick_cube_depth_eval.sh
#   bash scripts/run_pick_cube_depth_eval.sh training_2   # optional other run
#   NUM_ATTEMPTS=50 bash scripts/run_pick_cube_depth_eval.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TRAINING_ID="${1:-training_1}"
NUM_ATTEMPTS="${NUM_ATTEMPTS:-100}"
CHECKPOINT="${CHECKPOINT:-$ROOT/model/pick_cube_depth/${TRAINING_ID}.pt}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[ERROR] Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

echo "[INFO] task=ThesisRl-Pick-Cube-Depth-UR5E"
echo "[INFO] checkpoint=$CHECKPOINT"
echo "[INFO] training-id=$TRAINING_ID  attempts=$NUM_ATTEMPTS"
echo "[INFO] output → results/pick_cube_depth/testing_${TRAINING_ID#training_}/"

# Same entry point as move_cube eval. Current Isaac Lab auto-enables cameras when
# the scene has Kit camera sensors; headless is via HEADLESS=1 (no --headless flag).
export HEADLESS="${HEADLESS:-1}"
exec "$PYTHON_BIN" scripts/evaluate_policy.py \
  --task ThesisRl-Pick-Cube-Depth-UR5E \
  --checkpoint "$CHECKPOINT" \
  --training-id "$TRAINING_ID" \
  --num_envs 1 \
  --num-attempts "$NUM_ATTEMPTS" \
  physics=isaacsim_physx
