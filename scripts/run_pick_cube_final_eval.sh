#!/usr/bin/env bash
# Evaluate pick_cube_final (camera cube_xyz + pick_cube .pt) for 100 attempts.
#
# Writes:
#   results/pick_cube_final/testing/result.json
#   results/pick_cube_final/testing/plots/01_accuracy.png
#   results/pick_cube_final/testing/plots/02_completion_time.png
#   results/pick_cube_final/testing/plots/summary.txt
#
# Usage (from repo root)::
#
#   PYTHON_BIN=/path/to/isaac_env/bin/python bash scripts/run_pick_cube_final_eval.sh
#   CHECKPOINT=model/pick_cube/training_2.pt bash scripts/run_pick_cube_final_eval.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# pick_cube_final is play-only; load a pick_cube policy (obs layout matches).
CHECKPOINT="${CHECKPOINT:-$ROOT/model/pick_cube/training_3.pt}"
NUM_ATTEMPTS="${NUM_ATTEMPTS:-100}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUT_DIR:-$ROOT/results/pick_cube_final/testing}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[ERROR] Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

echo "[INFO] task=ThesisRl-Pick-Cube-Final-UR5E"
echo "[INFO] checkpoint=$CHECKPOINT"
echo "[INFO] attempts=$NUM_ATTEMPTS"
echo "[INFO] output → $OUT_DIR"

export HEADLESS="${HEADLESS:-1}"
exec "$PYTHON_BIN" scripts/evaluate_policy.py \
  --task ThesisRl-Pick-Cube-Final-UR5E \
  --checkpoint "$CHECKPOINT" \
  --num_envs 1 \
  --num-attempts "$NUM_ATTEMPTS" \
  --output-dir "$OUT_DIR" \
  physics=isaacsim_physx
