#!/bin/bash
#SBATCH --job-name=rq2-replay
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-9%10
#SBATCH --output=logs/rq2_replay_%A_%a.out
#SBATCH --error=logs/rq2_replay_%A_%a.err

set -euo pipefail

cd /global/home/hpc6237/Contexto

module load python/3.11.5
source /global/home/hpc6237/venvs/contexto/bin/activate

export PATH="$HOME/.local/bin:$PATH"
export OLLAMA_MODELS="$HOME/ollama_models"

export RANK_CACHE_DIR=data/rank_cache_A1
export RANK_CACHE_ENABLED=1

GAMES=(1303 1307 1319 1327 1335 1352 1364 1365 1372 1384)
GAME=${GAMES[$SLURM_ARRAY_TASK_ID]}

mkdir -p logs
mkdir -p "traces/rq2_replay/${GAME}"

PORT=$((20000 + (SLURM_JOB_ID + SLURM_ARRAY_TASK_ID) % 20000))

export OLLAMA_HOST="127.0.0.1:${PORT}"
export OLLAMA_BASE_URL="http://127.0.0.1:${PORT}/v1"
export OLLAMA_KEEP_ALIVE=12h
export OLLAMA_REQUEST_TIMEOUT_SECONDS=900

OLLAMA_LOG="logs/ollama_rq2_replay_game${GAME}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
METADATA="traces/rq2_replay/${GAME}/ollama_metadata_job${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt"

echo "===== RQ2 REPLAY JOB ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Game: $GAME"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "OLLAMA_BASE_URL=$OLLAMA_BASE_URL"
echo "RANK_CACHE_DIR=$RANK_CACHE_DIR"
echo "RANK_CACHE_ENABLED=$RANK_CACHE_ENABLED"

echo "Python: $(which python)"
python --version
python -c "import numpy; print('NumPy:', numpy.__version__)"

nvidia-smi || true

ollama serve > "$OLLAMA_LOG" 2>&1 &
OLLAMA_PID=$!

cleanup() {
    kill "$OLLAMA_PID" 2>/dev/null || true
}
trap cleanup EXIT

READY=0
for attempt in $(seq 1 24); do
    if ollama list >/dev/null 2>&1; then
        READY=1
        break
    fi
    sleep 5
done

if [ "$READY" -ne 1 ]; then
    echo "ERROR: Ollama failed to start."
    exit 1
fi

{
    echo "Job ID: $SLURM_JOB_ID"
    echo "Array task: $SLURM_ARRAY_TASK_ID"
    echo "Game: $GAME"
    echo "Host: $(hostname)"
    echo "Date: $(date)"
    echo
    echo "===== nvidia-smi ====="
    nvidia-smi
    echo
    echo "===== ollama --version ====="
    ollama --version
    echo
    echo "===== ollama list ====="
    ollama list
} > "$METADATA"

echo "===== START REPLAY ====="

python scripts/rq2_replay.py \
  "traces/rq1_A1/ea_llm_self_adaptive_api_${GAME}_run*_*.json" \
  --events-per-trace 40 \
  --seed 0 \
  --output "traces/rq2_replay/${GAME}"

STATUS=$?

echo "===== FINISHED ====="
echo "Game: $GAME"
echo "Exit status: $STATUS"
echo "Date: $(date)"

exit "$STATUS"
