#!/bin/bash
# resumable via per-threshold work directories.
# Start with smallest threshold (most expensive / most boxes).

set -e

VDB="$(pwd)/wdas_cloud/wdas_cloud.vdb"
WORKERS=10
MAX_LEVEL=12        # native levels are [11, 11, 12]
SPLIT_LEVELS="2 2 3 "  # 128 partitions    
BASE_DIR="$(pwd)/cloud_full_sweep"

THRESHOLDS=( 0 1e-8 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2)

mkdir -p "$BASE_DIR"

for THR in "${THRESHOLDS[@]}"; do
    WORK_DIR="$BASE_DIR/thr_${THR}"
    LOG="$BASE_DIR/log_${THR}.txt"

    echo "============================================================"
    echo "$(date '+%Y-%m-%d %H:%M:%S') — threshold=$THR"
    echo "  work-dir: $WORK_DIR"
    echo "  log: $LOG"
    echo "============================================================"

    $(pwd)/bin/micromamba run -n openvdb python -O \
        $(pwd)/parallel_cloud_compression.py \
        --vdb "$VDB" \
        --workers "$WORKERS" \
        --max-level "$MAX_LEVEL" \
        --split-levels $SPLIT_LEVELS \
        --threshold "$THR" \
        --skip-serial \
	--only-subs \
        --work-dir "$WORK_DIR" \
        2>&1 | tee "$LOG"

    echo ""
    echo "$(date '+%Y-%m-%d %H:%M:%S') — threshold=$THR DONE"
    echo ""
done

echo "============================================================"
echo "ALL THRESHOLDS COMPLETE"
echo "============================================================"
