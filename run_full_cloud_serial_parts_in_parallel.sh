#!/bin/bash
# Run the serial (compose + final coarsening) phase for each threshold in parallel.

set -e

VDB="$(pwd)/wdas_cloud/wdas_cloud.vdb"
MAX_LEVEL=12
SPLIT_LEVELS="2 2 3 "
BASE_DIR="$(pwd)/cloud_full_sweep"
CONCURRENCY="${CONCURRENCY:-8}"    

THRESHOLDS=(0 1e-8 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2 )

mkdir -p "$BASE_DIR"
MASTER_LOG="$BASE_DIR/master_serial_log.txt"

run_one_serial() {
    local THR=$1
    local WORK_DIR="$BASE_DIR/thr_${THR}"
    local LOG="$BASE_DIR/serial_log_${THR}.txt"

    if [ ! -d "$WORK_DIR" ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') SKIP  threshold=$THR (no work-dir $WORK_DIR)" | tee -a "$MASTER_LOG"
        return 0
    fi
    if [ -f "$WORK_DIR/final_3d.bin" ] && [ -f "$WORK_DIR/final_values.npy" ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') SKIP  threshold=$THR (final already exists)" | tee -a "$MASTER_LOG"
        return 0
    fi

    echo "$(date '+%Y-%m-%d %H:%M:%S') START threshold=$THR" | tee -a "$MASTER_LOG"
    /workspace/bin/micromamba run -n openvdb python -O \
        /workspace/wavelets_with_omnitrees/parallel_cloud_compression.py \
        --vdb "$VDB" \
        --workers 1 \
        --max-level "$MAX_LEVEL" \
        --split-levels "$SPLIT_LEVELS" \
        --threshold "$THR" \
        --skip-serial \
        --work-dir "$WORK_DIR" \
        > "$LOG" 2>&1
    echo "$(date '+%Y-%m-%d %H:%M:%S') DONE  threshold=$THR" | tee -a "$MASTER_LOG"
}

export -f run_one_serial
export VDB MAX_LEVEL SPLIT_LEVELS BASE_DIR MASTER_LOG

echo "$(date '+%Y-%m-%d %H:%M:%S') === Serial-phase sweep, concurrency=$CONCURRENCY ===" | tee -a "$MASTER_LOG"

# Fan out with a simple job-slot pool (no GNU parallel dependency).
pids=()
for THR in "${THRESHOLDS[@]}"; do
    # Wait until a slot frees up
    while [ "${#pids[@]}" -ge "$CONCURRENCY" ]; do
        new_pids=()
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                new_pids+=("$pid")
            fi
        done
        pids=("${new_pids[@]}")
        [ "${#pids[@]}" -ge "$CONCURRENCY" ] && sleep 1
    done
    run_one_serial "$THR" &
    pids+=($!)
done

# Wait for all remaining jobs
for pid in "${pids[@]}"; do
    wait "$pid" || true
done

echo "$(date '+%Y-%m-%d %H:%M:%S') ALL SERIAL PHASES COMPLETE" | tee -a "$MASTER_LOG"

# ── Size comparison ──────────────────────────────────────────────────────
if [ -x "$BASE_DIR/compare_sizes.sh" ]; then
    "$BASE_DIR/compare_sizes.sh" | tee -a "$MASTER_LOG"
fi
