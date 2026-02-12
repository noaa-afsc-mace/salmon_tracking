#!/bin/bash
# Launches 14 tmux windows to run all 28 tracker optimization jobs in parallel.
# Tracker pairs run sequentially in each window: botsort then ioutrack,
# bytetrack then centroidtrack. 7 jobs per pair = 14 windows total.
#
# Generated YAML configs are saved to tracking/optimization_jobs/.
#
# Usage: bash tracking/launch_all_optimizations.sh [gpu_ids]
#   gpu_ids: comma-separated GPU IDs to use (default: all GPUs)
#   Example: bash tracking/launch_all_optimizations.sh 1,2
# (run from the project root directory, requires pyenv with salmon_tracking env)
#
# Monitor: tmux attach -t salmon_opt
# Check status: tmux list-windows -t salmon_opt

set -e

# GPU selection: use argument if provided, otherwise all available GPUs
if [ -n "$1" ]; then
    IFS=',' read -ra GPU_IDS <<< "$1"
else
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    GPU_IDS=()
    for ((i=0; i<NUM_GPUS; i++)); do GPU_IDS+=("$i"); done
fi
if [ "${#GPU_IDS[@]}" -lt 1 ]; then
    GPU_IDS=(0)
fi
echo "Using GPU(s): ${GPU_IDS[*]} (${#GPU_IDS[@]} total), distributing jobs round-robin"
GPU_COUNTER=0

SESSION="salmon_opt"
JOBS_DIR="tracking/optimization_jobs"
NUM_ITERS=75
DET_TEMPLATE="model_type-{model}_data_source-clip_based_2_class_2_11_25_training_epochs-500_hyperparameter_source-default"

# Tracker pairs (each pair runs in one process)
PAIRS=(
    "botsort ioutrack"
    "bytetrack centroidtrack"
)

# Jobs: "model fps mode"
JOBS=(
    "yolo12n 30.0 train"
    "yolo12m 30.0 train"
    "yolo12x 30.0 train"
    "yolo12x 15.0 train"
    "yolo12x 10.0 train"
    "yolo12x 7.5 train"
    "yolo12x 30.0 test"
)

# --- Generate YAML configs ---
mkdir -p "$JOBS_DIR"

# Clean old generated configs
rm -f "$JOBS_DIR"/job_*.yaml

for pair in "${PAIRS[@]}"; do
    read -r t1 t2 <<< "$pair"
    pair_label="${t1}_${t2}"

    for job in "${JOBS[@]}"; do
        read -r model fps mode <<< "$job"
        job_label="${model}_fps${fps}_${mode}"
        yaml_file="$JOBS_DIR/job_${pair_label}_${job_label}.yaml"

        cat > "$yaml_file" <<EOF
trackers:
  - ${t1}
  - ${t2}

mode: ${mode}
num_search_iterations: ${NUM_ITERS}

detection_source_template: "${DET_TEMPLATE}"

jobs:
  - model: ${model}
    fps: [${fps}]
EOF
        echo "Generated: $yaml_file"
    done
done

# --- Launch tmux session ---
# Kill existing session if present
tmux kill-session -t "$SESSION" 2>/dev/null || true

FIRST=true
for pair in "${PAIRS[@]}"; do
    read -r t1 t2 <<< "$pair"
    pair_label="${t1}_${t2}"

    for job in "${JOBS[@]}"; do
        read -r model fps mode <<< "$job"
        job_label="${model}_fps${fps}_${mode}"
        yaml_file="$JOBS_DIR/job_${pair_label}_${job_label}.yaml"
        fps_label="${fps/./_}"
        window_name="${t1:0:3}${t2:0:3}_${model##yolo12}_${fps_label}_${mode:0:2}"

        gpu_id=${GPU_IDS[$((GPU_COUNTER % ${#GPU_IDS[@]}))]}
        cmd="cd $(pwd) && CUDA_VISIBLE_DEVICES=${gpu_id} PYENV_VERSION=salmon_tracking pyenv exec python tracking/smac_optimize_track.py ${yaml_file}"

        if [ "$FIRST" = true ]; then
            tmux new-session -d -s "$SESSION" -n "$window_name"
            tmux send-keys -t "$SESSION:$window_name" "$cmd" Enter
            FIRST=false
        else
            tmux new-window -t "$SESSION" -n "$window_name"
            tmux send-keys -t "$SESSION:$window_name" "$cmd" Enter
        fi

        echo "Launched: $window_name -> $yaml_file (GPU $gpu_id)"
        GPU_COUNTER=$((GPU_COUNTER + 1))
    done
done

echo ""
echo "All 14 windows launched in tmux session '$SESSION'"
echo "Attach with: tmux attach -t $SESSION"
echo "List windows: tmux list-windows -t $SESSION"
