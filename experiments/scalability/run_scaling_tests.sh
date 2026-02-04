#!/bin/bash

# EuroHPC JUPITER Scalability Test Launcher
# =========================================
# This script manages scaling experiments.
# Mode 1: Localhost Simulation (MPI/OpenMPI) - Default
# Mode 2: Slurm/GPU Cluster Execution - Triggered by --slurm flag

EXPERIMENT_DIR=$(dirname "$0")
SRC_DIR="$EXPERIMENT_DIR/src"
LOG_DIR="$EXPERIMENT_DIR/logs"
RESULTS_FILE="$EXPERIMENT_DIR/results_scaling.csv"

mkdir -p "$LOG_DIR"

MODE="local"
if [[ "$1" == "--slurm" ]]; then
    MODE="slurm"
fi

# Function to run Julia Benchmark
run_julia() {
    local NP=$1
    local MODE=$2
    local LOG_FILE="$LOG_DIR/julia_lux_${NP}proc.log"
    echo "  [Julia] Running with $NP Processes..."

    if [[ "$MODE" == "local" ]]; then
        mpiexecjl --project=experiments/scalability -n $NP julia --project=experiments/scalability "$SRC_DIR/train_lux_distributed.jl" > "$LOG_FILE" 2>&1
    else
        # Slurm Mode: Use srun
        # Assume 1 GPU per task
        srun --ntasks=$NP --gpus-per-task=1 --cpus-per-task=4 \
             julia --project=experiments/scalability "$SRC_DIR/train_lux_distributed.jl" > "$LOG_FILE" 2>&1
    fi

    if [ $? -eq 0 ]; then
        TIME=$(grep "Epoch 2" "$LOG_FILE" | head -n 1 | awk '{print $7}' | sed 's/s//')
        [ -z "$TIME" ] && TIME="N/A"
        echo "    Success: ${TIME}s"
        echo "Julia,$NP,$TIME" >> "$RESULTS_FILE"
    else
        echo "    Failed. Check log: $LOG_FILE"
    fi
}

# Function to run Python Benchmark
run_python() {
    local NP=$1
    local MODE=$2
    local LOG_FILE="$LOG_DIR/python_pl_${NP}proc.log"
    echo "  [Python] Running with $NP Processes..."

    if [[ "$MODE" == "local" ]]; then
        python3 "$SRC_DIR/train_lightning.py" --accelerator cpu --strategy ddp --num_processes $NP > "$LOG_FILE" 2>&1
    else
        # Slurm Mode: Use srun
        # PyTorch Lightning with DDP can be launched via srun (if using DDPStrategy)
        # or python (if using sbatch environment vars).
        # We use srun to be explicit about resource mapping.
        srun --ntasks=$NP --gpus-per-task=1 --cpus-per-task=4 \
             python3 "$SRC_DIR/train_lightning.py" --accelerator gpu --strategy ddp --gpus 1 --num_nodes $SLURM_JOB_NUM_NODES > "$LOG_FILE" 2>&1
    fi

    if [ $? -eq 0 ]; then
        TOTAL_TIME=$(grep "Training finished in" "$LOG_FILE" | awk '{print $4}')
        EPOCH_TIME=$(echo "$TOTAL_TIME / 2" | bc -l)
        echo "    Success: ${EPOCH_TIME}s"
        echo "Python,$NP,$EPOCH_TIME" >> "$RESULTS_FILE"
    else
        echo "    Failed. Check log: $LOG_FILE"
    fi
}


if [[ "$MODE" == "local" ]]; then
    echo "Starting Distributed Scalability Experiments (Localhost Simulation)..."
    echo "===================================================================="
    echo "Framework,Num_Procs,Epoch_Time_s" > "$RESULTS_FILE"

    STEPS=(1 2 4)
    export OMPI_MCA_rmaps_base_oversubscribe=1
    export OMPI_MCA_btl_vader_single_copy_mechanism=none

    for NP in "${STEPS[@]}"; do
        run_julia $NP "local"
    done

    for NP in "${STEPS[@]}"; do
        run_python $NP "local"
    done

elif [[ "$MODE" == "slurm" ]]; then
    echo "Starting Scalability Experiments (Slurm/GPU Mode)..."
    echo "===================================================="
    echo "Framework,Num_Procs,Epoch_Time_s" > "$RESULTS_FILE"

    # Detect available resources
    # SLURM_NTASKS might be set, or we infer from gpus.
    # We assume we are inside an allocation (e.g. 4 GPUs).
    # We want to run scaling: 1 GPU, 2 GPUs, 4 GPUs... up to limit.

    MAX_GPUS=${SLURM_GPUS_ON_NODE:-1}
    # If multiple nodes, multiply
    if [ -n "$SLURM_NNODES" ]; then
        MAX_GPUS=$((MAX_GPUS * SLURM_NNODES))
    fi

    # Or rely on user input if provided, else detect
    if [ -n "$2" ]; then MAX_GPUS=$2; fi

    echo "Max Available GPUs: $MAX_GPUS"

    # Generate Steps: 1, 2, 4, 8... <= MAX_GPUS
    STEPS=()
    curr=1
    while [ $curr -le $MAX_GPUS ]; do
        STEPS+=($curr)
        curr=$((curr * 2))
    done

    echo "Planned Steps: ${STEPS[@]}"

    for NP in "${STEPS[@]}"; do
        run_julia $NP "slurm"
    done

    for NP in "${STEPS[@]}"; do
        run_python $NP "slurm"
    done
fi

echo "Done. Results saved to $RESULTS_FILE"
