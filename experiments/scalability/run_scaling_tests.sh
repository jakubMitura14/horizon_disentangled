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
        # Slurm Mode: Single Node Scaling
        # We enforce --nodes=1 to test intra-node scaling on the provided node.
        # --ntasks=$NP (e.g., 1, 2, 4) matches the number of GPUs used.
        srun --nodes=1 --ntasks=$NP --gpus-per-task=1 --cpus-per-task=4 \
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
        # Slurm Mode: Single Node Scaling
        # Explicitly use --nodes=1 and vary --gpus (devices in PL)
        # Using srun to launch python directly can conflict with PL's DDP spawning if not careful.
        # But if we use srun --ntasks=1 and let PL spawn:
        # python script.py --gpus $NP --nodes 1 --strategy ddp

        # However, srun provides the resource isolation.
        # Let's use srun --ntasks=1 (one orchestrator) and let PL handle the GPUs visible.
        # But we need to ensure $NP GPUs are visible.

        srun --nodes=1 --ntasks=1 --gpus=$NP --cpus-per-task=$((4*NP)) \
             python3 "$SRC_DIR/train_lightning.py" --accelerator gpu --strategy ddp --gpus $NP --nodes 1 > "$LOG_FILE" 2>&1
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
    echo "Starting Scalability Experiments (Slurm/GPU Mode - Single Node)..."
    echo "=================================================================="
    echo "Framework,Num_Procs,Epoch_Time_s" > "$RESULTS_FILE"

    # Check max GPUs on this node
    MAX_GPUS=${SLURM_GPUS_ON_NODE:-4}
    # Or passed arg
    if [ -n "$2" ]; then MAX_GPUS=$2; fi

    echo "Max GPUs available: $MAX_GPUS"

    # Generate Steps: 1, 2, 4, ... <= MAX_GPUS
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
