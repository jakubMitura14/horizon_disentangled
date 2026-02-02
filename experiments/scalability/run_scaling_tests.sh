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

if [[ "$MODE" == "local" ]]; then
    echo "Starting Distributed Scalability Experiments (Localhost Simulation)..."
    echo "===================================================================="
    echo "Initialize results file"
    echo "Framework,Num_Procs,Epoch_Time_s" > "$RESULTS_FILE"

    STEPS=(1 2 4)
    export OMPI_MCA_rmaps_base_oversubscribe=1
    export OMPI_MCA_btl_vader_single_copy_mechanism=none

    # --- Julia (Lux + MPI) ---
    echo "--- Benchmarking Julia (Lux + MPI) ---"
    for NP in "${STEPS[@]}"; do
        echo "Running Julia with $NP Processes..."
        LOG_FILE="$LOG_DIR/julia_lux_${NP}proc.log"
        mpiexecjl --project=experiments/scalability -n $NP julia --project=experiments/scalability "$SRC_DIR/train_lux_distributed.jl" > "$LOG_FILE" 2>&1

        if [ $? -eq 0 ]; then
            TIME=$(grep "Epoch 2" "$LOG_FILE" | head -n 1 | awk '{print $7}' | sed 's/s//')
            [ -z "$TIME" ] && TIME="N/A"
            echo "  Time: ${TIME}s"
            echo "Julia,$NP,$TIME" >> "$RESULTS_FILE"
        else
            echo "  Failed. Check log: $LOG_FILE"
        fi
    done

    # --- Python (PyTorch Lightning DDP) ---
    echo "--- Benchmarking Python (PyTorch Lightning DDP) ---"
    for NP in "${STEPS[@]}"; do
        echo "Running Python with $NP Processes..."
        LOG_FILE="$LOG_DIR/python_pl_${NP}proc.log"
        python3 "$SRC_DIR/train_lightning.py" --accelerator cpu --strategy ddp --num_processes $NP > "$LOG_FILE" 2>&1

        if [ $? -eq 0 ]; then
            TOTAL_TIME=$(grep "Training finished in" "$LOG_FILE" | awk '{print $4}')
            EPOCH_TIME=$(echo "$TOTAL_TIME / 2" | bc -l)
            echo "  Time (Approx/Epoch): ${EPOCH_TIME}s"
            echo "Python,$NP,$EPOCH_TIME" >> "$RESULTS_FILE"
        else
            echo "  Failed. Check log: $LOG_FILE"
        fi
    done

elif [[ "$MODE" == "slurm" ]]; then
    echo "Starting Scalability Experiments (Slurm/GPU Mode)..."
    echo "===================================================="
    # Note: This block is intended to be run INSIDE an sbatch allocation or via sbatch submission loop.
    # Here we demonstrate the command structure.

    # We assume this script is called with specific params by the user or scheduler.
    # Usage: ./run_scaling_tests.sh --slurm <NODES> <GPUS_PER_NODE>

    NODES=${2:-1}
    GPUS_PER_NODE=${3:-4}
    TOTAL_GPUS=$((NODES * GPUS_PER_NODE))

    echo "Configuration: $NODES Nodes, $GPUS_PER_NODE GPUs/Node ($TOTAL_GPUS Total)"

    # Julia Run
    LOG_FILE_JL="$LOG_DIR/slurm_julia_${TOTAL_GPUS}gpu.log"
    echo "Launching Julia..."
    srun --ntasks=$TOTAL_GPUS --ntasks-per-node=$GPUS_PER_NODE --gpus-per-node=$GPUS_PER_NODE \
         julia --project=experiments/scalability "$SRC_DIR/train_lux_distributed.jl" > "$LOG_FILE_JL" 2>&1

    # Python Run
    LOG_FILE_PY="$LOG_DIR/slurm_python_${TOTAL_GPUS}gpu.log"
    echo "Launching Python..."
    srun --ntasks=$TOTAL_GPUS --ntasks-per-node=$GPUS_PER_NODE --gpus-per-node=$GPUS_PER_NODE \
         python3 "$SRC_DIR/train_lightning.py" --accelerator gpu --strategy ddp --gpus $GPUS_PER_NODE --nodes $NODES > "$LOG_FILE_PY" 2>&1

    echo "Jobs submitted/ran. Check logs in $LOG_DIR."
fi

echo "Done."
