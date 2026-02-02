#!/bin/bash

# EuroHPC JUPITER Scalability Test Launcher
# =========================================
# Runs distributed training simulations using OpenMPI on Localhost.

EXPERIMENT_DIR=$(dirname "$0")
SRC_DIR="$EXPERIMENT_DIR/src"
LOG_DIR="$EXPERIMENT_DIR/logs"
RESULTS_FILE="$EXPERIMENT_DIR/results_scaling.csv"

mkdir -p "$LOG_DIR"

# Initialize results file
echo "Framework,Num_Procs,Epoch_Time_s" > "$RESULTS_FILE"

# Define Scaling Steps (Processes)
STEPS=(1 2 4)

echo "Starting Distributed Scalability Experiments (Localhost Simulation)..."
echo "===================================================================="

export OMPI_MCA_rmaps_base_oversubscribe=1
export OMPI_MCA_btl_vader_single_copy_mechanism=none

# --- Julia (Lux + MPI) ---
echo "--- Benchmarking Julia (Lux + MPI) ---"
for NP in "${STEPS[@]}"; do
    echo "Running Julia with $NP Processes..."
    LOG_FILE="$LOG_DIR/julia_lux_${NP}proc.log"

    # Run using mpiexecjl
    mpiexecjl --project=experiments/scalability -n $NP julia --project=experiments/scalability "$SRC_DIR/train_lux_distributed.jl" > "$LOG_FILE" 2>&1

    if [ $? -eq 0 ]; then
        # Log format: "Epoch 2: Loss 70223.2891 | Time 1.6518s | Check Sum: 6.0"
        # We want the time value "1.6518".
        # awk '{print $7}' gives "1.6518s". sed 's/s//' removes the trailing 's'.
        # We assume the line contains "Epoch 2:".

        TIME=$(grep "Epoch 2" "$LOG_FILE" | head -n 1 | awk '{print $7}' | sed 's/s//')

        # Verify if TIME is extracted
        if [ -z "$TIME" ]; then
             echo "  Error parsing time. Check log."
             TIME="N/A"
        fi

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

    # Run directly with python (using DDP spawn or launch via torchrun?)
    # PyTorch Lightning handles DDP internally if we specify accelerator='cpu', strategy='ddp', devices=NP
    # But DDP on CPU with 'devices>1' spawns processes.

    python3 "$SRC_DIR/train_lightning.py" --accelerator cpu --strategy ddp --num_processes $NP > "$LOG_FILE" 2>&1

    if [ $? -eq 0 ]; then
        # Parse output "Training finished in X.XX seconds."
        TOTAL_TIME=$(grep "Training finished in" "$LOG_FILE" | awk '{print $4}')
        # Approx epoch time = Total / 2
        EPOCH_TIME=$(echo "$TOTAL_TIME / 2" | bc -l)
        echo "  Time (Approx/Epoch): ${EPOCH_TIME}s"
        echo "Python,$NP,$EPOCH_TIME" >> "$RESULTS_FILE"
    else
        echo "  Failed. Check log: $LOG_FILE"
    fi
done

echo "Experiments Complete. Results saved to $RESULTS_FILE"
