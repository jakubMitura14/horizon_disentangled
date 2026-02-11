#!/bin/bash
# monitor_memory.sh: Background GPU memory logger for Slurm jobs

OUTPUT_FILE=${1:-"memory_log_${SLURM_JOB_ID}.csv"}
echo "timestamp, gpu_idx, utilization_gpu, memory_used_mb, memory_total_mb" > "$OUTPUT_FILE"

while true; do
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total \
               --format=csv,noheader >> "$OUTPUT_FILE"
    sleep 2
done
