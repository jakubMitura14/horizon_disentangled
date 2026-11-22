#!/usr/bin/env bash
# ==============================================================================
# Docker Runner for 48h+ Stability & Convergence Training
# Runs container with `--restart unless-stopped` on NVIDIA H100 GPU
# ==============================================================================

CONTAINER_NAME="horizon_stability_trainer"
IMAGE_NAME="ubuntu:22.04"
WORKSPACE="/DATA/horizon_disentangled"
CONDA_PY="/DATA/conda/envs/nukl-to-ct/bin/python"
HOURS="${2:-48}"

ACTION="${1:-status}"

case "$ACTION" in
    start)
        echo "=== Starting $CONTAINER_NAME in Docker (restart: unless-stopped) ==="
        # Remove any existing container with same name if stopped
        docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

        docker run -d \
            --name "$CONTAINER_NAME" \
            --restart unless-stopped \
            --gpus all \
            --ipc=host \
            -u 1000:1000 \
            -v /etc/passwd:/etc/passwd:ro \
            -v /etc/group:/etc/group:ro \
            -v /DATA:/DATA \
            -w "$WORKSPACE" \
            -e LD_LIBRARY_PATH=/DATA/conda/envs/nukl-to-ct/lib:\$LD_LIBRARY_PATH \
            -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            -e PYTHONUNBUFFERED=1 \
            "$IMAGE_NAME" \
            $CONDA_PY scripts/training/stability_48h.py \
                --hours "$HOURS" \
                --checkpoint-interval 50 \
                --snapshot-interval 50 \
                --batch-b 36 \
                --batch-c 24 \
                --data-dir "$WORKSPACE/data/synthetic_stability" \
                --ckpt-dir "experiments/stability_checkpoints" \
                --resume

        echo "Container $CONTAINER_NAME started successfully."
        docker ps --filter "name=$CONTAINER_NAME"
        ;;

    stop)
        echo "=== Stopping $CONTAINER_NAME gracefully (sending SIGTERM for checkpoint save) ==="
        # Give 45 seconds for current epoch to finish and save complete checkpoint
        docker stop -t 45 "$CONTAINER_NAME"
        echo "Container $CONTAINER_NAME stopped cleanly."
        ;;

    restart)
        echo "=== Gracefully restarting $CONTAINER_NAME ==="
        docker stop -t 45 "$CONTAINER_NAME" 2>/dev/null || true
        $0 start "$HOURS"
        ;;

    extend)
        NEW_HOURS="${2:-72}"
        echo "=== Extending run duration to $NEW_HOURS hours ==="
        echo "Stopping current container gracefully to commit checkpoint..."
        docker stop -t 45 "$CONTAINER_NAME" 2>/dev/null || true
        echo "Starting container with target duration: $NEW_HOURS hours..."
        $0 start "$NEW_HOURS"
        ;;

    status)
        echo "=== Container Status ==="
        docker ps -a --filter "name=$CONTAINER_NAME"
        echo ""
        echo "=== GPU Memory Status ==="
        nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader 2>/dev/null || true
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free --format=csv,noheader 2>/dev/null || true
        echo ""
        echo "=== Latest Stability Log (tail -n 10) ==="
        tail -n 10 "$WORKSPACE/experiments/stability_48h.log" 2>/dev/null || true
        echo ""
        echo "=== Latest Checkpoints ==="
        ls -lh "$WORKSPACE/experiments/stability_checkpoints/" 2>/dev/null || true
        ;;

    logs)
        tail -f "$WORKSPACE/experiments/stability_48h.log"
        ;;

    *)
        echo "Usage: $0 {start [hours]|stop|restart [hours]|extend <new_hours>|status|logs}"
        exit 1
        ;;
esac
