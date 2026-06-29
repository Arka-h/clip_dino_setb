#!/bin/bash
# Blocks until GPU0 has >= THRESHOLD MiB free, then exits 0 (re-invokes the agent).
THRESHOLD=20000
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')
  if [ -n "$free" ] && [ "$free" -ge "$THRESHOLD" ]; then
    echo "GPU FREE: ${free} MiB free (>= ${THRESHOLD}). Resuming."
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>/dev/null
    exit 0
  fi
  sleep 60
done
