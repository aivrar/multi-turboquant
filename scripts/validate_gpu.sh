#!/bin/bash
# GPU Validation Script for Multi-TurboQuant
# Run inside WSL2 on a machine with NVIDIA GPUs
#
# Usage:
#   cd /mnt/e/Multi_TurboQuant
#   bash scripts/validate_gpu.sh [model_path] [context]
#
# Requirements:
#   - llama.cpp built with CUDA+FA (from app setup)
#   - A GGUF model file
#   - nvidia-smi available

set -e

MODEL="${1:-/opt/models/huihui-ai_Huihui-gemma-3n-E4B-it-abliterated-Q4_K_M.gguf}"
CONTEXT="${2:-4096}"
MAX_TOKENS=128
PROMPT="Write a detailed analysis of the benefits of KV cache compression for large language model inference."
SERVER="/opt/llama.cpp/build/bin/llama-server"
PORT=8190  # Use a test port that doesn't conflict

echo "================================================================="
echo "  MULTI-TURBOQUANT GPU VALIDATION"
echo "================================================================="
echo ""
echo "Model:    $MODEL"
echo "Context:  $CONTEXT"
echo "Server:   $SERVER"
echo ""

# Check prerequisites
if [ ! -f "$SERVER" ]; then
    echo "ERROR: llama-server not found at $SERVER"
    echo "Build it first: cd /opt/llama.cpp && cmake -B build ... && cmake --build build ..."
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model not found at $MODEL"
    exit 1
fi

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || {
    echo "ERROR: nvidia-smi not found"
    exit 1
}
echo ""

# Method configurations to test
# Format: "name cache_type_k cache_type_v"
CONFIGS=(
    "fp16_baseline f16 f16"
    "turbo3_sym turbo3 turbo3"
    "turbo3_tcq_sym turbo3_tcq turbo3_tcq"
    "turbo4_sym turbo4 turbo4"
    "iso3_k_only iso3 f16"
    "planar3_k_only planar3 f16"
    "iso3_sym iso3 iso3"
    "planar3_sym planar3 planar3"
)

# Results file
RESULTS="validation_results_$(date +%Y%m%d_%H%M%S).json"
echo "[" > "$RESULTS"
FIRST=true

for config_line in "${CONFIGS[@]}"; do
    read -r name k_type v_type <<< "$config_line"

    echo "─────────────────────────────────────────────────"
    echo "Testing: $name (K=$k_type, V=$v_type)"
    echo "─────────────────────────────────────────────────"

    # Start server
    $SERVER \
        --model "$MODEL" \
        --host 0.0.0.0 --port $PORT \
        -c "$CONTEXT" -ngl 99 -fa on \
        --cache-type-k "$k_type" --cache-type-v "$v_type" \
        > /tmp/mtq_server.log 2>&1 &
    SERVER_PID=$!

    # Wait for server to be ready
    echo -n "  Starting server..."
    for i in $(seq 1 30); do
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo " ready (${i}s)"
            break
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo " CRASHED"
            echo "  Server log:"
            tail -5 /tmp/mtq_server.log
            break 2
        fi
        sleep 1
    done

    if ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "  TIMEOUT - server didn't start"
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
        continue
    fi

    # Get VRAM usage
    VRAM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')

    # Run 3 inference passes
    BEST_DECODE=0
    BEST_PREFILL=0
    for run in 1 2 3; do
        RESPONSE=$(curl -s "http://localhost:$PORT/v1/completions" \
            -H "Content-Type: application/json" \
            -d "{\"prompt\": \"$PROMPT\", \"max_tokens\": $MAX_TOKENS, \"temperature\": 0.1}" \
            2>/dev/null)

        # Extract timing from response
        DECODE=$(echo "$RESPONSE" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    u = d.get('usage', {})
    ct = u.get('completion_tokens', 0)
    # Try to get timing from response
    if 'timings' in d:
        print(f\"{d['timings'].get('predicted_per_second', 0):.1f}\")
    elif ct > 0:
        print(f\"{ct}\")
    else:
        print('0')
except:
    print('0')
" 2>/dev/null || echo "0")

        echo "  Run $run: decode=$DECODE"
    done

    # Stop server
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    sleep 1

    # Record result
    if [ "$FIRST" = true ]; then
        FIRST=false
    else
        echo "," >> "$RESULTS"
    fi
    cat >> "$RESULTS" << ENTRY
  {"name": "$name", "k_type": "$k_type", "v_type": "$v_type", "vram_mb": $VRAM_USED, "context": $CONTEXT}
ENTRY

    echo "  VRAM: ${VRAM_USED} MB"
    echo ""
done

echo "]" >> "$RESULTS"

echo "================================================================="
echo "  VALIDATION COMPLETE"
echo "  Results: $RESULTS"
echo "================================================================="

# Kill any remaining server
fuser -k $PORT/tcp 2>/dev/null || true
