#!/usr/bin/env bash
# ==============================================================================
# Automated 4-Variant Ablation Benchmark Runner for Exact Minimum Vertex Cover
# Evaluates:
#   Variant A: 2022 Yamout Baseline (IPDPS)
#   Variant B: 2026 Amro Baseline (IEEE TPDS)
#   Variant C: 2026 + Addition 1 (Lazy BFS Gating)
#   Variant D: 2026 + Addition 1 + Addition 2 (Lazy Gating + Tree/Cycle Fast-Path)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

RESULTS_CSV="${SCRIPT_DIR}/results_ablation.csv"
DATA_DIR="${ROOT_DIR}/component-aware-vertex-cover-gpu/data"

echo "=============================================================================="
echo " Starting Exact Minimum Vertex Cover GPU 4-Variant Ablation Suite"
echo " Working Directory: ${ROOT_DIR}"
echo " Results Output:    ${RESULTS_CSV}"
echo "=============================================================================="

# Detect GPU Architecture for NVCC
GPU_ARCH="sm_70"
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | head -n 1)
    echo "Detected GPU: ${GPU_NAME}"
    if [[ "${GPU_NAME}" == *"T4"* ]]; then
        GPU_ARCH="sm_75"
    elif [[ "${GPU_NAME}" == *"V100"* ]]; then
        GPU_ARCH="sm_70"
    elif [[ "${GPU_NAME}" == *"A100"* ]]; then
        GPU_ARCH="sm_80"
    elif [[ "${GPU_NAME}" == *"RTX 30"* ]] || [[ "${GPU_NAME}" == *"A10"* ]]; then
        GPU_ARCH="sm_86"
    elif [[ "${GPU_NAME}" == *"H100"* ]]; then
        GPU_ARCH="sm_90"
    fi
fi
echo "Using CUDA Compute Target: ${GPU_ARCH}"

# List of Benchmark Graphs (Category 1: Scale-Free / Sparse; Category 2: Dense / Cliques)
BENCHMARKS=(
    "qc324.txt"
    "mhda416.txt"
    "c-fat500-5.txt"
    "US-power-grid.txt"
    "power-eris1176.txt"
    "LastFM-Asia.txt"
    "web-spam.txt"
    "web-webbase-2001.txt"
)

# Header for Results CSV
echo "Graph,Variant,MVC_Size,Status,Time_Sec,Nodes_Explored,Total_BFS,Unproductive_BFS_Pct,FastPath_Hits" > "${RESULTS_CSV}"

# Function to compile a variant
compile_variant() {
    local variant_name=$1
    local variant_dir=$2
    local use_counters=$3

    echo "Compiling ${variant_name} in ${variant_dir} (USE_COUNTERS=${use_counters})..."
    cd "${variant_dir}/src"
    make clean > /dev/null 2>&1 || true
    # Update Makefile compute arch
    sed -i "s/sm_[0-9][0-9]/${GPU_ARCH}/g" Makefile
    sed -i "s/compute_[0-9][0-9]/compute_${GPU_ARCH:3:2}/g" Makefile
    make USE_COUNTERS=${use_counters} -j$(nproc) > /dev/null
    cd "${ROOT_DIR}"
}

VARIANTS=(
    "Variant_A_2022:${ROOT_DIR}/Parallel-Vertex-Cover-Algorithms-on-GPUs"
    "Variant_B_2026:${ROOT_DIR}/component-aware-vertex-cover-gpu"
    "Variant_C_Lazy:${ROOT_DIR}/solver_variant_c_lazy"
    "Variant_D_Full:${ROOT_DIR}/solver_variant_d_full"
)

# 1. TIMING PASS (Accurate Wall-Clock Execution without atomic counter overhead)
echo ""
echo ">>> EXECUTING PASS 1: TIMING PASS (USE_COUNTERS=0) <<<"
for V_INFO in "${VARIANTS[@]}"; do
    V_NAME="${V_INFO%%:*}"
    V_DIR="${V_INFO##*:}"
    
    compile_variant "${V_NAME}" "${V_DIR}" 0

    for G_NAME in "${BENCHMARKS[@]}"; do
        G_PATH="${DATA_DIR}/${G_NAME}"
        if [ ! -f "${G_PATH}" ]; then
            echo "Skipping ${G_NAME} (not found)"
            continue
        fi

        echo "  Running ${V_NAME} on ${G_NAME}..."
        START_TIME=$(date +%s%N)
        OUTPUT=$("${V_DIR}/src/output" -f "${G_PATH}" -v HYBRID || true)
        END_TIME=$(date +%s%N)

        ELAPSED_SEC=$(echo "scale=4; (${END_TIME} - ${START_TIME}) / 1000000000" | bc)
        MVC_VAL=$(echo "${OUTPUT}" | grep -i "Minimum is:" | head -n 1 | awk '{print $NF}')
        if [ -z "${MVC_VAL}" ]; then
            MVC_VAL=$(echo "${OUTPUT}" | grep -i "minimum" | tail -n 1 | awk '{print $NF}')
        fi
        MVC_VAL=${MVC_VAL:-"0"}

        echo "${G_NAME},${V_NAME},${MVC_VAL},SUCCESS,${ELAPSED_SEC},N/A,N/A,N/A,N/A" >> "${RESULTS_CSV}"
    done
done

echo ""
echo "Timing pass complete. Results logged to ${RESULTS_CSV}."
python3 "${SCRIPT_DIR}/verify_correctness.py"

