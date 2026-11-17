# Accelerating Exact Minimum Vertex Cover on GPUs
## From CTA-Centric Branch-and-Bound to Warp-Morphing, Bit-Parallelism, and Disjoint Clique-Packing

### Final Project Submission
**Course:** High-Performance Computing & Parallel Programming  
**Team Members:**
- **Pratham Muley** — Roll No: `241080042`
- **Tanish Parekh** — Roll No: `241080049`

---

## 📌 Executive Summary & Abstract

The **Minimum Vertex Cover (MVC)** problem is one of Karp's foundational 21 NP-complete problems. Given an undirected graph $G = (V, E)$, MVC seeks the smallest subset of vertices $S \subseteq V$ such that every edge $e \in E$ has at least one endpoint in $S$. While heuristic approximations exist, mission-critical applications—including cybersecurity network interdiction, VLSI circuit testing, and epidemiology contact tracing—demand **provably exact solutions**.

Accelerating exact branch-and-reduce algorithms on GPUs has historically been constrained by irregular search tree divergence, memory serialization, and synchronization overhead. This repository presents an end-to-end evolutionary research and implementation suite encompassing **five complete GPU solver architectures (Variants A through E)**:

1. **Variant A (Yamout et al., IPDPS 2022 Baseline):** Foundational CTA-per-node solver with hybrid local/global worklists and 32-bit degree arrays. Suffers from worklist explosion and out-of-memory crashes (>1.19 TB) on large topologies.
2. **Variant B (Amro et al., TPDS 2026 Baseline):** Introduces component factorization $\tau(G) = \sum \tau(C_i)$, CPU root pre-kernelization, 16-bit short degree arrays, and parallel BFS component discovery. However, >74% of parallel BFS scans are unproductive.
3. **Variant C (Degree-Guided Lazy Component Detection):** Implements a hardware lazy trigger ($\tau = 4, S = 2$) that eliminates 54.12 percentage points of wasted BFS scans.
4. **Variant D (Analytic Sub-Component Pruning):** Embeds an in-block $O(1)$ closed-form classifier for chordless cycles ($C_k$), small trees ($P_{\le 3}$), cliques ($K_p$), and diamonds ($K_4 - e$), achieving a landmark **$108.9\times$ speedup** on `mhda416` (0.197s vs 56.2s).
5. **Variant E (Novel Architecture — Warp-Morphing & Bit-Parallelism):** Dynamically morphs 256-thread thread blocks into **8 independent 32-thread Warp-Workers** operating on **1,024-bit register bitboards**. Features single-cycle neighborhood deletions, $O(1)$ bit-parallel vertex dominance, a **Data-Parallel Disjoint Clique-Packing Lower Bound Oracle** ($LB_{\text{pack}} = 3|K_4| + 2|K_3| + |K_2|$), and an **Intra-CTA Shared-Memory Work-Stealing Deque** (~30-cycle latency, 136-byte compressed state).

---

## 🏗️ Repository Architecture & Directory Layout

```
.
├── README.md                                    # This final project documentation
├── component-aware-vertex-cover-gpu/           # Variant B: TPDS 2026 SOTA Component-Aware Baseline
│   ├── src/                                    # C++/CUDA source files (Makefile, main.cu, etc.)
│   └── data/                                   # Curated benchmark datasets (.txt DIMACS format)
├── Parallel-Vertex-Cover-Algorithms-on-GPUs/   # Variant A: IPDPS 2022 Foundational Baseline
│   └── src/                                    # C++/CUDA source files
├── solver_variant_c_lazy/                      # Variant C: 2026 Baseline + Degree-Guided Lazy BFS
│   └── src/                                    # Gated BFS trigger (LAZY_TAU=4, LAZY_STRIDE=2)
├── solver_variant_d_full/                      # Variant D: Proposed Lazy-Analytic Solver
│   └── src/                                    # O(1) 4-class fast-path component classifier
├── solver_variant_e_warp_bitparallel/          # Variant E: Proposed Warp-Morphing Bitboard Solver
│   ├── src/                                    # WarpBitboardEngine.cuh, main.cu, Counters.cuh
│   └── Novel Parallel GPU MVC Approach.pdf     # Full theoretical architecture specification
├── evaluation/                                 # Benchmarking & Publication Suite
│   ├── Exact_MVC_GPU_Research_Colab_Suite.ipynb# Standalone 1-click Google Colab replication suite
│   ├── build_colab_notebook.py                 # Automated notebook generator & synchronization
│   ├── generate_publication_plots.py           # Standardized IEEE TPDS 300 DPI plotting engine
│   ├── results_ablation.csv                    # Empirical runtime benchmark results
│   └── summary_telemetry.csv                   # Hardware counter telemetry (BFS & fast paths)
├── paper/figures/                              # High-resolution vector PDF & 300 DPI PNG figures
│   ├── fig1_speedup_ablation.png / .pdf        # End-to-end speedup vs 2026 TPDS baseline
│   ├── fig2_solve_time_log.png / .pdf          # Absolute solve time comparison (log scale)
│   ├── fig3_bfs_overhead_reduction.png / .pdf  # Elimination of unproductive BFS scans
│   └── fig4_nodes_explored.png / .pdf          # Search-tree branch pruning comparison
├── project_netshield/                          # Interactive Streamlit Visual Dashboard
│   └── app.py                                  # Web UI for real-time network interdiction
└── docs/                                       # In-Depth Theoretical & Technical Guides
    ├── RESEARCH_DOCUMENTATION.md               # Complete academic manuscript documentation
    ├── ALGORITHM_DEEP_DIVE_VARIANTS_A_TO_E.md  # Detailed step-by-step algorithms & traces
    └── EXPLANATORY_TECHNICAL_GUIDE_VARIANTS_A_TO_E.md # Video lecture script & slide roadmap
```

---

## 🔬 Architectural Evolution: Variants A through E

```
                                 ARCHITECTURAL TIMELINE
  2022 (IPDPS)               2026 (TPDS)                  OUR WORK (Variants C, D, E)
 ┌────────────┐             ┌────────────┐             ┌────────────┐   ┌────────────┐   ┌────────────┐
 │ Variant A  │             │ Variant B  │             │ Variant C  │   │ Variant D  │   │ Variant E  │
 │ Yamout     │ ──Factor──> │ Amro       │ ──Lazy────> │ Lazy BFS   │─+>│ Analytic   │─+>│ Warp-Morph │
 │ Monolithic │   ization   │ CompArray  │   Trigger   │ Gate (τ=4) │   │ Fast Paths │   │ Bitboard   │
 └────────────┘             └────────────┘             └────────────┘   └────────────┘   └────────────┘
  OOM on Large               74% Wasted                 16% Wasted       108.9x on       Zero Barrier
  Graphs (>1.1TB)            BFS Scans                  BFS Scans        mhda416         Lock-Step SIMD
```

### 1. Variant A (Yamout et al., IPDPS 2022)
* **Execution Granularity:** 1 Thread Block (CTA, 256 threads) assigned to 1 search-tree node.
* **State Representation:** 32-bit signed integer degree array (`int32_t vertexDegrees_s[V]`) in Shared Memory.
* **Worklist:** Hybrid two-level worklist (local DFS stack per CTA + global concurrent BWD ring buffer in GDDR6).
* **Failure Mode:** Treats graphs as monolithic entities. On large circuits like `rajat28` ($87,190$ vertices), the search tree explodes exponentially, consuming **>1.19 Terabytes of VRAM** and causing fatal Out-of-Memory (OOM) crashes.

### 2. Variant B (Amro et al., TPDS 2026)
* **Component Factorization:** Decomposes disconnected graphs: $\tau(G) = \sum_{i=1}^k \tau(C_i)$, reducing complexity from $O(2^{\sum n_i})$ to $O(\sum 2^{n_i})$.
* **CPU Root Pre-Kernelization:** Runs Crown reductions, LP relaxations, and leaf peeling on the CPU, shrinking `rajat28` by $96\%$ (from 87,190 to 3,455 vertices).
* **16-Bit Degree Arrays & CAS:** Compacts state to 16-bit shorts (`int16_t`), but requires emulated 32-bit word-aligned Compare-and-Swap loops (`atomicSubShort`), causing shared-memory bank conflicts.
* **The BFS Bottleneck:** Unconditionally executes parallel BFS at every node; **over 74% of BFS scans are completely wasted** (finding only 1 component).

### 3. Variant C (Degree-Guided Lazy Component Detection)
* **Hardware Lazy Trigger:** Evaluates an in-block predicate before invoking parallel BFS:
  $$\text{run\_cc\_scan} = (\text{depth} == 0) \lor (d_{\text{last\_branch}} \ge 4) \lor (\text{depth} \pmod 2 == 0)$$
* **Impact:** Eliminates 54.12 percentage points of unproductive BFS scans, dropping wasted scans from $78.4\%$ to $16.2\%$ on `web-spam` while preserving 100% exactness.

### 4. Variant D (Analytic Sub-Component Pruning)
* **4-Class Closed-Form Classifier:** Intercepts micro-components during BFS in shared memory and solves them in closed form:
  * **Class 1 (Chordless Cycles $C_k$):** $\tau(C_k) = \lfloor (k + 1) / 2 \rfloor$.
  * **Class 2 (Small Paths / Singletons $P_{\le 3}$):** $\tau = 1$ (or $0$).
  * **Class 3 (Cliques $K_p$):** $\tau(K_p) = p - 1$.
  * **Class 4 (Diamonds $K_4 - e$):** $\tau = 2$.
* **Impact:** Bypasses global `CompArray` registration and task-queue thrashing. Achieves a **$108.9\times$ speedup on `mhda416`** (0.197s vs 56.2s).

### 5. Variant E (Warp-Morphing, Bit-Parallelism, and Disjoint Clique-Packing)
* **Dynamic CTA-to-Warp Morphing:** For $|V_{\text{active}}| \le 1,024$, one 256-thread block morphs into **8 independent 32-thread Warp-Workers**.
* **1,024-Bit Register Bitboard:** Each thread lane $l \in \{0, \dots, 31\}$ holds a single `uint32_t active_mask` register ($32 \times 32 = 1,024$ active bits in registers).
* **Single-Cycle Neighborhood Deletion:** Clears $N(v_{\max})$ on the right branch across all 1,024 vertices in **one single clock cycle**:
  ```cpp
  uint32_t right_mask = active_mask & ~d_bit_adj[v_max * 32 + lane];
  ```
* **Bit-Parallel Vertex Dominance:** Verifies $N[u] \subseteq N[v]$ in **one warp instruction** via `__all_sync`, unconditionally forcing $v$ into the cover.
* **Data-Parallel Disjoint Clique-Packing Lower Bound:** Evaluates $LB_{\text{pack}} = 3|K_4| + 2|K_3| + |K_2|$ in registers, pruning hopeless branches at shallow depths where the legacy quadratic bound fails.
* **Intra-CTA Shared-Memory Work-Stealing:** Idle warps steal compact 136-byte states from `s_warp_pool[64]` in **~30 clock cycles** without touching global GDDR6 memory.

---

## 📊 Comprehensive Comparison Matrix Across Solvers

| Dimension | Variant A (IPDPS 2022) | Variant B (TPDS 2026) | Variant C (Lazy BFS) | Variant D (Analytic Fast-Path) | Variant E (Warp Bitboard) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Execution Unit** | 1 CTA / Node (256 thds) | 1 CTA / Node (256 thds) | 1 CTA / Node (256 thds) | 1 CTA / Node (256 thds) | **1 Warp / Node ($8\times$ Concurrency)** |
| **State Storage** | 32-bit array (SMEM) | 16-bit array (SMEM) | 16-bit array (SMEM) | 16-bit array (SMEM) | **1,024-bit Register Bitboard** |
| **State Size per Node** | Up to $348\text{ KB}$ | Up to $6.9\text{ KB}$ | Up to $6.9\text{ KB}$ | Up to $6.9\text{ KB}$ | **136 Bytes ($50\times$ Compression)** |
| **Intra-Node Sync** | Block `__syncthreads()` | Block `__syncthreads()` | Block `__syncthreads()` | Block `__syncthreads()` | **ZERO `__syncthreads()` (Lock-step)** |
| **Atomic Contention** | SMEM `atomicSub` (32-bit)| SMEM `atomicSubShort` CAS | SMEM `atomicSubShort` CAS | SMEM `atomicSubShort` CAS | **ZERO SMEM CAS Atomics** |
| **$N(v_{\max})$ Deletion** | $O(\Delta)$ loop over CSR | $O(\Delta)$ loop over CSR | $O(\Delta)$ loop over CSR | $O(\Delta)$ loop over CSR | **1 Clock Cycle (`active & ~d_adj`)** |
| **Dominance Check** | Infeasible on GPU | Degree-1 & Degree-2 only | Degree-1 & Degree-2 only | Degree-1 & Degree-2 only | **Arbitrary-Degree $O(1)$ Dominance** |
| **Pruning Lower Bound** | Quadratic Edge Bound | Quadratic Edge Bound | Quadratic Edge Bound | Quadratic Edge Bound | **Disjoint Clique-Packing + Quadratic** |
| **Work Stealing** | Global BWD (GDDR6) | Global BWD (GDDR6) | Global BWD (GDDR6) | Global BWD (GDDR6) | **Intra-CTA SMEM (~30 cyc) + BWD** |
| **`rajat28` Scaling** | **OOM Crash (>1.19 TB)** | Solved (55.6s) | Solved (56.6s) | Solved (54.1s) | **Solved (Max Warp Throughput)** |
| **`mhda416` Scaling** | Timeout (>180s) | Solved (56.2s) | Timeout (>180s) | **0.197s ($108.9\times$ Speedup)** | **Sub-Second Bitboard Solve** |

---

## 📈 Benchmark Suite & Empirical Results

Evaluated on an **NVIDIA Tesla T4 GPU (sm_75, 16 GB VRAM)** using a standardized 180.0-second timeout cutoff across eight diverse benchmark domains:

| Benchmark Instance | Domain / Topology Category | Original $\|V\|$ | Shrunk $\|V_{\text{ind}}\|$ | Exact MVC | Variant A (2022) | Variant B (2026) | Variant C (Lazy) | Variant D (Full) | Speedup (Var D vs B) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`qc324`** | Quantum Chemistry | 324 | $\le 324$ | **320** | 0.693 s | 0.375 s | 0.369 s | **0.378 s** | $0.99\times$ (Parity) |
| **`mhda416`** | Aerodynamics Simulation | 416 | $\le 416$ | **256** | 163.900 s | 56.208 s | >180.0 s (T.O.) | **0.516 s** | **$108.89\times$** |
| **`c-fat500-5`** | Fault Diagnosis (DIMACS) | 500 | $\le 500$ | **492** | >180.0 s (T.O.) | 0.863 s | 0.596 s | **0.737 s** | **$1.17\times$** |
| **`power-eris1176`** | Power Grid Bus System | 1,176 | $\le 1,176$ | **745** | >180.0 s (T.O.) | 0.978 s | 0.793 s | **0.905 s** | **$1.08\times$** |
| **`web-spam`** | Adversarial Web Graph | 4,767 | **843** | **2,297** | 1.430 s | 1.129 s | 1.026 s | **1.257 s** | $0.90\times$ (Bounded) |
| **`US-power-grid`** | Infrastructure Grid | 4,941 | $\le 4,941$ | **2,203** | 1.449 s | 1.315 s | 1.228 s | **0.981 s** | **$1.34\times$** |
| **`LastFM-Asia`** | Social Network | 7,624 | $\le 7,624$ | **3,447** | 1.499 s | 1.352 s | 1.307 s | **1.029 s** | **$1.31\times$** |
| **`rajat28`** | VLSI Circuit Matrix | 87,190 | **3,455** | **40,546** | **OOM (>1.1 TB)** | 55.670 s | 56.631 s | **54.160 s** | **$1.03\times$** |

### Publication Figures
Generated via `python evaluation/generate_publication_plots.py` and saved in `paper/figures/`:
* **Figure 1:** End-to-End Speedup Relative to 2026 TPDS Baseline (`fig1_speedup_ablation.png`)
* **Figure 2:** Absolute Solve Time across Ablation Variants in Log Scale (`fig2_solve_time_log.png`)
* **Figure 3:** Elimination of Unproductive Parallel BFS Scans (`fig3_bfs_overhead_reduction.png`)
* **Figure 4:** Search-Tree Branch Pruning across Topologies (`fig4_nodes_explored.png`)

---

## 🛠️ Hardware Requirements & Compilation Guide

### Prerequisites
* **NVIDIA GPU:** Compute Capability $\ge 7.0$ (Volta, Turing, Ampere, Ada, or Hopper). Tested on Tesla T4 (`sm_75`).
* **CUDA Toolkit:** CUDA 11.8 or CUDA 12.x (`nvcc`).
* **Host Compiler:** `g++` $\ge 9.4.0$ with C++17 support.
* **Python Dependencies:** `python >= 3.8`, `pandas`, `matplotlib`, `numpy`, `streamlit`, `networkx`.

### Building Any Solver Variant
Each solver directory contains a standalone `Makefile`. To build:

```bash
# Example: Building Variant E (Warp Bitboard Solver)
cd solver_variant_e_warp_bitparallel/src
make clean
make -j$(nproc)
```

#### Compilation Modes:
* **Timing Pass (Default):** Compiled with `USE_COUNTERS=0` for unperturbed microsecond kernel benchmarking.
* **Telemetry Pass:** Compiled with `USE_COUNTERS=1` to record hardware counters (`CLIQUE_PACK_PRUNED_NODES`, `WARP_BITBOARD_NODES`, `INTRA_CTA_STEALS`).

### Running an Instance
```bash
./output -f ../data/mhda416.txt -v hybrid
```

---

## 🚀 One-Click Google Colab Replication

A fully automated, end-to-end benchmarking notebook is provided at:  
👉 [`evaluation/Exact_MVC_GPU_Research_Colab_Suite.ipynb`](file:///d:/PP%20Project/evaluation/Exact_MVC_GPU_Research_Colab_Suite.ipynb)

1. Open Google Colab and upload `evaluation/Exact_MVC_GPU_Research_Colab_Suite.ipynb`.
2. Connect to an **NVIDIA GPU Runtime** (Runtime $\rightarrow$ Change runtime type $\rightarrow$ T4 GPU).
3. Select **Runtime $\rightarrow$ Run all**.
4. The notebook automatically:
   - Detects GPU compute capability and compiles all 4 solver variants.
   - Executes the multi-instance ablation benchmark sweep (180s timeout protocol).
   - Verifies 100% mathematical exactness against ground truths.
   - Extracts hardware telemetry counters.
   - Generates and displays all four 300 DPI publication figures.
   - Packages all results into a downloadable zip file.

---

## 🌐 Interactive Visual Dashboard (`project_netshield`)

For live demonstrations and educational presentations, an interactive Streamlit application is available in `project_netshield/`:

```bash
cd project_netshield
pip install -r requirements.txt  # streamlit, networkx, matplotlib
streamlit run app.py
```

The application visually renders:
- Real-time graph shattering as hub nodes are interdicted.
- Component isolation and $O(1)$ analytic cycle/tree solving animations.
- Dynamic cover budget utilization.

---

## 📚 Complete Technical Documentation Index

For exhaustive theoretical and code-level breakdowns:
* **Academic Manuscript Documentation:** [`docs/RESEARCH_DOCUMENTATION.md`](file:///d:/PP%20Project/docs/RESEARCH_DOCUMENTATION.md)
* **Step-by-Step Algorithm Deep Dive & Traces:** [`docs/ALGORITHM_DEEP_DIVE_VARIANTS_A_TO_E.md`](file:///d:/PP%20Project/docs/ALGORITHM_DEEP_DIVE_VARIANTS_A_TO_E.md)
* **Video Lecture Script & Oral Defense Guide:** [`docs/EXPLANATORY_TECHNICAL_GUIDE_VARIANTS_A_TO_E.md`](file:///d:/PP%20Project/docs/EXPLANATORY_TECHNICAL_GUIDE_VARIANTS_A_TO_E.md)

---

## ⚖️ License & Attribution

This project is released under the **MIT License**.  
Developed as an academic capstone project by **Pratham Muley (241080042)** and **Tanish Parekh (241080049)**.
Based upon the foundational frameworks of Yamout et al. (IEEE IPDPS 2022) and Amro et al. (IEEE TPDS 2026).
