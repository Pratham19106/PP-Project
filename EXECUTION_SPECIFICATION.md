# AI Agent Execution & Implementation Specification: Optimizing Component-Aware GPU Branch-and-Bound for Exact Minimum Vertex Cover

**Target Research Title:** Accelerating Component-Aware GPU Branch-and-Bound for Minimum Vertex Cover via Lazy Splitting and Analytic Sub-Component Pruning  
**Execution Domain:** Parallel Processing & GPU Computing (CUDA / C++)  
**Target Application Domain:** Network Interdiction & Cybersecurity (Critical Node & Botnet Disruption in Scale-Free Topologies)  

---

## Module 1: System Role, Problem Formulation & Reference Resources

### 1.1 Agent Mission & High-Level Abstract
Implement, verify, benchmark, and demonstrate two in-block CUDA kernel optimizations (~100–150 lines of C++/CUDA code) for solving the exact Minimum Vertex Cover (MVC) problem on GPUs:
- **Bottleneck A (Wasted Component Scans):** Expensive parallel BFS/Label Propagation at every search-tree node even when connected.
- **Bottleneck B (Registry Contention on Trivial Components):** Shattered trees and simple cycles pushed to global Component Branch Registry causing atomic lock contention and redundant search-tree nodes.
- **Addition 1 (Lazily Triggered Component Detection):** Gated BFS behind shared-memory heuristic: $\text{deg}(v) \ge \tau$ or $d \pmod S == 0$.
- **Addition 2 (Analytic Tree & Cycle Fast-Path):** In-block exact $O(1)$ cycle formula ($\lceil n_i / 2 \rceil$) and $O(n_i)$ tree leaf peeling in shared memory.

### 1.2 Repositories, Papers & Datasets
1. **2026 SOTA GitHub Repo (Amro et al. — Target):**  
   `https://github.com/HusseinAmro/component-aware-vertex-cover-gpu`
2. **2022 Baseline GitHub Repo (Yamout et al. — Comparison):**  
   `https://github.com/pky00/Parallel-Vertex-Cover-Algorithms-on-GPUs`
3. **2026 Paper & Thesis:** Hussein Amro et al., *"Faster Vertex Cover Algorithms on GPUs with Component-Aware Parallel Branching,"* IEEE TPDS, Feb 2026.
4. **2022 Base Paper:** Yamout et al., *"Parallel Vertex Cover Algorithms on GPUs,"* IEEE IPDPS 2022.
5. **Benchmarks:** Built-in `data/` graphs, DIMACS, SNAP (e.g. `ca-AstroPh`), Network Repository.

---

## Module 2: Architectural Context & Baseline Analysis

### 2.1 2022 Baseline (Yamout et al.)
- Branch-and-Reduce paradigm, shared memory local stack, global worklist with CUDA atomics.
- Flaw: Multiplicative complexity $O(b^{dk})$ on $k$ disconnected components.

### 2.2 2026 Baseline (Amro et al.)
- Additive decomposition: $\text{MVC}(G) = \sum_{i=1}^k \text{MVC}(C_i) \implies O(k \cdot b^d)$.
- Component Branch Registry for non-tail recursion (AND-nodes).
- Root graph compression.

---

## Module 3: Detailed Implementation Specification

### 3.1 Global Implementation Guardrails
- Build directly on `HusseinAmro/component-aware-vertex-cover-gpu`.
- Prevent warp divergence: Thread 0 writes `__shared__` flags following block-wide reductions and calls `__syncthreads()`.
- Preserve exactness: 100% agreement on MVC size with baseline.

### 3.2 Addition 1: Lazily Triggered Component Detection
```cpp
#ifndef LAZY_TAU
#define LAZY_TAU 4        // Hub-removal degree threshold (tau)
#endif
#ifndef LAZY_STRIDE
#define LAZY_STRIDE 2     // Search-tree depth stride (S)
#endif

// In __shared__ memory:
__shared__ bool run_cc_scan;
if (threadIdx.x == 0) {
    run_cc_scan = (current_depth == 0) || 
                  (last_branched_degree >= LAZY_TAU) || 
                  (current_depth % LAZY_STRIDE == 0);
}
__syncthreads();
```
- If `run_cc_scan == true`: execute BFS scan. (Telemetry: count total and unproductive scans).
- If `run_cc_scan == false`: bypass BFS, treat active vertices as $k=1$, continue standard branching.

### 3.3 Addition 2: Analytic Tree & Cycle Fast-Path
```cpp
__shared__ int s_n_i;
__shared__ int s_degree_sum_i;
__shared__ int s_max_deg_i;
__shared__ int s_fast_path_type; // 0 = None, 1 = Cycle, 2 = Tree

if (threadIdx.x == 0) {
    s_n_i = 0;
    s_degree_sum_i = 0;
    s_max_deg_i = 0;
    s_fast_path_type = 0;
}
__syncthreads();

// Parallel reduction over vertices in C_i
int local_n = 0;
int local_deg_sum = 0;
int local_max_deg = 0;

for (int v = threadIdx.x; v < num_vertices; v += blockDim.x) {
    if (component_id[v] == current_comp_label && deg[v] > 0) {
        int d = deg[v];
        local_n += 1;
        local_deg_sum += d;
        if (d > local_max_deg) local_max_deg = d;
    }
}

atomicAdd(&s_n_i, local_n);
atomicAdd(&s_degree_sum_i, local_deg_sum);
atomicMax(&s_max_deg_i, local_max_deg);
__syncthreads();

if (threadIdx.x == 0) {
    int m_i = s_degree_sum_i / 2; // undirected edges
    if (s_n_i >= 3 && m_i == s_n_i && s_max_deg_i == 2) {
        s_fast_path_type = 1; // Simple Cycle
    } else if (s_n_i >= 1 && m_i == s_n_i - 1) {
        s_fast_path_type = 2; // Acyclic Tree / Star
    }
}
__syncthreads();
```

#### Cycle Fast-Path ($m_i == n_i \land \Delta_i == 2$):
- $\text{MVC}(C_i) = \lceil n_i / 2 \rceil = (n_i + 1) / 2$.
- Thread 0 adds to `local_cover_size`.
- Block zeroes out degrees of all $v \in C_i$.
- Telemetry: increment `cycle_fast_path_hits`. Bypass registry.

#### Tree Fast-Path ($m_i == n_i - 1$):
- Fast-path for $K_2$ or star ($n_i == 2$ or $\Delta_i == n_i - 1$): cover size + 1.
- General tree: race-free iterative leaf peeling:
  - Thread 0 selects degree-1 leaf.
  - Parallel scan finds active unique parent $p$.
  - All threads decrement degrees of neighbors of $p$.
  - Thread 0 covers $p$ (`deg[p] = 0`, `local_cover_size += 1`).
  - Repeat until no edges remain.

#### Component Registry Count After Fast-Path:
- $k_{\text{rem}} = k - k_{\text{fast}}$:
  - If $k_{\text{rem}} == 0$: all solved analytically, terminate branch.
  - If $k_{\text{rem}} == 1$: continue standard OR-branching on single component.
  - If $k_{\text{rem}} \ge 2$: register only the $k_{\text{rem}}$ non-trivial components.

---

## Module 4: Compilation, Telemetry & Testing Protocol
- Two-pass compilation: Timing pass vs. Telemetry pass (`make USE_COUNTERS=1`).
- Custom counters: `unproductive_bfs_scans`, `cycle_fast_path_hits`, `tree_fast_path_hits`, `registry_atomic_writes`.
- 4 Ablation Variants:
  - **Variant A:** 2022 Yamout Baseline
  - **Variant B:** 2026 Amro Baseline
  - **Variant C:** 2026 + Addition 1 (Lazy Trigger)
  - **Variant D:** 2026 + Addition 1 + Addition 2 (Full Solver)
- Verification: 100% exact cover size match across all test instances.

---

## Module 5: Real-World Application & "Project NetShield"
- Domain: P2P Botnet Disruption & Network Interdiction.
- Interactive Dashboard (Streamlit / Plotly):
  - Stage 1: Load 500-node malicious topology.
  - Stage 2: Side-by-side solver execution.
  - Stage 3: Visual hub isolation & shattering.
  - Stage 4: Analytic fast-path callout (green clusters).
  - Stage 5: 100% link interception & telemetry comparison table.

