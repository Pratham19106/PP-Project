#ifndef WARP_BITBOARD_ENGINE_CUH
#define WARP_BITBOARD_ENGINE_CUH

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "Counters.cuh"
#include "compArray.cuh"
#include "CSRGraphRep.cuh"

#ifndef BITBOARD_MAX_V
#define BITBOARD_MAX_V 1024
#endif

#define BITBOARD_WORDS (BITBOARD_MAX_V / 32)
#define FULL_MASK 0xFFFFFFFFu
#define CTA_POOL_CAP 64
#define WARP_STACK_CAP 64

#ifndef LAZY_TAU
#define LAZY_TAU 4
#endif

#ifndef LAZY_STRIDE
#define LAZY_STRIDE 2
#endif

// -----------------------------------------------------------------------------
// 1. Warp-Level Reduction and Collective Helpers
// -----------------------------------------------------------------------------
__device__ __forceinline__ int warpReduceSum(int val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(FULL_MASK, val, offset);
    }
    return val;
}

__device__ __forceinline__ void warpReduceMaxDegree(
    int local_max_deg, int local_best_v, int &out_max_deg, int &out_best_v)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        int other_deg = __shfl_xor_sync(FULL_MASK, local_max_deg, offset);
        int other_v   = __shfl_xor_sync(FULL_MASK, local_best_v, offset);
        if (other_deg > local_max_deg || (other_deg == local_max_deg && other_v < local_best_v)) {
            local_max_deg = other_deg;
            local_best_v  = other_v;
        }
    }
    out_max_deg = local_max_deg;
    out_best_v  = local_best_v;
}

__device__ __forceinline__ int warpReduceMin(int val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        int other = __shfl_xor_sync(FULL_MASK, val, offset);
        if (other < val) val = other;
    }
    return val;
}

__device__ __forceinline__ int warpReduceMax(int val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        int other = __shfl_xor_sync(FULL_MASK, val, offset);
        if (other > val) val = other;
    }
    return val;
}

// -----------------------------------------------------------------------------
// 2. Compact 136-Byte Warp Bitboard Search-Tree Node State
// -----------------------------------------------------------------------------
struct __align__(16) BitboardNodeState {
    uint32_t active_mask[32]; // 1024 bits = 128 bytes (lane i owns word i)
    int16_t  numDeleted;      // Current cover size |S|
    int16_t  depth;           // Search tree depth
    int32_t  compIndex;       // Index in CompArray (-1 if root/monolithic)
};

// -----------------------------------------------------------------------------
// 3. Data-Parallel Disjoint Clique-Packing Lower-Bound Oracle (Warp Level)
// Packs vertex-disjoint triangles (K_3, +2) and edges (K_2, +1) in registers.
// Prunes immediately when numDeleted + lb_pack >= best_bound.
// -----------------------------------------------------------------------------
__device__ __forceinline__ bool computeCliquePackingLB_Warp(
    uint32_t my_mask,
    int numDeleted,
    int best_bound,
    const uint32_t* __restrict__ d_bit_adj,
    int lane,
    Counters* blockCounters)
{
    if (numDeleted >= best_bound) return true;

    uint32_t rem_mask = my_mask;
    int lb_pack = 0;

    while (__any_sync(FULL_MASK, rem_mask != 0)) {
        uint32_t rem_ballot = __ballot_sync(FULL_MASK, rem_mask != 0);
        if (rem_ballot == 0) break;

        // Select first unpacked vertex u
        int u_lane = __ffs(rem_ballot) - 1;
        uint32_t u_word = __shfl_sync(FULL_MASK, rem_mask, u_lane);
        int u_bit = __ffs(u_word) - 1;
        int u = (u_lane << 5) + u_bit;

        // Clear u from rem_mask on u_lane
        if (lane == u_lane) {
            rem_mask &= ~(1u << u_bit);
        }

        // Load u's unpacked neighborhood across the warp
        uint32_t nu_rem = d_bit_adj[u * 32 + lane] & rem_mask;
        uint32_t nu_ballot = __ballot_sync(FULL_MASK, nu_rem != 0);
        if (nu_ballot == 0) {
            // Vertex u has no active unpacked neighbors (isolated in remainder)
            continue;
        }

        // Select first unpacked neighbor v
        int v_lane = __ffs(nu_ballot) - 1;
        uint32_t v_word = __shfl_sync(FULL_MASK, nu_rem, v_lane);
        int v_bit = __ffs(v_word) - 1;
        int v = (v_lane << 5) + v_bit;

        // Clear v from rem_mask on v_lane
        if (lane == v_lane) {
            rem_mask &= ~(1u << v_bit);
        }

        // Intersect neighborhoods in one bitwise AND to detect a vertex-disjoint triangle K_3
        uint32_t common_uv = nu_rem & d_bit_adj[v * 32 + lane] & rem_mask;
        uint32_t w_ballot = __ballot_sync(FULL_MASK, common_uv != 0);
        if (w_ballot != 0) {
            // Found triangle K_3: {u, v, w}
            int w_lane = __ffs(w_ballot) - 1;
            uint32_t w_word = __shfl_sync(FULL_MASK, common_uv, w_lane);
            int w_bit = __ffs(w_word) - 1;
            if (lane == w_lane) {
                rem_mask &= ~(1u << w_bit);
            }
            lb_pack += 2; // Every triangle requires >= 2 vertices in any cover
        } else {
            // Disjoint edge K_2: {u, v}
            lb_pack += 1; // Every disjoint edge requires >= 1 vertex
        }

        // Structural pruning check
        if (numDeleted + lb_pack >= best_bound) {
            #if USE_COUNTERS
            if (lane == 0 && blockCounters != NULL) {
                blockCounters->totalTime[CLIQUE_PACK_PRUNED_NODES]++;
            }
            #endif
            return true;
        }
    }

    return false;
}

// -----------------------------------------------------------------------------
// 4. CTA-Level Disjoint Clique-Packing Lower-Bound Oracle (Coarse CTA Mode)
// For graphs with |V| > 1024 running in CTA mode using a shared memory bitmask.
// -----------------------------------------------------------------------------
__device__ __forceinline__ bool computeCliquePackingLB_CTA(
    unsigned int numVertices,
    const short* __restrict__ vertexDegrees_s,
    CSRGraph graph,
    unsigned int numDeletedVertices,
    unsigned int minimum_s,
    unsigned int left,
    unsigned int right,
    uint32_t* __restrict__ s_visited_mask,
    Counters* blockCounters)
{
    if (numDeletedVertices >= minimum_s) return true;

    // Zero out visited mask up to right (up to 256 words = 8192 vertices)
    unsigned int numWords = (right + 32) >> 5;
    if (numWords > 256) numWords = 256;
    for (unsigned int w = threadIdx.x; w < numWords; w += blockDim.x) {
        s_visited_mask[w] = 0;
    }
    __syncthreads();

    __shared__ int s_lb_pack;
    __shared__ bool s_pruned;
    if (threadIdx.x == 0) {
        s_lb_pack = 0;
        s_pruned = false;
    }
    __syncthreads();

    // Single-thread or cooperative greedy packing
    if (threadIdx.x == 0) {
        int lb = 0;
        for (unsigned int u = left; u <= right; ++u) {
            if (vertexDegrees_s[u] <= 0) continue;
            unsigned int u_word = u >> 5;
            uint32_t u_bit = (1u << (u & 31));
            if (u_word < 256 && (s_visited_mask[u_word] & u_bit)) continue;

            // Mark u visited
            if (u_word < 256) s_visited_mask[u_word] |= u_bit;

            // Find first unvisited neighbor v
            int v_match = -1;
            for (unsigned int e = graph.srcPtr[u]; e < graph.srcPtr[u + 1]; ++e) {
                unsigned int v = graph.dst[e];
                if (v >= left && v <= right && vertexDegrees_s[v] > 0) {
                    unsigned int v_word = v >> 5;
                    uint32_t v_bit = (1u << (v & 31));
                    if (v_word < 256 && !(s_visited_mask[v_word] & v_bit)) {
                        v_match = v;
                        s_visited_mask[v_word] |= v_bit;
                        break;
                    }
                }
            }

            if (v_match == -1) {
                continue;
            }

            // Check if u and v have a common unvisited neighbor w (Triangle K_3)
            int w_match = -1;
            for (unsigned int e = graph.srcPtr[u]; e < graph.srcPtr[u + 1]; ++e) {
                unsigned int w = graph.dst[e];
                if ((int)w == v_match) continue;
                if (w >= left && w <= right && vertexDegrees_s[w] > 0) {
                    unsigned int w_word = w >> 5;
                    uint32_t w_bit = (1u << (w & 31));
                    if (w_word < 256 && !(s_visited_mask[w_word] & w_bit)) {
                        for (unsigned int ev = graph.srcPtr[v_match]; ev < graph.srcPtr[v_match + 1]; ++ev) {
                            if (graph.dst[ev] == w) {
                                w_match = w;
                                s_visited_mask[w_word] |= w_bit;
                                break;
                            }
                        }
                        if (w_match != -1) break;
                    }
                }
            }

            if (w_match != -1) {
                lb += 2; // Triangle K_3
            } else {
                lb += 1; // Edge K_2
            }

            if (numDeletedVertices + lb >= minimum_s) {
                s_pruned = true;
                #if USE_COUNTERS
                if (blockCounters != NULL) {
                    blockCounters->totalTime[CLIQUE_PACK_PRUNED_NODES]++;
                }
                #endif
                break;
            }
        }
        s_lb_pack = lb;
    }
    __syncthreads();

    return s_pruned;
}

// -----------------------------------------------------------------------------
// 5. Zero-Barrier Warp Fixed-Point Reductions (reduceSubgraph_Warp)
// Isolated vertex, High-degree, Degree-1 leaf, Degree-2 Triangle & Dominance.
// -----------------------------------------------------------------------------
__device__ __forceinline__ void reduceSubgraph_Warp(
    uint32_t &my_mask,
    int16_t &numDeleted,
    int best_bound,
    const uint32_t* __restrict__ d_bit_adj,
    int num_words,
    int lane)
{
    bool changed = true;
    while (changed) {
        changed = false;

        for (int w_idx = 0; w_idx < num_words; ++w_idx) {
            uint32_t word = __shfl_sync(FULL_MASK, my_mask, w_idx);
            while (word != 0) {
                int bit_pos = __ffs(word) - 1;
                word &= (word - 1);
                int u = (w_idx << 5) + bit_pos;

                // Re-verify that u is still active
                uint32_t check_word = __shfl_sync(FULL_MASK, my_mask, w_idx);
                if (!((check_word >> bit_pos) & 1u)) {
                    continue;
                }

                // Compute u's active neighborhood and degree across 32 lanes
                uint32_t nu_lane = d_bit_adj[u * 32 + lane] & my_mask;
                int deg_u = warpReduceSum(__popc(nu_lane));

                // Rule 0: Isolated Vertex Rule (deg == 0)
                if (deg_u == 0) {
                    if (lane == w_idx) my_mask &= ~(1u << bit_pos);
                    continue;
                }

                // Rule 1: High-Degree Rule (deg_u + numDeleted >= best_bound)
                if (deg_u + numDeleted >= best_bound) {
                    if (lane == w_idx) my_mask &= ~(1u << bit_pos);
                    numDeleted++;
                    changed = true;
                    continue;
                }

                // Rule 2: Degree-1 Leaf Peeling Rule
                if (deg_u == 1) {
                    uint32_t ballot = __ballot_sync(FULL_MASK, nu_lane != 0);
                    int v_lane = __ffs(ballot) - 1;
                    uint32_t v_word = __shfl_sync(FULL_MASK, nu_lane, v_lane);
                    int v_bit = __ffs(v_word) - 1;

                    // Include neighbor v into cover, delete both u and v
                    if (lane == w_idx)  my_mask &= ~(1u << bit_pos);
                    if (lane == v_lane) my_mask &= ~(1u << v_bit);
                    numDeleted++;
                    changed = true;
                    continue;
                }

                // Rule 3: Degree-2 Triangle Rule
                if (deg_u == 2) {
                    uint32_t ballot = __ballot_sync(FULL_MASK, nu_lane != 0);
                    int v1_lane = __ffs(ballot) - 1;
                    uint32_t v1_word = __shfl_sync(FULL_MASK, nu_lane, v1_lane);
                    int v1_bit = __ffs(v1_word) - 1;
                    int v1 = (v1_lane << 5) + v1_bit;

                    v1_word &= ~(1u << v1_bit);
                    int v2_lane, v2_bit;
                    if (v1_word != 0) {
                        v2_lane = v1_lane;
                        v2_bit = __ffs(v1_word) - 1;
                    } else {
                        ballot &= ~(1u << v1_lane);
                        v2_lane = __ffs(ballot) - 1;
                        uint32_t v2_word = __shfl_sync(FULL_MASK, nu_lane, v2_lane);
                        v2_bit = __ffs(v2_word) - 1;
                    }
                    int v2 = (v2_lane << 5) + v2_bit;

                    uint32_t v1_row_word = d_bit_adj[v1 * 32 + (v2 >> 5)];
                    bool is_triangle = (v1_row_word >> (v2 & 31)) & 1u;

                    if (is_triangle) {
                        if (lane == w_idx)  my_mask &= ~(1u << bit_pos);
                        if (lane == v1_lane) my_mask &= ~(1u << v1_bit);
                        if (lane == v2_lane) my_mask &= ~(1u << v2_bit);
                        numDeleted += 2;
                        changed = true;
                        continue;
                    }
                }

                // Rule 4: Bit-Parallel Vertex Dominance Reduction
                if (deg_u <= 3) {
                    uint32_t temp_ballot = __ballot_sync(FULL_MASK, nu_lane != 0);
                    bool dominated = false;
                    while (temp_ballot != 0 && !dominated) {
                        int cand_v_lane = __ffs(temp_ballot) - 1;
                        uint32_t cand_v_word = __shfl_sync(FULL_MASK, nu_lane, cand_v_lane);
                        while (cand_v_word != 0 && !dominated) {
                            int cand_v_bit = __ffs(cand_v_word) - 1;
                            cand_v_word &= ~(1u << cand_v_bit);
                            int v = (cand_v_lane << 5) + cand_v_bit;

                            uint32_t nu_without_v = nu_lane;
                            if (lane == cand_v_lane) nu_without_v &= ~(1u << cand_v_bit);
                            uint32_t nv_lane = d_bit_adj[v * 32 + lane];
                            bool lane_subset = ((nu_without_v & ~nv_lane) == 0);
                            if (__all_sync(FULL_MASK, lane_subset)) {
                                if (lane == cand_v_lane) my_mask &= ~(1u << cand_v_bit);
                                numDeleted++;
                                changed = true;
                                dominated = true;
                                break;
                            }
                        }
                        temp_ballot &= ~(1u << cand_v_lane);
                    }
                    if (dominated) continue;
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// 6. Lock-Step Warp Lazy BFS & O(1) Analytic Fast-Path Component Classifier
// -----------------------------------------------------------------------------
__device__ __forceinline__ int bfsAndFastPath_Warp(
    uint32_t &my_mask,
    int16_t &numDeleted,
    const uint32_t* __restrict__ d_bit_adj,
    int lane,
    Counters* blockCounters,
    uint32_t &out_second_comp_mask)
{
    out_second_comp_mask = 0;
    uint32_t unvisited = my_mask;
    int num_nontrivial_comps = 0;

    while (__any_sync(FULL_MASK, unvisited != 0)) {
        uint32_t unv_ballot = __ballot_sync(FULL_MASK, unvisited != 0);
        if (unv_ballot == 0) break;

        int seed_lane = __ffs(unv_ballot) - 1;
        uint32_t seed_word = __shfl_sync(FULL_MASK, unvisited, seed_lane);
        int seed_bit = __ffs(seed_word) - 1;
        int seed_v = (seed_lane << 5) + seed_bit;

        uint32_t frontier = (lane == seed_lane) ? (1u << seed_bit) : 0u;
        uint32_t comp_mask = frontier;
        unvisited &= ~frontier;

        // Warp BFS expansion
        while (__any_sync(FULL_MASK, frontier != 0)) {
            uint32_t next_frontier = 0;
            for (int w = 0; w < 32; ++w) {
                uint32_t f_word = __shfl_sync(FULL_MASK, frontier, w);
                while (f_word != 0) {
                    int b = __ffs(f_word) - 1;
                    f_word &= (f_word - 1);
                    int f_v = (w << 5) + b;
                    next_frontier |= (d_bit_adj[f_v * 32 + lane] & unvisited);
                }
            }
            unvisited &= ~next_frontier;
            comp_mask |= next_frontier;
            frontier = next_frontier;
        }

        int compSize = warpReduceSum(__popc(comp_mask));
        if (compSize <= 1) {
            my_mask &= ~comp_mask;
            continue;
        }

        int min_d = 999999;
        int max_d = 0;
        int edge_sum = 0;

        for (int w = 0; w < 32; ++w) {
            uint32_t c_word = __shfl_sync(FULL_MASK, comp_mask, w);
            while (c_word != 0) {
                int b = __ffs(c_word) - 1;
                c_word &= (c_word - 1);
                int cv = (w << 5) + b;
                uint32_t nu_comp = d_bit_adj[cv * 32 + lane] & comp_mask;
                int deg_cv = warpReduceSum(__popc(nu_comp));
                if (deg_cv < min_d) min_d = deg_cv;
                if (deg_cv > max_d) max_d = deg_cv;
                edge_sum += deg_cv;
            }
        }

        int m = edge_sum / 2;

        if (compSize >= 3 && min_d == 2 && max_d == 2 && m == compSize) {
            // Class 1: Chordless Cycle C_k
            int cycle_cover = (compSize + 1) / 2;
            numDeleted += cycle_cover;
            my_mask &= ~comp_mask;
            #if USE_COUNTERS
            if (lane == 0 && blockCounters != NULL) {
                blockCounters->totalTime[CYCLE_FAST_PATH_HITS]++;
            }
            #endif
        } else if (compSize <= 3 && m < compSize) {
            // Class 2: Small Path / Tree P_<=3 (MVC = 1)
            numDeleted += 1;
            my_mask &= ~comp_mask;
            #if USE_COUNTERS
            if (lane == 0 && blockCounters != NULL) {
                blockCounters->totalTime[TREE_FAST_PATH_HITS]++;
            }
            #endif
        } else if (compSize >= 2 && min_d == compSize - 1 && max_d == compSize - 1) {
            // Class 3: Clique K_p (MVC = compSize - 1)
            numDeleted += (compSize - 1);
            my_mask &= ~comp_mask;
        } else if (compSize == 4 && m == 5) {
            // Class 4: Diamond Subgraph K_4 - e (MVC = 2)
            numDeleted += 2;
            my_mask &= ~comp_mask;
            #if USE_COUNTERS
            if (lane == 0 && blockCounters != NULL) {
                blockCounters->totalTime[CYCLE_FAST_PATH_HITS]++;
            }
            #endif
        } else {
            num_nontrivial_comps++;
            if (num_nontrivial_comps == 2) {
                out_second_comp_mask = comp_mask;
            }
        }
    }

    return num_nontrivial_comps;
}

// -----------------------------------------------------------------------------
// 7. Dynamic CTA-to-Warp Morphing Bitboard Kernel (GlobalWorkList / Standalone)
// -----------------------------------------------------------------------------
__global__ void GlobalWorkList_warp_bitboard_kernel(
    unsigned int* __restrict__ minimum,
    CSRGraph graph,
    const uint32_t* __restrict__ d_bit_adj,
    unsigned int numVertices,
    Counters* counters,
    unsigned long long* NODES_PER_SM,
    CompArray compArray,
    unsigned int* compArraySize,
    int* first_to_dequeue_global,
    BitboardNodeState* __restrict__ d_warp_stacks,
    int* __restrict__ global_active_warps)
{
    Counters blockCounters;
    initializeCounters(&blockCounters);

    #if USE_COUNTERS
    __shared__ unsigned int sm_id;
    if (threadIdx.x == 0) {
        sm_id = get_smid();
    }
    #endif

    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int num_warps = blockDim.x >> 5;
    int num_words = (numVertices + 31) >> 5;

    // Intra-CTA Shared-Memory Bitmask Work-Stealing Pool
    __shared__ BitboardNodeState s_warp_pool[CTA_POOL_CAP];
    __shared__ int s_pool_top;
    __shared__ unsigned int s_idle_warps;
    __shared__ bool s_is_root_seed;

    if (threadIdx.x == 0) {
        s_pool_top = 0;
        s_idle_warps = 0;
        if (atomicCAS(first_to_dequeue_global, 0, 1) == 0) {
            s_is_root_seed = true;
        } else {
            s_is_root_seed = false;
        }
    }
    __syncthreads();

    uint32_t my_mask = 0;
    int16_t numDeleted = 0;
    int16_t depth = 0;
    int32_t compIndex = -1;
    int last_branched_deg = 9999;
    bool has_work = false;

    // Root node initialization on Block 0, Warp 0
    if (s_is_root_seed && warp_id == 0) {
        uint32_t init_mask = 0;
        int v_base = lane << 5;
        for (int b = 0; b < 32; ++b) {
            if ((unsigned int)(v_base + b) < numVertices) {
                init_mask |= (1u << b);
            }
        }
        my_mask = init_mask;
        numDeleted = 0;
        depth = 0;
        compIndex = -1;
        has_work = true;
        if (lane == 0) {
            atomicAdd(global_active_warps, 1);
        }
    }

    int warp_stack_top = -1;
    int warp_global_id = blockIdx.x * num_warps + warp_id;
    BitboardNodeState* my_stack = &d_warp_stacks[warp_global_id * WARP_STACK_CAP];

    int idle_spins = 0;

    while (true) {
        if (!has_work) {
            // 1. Try popping from private stack
            if (warp_stack_top >= 0) {
                my_mask = my_stack[warp_stack_top].active_mask[lane];
                numDeleted = my_stack[warp_stack_top].numDeleted;
                depth = my_stack[warp_stack_top].depth;
                compIndex = my_stack[warp_stack_top].compIndex;
                last_branched_deg = 0;
                warp_stack_top--;
                has_work = true;
                idle_spins = 0;
            } else {
                // 2. Try stealing from Intra-CTA shared-memory pool
                int steal_idx = -1;
                if (lane == 0) {
                    if (s_pool_top > 0) {
                        steal_idx = atomicSub(&s_pool_top, 1) - 1;
                    }
                }
                steal_idx = __shfl_sync(FULL_MASK, steal_idx, 0);

                if (steal_idx >= 0 && steal_idx < CTA_POOL_CAP) {
                    my_mask = s_warp_pool[steal_idx].active_mask[lane];
                    numDeleted = s_warp_pool[steal_idx].numDeleted;
                    depth = s_warp_pool[steal_idx].depth;
                    compIndex = s_warp_pool[steal_idx].compIndex;
                    last_branched_deg = 0;
                    has_work = true;
                    idle_spins = 0;
                    #if USE_COUNTERS
                    if (lane == 0) {
                        blockCounters.totalTime[INTRA_CTA_STEALS]++;
                    }
                    #endif
                } else {
                    // Check if entire GPU search tree has completed
                    int active_cnt = *global_active_warps;
                    if (active_cnt <= 0 && s_pool_top <= 0) {
                        break;
                    }

                    // Mark warp idle and spin briefly
                    if (lane == 0) {
                        atomicOr(&s_idle_warps, (1u << warp_id));
                    }
                    idle_spins++;
                    if (idle_spins > 50000 && active_cnt <= 0) {
                        break;
                    }
                    continue;
                }
            }
        }

        // Warp is now active
        if (lane == 0) {
            atomicAnd(&s_idle_warps, ~(1u << warp_id));
        }

        #if USE_COUNTERS
        if (lane == 0) {
            atomicAdd(&NODES_PER_SM[sm_id], 1);
            blockCounters.totalTime[WARP_BITBOARD_NODES]++;
        }
        #endif

        int current_best = *minimum;
        if (compIndex > -1) {
            current_best = compArray.entries[compIndex].minimum;
        }

        // Fixed-Point Reductions in Registers
        reduceSubgraph_Warp(my_mask, numDeleted, current_best, d_bit_adj, num_words, lane);

        // Disjoint Clique-Packing Lower Bound Oracle (K_3 + K_2)
        bool pruned = computeCliquePackingLB_Warp(
            my_mask, numDeleted, current_best, d_bit_adj, lane, &blockCounters
        );

        if (pruned || numDeleted >= current_best) {
            // Branch pruned
            has_work = false;
            continue;
        }

        // Lazy BFS & Fast-Path Classifier
        bool run_cc = (depth == 0 || last_branched_deg >= LAZY_TAU || (depth % LAZY_STRIDE == 0));
        if (run_cc) {
            uint32_t second_comp = 0;
            int num_comps = bfsAndFastPath_Warp(
                my_mask, numDeleted, d_bit_adj, lane, &blockCounters, second_comp
            );

            if (num_comps == 0) {
                // Completely solved analytically
                if (lane == 0) {
                    if (compIndex > -1) {
                        atomicMin((int*)&compArray.entries[compIndex].minimum, (int)numDeleted);
                    } else {
                        atomicMin(minimum, (unsigned int)numDeleted);
                    }
                }
                has_work = false;
                continue;
            }
        }

        // Max-Degree Selection
        int local_max_deg = 0, local_best_v = 0;
        uint32_t temp_mask = my_mask;
        while (temp_mask != 0) {
            int b = __ffs(temp_mask) - 1;
            temp_mask &= (temp_mask - 1);
            int v = (lane << 5) + b;
            uint32_t nu = d_bit_adj[v * 32 + lane] & my_mask;
            int deg_v = warpReduceSum(__popc(nu));
            if (deg_v > local_max_deg || (deg_v == local_max_deg && v < local_best_v)) {
                local_max_deg = deg_v;
                local_best_v = v;
            }
        }

        int out_max_deg = 0, out_best_v = 0;
        warpReduceMaxDegree(local_max_deg, local_best_v, out_max_deg, out_best_v);

        if (out_max_deg == 0) {
            // All edges covered: update solution
            if (lane == 0) {
                if (compIndex > -1) {
                    atomicMin((int*)&compArray.entries[compIndex].minimum, (int)numDeleted);
                } else {
                    atomicMin(minimum, (unsigned int)numDeleted);
                }
            }
            has_work = false;
            continue;
        }

        // Construct Right Child: G - N(v_max)
        uint32_t right_mask = my_mask & ~d_bit_adj[out_best_v * 32 + lane];
        if (lane == (out_best_v >> 5)) {
            right_mask &= ~(1u << (out_best_v & 31));
        }
        int right_deleted = numDeleted + out_max_deg;

        // Push Right Child: prefer Intra-CTA shared-memory pool if sibling is idle
        bool pushed_to_pool = false;
        if (s_idle_warps != 0 && s_pool_top < CTA_POOL_CAP) {
            int pool_slot = -1;
            if (lane == 0) {
                if (s_pool_top < CTA_POOL_CAP) {
                    pool_slot = atomicAdd(&s_pool_top, 1);
                }
            }
            pool_slot = __shfl_sync(FULL_MASK, pool_slot, 0);

            if (pool_slot >= 0 && pool_slot < CTA_POOL_CAP) {
                s_warp_pool[pool_slot].active_mask[lane] = right_mask;
                if (lane == 0) {
                    s_warp_pool[pool_slot].numDeleted = right_deleted;
                    s_warp_pool[pool_slot].depth = depth + 1;
                    s_warp_pool[pool_slot].compIndex = compIndex;
                }
                pushed_to_pool = true;
            }
        }

        if (!pushed_to_pool && warp_stack_top + 1 < WARP_STACK_CAP) {
            warp_stack_top++;
            my_stack[warp_stack_top].active_mask[lane] = right_mask;
            if (lane == 0) {
                my_stack[warp_stack_top].numDeleted = right_deleted;
                my_stack[warp_stack_top].depth = depth + 1;
                my_stack[warp_stack_top].compIndex = compIndex;
            }
        }

        // Construct Left Child: G - {v_max}
        if (lane == (out_best_v >> 5)) {
            my_mask &= ~(1u << (out_best_v & 31));
        }
        numDeleted++;
        depth++;
        last_branched_deg = out_max_deg;
    }

    #if USE_COUNTERS
    if (threadIdx.x == 0) {
        counters[blockIdx.x] = blockCounters;
    }
    #endif
}

#endif // WARP_BITBOARD_ENGINE_CUH
