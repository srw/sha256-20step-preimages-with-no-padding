// Heterogeneous CPU + Multi-GPU CUDA Solver for 20-Step Padded SHA-256
//
// Complementary Architecture (CPU + CUDA):
//   1. Host CPU Cores:
//      - Build/load the 6.00 GB 2-Tier Nested 4-Slot Sudoku-Augmented Q0^-1 table (/tmp/q0_u16_sudoku6.bin).
//      - Run the 7-Stage Causal Carry-Sudoku tree search (ctx_worker) in parallel to maintain
//        a large T16-sorted pool (CTX_POOL_CAP = 8192) of exact a5 == a6 (D = 0) Right contexts.
//   2. NVIDIA GPU(s) (L4 / A100 / H100, single or multi-GPU):
//      - Holds the 6.00 GB H_pack6 table in high-bandwidth VRAM (300 - 3,350 GB/s) + 128 KB REC_HI16 in L2/L1.
//      - Evaluates T16-clustered context pairs (B=2) across all 2^32 a1 values using hardware
//        funnel-shift (__funnelshift_r), hardware popcount (__popc), and 64-bit SWAR 12-slot Level-3 filtering.
//      - Checks both the 19-round match and the 20-round target (s0(W4) + W3 == T19) directly in registers on the GPU.
//
// Compile:
//   nvcc -O3 -arch=native -Xcompiler -pthread solve20_padded_cuda.cu -o solve20_padded_cuda

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

typedef uint32_t u32;
typedef uint64_t u64;
typedef uint16_t u16;
typedef uint8_t  u8;

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[CUDA ERROR] %s at %s:%d: %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

static const u32 K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};
static u32 IV[8] = {
    0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
    0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19
};

static u64 g_padlen_bits = 440ULL;
__constant__ u32 d_padlen_bits = 440u;
#ifdef __CUDA_ARCH__
#define PADLEN_BITS ((u64)d_padlen_bits)
#else
#define PADLEN_BITS g_padlen_bits
#endif
#define R_STEPS 20
#define OFF 7
#define MAX_BATCH_CTX 256

#define ROR(x,n) (((x)>>((n)&31u))|((x)<<((32u-((n)&31u))&31u)))
static inline u32 S0(u32 x){ return ROR(x,2)^ROR(x,13)^ROR(x,22); }
static inline u32 S1(u32 x){ return ROR(x,6)^ROR(x,11)^ROR(x,25); }
static inline u32 s0(u32 x){ return ROR(x,7)^ROR(x,18)^(x>>3); }
static inline u32 s1(u32 x){ return ROR(x,17)^ROR(x,19)^(x>>10); }
static inline u32 CH(u32 e,u32 f,u32 g){ return g ^ (e & (f ^ g)); }
static inline u32 MAJ(u32 a,u32 b,u32 c){ return (a & (b ^ c)) ^ (b & c); }
static inline u32 Qf(u32 u){ return s0(u)-u; }

static double now_sec(void){
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

static inline u64 splitmix64(u64 *state){
    u64 z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static inline u32 E_arr(const u32 *A, int s){
    return A[s+OFF] - S0(A[s-1+OFF]) - MAJ(A[s-1+OFF],A[s-2+OFF],A[s-3+OFF]) + A[s-4+OFF];
}
static inline u32 Wt_arr(const u32 *A, int t){
    u32 et = E_arr(A, t);
    return E_arr(A, t+1) - A[t-3+OFF] - E_arr(A, t-3) - S1(et) - CH(et, E_arr(A, t-1), E_arr(A, t-2)) - K[t];
}

static void window_from_state(const u32 st[8], u32 w[8]){
    u32 Aa[8], Ee[4];
    for(int i=0;i<4;i++){ Aa[i]=st[i]; Ee[i]=st[4+i]; }
    for(int k=0;k<4;k++) Aa[4+k] = Ee[k] - Aa[k] + S0(Aa[k+1]) + MAJ(Aa[k+1],Aa[k+2],Aa[k+3]);
    for(int i=0;i<8;i++) w[i]=Aa[7-i];
}

static void compress_r(const u32 Wm[16], int r, u32 out[8]){
    u32 W[64]; memcpy(W, Wm, 64);
    for(int t=16;t<r;t++) W[t] = s1(W[t-2]) + W[t-7] + s0(W[t-15]) + W[t-16];
    u32 a=IV[0],b=IV[1],c=IV[2],d=IV[3],e=IV[4],f=IV[5],g=IV[6],h=IV[7];
    for(int t=0;t<r;t++){
        u32 T1 = h + S1(e) + CH(e,f,g) + K[t] + W[t];
        u32 T2 = S0(a) + MAJ(a,b,c);
        h=g; g=f; f=e; e=d+T1; d=c; c=b; b=a; a=T1+T2;
    }
    u32 st[8]={a,b,c,d,e,f,g,h};
    for(int i=0;i<8;i++) out[i] = st[i] + IV[i];
}

// ---------- Host 6.00 GB 2-Tier Sudoku-Augmented Q_0^{-1} Table Builder ----------
#define CACHE_FILE_HIER6   "/tmp/q0_u16_hier6.bin"
#define CACHE_FILE_SUDOKU6 "/tmp/q0_u16_sudoku6.bin"

static u16 *H_pack6 = NULL;
static u16 REC_HI16[65536];

static void init_rec_tables(void){
    for(u32 uhi = 0; uhi < 65536; uhi++){
        u32 v17 = (s0(uhi << 16) & 0x1ffffu) ^ ((uhi & 1u) << 16);
        REC_HI16[(v17 >> 1) & 0xffffu] = (u16)uhi;
    }
}

static inline u32 rec_idx(u32 ulo, u32 y){
    return (((ulo + y) ^ s0(ulo)) >> 1) & 0xffffu;
}

static const int S_SLIDE3[16]      = {0,0,0,1, 2,3,3,4, 5,6,6,7, 8,9,9,9};
static const int S_SLIDE4_NEST[16] = {0,0,0,1, 2,3,3,4, 5,6,6,7, 8,8,8,8};

static inline size_t slide3_off(u32 y){
    u32 d = ROR(0x6AAA6AAAu, y & 31u) >> 31;
    return (size_t)((y >> 1) + (y >> 2) - d);
}

static inline size_t pf12_off(u32 y){
    return (size_t)(y >> 1) + (size_t)(y >> 2);
}

typedef struct {
    int tid;
    int nthreads;
    u64 start_u;
    u64 end_u;
    u64 stored;
} build_arg_t;

#define BUILD_PF 64

static void *prefault_worker_6gb(void *arg){
    build_arg_t *ba = (build_arg_t *)arg;
    u64 total_u16 = 3ULL << 30;
    u64 start = ((u64)ba->tid * total_u16) / (u64)ba->nthreads;
    u64 end   = ((u64)(ba->tid + 1) * total_u16) / (u64)ba->nthreads;
    volatile u64 sum = 0;
    for(u64 idx = start; idx < end; idx += 32){
        sum += H_pack6[idx];
    }
    (void)sum;
    return NULL;
}

static void *build_worker_6gb(void *arg){
    build_arg_t *ba = (build_arg_t *)arg;
    u64 span = ba->end_u - ba->start_u;
    u64 start = ba->start_u + ((u64)ba->tid * span) / (u64)ba->nthreads;
    u64 end   = ba->start_u + ((u64)(ba->tid + 1) * span) / (u64)ba->nthreads;
    start &= ~(u64)(BUILD_PF - 1);
    if(ba->tid < ba->nthreads - 1) end &= ~(u64)(BUILD_PF - 1);
    u32 b_idx[BUILD_PF];
    u16 b_ulo[BUILD_PF];
    u64 local_stored = 0;

    for(u64 base = start; base + BUILD_PF <= end; base += BUILD_PF){
        for(int i=0; i<BUILD_PF; i++){
            u32 u = (u32)(base + (u64)i);
            u32 y = Qf(u);
            u32 idx = y >> 2;
            b_idx[i] = idx;
            b_ulo[i] = (u16)(u & 0xffffu);
            __builtin_prefetch(&H_pack6[(size_t)idx * 3], 1, 0);
        }
        for(int i=0; i<BUILD_PF; i++){
            u16 ulo = b_ulo[i];
            if(!ulo) continue;
            size_t off = (size_t)b_idx[i] * 3;
            for(int s=0; s<3; s++){
                u16 cur = __atomic_load_n(&H_pack6[off + s], __ATOMIC_RELAXED);
                if(cur == ulo) break;
                if(cur == 0u){
                    if(__atomic_compare_exchange_n(&H_pack6[off + s], &cur, ulo, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED)){
                        local_stored++;
                        break;
                    }
                    if(cur == ulo) break;
                }
            }
        }
    }
    ba->stored = local_stored;
    return NULL;
}

static void *convert_pack6_to_pair6_pass1_inplace(void *arg){
    build_arg_t *ba = (build_arg_t *)arg;
    u64 n_pairs = 1ULL << 29;
    u64 start = ((u64)ba->tid * n_pairs) / (u64)ba->nthreads;
    u64 end   = ((u64)(ba->tid + 1) * n_pairs) / (u64)ba->nthreads;
    for(u64 p = start; p < end; p++){
        size_t off = (size_t)p * 6u;
        u16 o0 = H_pack6[off + 3], o2 = H_pack6[off + 5];
        H_pack6[off + 3] = o2;
        H_pack6[off + 5] = o0;
    }
    return NULL;
}

static void *build_pair6_pass2_ovf(void *arg){
    build_arg_t *ba = (build_arg_t *)arg;
    u64 span = ba->end_u - ba->start_u;
    u64 start = ba->start_u + ((u64)ba->tid * span) / (u64)ba->nthreads;
    u64 end   = ba->start_u + ((u64)(ba->tid + 1) * span) / (u64)ba->nthreads;
    start &= ~63ULL;
    if(ba->tid < ba->nthreads - 1) end &= ~63ULL;
    u32 b_idx[64];
    u16 b_ulo[64];
    u64 ovf = 0;

    for(u64 base = start; base + 64 <= end; base += 64){
        for(int i=0; i<64; i++){
            u32 u = (u32)(base + (u64)i);
            u32 y = Qf(u);
            u32 idx = y >> 2;
            b_idx[i] = idx;
            b_ulo[i] = (u16)(u & 0xffffu);
            __builtin_prefetch(&H_pack6[(size_t)(idx >> 1) * 6u], 1, 0);
        }
        for(int i=0; i<64; i++){
            u16 ulo = b_ulo[i];
            if(!ulo) continue;
            u32 idx = b_idx[i];
            size_t poff = (size_t)(idx >> 1) * 6u;
            if((idx & 1u) == 0u){
                u16 s2 = __atomic_load_n(&H_pack6[poff + 2], __ATOMIC_RELAXED);
                if(s2 == 0u) continue;
                u16 s0 = __atomic_load_n(&H_pack6[poff + 0], __ATOMIC_RELAXED);
                u16 s1 = __atomic_load_n(&H_pack6[poff + 1], __ATOMIC_RELAXED);
                if(s0 == ulo || s1 == ulo || s2 == ulo) continue;
                u16 cur = __atomic_load_n(&H_pack6[poff + 3], __ATOMIC_RELAXED);
                if(cur == 0u){
                    if(__atomic_compare_exchange_n(&H_pack6[poff + 3], &cur, ulo, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED)){
                        ovf++;
                    }
                }
            } else {
                u16 s3 = __atomic_load_n(&H_pack6[poff + 3], __ATOMIC_RELAXED);
                if(s3 == 0u) continue;
                u16 s5 = __atomic_load_n(&H_pack6[poff + 5], __ATOMIC_RELAXED);
                u16 s4 = __atomic_load_n(&H_pack6[poff + 4], __ATOMIC_RELAXED);
                if(s5 == ulo || s4 == ulo || s3 == ulo) continue;
                u16 cur = __atomic_load_n(&H_pack6[poff + 2], __ATOMIC_RELAXED);
                if(cur == 0u){
                    if(__atomic_compare_exchange_n(&H_pack6[poff + 2], &cur, ulo, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED)){
                        ovf++;
                    }
                }
            }
        }
    }
    ba->stored = ovf;
    return NULL;
}

static void *build_hier6_pass3_pair_ovf(void *arg){
    build_arg_t *ba = (build_arg_t *)arg;
    u64 span = ba->end_u - ba->start_u;
    u64 start = ba->start_u + ((u64)ba->tid * span) / (u64)ba->nthreads;
    u64 end   = ba->start_u + ((u64)(ba->tid + 1) * span) / (u64)ba->nthreads;
    start &= ~63ULL;
    if(ba->tid < ba->nthreads - 1) end &= ~63ULL;
    u32 b_idx[64];
    u16 b_ulo[64];
    u64 ovf = 0;

    for(u64 base = start; base + 64 <= end; base += 64){
        for(int i=0; i<64; i++){
            u32 u = (u32)(base + (u64)i);
            u32 y = Qf(u);
            u32 idx = y >> 2;
            b_idx[i] = idx;
            b_ulo[i] = (u16)(u & 0xffffu);
            __builtin_prefetch(&H_pack6[(size_t)(idx >> 1) * 6u], 1, 0);
        }
        for(int i=0; i<64; i++){
            u16 ulo = b_ulo[i];
            if(!ulo) continue;
            u32 idx = b_idx[i];
            size_t poff = (size_t)(idx >> 1) * 6u;
            u16 s0 = __atomic_load_n(&H_pack6[poff + 0], __ATOMIC_RELAXED);
            u16 s1 = __atomic_load_n(&H_pack6[poff + 1], __ATOMIC_RELAXED);
            u16 s2 = __atomic_load_n(&H_pack6[poff + 2], __ATOMIC_RELAXED);
            u16 s3 = __atomic_load_n(&H_pack6[poff + 3], __ATOMIC_RELAXED);
            u16 s4 = __atomic_load_n(&H_pack6[poff + 4], __ATOMIC_RELAXED);
            u16 s5 = __atomic_load_n(&H_pack6[poff + 5], __ATOMIC_RELAXED);
            if(s0 == ulo || s1 == ulo || s2 == ulo || s3 == ulo || s4 == ulo || s5 == ulo) continue;
            if((s0 != 0u) & (s1 != 0u) & (s2 != 0u) & (s3 != 0u) & (s4 != 0u) & (s5 != 0u)) continue;
            for(int s=0; s<6; s++){
                u16 cur = __atomic_load_n(&H_pack6[poff + (size_t)s], __ATOMIC_RELAXED);
                if(cur == ulo) break;
                if(cur == 0u){
                    if(__atomic_compare_exchange_n(&H_pack6[poff + (size_t)s], &cur, ulo, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED)){
                        ovf++;
                        break;
                    }
                }
            }
        }
    }
    ba->stored = ovf;
    return NULL;
}

static void *build_hier6_pass4_quad_ovf(void *arg){
    build_arg_t *ba = (build_arg_t *)arg;
    u64 span = ba->end_u - ba->start_u;
    u64 start = ba->start_u + ((u64)ba->tid * span) / (u64)ba->nthreads;
    u64 end   = ba->start_u + ((u64)(ba->tid + 1) * span) / (u64)ba->nthreads;
    start &= ~63ULL;
    if(ba->tid < ba->nthreads - 1) end &= ~63ULL;
    u32 b_idx[64];
    u16 b_ulo[64];
    u64 ovf = 0;

    for(u64 base = start; base + 64 <= end; base += 64){
        for(int i=0; i<64; i++){
            u32 u = (u32)(base + (u64)i);
            u32 y = Qf(u);
            u32 idx = y >> 2;
            b_idx[i] = idx;
            b_ulo[i] = (u16)(u & 0xffffu);
            __builtin_prefetch(&H_pack6[(size_t)(idx >> 2) * 12u], 1, 0);
        }
        for(int i=0; i<64; i++){
            u16 ulo = b_ulo[i];
            if(!ulo) continue;
            u32 idx = b_idx[i];
            size_t qoff = (size_t)(idx >> 2) * 12u;
            int found = 0, has_zero = 0;
            for(int s=0; s<12; s++){
                u16 v = __atomic_load_n(&H_pack6[qoff + (size_t)s], __ATOMIC_RELAXED);
                if(v == ulo){ found = 1; break; }
                if(v == 0u) has_zero = 1;
            }
            if(found || !has_zero) continue;
            for(int s=0; s<12; s++){
                u16 cur = __atomic_load_n(&H_pack6[qoff + (size_t)s], __ATOMIC_RELAXED);
                if(cur == ulo) break;
                if(cur == 0u){
                    if(__atomic_compare_exchange_n(&H_pack6[qoff + (size_t)s], &cur, ulo, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED)){
                        ovf++;
                        break;
                    }
                }
            }
        }
    }
    ba->stored = ovf;
    return NULL;
}

static void edf_reorder_quad_slide3(u16 q[12], u32 quad_idx){
    u16 items_by_r[16][12];
    u8  cnt_r[16] = {0};
    u16 unmatched[12];
    int n_unmatched = 0;

    u32 base_y = quad_idx << 4;
    for(int s = 0; s < 12; s++){
        u32 ulo = q[s];
        if(!ulo) continue;
        int found_r = -1;
        u32 r0 = (u32)(s / 3) * 4u;
        for(u32 k = 0; k < 16; k++){
            u32 r = (r0 + k) & 15u;
            u32 y = base_y | r;
            u32 u = ulo | ((u32)REC_HI16[rec_idx(ulo, y)] << 16);
            if(Qf(u) == y){
                found_r = (int)r;
                break;
            }
        }
        if(found_r >= 0){
            items_by_r[found_r][cnt_r[found_r]++] = (u16)ulo;
        } else {
            unmatched[n_unmatched++] = (u16)ulo;
        }
    }

    u16 out[12] = {0};
    for(int r = 0; r < 16; r++){
        int start = S_SLIDE3[r], end = start + 2;
        for(int k = 0; k < cnt_r[r]; k++){
            u16 ulo = items_by_r[r][k];
            int placed = 0;
            for(int s = start; s <= end; s++){
                if(out[s] == 0){
                    out[s] = ulo;
                    placed = 1;
                    break;
                }
            }
            if(!placed){
                unmatched[n_unmatched++] = ulo;
            }
        }
    }
    int u_idx = 0;
    for(int s = 0; s < 12 && u_idx < n_unmatched; s++){
        if(out[s] == 0){
            out[s] = unmatched[u_idx++];
        }
    }
    memcpy(q, out, 12 * sizeof(u16));
}

static void *edf_slide3_worker(void *arg){
    build_arg_t *ba = (build_arg_t *)arg;
    u64 n_quads = 1ULL << 28;
    u64 start = ((u64)ba->tid * n_quads) / (u64)ba->nthreads;
    u64 end   = ((u64)(ba->tid + 1) * n_quads) / (u64)ba->nthreads;
    for(u64 q = start; q < end; q++){
        edf_reorder_quad_slide3(&H_pack6[(size_t)q * 12u], (u32)q);
    }
    return NULL;
}

static const u8 SLIDE_R_MIN[12] = {0, 0, 0, 3, 4, 5, 7, 8, 9, 11, 12, 13};
static const u8 SLIDE_R_MAX[12] = {2, 3, 4, 6, 7, 8, 10, 11, 12, 15, 15, 15};
static volatile u8 g_quad_locks[65536];

static inline void quad_lock(u32 quad_idx){
    u32 h = (quad_idx ^ (quad_idx >> 16)) & 0xffffu;
    while(__atomic_test_and_set(&g_quad_locks[h], __ATOMIC_ACQUIRE)){}
}

static inline void quad_unlock(u32 quad_idx){
    u32 h = (quad_idx ^ (quad_idx >> 16)) & 0xffffu;
    __atomic_clear(&g_quad_locks[h], __ATOMIC_RELEASE);
}

static inline int slot_matched_rem(u32 v, int s, u32 base_y){
    if(!v) return -1;
    int rmin = SLIDE_R_MIN[s], rmax = SLIDE_R_MAX[s];
    for(int rem = rmin; rem <= rmax; rem++){
        u32 y = base_y | (u32)rem;
        u32 u = v | ((u32)REC_HI16[rec_idx(v, y)] << 16);
        if(Qf(u) == y) return rem;
    }
    return -1;
}

static int sudoku_augment_dfs(u16 q[12], int s_R[12], u16 item, int r, u32 visited_mask, u16 *evicted_ovf){
    int start = S_SLIDE3[r];
    for(int k = 0; k < 3; k++){
        int s = start + k;
        if(s_R[s] < 0){
            *evicted_ovf = q[s];
            q[s]   = item;
            s_R[s] = r;
            return 1;
        }
    }
    for(int k = 0; k < 3; k++){
        int s = start + k;
        if(visited_mask & (1u << s)) continue;
        int r_occ = s_R[s];
        if(r_occ == r) continue;
        u16 item_occ = q[s];
        if(sudoku_augment_dfs(q, s_R, item_occ, r_occ, visited_mask | (1u << s), evicted_ovf)){
            q[s]   = item;
            s_R[s] = r;
            return 1;
        }
    }
    return 0;
}

static void *sudoku_upgrade_worker_6gb(void *arg){
    build_arg_t *ba = (build_arg_t *)arg;
    u64 span = ba->end_u - ba->start_u;
    u64 start = ba->start_u + ((u64)ba->tid * span) / (u64)ba->nthreads;
    u64 end   = ba->start_u + ((u64)(ba->tid + 1) * span) / (u64)ba->nthreads;
    start &= ~63ULL;
    if(ba->tid < ba->nthreads - 1) end &= ~63ULL;
    u32 b_y[64];
    u16 b_ulo[64];
    u64 upgraded = 0;

    for(u64 base = start; base + 64 <= end; base += 64){
        for(int i=0; i<64; i++){
            u32 u = (u32)(base + (u64)i);
            u32 y = Qf(u);
            b_y[i]   = y;
            b_ulo[i] = (u16)(u & 0xffffu);
            __builtin_prefetch(&H_pack6[pf12_off(y)], 0, 0);
        }
        for(int i=0; i<64; i++){
            u16 ulo = b_ulo[i];
            if(!ulo) continue;
            u32 y = b_y[i];
            const u16 *win = &H_pack6[slide3_off(y)];
            if(win[0] == ulo || win[1] == ulo || win[2] == ulo) continue;

            u32 quad_idx = y >> 4;
            u32 base_y   = quad_idx << 4;
            int r        = (int)(y & 15u);
            int wstart   = S_SLIDE3[r];

            u16 w0 = win[0], w1 = win[1], w2 = win[2];
            int r0 = slot_matched_rem(w0, wstart + 0, base_y);
            int r1 = slot_matched_rem(w1, wstart + 1, base_y);
            int r2 = slot_matched_rem(w2, wstart + 2, base_y);
            if(r0 == r && r1 == r && r2 == r) continue;

            quad_lock(quad_idx);
            u16 *q = &H_pack6[(size_t)quad_idx * 12u];
            if(q[wstart] != ulo && q[wstart + 1] != ulo && q[wstart + 2] != ulo){
                int prev_slot = -1;
                for(int s = 0; s < 12; s++){
                    if(q[s] == ulo){ prev_slot = s; break; }
                }
                int s_R[12];
                for(int s = 0; s < 12; s++){
                    s_R[s] = slot_matched_rem(q[s], s, base_y);
                }
                u16 evicted = 0;
                if(sudoku_augment_dfs(q, s_R, ulo, r, 0u, &evicted)){
                    if(prev_slot >= 0 && s_R[prev_slot] < 0){
                        q[prev_slot] = evicted;
                    } else if(evicted != 0){
                        for(int s = 0; s < 12; s++){
                            if(q[s] == 0){ q[s] = evicted; break; }
                        }
                    }
                    upgraded++;
                }
            }
            quad_unlock(quad_idx);
        }
    }
    ba->stored = upgraded;
    return NULL;
}

static int sudoku_augment_tier2_dfs(u16 q[12], int s_R[12], u8 s_tier[12],
                                    u16 item, int r, u8 tier, u32 visited_mask){
    int start = (tier == 1) ? S_SLIDE3[r] : S_SLIDE4_NEST[r];
    int width = (tier == 1) ? 3 : 4;
    for(int k = 0; k < width; k++){
        int s = start + k;
        if(s_tier[s] == 0){
            q[s]      = item;
            s_R[s]    = r;
            s_tier[s] = tier;
            return 1;
        }
    }
    for(int k = 0; k < width; k++){
        int s = start + k;
        if(visited_mask & (1u << s)) continue;
        int r_occ    = s_R[s];
        u8  tier_occ = s_tier[s];
        if(r_occ == r && tier_occ <= tier) continue;
        u16 item_occ = q[s];
        if(sudoku_augment_tier2_dfs(q, s_R, s_tier, item_occ, r_occ, tier_occ, visited_mask | (1u << s))){
            q[s]      = item;
            s_R[s]    = r;
            s_tier[s] = tier;
            return 1;
        }
    }
    return 0;
}

static inline int tier2_reorder_single_quad(u16 *q, u32 quad_idx, int apply){
    u32 base_y = quad_idx << 4;
    u16 q_tmp[12];
    int s_R[12];
    u8  s_tier[12];
    u16 ovf_v_in[12], ovf_v_out[12];
    int ovf_r_in[12], ovf_r_out[12];
    int n_in = 0, n_out = 0;

    for(int s = 0; s < 12; s++){
        u16 v = q[s];
        if(!v){
            q_tmp[s]  = 0;
            s_R[s]    = -1;
            s_tier[s] = 0;
            continue;
        }
        int r = slot_matched_rem(v, s, base_y);
        if(r >= 0){
            q_tmp[s]  = v;
            s_R[s]    = r;
            s_tier[s] = 1;
        } else {
            q_tmp[s]  = 0;
            s_R[s]    = -1;
            s_tier[s] = 0;
            int rmin = SLIDE_R_MIN[s], rmax = SLIDE_R_MAX[s];
            int found_r = -1;
            for(int rem = 0; rem < 16; rem++){
                if(rem >= rmin && rem <= rmax) continue;
                u32 y = base_y | (u32)rem;
                u32 u = (u32)v | ((u32)REC_HI16[rec_idx(v, y)] << 16);
                if(Qf(u) == y){ found_r = rem; break; }
            }
            if(found_r >= 0 && s >= S_SLIDE4_NEST[found_r] && s <= S_SLIDE4_NEST[found_r] + 3){
                ovf_v_in[n_in] = v;
                ovf_r_in[n_in] = found_r;
                n_in++;
            } else {
                ovf_v_out[n_out] = v;
                ovf_r_out[n_out] = found_r;
                n_out++;
            }
        }
    }
    if(n_out == 0) return 0;

    u16 unplaced[12];
    int n_unplaced = 0;
    int after_t2 = 0;
    for(int k = 0; k < n_in; k++){
        if(sudoku_augment_tier2_dfs(q_tmp, s_R, s_tier, ovf_v_in[k], ovf_r_in[k], 2, 0u)){
            after_t2++;
        } else {
            unplaced[n_unplaced++] = ovf_v_in[k];
        }
    }
    for(int k = 0; k < n_out; k++){
        if(ovf_r_out[k] >= 0 && sudoku_augment_tier2_dfs(q_tmp, s_R, s_tier, ovf_v_out[k], ovf_r_out[k], 2, 0u)){
            after_t2++;
        } else {
            unplaced[n_unplaced++] = ovf_v_out[k];
        }
    }
    if(after_t2 <= n_in) return 0;

    if(apply){
        int u_idx = 0;
        for(int s = 0; s < 12 && u_idx < n_unplaced; s++){
            if(s_tier[s] == 0){
                q_tmp[s] = unplaced[u_idx++];
            }
        }
        memcpy(q, q_tmp, 12 * sizeof(u16));
    }
    return after_t2 - n_in;
}

static void *tier2_nest4_quad_worker(void *arg){
    build_arg_t *ba = (build_arg_t *)arg;
    u64 n_quads = 1ULL << 28;
    u64 start = ((u64)ba->tid * n_quads) / (u64)ba->nthreads;
    u64 end   = ((u64)(ba->tid + 1) * n_quads) / (u64)ba->nthreads;
    u64 gained = 0;
    for(u64 qi = start; qi < end; qi++){
        gained += (u64)tier2_reorder_single_quad(&H_pack6[(size_t)qi * 12u], (u32)qi, 1);
    }
    ba->stored = gained;
    return NULL;
}

static void init_q0_table_6gb(int nthreads){
    init_rec_tables();
    u64 bytes = (3ULL << 30) * sizeof(u16); // 6.00 GB

    int sfd = open(CACHE_FILE_SUDOKU6, O_RDONLY);
    if(sfd >= 0){
        struct stat st;
        if(fstat(sfd, &st) == 0 && (u64)st.st_size == bytes){
            void *p = mmap(NULL, bytes, PROT_READ, MAP_SHARED, sfd, 0);
            close(sfd);
            if(p != MAP_FAILED){
                H_pack6 = (u16 *)p;
                pthread_t *th = (pthread_t *)malloc(nthreads * sizeof(pthread_t));
                build_arg_t *args = (build_arg_t *)calloc(nthreads, sizeof(build_arg_t));
                for(int i=0;i<nthreads;i++){
                    args[i].tid = i;
                    args[i].nthreads = nthreads;
                    pthread_create(&th[i], NULL, prefault_worker_6gb, &args[i]);
                }
                for(int i=0;i<nthreads;i++) pthread_join(th[i], NULL);
                free(th);
                free(args);
                fprintf(stderr, "[init] Mapped & pre-faulted cached 6.00 GB 2-Tier Sudoku-Augmented Q0^-1 table from %s\n",
                        CACHE_FILE_SUDOKU6);
                return;
            }
        } else {
            close(sfd);
        }
    }

    fprintf(stderr, "[init] Building 6.00 GB 2-Tier Sudoku-Augmented Q0^-1 table (%s) across %d CPU threads...\n",
            CACHE_FILE_SUDOKU6, nthreads);
    void *p = mmap(NULL, bytes + 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if(p == MAP_FAILED){ perror("mmap 6GB"); exit(1); }
    H_pack6 = (u16 *)p;
    pthread_t *th = (pthread_t *)malloc(nthreads * sizeof(pthread_t));
    build_arg_t *args = (build_arg_t *)calloc(nthreads, sizeof(build_arg_t));
    for(int i=0;i<nthreads;i++){
        args[i].tid = i; args[i].nthreads = nthreads;
        args[i].start_u = 0; args[i].end_u = (1ULL << 32);
        pthread_create(&th[i], NULL, build_worker_6gb, &args[i]);
    }
    for(int i=0;i<nthreads;i++) pthread_join(th[i], NULL);
    for(int i=0;i<nthreads;i++) pthread_create(&th[i], NULL, convert_pack6_to_pair6_pass1_inplace, &args[i]);
    for(int i=0;i<nthreads;i++) pthread_join(th[i], NULL);
    for(int i=0;i<nthreads;i++){ args[i].stored = 0; pthread_create(&th[i], NULL, build_pair6_pass2_ovf, &args[i]); }
    for(int i=0;i<nthreads;i++) pthread_join(th[i], NULL);
    for(int i=0;i<nthreads;i++){ args[i].stored = 0; pthread_create(&th[i], NULL, build_hier6_pass3_pair_ovf, &args[i]); }
    for(int i=0;i<nthreads;i++) pthread_join(th[i], NULL);
    for(int i=0;i<nthreads;i++){ args[i].stored = 0; pthread_create(&th[i], NULL, build_hier6_pass4_quad_ovf, &args[i]); }
    for(int i=0;i<nthreads;i++) pthread_join(th[i], NULL);
    for(int i=0;i<nthreads;i++) pthread_create(&th[i], NULL, edf_slide3_worker, &args[i]);
    for(int i=0;i<nthreads;i++) pthread_join(th[i], NULL);
    for(int i=0;i<nthreads;i++){ args[i].stored = 0; pthread_create(&th[i], NULL, sudoku_upgrade_worker_6gb, &args[i]); }
    for(int i=0;i<nthreads;i++) pthread_join(th[i], NULL);
    for(int i=0;i<nthreads;i++){ args[i].stored = 0; pthread_create(&th[i], NULL, tier2_nest4_quad_worker, &args[i]); }
    for(int i=0;i<nthreads;i++) pthread_join(th[i], NULL);
    free(th);
    free(args);

    int wfd = open(CACHE_FILE_SUDOKU6, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if(wfd >= 0){
        u64 off = 0;
        const u8 *buf = (const u8 *)H_pack6;
        while(off < bytes){
            size_t chunk = (bytes - off > (1ULL << 26)) ? (1ULL << 26) : (size_t)(bytes - off);
            ssize_t nw = write(wfd, buf + off, chunk);
            if(nw <= 0) break;
            off += (u64)nw;
        }
        close(wfd);
        fprintf(stderr, "[init] Cached 6.00 GB 2-Tier Sudoku-Augmented table to %s\n", CACHE_FILE_SUDOKU6);
    }
}

// ---------- Host 7-Stage Causal Carry-Sudoku Right-Context Generator ----------
typedef struct {
    u32 A_base[R_STEPS + OFF + 2];
    u32 a5, a6;
    u32 T16, T17, T18, T19;
    int target_id;
} right_ctx_t;

typedef struct {
    u32 T16;
    u32 a12;
    u32 a9;
    short target_id;
    u8  used;
    u8  pad;
} compact_ctx_t;

typedef struct {
    u32 A_template[R_STEPS + OFF + 2];
    u32 a0, am1, am2, am3;
    u32 e0, em1, em2, em3;
    u32 C_e1, C_W0, C_G1, C_G2, C_G3;
    u32 dig[8];
    int w_thresh;
    int batch_ctx;
    right_ctx_t rctx[MAX_BATCH_CTX];
} solver_state_t;

typedef struct {
    const char *label;
    const char *hex;
    u32 dig[8];
} target14_entry_t;

static const target14_entry_t G_TARGETS_14[14] = {
    { "abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", { 0xba7816bfu, 0x8f01cfeau, 0x414140deu, 0x5dae2223u, 0xb00361a3u, 0x96177a9cu, 0xb410ff61u, 0xf20015adu } },
    { "hello", "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824", { 0x2cf24dbau, 0x5fb0a30eu, 0x26e83b2au, 0xc5b9e29eu, 0x1b161e5cu, 0x1fa7425eu, 0x73043362u, 0x938b9824u } },
    { "padded test 1", "146f48dd2a3324d46feb7ad2069e05ddb9530be67f8f823f9d1c2ed3ea4f1d19", { 0x146f48ddu, 0x2a3324d4u, 0x6feb7ad2u, 0x069e05ddu, 0xb9530be6u, 0x7f8f823fu, 0x9d1c2ed3u, 0xea4f1d19u } },
    { "padded test 2", "4c51f9dde95fa3ec5545b46ca0f2d8968b2399f81f208e07304f4b00117c02df", { 0x4c51f9ddu, 0xe95fa3ecu, 0x5545b46cu, 0xa0f2d896u, 0x8b2399f8u, 0x1f208e07u, 0x304f4b00u, 0x117c02dfu } },
    { "padded test 3", "cb9ace26d0eee1034f838de327a72598832bfa4fa1644493ad4ef004a5a2dab9", { 0xcb9ace26u, 0xd0eee103u, 0x4f838de3u, 0x27a72598u, 0x832bfa4fu, 0xa1644493u, 0xad4ef004u, 0xa5a2dab9u } },
    { "padded test 4", "90b183e68a4966c5c268807fe3f49a4f4cf890f9f25b00bb1f836552496192f4", { 0x90b183e6u, 0x8a4966c5u, 0xc268807fu, 0xe3f49a4fu, 0x4cf890f9u, 0xf25b00bbu, 0x1f836552u, 0x496192f4u } },
    { "padded test 5", "286d96b78dc5b71f90b61e6374de3ffc82939941955e7b1c5d00fd9af345e80a", { 0x286d96b7u, 0x8dc5b71fu, 0x90b61e63u, 0x74de3ffcu, 0x82939941u, 0x955e7b1cu, 0x5d00fd9au, 0xf345e80au } },
    { "padded test 6", "efa23b479b7518cb4117dfdf041a694b46441fa5796e6d4b8a7443a724a0d8f1", { 0xefa23b47u, 0x9b7518cbu, 0x4117dfdfu, 0x041a694bu, 0x46441fa5u, 0x796e6d4bu, 0x8a7443a7u, 0x24a0d8f1u } },
    { "padded test 7", "2f507443a19f9f39300d5905b02aa1cd77c744f0424c684f75a5b2cc4a0516a8", { 0x2f507443u, 0xa19f9f39u, 0x300d5905u, 0xb02aa1cdu, 0x77c744f0u, 0x424c684fu, 0x75a5b2ccu, 0x4a0516a8u } },
    { "padded test 8", "1fad9d7e79a132660d8536e6b151a4cef494bdfdc2962e0f7bae20acd8aed5c8", { 0x1fad9d7eu, 0x79a13266u, 0x0d8536e6u, 0xb151a4ceu, 0xf494bdfdu, 0xc2962e0fu, 0x7bae20acu, 0xd8aed5c8u } },
    { "padded test 9", "849564c85e22852e6af2c6ec60bd891de79298bbc8d5106a221bca536a908d1f", { 0x849564c8u, 0x5e22852eu, 0x6af2c6ecu, 0x60bd891du, 0xe79298bbu, 0xc8d5106au, 0x221bca53u, 0x6a908d1fu } },
    { "padded test 10", "91e12eb79793591d6773d8c3b5090f07a69742c7031bbd514273c5de1c241c87", { 0x91e12eb7u, 0x9793591du, 0x6773d8c3u, 0xb5090f07u, 0xa69742c7u, 0x031bbd51u, 0x4273c5deu, 0x1c241c87u } },
    { "padded test 11", "7b66f998c1c60bdd65744e43b69190481502cee718245876fc2c535f4b7cf5a8", { 0x7b66f998u, 0xc1c60bddu, 0x65744e43u, 0xb6919048u, 0x1502cee7u, 0x18245876u, 0xfc2c535fu, 0x4b7cf5a8u } },
    { "hand-off cross-check 2026-09-29", "d374822f21eb0d4dbc809ae0443c2c6c66f8ce5d07c8779bcc5fc7849f44e3c2", { 0xd374822fu, 0x21eb0d4du, 0xbc809ae0u, 0x443c2c6cu, 0x66f8ce5du, 0x07c8779bu, 0xcc5fc784u, 0x9f44e3c2u } }
};

static solver_state_t G_STATE;
static solver_state_t G_STATES_14[14];
static volatile int   g_target14_solved[14];
static volatile int   g_targets14_solved_count = 0;
static int            g_all_14_mode = 0;
static int            g_target_first = 0;
static int            g_target_last  = 13;
static const char    *g_single_label = "target";
static volatile int g_ctx_count = 0;
static volatile u64 g_ctx_trials = 0;
static volatile u64 g_ctx_seed_gen = 0;
static pthread_mutex_t g_ctx_mu = PTHREAD_MUTEX_INITIALIZER;

#define CTX_POOL_CAP   20971520
#define CTX_BATCH_ADD  3000000
static compact_ctx_t g_ctx_pool[CTX_POOL_CAP];
static int           g_ctx_pool_len = 0;
static int           g_exact_pairs_avail = 0;

static void reconstruct_right_ctx_for_state(const solver_state_t *st_ctx, int target_id, u32 a12, u32 a9, right_ctx_t *rc){
    u32 A[R_STEPS + OFF + 2];
    memcpy(A, st_ctx->A_template, sizeof(A));
    const u32 a13 = A[13+OFF], a14 = A[14+OFF], a15 = A[15+OFF], a16 = A[16+OFF];
    const u32 e15 = 0xffffffffu;
    u32 a11 = e15 - a15 + S0(a14) + MAJ(a14, a13, a12);
    u32 a10 = a11;
    u32 e14 = a14 - S0(a13) - MAJ(a13, a12, a11) + a10;
    u32 e16 = a16 - S0(a15) - MAJ(a15, a14, a13) + a12;
    u32 c_e13 = a13 - S0(a12) - a11;
    u32 e13 = a9 + c_e13;
    u32 c_e12 = e16 - a12 - 0xffffffffu - K[15] - (u32)PADLEN_BITS;
    u32 e12 = c_e12 - e14;
    u32 a8  = e12 + S0(a11) - a12 + a11;
    u32 c_e11 = e15 - a11 - S1(e14) - K[14];
    u32 e11 = c_e11 - CH(e14, e13, e12);
    u32 a7  = e11 + S0(a10) - a11 + MAJ(a10, a9, a8);
    u32 a6  = 0xffffffffu - a10 + S0(a9) + MAJ(a9, a8, a7);
    u32 a5  = a6;

    A[12+OFF] = a12; A[11+OFF] = a11; A[10+OFF] = a10; A[9+OFF] = a9;
    A[8+OFF]  = a8;  A[7+OFF]  = a7;  A[6+OFF]  = a6;  A[5+OFF] = a5;
    A[4+OFF]  = 0;   A[3+OFF]  = 0;   A[2+OFF]  = 0;   A[1+OFF] = 0;

    memcpy(rc->A_base, A, sizeof(A));
    rc->a5 = a5;
    rc->a6 = a6;
    rc->target_id = target_id;

    u32 c9  = Wt_arr(A, 9);
    u32 c10 = Wt_arr(A, 10);
    u32 c11 = Wt_arr(A, 11);
    u32 W12 = Wt_arr(A, 12);
    u32 W14 = Wt_arr(A, 14);
    u32 W15 = Wt_arr(A, 15);
    u32 W16 = Wt_arr(A, 16);
    u32 W17 = Wt_arr(A, 17);
    u32 W18 = Wt_arr(A, 18);
    u32 W19 = Wt_arr(A, 19);

    rc->T16 = W16 - s1(W14) - c9;
    rc->T17 = W17 - s1(W15) - c10;
    rc->T18 = W18 - s1(W16) - c11;
    rc->T19 = W19 - s1(W17) - W12;
}

static void reconstruct_right_ctx(u32 a12, u32 a9, right_ctx_t *rc){
    reconstruct_right_ctx_for_state(&G_STATE, 0, a12, a9, rc);
}

static void fill_compact_ctx_for_state(const solver_state_t *st_ctx, int target_id, u32 a12, u32 a9, compact_ctx_t *cc){
    right_ctx_t rc;
    reconstruct_right_ctx_for_state(st_ctx, target_id, a12, a9, &rc);
    cc->T16       = rc.T16;
    cc->a12       = a12;
    cc->a9        = a9;
    cc->target_id = (short)target_id;
    cc->used      = 0;
    cc->pad       = 0;
}

static void fill_compact_ctx(u32 a12, u32 a9, compact_ctx_t *cc){
    fill_compact_ctx_for_state(&G_STATE, 0, a12, a9, cc);
}

static void init_iv_and_target(solver_state_t *st_ctx, const u32 dig[8]){
    memset(st_ctx->A_template, 0, sizeof(st_ctx->A_template));
    memcpy(st_ctx->dig, dig, 32);
    u32 wiv[8]; window_from_state(IV, wiv);
    for(int i=0;i<8;i++) st_ctx->A_template[i] = wiv[i];

    u32 st[8];
    for(int i=0;i<8;i++) st[i] = dig[i] - IV[i];
    u32 wt[8]; window_from_state(st, wt);
    for(int i=0;i<8;i++) st_ctx->A_template[(R_STEPS - 7 + i) + OFF] = wt[i];

    st_ctx->a0  = st_ctx->A_template[0 + OFF];
    st_ctx->am1 = st_ctx->A_template[-1 + OFF];
    st_ctx->am2 = st_ctx->A_template[-2 + OFF];
    st_ctx->am3 = st_ctx->A_template[-3 + OFF];

    st_ctx->e0  = E_arr(st_ctx->A_template, 0);
    st_ctx->em1 = E_arr(st_ctx->A_template, -1);
    st_ctx->em2 = E_arr(st_ctx->A_template, -2);
    st_ctx->em3 = E_arr(st_ctx->A_template, -3);

    st_ctx->C_e1 = st_ctx->am3 - S0(st_ctx->a0) - MAJ(st_ctx->a0, st_ctx->am1, st_ctx->am2);
    st_ctx->C_W0 = st_ctx->C_e1 - st_ctx->am3 - st_ctx->em3 - S1(st_ctx->e0) - CH(st_ctx->e0, st_ctx->em1, st_ctx->em2) - K[0];
    st_ctx->C_G1 = st_ctx->em2 + K[1];
    st_ctx->C_G2 = st_ctx->em1 + K[2];
    st_ctx->C_G3 = st_ctx->e0  + K[3];
}

// ---------- CUDA Device Kernel & Data Structures ----------
typedef struct {
    u32 a0, am1, am2;
    u32 e0, em1;
    u32 C_e1, C_W0, C_G1, C_G2, C_G3;
    int w_thresh;
    int batch_ctx;
    u32 a5[MAX_BATCH_CTX];
    u32 T16[MAX_BATCH_CTX];
    u32 T17[MAX_BATCH_CTX];
    u32 T18[MAX_BATCH_CTX];
    u32 T19[MAX_BATCH_CTX];
} gpu_params_t;

typedef struct {
    unsigned long long l2_nodes;
    unsigned long long l3_lookups;
    unsigned long long sol19_count;
    int solved20;
    int sol_ctx_id;
    u32 sol_a1, sol_a2, sol_a3, sol_a4;
} gpu_stats_t;

__device__ __forceinline__ u32 rotr_dev(u32 x, u32 n){
    return __funnelshift_r(x, x, n);
}
__device__ __forceinline__ u32 S0_dev(u32 x){ return rotr_dev(x,2) ^ rotr_dev(x,13) ^ rotr_dev(x,22); }
__device__ __forceinline__ u32 S1_dev(u32 x){ return rotr_dev(x,6) ^ rotr_dev(x,11) ^ rotr_dev(x,25); }
__device__ __forceinline__ u32 s0_dev(u32 x){ return rotr_dev(x,7) ^ rotr_dev(x,18) ^ (x >> 3); }
__device__ __forceinline__ u32 CH_dev(u32 e, u32 f, u32 g){ return g ^ (e & (f ^ g)); }
__device__ __forceinline__ u32 MAJ_dev(u32 a, u32 b, u32 c){ return (a & (b ^ c)) ^ (b & c); }
__device__ __forceinline__ u32 Qf_dev(u32 u){ return s0_dev(u) - u; }

__device__ __forceinline__ u32 perm32_dev(u32 x){
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// Warp-Cooperative 7-Stage Causal Carry-Sudoku Right-Context Generator on GPU
// Uses a monotonic persistent counter + 32-bit bijection so zero duplicate a12 values are ever tested.
__global__ void __launch_bounds__(256, 4)
ctx_gen_cuda_kernel(
    u32 a13, u32 a14, u32 a15, u32 a16,
    u32 a12_perm_key,
    int target_count,
    int max_cap,
    u32 * __restrict__ d_out_a12,
    u32 * __restrict__ d_out_a9,
    int * __restrict__ d_ctx_count,
    unsigned long long * __restrict__ d_a12_counter,
    unsigned long long * __restrict__ d_trials)
{
    u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    u32 lane = tid & 31u;
    unsigned long long local_a12_tried = 0;

    while(1){
        int cur = 0;
        unsigned long long my_idx = 0;
        if(lane == 0){
            cur = atomicAdd(d_ctx_count, 0);
            if(cur < target_count){
                my_idx = atomicAdd(d_a12_counter, 1ULL);
            }
        }
        cur = __shfl_sync(0xffffffffu, cur, 0);
        if(cur >= target_count) break;
        my_idx = __shfl_sync(0xffffffffu, my_idx, 0);

        u32 a12 = perm32_dev((u32)my_idx + a12_perm_key);
        local_a12_tried++;

        const u32 e15 = 0xffffffffu;
        u32 a11 = e15 - a15 + S0_dev(a14) + MAJ_dev(a14, a13, a12);
        u32 a10 = a11;
        u32 e14 = a14 - S0_dev(a13) - MAJ_dev(a13, a12, a11) + a10;
        u32 e16 = a16 - S0_dev(a15) - MAJ_dev(a15, a14, a13) + a12;

        u32 c_e13 = a13 - S0_dev(a12) - a11;
        u32 c_e12 = e16 - a12 - 0xffffffffu - 0xc19bf174u - (u32)PADLEN_BITS; // K[15]=0xc19bf174
        u32 c_e11 = e15 - a11 - S1_dev(e14) - 0x9bdc06a7u; // K[14]=0x9bdc06a7
        u32 c_W13 = e14 - a10 - 0xffffffffu - 0x80deb1feu; // K[13]=0x80deb1fe

        u32 c_a8 = S0_dev(a11) - a12;
        u32 c_a7 = S0_dev(a10) - a11;
        u32 c_a6 = 0xffffffffu - a10;

        u32 e12 = c_e12 - e14;
        u32 a8  = e12 + c_a8 + a11;
        u32 c_a5_s0a8  = 0xffffffffu + S0_dev(a8);
        u32 m_10_8     = a10 ^ a8;
        u32 a10_and_a8 = a10 & a8;
        u32 c_e11_base = c_e11 - (~e14 & e12);
        u32 c_a7_full  = c_e11_base + c_a7 + a10_and_a8;

        for(u32 base_lo = 0; base_lo < 256u; base_lo += 32u){
            u32 my_e13_lo = base_lo + lane;
            u32 s0 = my_e13_lo & 0xfu;
            u32 s4 = my_e13_lo & 0xf0u;

            u32 a9_0a  = s0 - c_e13;
            u32 X_0a   = c_a5_s0a8 - a9_0a;
            u32 a7_b0a = c_a7_full - (e14 & s0);
            u32 a7_0a  = a7_b0a + (m_10_8 & a9_0a);
            u32 D_0a   = a8 ^ a7_0a;
            int ok4    = ((((X_0a + CH_dev(D_0a, X_0a, a8)) ^ X_0a) & D_0a & 0xfu) == 0u);

            u32 a9_0  = a9_0a + s4;
            u32 X_0   = X_0a  - s4;
            u32 a7_b0 = a7_b0a - (e14 & s4);
            u32 a7_0  = a7_b0 + (m_10_8 & a9_0);
            u32 D_0   = a8 ^ a7_0;
            int ok8   = ok4 && ((((X_0 + CH_dev(D_0, X_0, a8)) ^ X_0) & D_0 & 0xffu) == 0u);

            u32 mask = __ballot_sync(0xffffffffu, ok8);
            while(mask){
                int src = __ffs(mask) - 1;
                mask &= mask - 1u;

                u32 e13_lo  = __shfl_sync(0xffffffffu, my_e13_lo, src);
                u32 a9_0_w  = __shfl_sync(0xffffffffu, a9_0, src);
                u32 X_0_w   = __shfl_sync(0xffffffffu, X_0, src);
                u32 a7_b0_w = __shfl_sync(0xffffffffu, a7_b0, src);

                u32 e11_lo = (c_e11_base - (e14 & e13_lo)) & 0xffu;
                u32 req_S1_lo = (c_W13 - CH_dev(e13_lo, e12, e11_lo) - 0x80u) & 0xffu;
                u32 req_b7_xor_lo0 = ((req_S1_lo >> 7) ^ e13_lo) & 1u;

                // Stage 1: All 32 lanes cooperatively evaluate the 128 (s8 x s12) combinations in 4 lockstep steps
                for(u32 k128 = 0; k128 < 128u; k128 += 32u){
                    u32 idx128  = k128 + lane;
                    u32 s8      = (idx128 & 15u) << 8;
                    u32 s12_idx = idx128 >> 4;

                    u32 a9_1  = a9_0_w + s8;
                    u32 X_1   = X_0_w  - s8;
                    u32 a7_b1 = a7_b0_w - (e14 & s8);
                    u32 a7_1  = a7_b1 + (m_10_8 & a9_1);
                    u32 D_1   = a8 ^ a7_1;
                    int ok12  = ((((X_1 + CH_dev(D_1, X_1, a8)) ^ X_1) & D_1 & 0xfffu) == 0u);

                    u32 s12   = s12_idx << 12;
                    u32 b18   = ((req_b7_xor_lo0 ^ (s12 >> 13)) & 1u) << 18;
                    u32 s12_b = s12 | b18;
                    u32 a9_2  = a9_1 + s12_b;
                    u32 X_2   = X_1  - s12_b;
                    u32 a7_b2 = a7_b1 - (e14 & s12_b);
                    u32 a7_2  = a7_b2 + (m_10_8 & a9_2);
                    u32 D_2   = a8 ^ a7_2;
                    int ok15  = ok12 && ((((X_2 + CH_dev(D_2, X_2, a8)) ^ X_2) & D_2 & 0x7fffu) == 0u);
                    u32 e13_14_lane = e13_lo | s8 | s12;

                    // Stage 2: Broadcast each 15-bit survivor so all 32 lanes evaluate 64 (s15 x s19) pairs in 2 steps
                    u32 mask15 = __ballot_sync(0xffffffffu, ok15);
                    while(mask15){
                        int src15 = __ffs(mask15) - 1;
                        mask15 &= mask15 - 1u;

                        u32 a9_2_w   = __shfl_sync(0xffffffffu, a9_2, src15);
                        u32 X_2_w    = __shfl_sync(0xffffffffu, X_2, src15);
                        u32 a7_b2_w  = __shfl_sync(0xffffffffu, a7_b2, src15);
                        u32 e13_14_w = __shfl_sync(0xffffffffu, e13_14_lane, src15);

                        for(u32 k64 = 0; k64 < 64u; k64 += 32u){
                            u32 idx64 = k64 + lane;
                            u32 s15   = (idx64 & 7u) << 15;
                            u32 s19   = (idx64 >> 3) << 19;

                            u32 a9_5  = a9_2_w + s15;
                            u32 X_5   = X_2_w  - s15;
                            u32 a7_b5 = a7_b2_w - (e14 & s15);
                            u32 a7_5  = a7_b5 + (m_10_8 & a9_5);
                            u32 D_5   = a8 ^ a7_5;
                            u32 S_5   = X_5 + CH_dev(D_5, X_5, a8);
                            if(((S_5 ^ X_5) & D_5 & 0x7ffffu) != 0u) continue;

                            u32 e13_base = e13_14_w | s15;
                            u32 diff     = (req_S1_lo ^ S1_dev(e13_base)) & 0x7fu;
                            u32 hi25     = diff << 25;
                            u32 a9_base  = a9_5 + hi25;
                            u32 X0       = X_5  - hi25;
                            u32 a7_base  = a7_b5 - (e14 & hi25);

                            u32 a9_i1  = a9_base + s19;
                            u32 X_i1   = X0 - s19;
                            u32 a7_bi1 = a7_base - (e14 & s19);
                            u32 a7_i1  = a7_bi1 + (m_10_8 & a9_i1);
                            u32 D_i1   = a8 ^ a7_i1;
                            if((((X_i1 + CH_dev(D_i1, X_i1, a8)) ^ X_i1) & D_i1 & 0x3fffffu) != 0u) continue;

                            for(u32 s22 = 0; s22 < (8u << 22); s22 += (1u << 22)){
                                u32 a9 = a9_i1 + s22;
                                u32 X  = X_i1  - s22;
                                u32 a7 = a7_bi1 - (e14 & s22) + (m_10_8 & a9);
                                u32 D  = a8 ^ a7;
                                u32 S  = X + CH_dev(D, X, a8);
                                if(((S ^ X) & D) != 0u) continue;
                                u32 a6 = c_a6 + S0_dev(a9) + MAJ_dev(a9, a8, a7);
                                if(((a6 ^ S) & ~D) == 0u){
                                    int idx = atomicAdd(d_ctx_count, 1);
                                    if(idx < max_cap){
                                        d_out_a12[idx] = a12;
                                        d_out_a9[idx]  = a9;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if(lane == 0 && local_a12_tried){
        atomicAdd(d_trials, local_a12_tried << 24);
    }
}

static u32 *d_gen_a12 = NULL;
static u32 *d_gen_a9  = NULL;
static u32 *d_gen_T16 = NULL;
static int *d_gen_cnt = NULL;
static unsigned long long *d_gen_a12_counter = NULL;
static unsigned long long *d_gen_trials = NULL;
static u32 *h_gen_a12 = NULL;
static u32 *h_gen_a9  = NULL;
static u32 *h_gen_T16 = NULL;
static u32 g_a12_perm_key = 0;
static unsigned long long g_a12_ctr_14[14] = {0};

__global__ void __launch_bounds__(256, 4)
ctx_gen_fast_cuda_kernel(
    u32 a13, u32 a14, u32 a15, u32 a16, u32 e17,
    u32 a12_perm_key,
    int target_count,
    int max_cap,
    u32 * __restrict__ d_out_a12,
    u32 * __restrict__ d_out_a9,
    u32 * __restrict__ d_out_T16,
    int * __restrict__ d_ctx_count,
    unsigned long long * __restrict__ d_a12_counter,
    unsigned long long * __restrict__ d_trials);

static int cmp_compact_ctx(const void *a, const void *b){
    const compact_ctx_t *ca = (const compact_ctx_t *)a;
    const compact_ctx_t *cb = (const compact_ctx_t *)b;
    if(ca->T16 != cb->T16) return (ca->T16 > cb->T16) - (ca->T16 < cb->T16);
    if(ca->target_id != cb->target_id) return (ca->target_id > cb->target_id) - (ca->target_id < cb->target_id);
    if(ca->a12 != cb->a12) return (ca->a12 > cb->a12) - (ca->a12 < cb->a12);
    return (ca->a9 > cb->a9) - (ca->a9 < cb->a9);
}

static u64 generate_right_contexts_parallel(int nthreads){
    (void)nthreads;
    int B = G_STATE.batch_ctx;
    char cache_path[128];
    snprintf(cache_path, sizeof(cache_path), "/tmp/ctx_pool_%d_%d_%u.bin",
             g_all_14_mode ? g_target_first : -1,
             g_all_14_mode ? g_target_last  : -1,
             (u32)g_padlen_bits);

    // Load cached reservoir on first call if available
    if(g_ctx_pool_len == 0){
        FILE *fp = fopen(cache_path, "rb");
        if(fp){
            int saved_len = 0;
            u32 saved_key = 0;
            unsigned long long saved_ctr[14] = {0};
            if(fread(&saved_len, sizeof(int), 1, fp) == 1 &&
               saved_len > 0 && saved_len <= CTX_POOL_CAP &&
               fread(&saved_key, sizeof(u32), 1, fp) == 1 &&
               fread(saved_ctr, sizeof(saved_ctr), 1, fp) == 1 &&
               (int)fread(g_ctx_pool, sizeof(compact_ctx_t), (size_t)saved_len, fp) == saved_len){
                g_ctx_pool_len = saved_len;
                g_a12_perm_key = saved_key;
                memcpy(g_a12_ctr_14, saved_ctr, sizeof(saved_ctr));
                fprintf(stderr, "[ctx-cache] Loaded %d cached contexts from %s\n", g_ctx_pool_len, cache_path);
            }
            fclose(fp);
        }
    }

    // 1. Compact out used contexts or contexts belonging to newly solved or out-of-range targets.
    //    Since g_ctx_pool is already sorted by T16, linear compaction preserves T16 sort order!
    int w = 0;
    for(int i = 0; i < g_ctx_pool_len; i++){
        if(g_ctx_pool[i].used) continue;
        int tid = (int)g_ctx_pool[i].target_id;
        if(g_all_14_mode && (g_target14_solved[tid] || tid < g_target_first || tid > g_target_last)) continue;
        g_ctx_pool[w++] = g_ctx_pool[i];
    }
    g_ctx_pool_len = w;

    // 2. Count available Exact-T16 collision pairs in O(N) without sorting
    int exact_pairs_avail = 0;
    for(int i = 0; i + 1 < g_ctx_pool_len; i++){
        if(g_ctx_pool[i].T16 == g_ctx_pool[i+1].T16){
            exact_pairs_avail++;
            i++;
        }
    }

    int gen_target = 0;
    if(B == 1){
        gen_target = (g_ctx_pool_len >= 1) ? 0 : 4096;
    } else if(g_ctx_pool_len < B * 64){
        // Only generate a new 3M batch if the pool of valid contexts for our target range runs low.
        // We do NOT require exact_pairs_avail >= B/2 once a pool is populated, because across all 2^32
        // a12 values there are only ~187 sweeps of Exact-T16 twins per target, whereas there are ~79,000
        // sweeps (10.14M) of valid non-twin contexts in the same pool!
        if(g_ctx_pool_len > CTX_POOL_CAP - CTX_BATCH_ADD - 65536){
            int keep = CTX_POOL_CAP / 2;
            memmove(&g_ctx_pool[0], &g_ctx_pool[g_ctx_pool_len - keep], (size_t)keep * sizeof(compact_ctx_t));
            g_ctx_pool_len = keep;
        }
        gen_target = CTX_BATCH_ADD;
    }

    unsigned long long h_trials = 0;
    if(gen_target > 0){
        CUDA_CHECK(cudaSetDevice(0));
        const int gpu_buf_cap = CTX_BATCH_ADD + 65536;
        if(!d_gen_a12){
            CUDA_CHECK(cudaMalloc(&d_gen_a12, (size_t)gpu_buf_cap * sizeof(u32)));
            CUDA_CHECK(cudaMalloc(&d_gen_a9,  (size_t)gpu_buf_cap * sizeof(u32)));
            CUDA_CHECK(cudaMalloc(&d_gen_T16, (size_t)gpu_buf_cap * sizeof(u32)));
            CUDA_CHECK(cudaMalloc(&d_gen_cnt, sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_gen_a12_counter, sizeof(unsigned long long)));
            CUDA_CHECK(cudaMalloc(&d_gen_trials, sizeof(unsigned long long)));
            u64 s = __atomic_load_n(&g_ctx_seed_gen, __ATOMIC_RELAXED);
            g_a12_perm_key = (u32)(s ^ (s >> 32));
            h_gen_a12 = (u32 *)malloc((size_t)gpu_buf_cap * sizeof(u32));
            h_gen_a9  = (u32 *)malloc((size_t)gpu_buf_cap * sizeof(u32));
            h_gen_T16 = (u32 *)malloc((size_t)gpu_buf_cap * sizeof(u32));
        }

        int active_tids[14];
        int n_active = 0;
        if(g_all_14_mode){
            for(int t = g_target_first; t <= g_target_last; t++){
                if(!g_target14_solved[t] && g_a12_ctr_14[t] < (1ULL << 32)) active_tids[n_active++] = t;
            }
            if(n_active == 0){
                active_tids[n_active++] = g_target_first;
            }
        } else {
            active_tids[n_active++] = 0;
        }

        int per_target = (gen_target + n_active - 1) / n_active;
        for(int k = 0; k < n_active && g_ctx_pool_len < CTX_POOL_CAP - 1024; k++){
            int tid = active_tids[k];
            const solver_state_t *st_ptr = g_all_14_mode ? &G_STATES_14[tid] : &G_STATE;
            int room = CTX_POOL_CAP - g_ctx_pool_len;
            int this_target = (per_target < room) ? per_target : room;
            if(this_target > gpu_buf_cap) this_target = gpu_buf_cap;
            if(this_target <= 0) break;

            int zero_cnt = 0;
            unsigned long long zero_tr = 0;
            CUDA_CHECK(cudaMemcpy(d_gen_cnt, &zero_cnt, sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_gen_trials, &zero_tr, sizeof(unsigned long long), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_gen_a12_counter, &g_a12_ctr_14[tid], sizeof(unsigned long long), cudaMemcpyHostToDevice));

            const u32 a13 = st_ptr->A_template[13+OFF];
            const u32 a14 = st_ptr->A_template[14+OFF];
            const u32 a15 = st_ptr->A_template[15+OFF];
            const u32 a16 = st_ptr->A_template[16+OFF];
            const u32 e17 = E_arr(st_ptr->A_template, 17);

            ctx_gen_fast_cuda_kernel<<<232, 256>>>(
                a13, a14, a15, a16, e17, g_a12_perm_key ^ ((u32)tid * 0x9e3779b9u),
                this_target, gpu_buf_cap,
                d_gen_a12, d_gen_a9, d_gen_T16, d_gen_cnt, d_gen_a12_counter, d_gen_trials);
            CUDA_CHECK(cudaDeviceSynchronize());

            int h_cnt = 0;
            unsigned long long sub_tr = 0;
            CUDA_CHECK(cudaMemcpy(&h_cnt, d_gen_cnt, sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&sub_tr, d_gen_trials, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&g_a12_ctr_14[tid], d_gen_a12_counter, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            h_trials += sub_tr;
            if(h_cnt > this_target) h_cnt = this_target;

            if(h_cnt > 0){
                CUDA_CHECK(cudaMemcpy(h_gen_a12, d_gen_a12, (size_t)h_cnt * sizeof(u32), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_gen_a9,  d_gen_a9,  (size_t)h_cnt * sizeof(u32), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_gen_T16, d_gen_T16, (size_t)h_cnt * sizeof(u32), cudaMemcpyDeviceToHost));
                for(int i = 0; i < h_cnt; i++){
                    compact_ctx_t *cc = &g_ctx_pool[g_ctx_pool_len + i];
                    cc->T16       = h_gen_T16[i];
                    cc->a12       = h_gen_a12[i];
                    cc->a9        = h_gen_a9[i];
                    cc->target_id = (short)tid;
                    cc->used      = 0;
                    cc->pad       = 0;
                }
                g_ctx_pool_len += h_cnt;
            }
        }

        // Sort entire pool once by (T16, target_id, a12, a9) and deduplicate in-place:
        qsort(g_ctx_pool, (size_t)g_ctx_pool_len, sizeof(compact_ctx_t), cmp_compact_ctx);
        int u_len = 0;
        for(int i = 0; i < g_ctx_pool_len; i++){
            if(u_len > 0 &&
               g_ctx_pool[i].target_id == g_ctx_pool[u_len - 1].target_id &&
               g_ctx_pool[i].a12       == g_ctx_pool[u_len - 1].a12 &&
               g_ctx_pool[i].a9        == g_ctx_pool[u_len - 1].a9){
                continue;
            }
            g_ctx_pool[u_len++] = g_ctx_pool[i];
        }
        g_ctx_pool_len = u_len;

        exact_pairs_avail = 0;
        for(int i = 0; i + 1 < g_ctx_pool_len; i++){
            if(g_ctx_pool[i].T16 == g_ctx_pool[i+1].T16){
                exact_pairs_avail++;
                i++;
            }
        }
    }

    int chosen = 0;
    if(g_ctx_pool_len > 0){
        u64 rng_st = __atomic_fetch_add(&g_ctx_seed_gen, 0x9E3779B97F4A7C15ULL, __ATOMIC_RELAXED);
        for(int attempt = 0; chosen < B && attempt < g_ctx_pool_len * 2; attempt++){
            int i = (int)(splitmix64(&rng_st) % (u64)g_ctx_pool_len);
            if(!g_ctx_pool[i].used){
                int tid = (int)g_ctx_pool[i].target_id;
                const solver_state_t *st_ptr = g_all_14_mode ? &G_STATES_14[tid] : &G_STATE;
                reconstruct_right_ctx_for_state(st_ptr, tid, g_ctx_pool[i].a12, g_ctx_pool[i].a9, &G_STATE.rctx[chosen++]);
                g_ctx_pool[i].used = 1;
            }
        }
    }
    g_exact_pairs_avail = exact_pairs_avail;

    // Persist reservoir state to /tmp when new contexts are generated so restarts take 0.01s
    if(gen_target > 0){
        char tmp_path[144];
        snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", cache_path);
        FILE *fp = fopen(tmp_path, "wb");
        if(fp){
            fwrite(&g_ctx_pool_len, sizeof(int), 1, fp);
            fwrite(&g_a12_perm_key, sizeof(u32), 1, fp);
            fwrite(g_a12_ctr_14, sizeof(g_a12_ctr_14), 1, fp);
            fwrite(g_ctx_pool, sizeof(compact_ctx_t), (size_t)g_ctx_pool_len, fp);
            fclose(fp);
            rename(tmp_path, cache_path);
        }
    }
    return (u64)h_trials;
}

// Zero-table closed-form GF(2) inverse for upper 16 bits of u from (ulo, y):
// Replaces 128 KB d_REC_HI16 global/L1/L2 memory lookups with ~19 bitwise/ALU register operations (5 logical stages)!
__device__ __forceinline__ u32 reconstruct_u_dev(u32 ulo, u32 y){
    u32 v   = (ulo + y) ^ s0_dev(ulo);
    u32 b0  = ((v << 15) ^ (v << 11) ^ v) & 0x00010000u;
    u32 b12 = ((v << 14) ^ (v << 3)) & 0x00060000u;
    u32 T   = b0 | b12 | ((v << 18) & 0xfff80000u);
    return ulo | (T ^ (T << 11) ^ (T << 15));
}

__device__ __forceinline__ size_t slide3_off_dev(u32 y){
    u32 d = rotr_dev(0x6AAA6AAAu, y & 31u) >> 31;
    return (size_t)((y >> 1) + (y >> 2) - d);
}

__device__ __forceinline__ void check_sol19_dev(
    u32 W3, u32 G3, u32 a1, u32 e1, u32 a2, u32 e2, u32 a3, u32 e3, u32 sm3,
    u32 a0, u32 a5, u32 T19, int ctx_id,
    u32 *local_sol19, gpu_stats_t *d_stats)
{
    (*local_sol19)++;
    u32 a4  = W3 + G3;
    u32 e4  = a4 - sm3 + a0;
    u32 sm4 = S0_dev(a4) + MAJ_dev(a4, a3, a2);
    u32 W4  = a5 - sm4 - e1 - S1_dev(e4) - CH_dev(e4, e3, e2) - 0x3956c25bu; // K[4]
    if(s0_dev(W4) + W3 == T19){
        if(atomicCAS(&d_stats->solved20, 0, 1) == 0){
            d_stats->sol_ctx_id = ctx_id;
            d_stats->sol_a1 = a1;
            d_stats->sol_a2 = a2;
            d_stats->sol_a3 = a3;
            d_stats->sol_a4 = a4;
        }
    }
}

__global__ void __launch_bounds__(256, 4)
sweep_6gb_cuda_kernel(
    const u16 * __restrict__ d_H_pack6,
    const u16 * __restrict__ d_REC_HI16,
    const gpu_params_t p,
    u64 start_a1,
    u64 count_a1,
    gpu_stats_t * __restrict__ d_stats)
{
    (void)d_REC_HI16;
    u64 tid = (u64)blockIdx.x * (u64)blockDim.x + (u64)threadIdx.x;
    u32 local_l2 = 0, local_l3 = 0, local_sol19 = 0;

    if(tid < count_a1){
        const u32 a0 = p.a0, am1 = p.am1, am2 = p.am2;
        const u32 e0 = p.e0, em1 = p.em1;
        const u32 C_e1 = p.C_e1, C_W0 = p.C_W0;
        const u32 C_G1 = p.C_G1, C_G2 = p.C_G2, C_G3 = p.C_G3;
        const int w_thresh = p.w_thresh;

        const u32 a1  = (u32)(start_a1 + tid);
        const u32 e1  = a1 + C_e1;
        const u32 sm1 = S0_dev(a1) + MAJ_dev(a1, a0, am1);
        const u32 G1  = sm1 + S1_dev(e1) + CH_dev(e1, e0, em1) + C_G1;
        const u32 f1  = G1 - a1 - C_W0;

        const u64 K7 = 0x7fff7fff7fff7fffULL;
        const u64 K8 = 0x8000800080008000ULL;
        const u64 K1 = 0x0001000100010001ULL;

        for(int pair_base = 0; pair_base < p.batch_ctx; pair_base += 2){
            const int c0 = pair_base;
            const int c1 = (pair_base + 1 < p.batch_ctx) ? (pair_base + 1) : c0;
            const u32 T16_0 = p.T16[c0];
            const u32 T16_1 = p.T16[c1];
            const u32 t16_0 = T16_0 + f1;
            const u32 t16_1 = T16_1 + f1;

            // Upfront Same-Sector Pair Coalescing with Same-Window Elision:
            const size_t off1_0 = slide3_off_dev(t16_0);
            const size_t off1_1 = slide3_off_dev(t16_1);
            const u16 *b1_0 = &d_H_pack6[off1_0];
            u32 u00 = __ldg(&b1_0[0]);
            u32 u01 = __ldg(&b1_0[1]);
            u32 u02 = __ldg(&b1_0[2]);
            u32 pair_pack01_0 = u00 | (u01 << 16);
            u32 pair_pack01_1 = pair_pack01_0;
            u32 u12 = u02;
            if(off1_1 != off1_0){
                const u16 *b1_1 = &d_H_pack6[off1_1];
                u32 u10 = __ldg(&b1_1[0]);
                u32 u11 = __ldg(&b1_1[1]);
                u12     = __ldg(&b1_1[2]);
                pair_pack01_1 = u10 | (u11 << 16);
            }
            asm volatile("" : "+r"(pair_pack01_0), "+r"(u02), "+r"(pair_pack01_1), "+r"(u12));

            u32 c1_0 = 0, c1_1 = 0, c1_2 = 0, c1_3 = 0;
            u32 mask1_saved = 0;

            for(int sub = 0; sub < 2; sub++){
                int c = pair_base + sub;
                if(c >= p.batch_ctx) break;

                const u32 a5  = p.a5[c];
                const u32 T16 = p.T16[c];
                const u32 T17 = p.T17[c];
                const u32 T18 = p.T18[c];
                const u32 T19 = p.T19[c];
                const u64 A64 = (u64)(a5 & 0x7fffu) * K1;
                const u32 t16 = T16 + f1;

                u32 cur_pack01 = pair_pack01_0;
                u32 cur_u2     = u02;
                pair_pack01_0  = pair_pack01_1;
                u02            = u12;

                // Reuse Level-1 candidates directly if sub == 1 is an Exact-T16 Twin (T16_1 == T16_0)
                if(sub == 0 || T16 != T16_0){
                    u32 ulo1_0 = cur_pack01 & 0xffffu;
                    u32 ulo1_1 = cur_pack01 >> 16;
                    u32 ulo1_2 = cur_u2;

                    u32 m1 = 0;
                    if(ulo1_0){
                        c1_0 = reconstruct_u_dev(ulo1_0, t16);
                        if(Qf_dev(c1_0) == t16) m1 |= 1u;
                    }
                    if(ulo1_1){
                        c1_1 = reconstruct_u_dev(ulo1_1, t16);
                        if(Qf_dev(c1_1) == t16) m1 |= 2u;
                    }
                    if(ulo1_2){
                        c1_2 = reconstruct_u_dev(ulo1_2, t16);
                        if(Qf_dev(c1_2) == t16) m1 |= 4u;
                    }
                    if(m1 == 7u){
                        const u16 *b1 = &d_H_pack6[slide3_off_dev(t16)];
                        u32 ulo1_3 = __ldg(&b1[(t16 & 15u) >= 13u ? -1 : 3]);
                        if(ulo1_3){
                            c1_3 = reconstruct_u_dev(ulo1_3, t16);
                            if(Qf_dev(c1_3) == t16) m1 |= 8u;
                        }
                    }
                    mask1_saved = m1;
                }

                u32 mask1 = mask1_saved;
                while(mask1){
                    int bit1 = __ffs(mask1) - 1;
                    mask1 &= mask1 - 1u;
                    u32 W1  = (bit1 == 0) ? c1_0 : ((bit1 == 1) ? c1_1 : ((bit1 == 2) ? c1_2 : c1_3));
                    u32 a2  = W1 + G1;
                    u32 e2  = a2 - sm1 + am2;
                    u32 sm2 = S0_dev(a2) + MAJ_dev(a2, a1, a0);
                    u32 G2  = sm2 + S1_dev(e2) + CH_dev(e2, e1, e0) + C_G2;
                    u32 t17 = T17 + G2 - W1;

                    // Level 2: 2-Tier Nested 4-Slot Lookup for y = t17 + immediate popcount(M3) filter
                    const u16 *b2 = &d_H_pack6[slide3_off_dev(t17)];
                    u32 ulo2_0 = __ldg(&b2[0]);
                    u32 ulo2_1 = __ldg(&b2[1]);
                    u32 ulo2_2 = __ldg(&b2[2]);

                    u32 c2_0 = 0, c2_1 = 0, c2_2 = 0, c2_3 = 0;
                    u32 mask2 = 0;
                    int n2_raw = 0;
                    if(ulo2_0){
                        c2_0 = reconstruct_u_dev(ulo2_0, t17);
                        if(Qf_dev(c2_0) == t17){
                            n2_raw++;
                            if(__popc(~((c2_0 + G2) ^ a5)) <= w_thresh) mask2 |= 1u;
                        }
                    }
                    if(ulo2_1){
                        c2_1 = reconstruct_u_dev(ulo2_1, t17);
                        if(Qf_dev(c2_1) == t17){
                            n2_raw++;
                            if(__popc(~((c2_1 + G2) ^ a5)) <= w_thresh) mask2 |= 2u;
                        }
                    }
                    if(ulo2_2){
                        c2_2 = reconstruct_u_dev(ulo2_2, t17);
                        if(Qf_dev(c2_2) == t17){
                            n2_raw++;
                            if(__popc(~((c2_2 + G2) ^ a5)) <= w_thresh) mask2 |= 4u;
                        }
                    }
                    if(n2_raw == 3){
                        u32 ulo2_3 = __ldg(&b2[(t17 & 15u) >= 13u ? -1 : 3]);
                        if(ulo2_3){
                            c2_3 = reconstruct_u_dev(ulo2_3, t17);
                            if(Qf_dev(c2_3) == t17){
                                if(__popc(~((c2_3 + G2) ^ a5)) <= w_thresh) mask2 |= 8u;
                            }
                        }
                    }

                    while(mask2){
                        int bit2 = __ffs(mask2) - 1;
                        mask2 &= mask2 - 1u;
                        u32 W2 = (bit2 == 0) ? c2_0 : ((bit2 == 1) ? c2_1 : ((bit2 == 2) ? c2_2 : c2_3));
                        u32 a3 = W2 + G2;
                        u32 M3 = ~(a3 ^ a5);

                        local_l2++;
                        local_l3++;
                        u32 e3  = a3 - sm2 + am1;
                        u32 sm3 = S0_dev(a3) + MAJ_dev(a3, a2, a1);
                        u32 G3  = sm3 + S1_dev(e3) + CH_dev(e3, e2, e1) + C_G3;
                        u32 t18 = T18 + G3 - W2;

                        // Level 3: 12-Slot Aligned Quad 64-bit SWAR Filter
                        size_t qoff = (size_t)(t18 >> 4) * 12u;
                        const unsigned long long *q64 = (const unsigned long long *)&d_H_pack6[qoff];
                        u64 w0 = __ldg(&q64[0]);
                        u64 w1 = __ldg(&q64[1]);
                        u64 w2 = __ldg(&q64[2]);
                        u64 g64 = (u64)(G3 & 0x7fffu) * K1;
                        u64 m64 = (u64)(M3 & 0x7fffu) * K1;
                        u64 d0 = ((((w0 & K7) + g64) ^ A64) & m64) + K7;
                        u64 d1 = ((((w1 & K7) + g64) ^ A64) & m64) + K7;
                        u64 d2 = ((((w2 & K7) + g64) ^ A64) & m64) + K7;

                        if(((d0 & d1 & d2) & K8) != K8){
                            u32 M3_lo = M3 & 0xffffu;
                            u64 h0 = ~d0 & K8;
                            while(h0){
                                int s = (__ffsll((long long)h0) - 1) >> 4;
                                h0 &= h0 - 1ULL;
                                u32 ulo = (u32)((w0 >> (s * 16)) & 0xffffULL);
                                if(ulo && (((ulo + G3) ^ a5) & M3_lo) == 0u){
                                    u32 u = reconstruct_u_dev(ulo, t18);
                                    if((((u + G3) ^ a5) & M3) == 0u && Qf_dev(u) == t18){
                                        check_sol19_dev(u, G3, a1, e1, a2, e2, a3, e3, sm3, a0, a5, T19, c, &local_sol19, d_stats);
                                    }
                                }
                            }
                            u64 h1 = ~d1 & K8;
                            while(h1){
                                int s = (__ffsll((long long)h1) - 1) >> 4;
                                h1 &= h1 - 1ULL;
                                u32 ulo = (u32)((w1 >> (s * 16)) & 0xffffULL);
                                if(ulo && (((ulo + G3) ^ a5) & M3_lo) == 0u){
                                    u32 u = reconstruct_u_dev(ulo, t18);
                                    if((((u + G3) ^ a5) & M3) == 0u && Qf_dev(u) == t18){
                                        check_sol19_dev(u, G3, a1, e1, a2, e2, a3, e3, sm3, a0, a5, T19, c, &local_sol19, d_stats);
                                    }
                                }
                            }
                            u64 h2 = ~d2 & K8;
                            while(h2){
                                int s = (__ffsll((long long)h2) - 1) >> 4;
                                h2 &= h2 - 1ULL;
                                u32 ulo = (u32)((w2 >> (s * 16)) & 0xffffULL);
                                if(ulo && (((ulo + G3) ^ a5) & M3_lo) == 0u){
                                    u32 u = reconstruct_u_dev(ulo, t18);
                                    if((((u + G3) ^ a5) & M3) == 0u && Qf_dev(u) == t18){
                                        check_sol19_dev(u, G3, a1, e1, a2, e2, a3, e3, sm3, a0, a5, T19, c, &local_sol19, d_stats);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Warp-level aggregation before atomicAdd to global stats
    unsigned int mask = 0xffffffffu;
    for(int offset = 16; offset > 0; offset >>= 1){
        local_l2    += __shfl_down_sync(mask, local_l2, offset);
        local_l3    += __shfl_down_sync(mask, local_l3, offset);
        local_sol19 += __shfl_down_sync(mask, local_sol19, offset);
    }
    if((threadIdx.x & 31) == 0){
        if(local_l2)    atomicAdd(&d_stats->l2_nodes, (unsigned long long)local_l2);
        if(local_l3)    atomicAdd(&d_stats->l3_lookups, (unsigned long long)local_l3);
        if(local_sol19) atomicAdd(&d_stats->sol19_count, (unsigned long long)local_sol19);
    }
}

// ============================================================================
//  SUDOKU ROUND-20 EXTENSIONS (IDEAS 1, 2, 3, 4 & COMBINED L4 KERNEL)
// ============================================================================

// --- IDEA 2: Exact GF(2)^32 Inverse of s0(x) = ROR(x,7) ^ ROR(x,18) ^ (x>>3) ---
__constant__ u32 d_inv_s0_lut[4][256];
static u32 h_inv_s0_lut[4][256];

static void init_inv_s0_lut(void){
    // Build 32x32 GF(2) matrix for s0 and invert via Gauss-Jordan
    u32 mat[32], inv[32];
    for(int j = 0; j < 32; j++){
        u32 x = 1u << j;
        mat[j] = s0(x); // column j is s0(1 << j)
        inv[j] = 1u << j;
    }
    // Transpose so row[i] is equation for output bit i
    u32 row[32], rinv[32];
    for(int i = 0; i < 32; i++){
        u32 m = 0, iv = 0;
        for(int j = 0; j < 32; j++){
            if((mat[j] >> i) & 1u) m |= (1u << j);
            if((inv[j] >> i) & 1u) iv |= (1u << j);
        }
        row[i] = m;
        rinv[i] = iv;
    }
    for(int col = 0; col < 32; col++){
        int piv = -1;
        for(int r = col; r < 32; r++){
            if((row[r] >> col) & 1u){ piv = r; break; }
        }
        if(piv != col){
            u32 t = row[col]; row[col] = row[piv]; row[piv] = t;
            t = rinv[col]; rinv[col] = rinv[piv]; rinv[piv] = t;
        }
        for(int r = 0; r < 32; r++){
            if(r != col && ((row[r] >> col) & 1u)){
                row[r]  ^= row[col];
                rinv[r] ^= rinv[col];
            }
        }
    }
    // Precompute 4 byte-slice LUTs: inv_s0(y) = XOR_{b=0..3} lut[b][(y >> (8*b)) & 0xff]
    for(int b = 0; b < 4; b++){
        for(int v = 0; v < 256; v++){
            u32 y = (u32)v << (8 * b);
            u32 x = 0;
            for(int i = 0; i < 32; i++){
                u32 parity = __builtin_parity(rinv[i] & y);
                x |= (parity << i);
            }
            h_inv_s0_lut[b][v] = x;
        }
    }
}

__device__ __forceinline__ u32 inv_s0_dev(u32 y){
    return d_inv_s0_lut[0][y & 0xffu]
         ^ d_inv_s0_lut[1][(y >> 8) & 0xffu]
         ^ d_inv_s0_lut[2][(y >> 16) & 0xffu]
         ^ d_inv_s0_lut[3][y >> 24];
}

__device__ __forceinline__ void check_sol19_idea2_dev(
    u32 W3, u32 G3, u32 a1, u32 e1, u32 a2, u32 e2, u32 a3, u32 e3, u32 sm3,
    u32 a0, u32 a5, u32 T19, int ctx_id,
    const u32 * __restrict__ d_multi_t19_bitmap,
    u32 *local_sol19, gpu_stats_t *d_stats)
{
    (*local_sol19)++;
    // Idea 2a: Exact GF(2)^32 algebraic inversion W4_req = s0^-1(T19 - W3)
    u32 W4_req = inv_s0_dev(T19 - W3);
    u32 a4  = W3 + G3;
    u32 e4  = a4 - sm3 + a0;
    u32 sm4 = S0_dev(a4) + MAJ_dev(a4, a3, a2);
    u32 W4  = a5 - sm4 - e1 - S1_dev(e4) - CH_dev(e4, e3, e2) - 0x3956c25bu; // K[4]
    if(W4 == W4_req){
        if(atomicCAS(&d_stats->solved20, 0, 1) == 0){
            d_stats->sol_ctx_id = ctx_id;
            d_stats->sol_a1 = a1;
            d_stats->sol_a2 = a2;
            d_stats->sol_a3 = a3;
            d_stats->sol_a4 = a4;
        }
    } else if(d_stats->sol_a1 == 0u && d_stats->solved20 == 0){
        if(atomicCAS(&d_stats->sol_a1, 0u, a1) == 0u){
            d_stats->sol_ctx_id = ctx_id;
            d_stats->sol_a2 = a2;
            d_stats->sol_a3 = a3;
            d_stats->sol_a4 = a4;
        }
    }
    // Idea 2b: Optional O(1) Multi-Target T19 match (e.g., 2^16 Block-2 targets varying in a20)
    if(d_multi_t19_bitmap){
        u32 req_T19 = s0_dev(W4) + W3;
        u32 word_idx = req_T19 >> 5;
        u32 bit_mask = 1u << (req_T19 & 31u);
        if((__ldg(&d_multi_t19_bitmap[word_idx & 0x7ffffu]) & bit_mask) != 0u && ((req_T19 >> 24) == (T19 >> 24))){
            atomicAdd((unsigned int *)&d_stats->solved20, 1u);
        }
    }
}

// --- IDEA 1: Fused Exact-T16 Collision Pair Kernel (Shares Level-1 & W1->a2->e2->sm2->G2) ---
__global__ void __launch_bounds__(256, 4)
sweep_idea1_fused_6gb_kernel(
    const u16 * __restrict__ d_H_pack6,
    const gpu_params_t p,
    const u32 * __restrict__ d_multi_t19_bitmap,
    u64 start_a1,
    u64 count_a1,
    gpu_stats_t * __restrict__ d_stats)
{
    u64 tid = (u64)blockIdx.x * (u64)blockDim.x + (u64)threadIdx.x;
    u32 local_l2 = 0, local_l3 = 0, local_sol19 = 0;

    if(tid < count_a1){
        const u32 a0 = p.a0, am1 = p.am1, am2 = p.am2;
        const u32 e0 = p.e0, em1 = p.em1;
        const u32 C_e1 = p.C_e1, C_W0 = p.C_W0;
        const u32 C_G1 = p.C_G1, C_G2 = p.C_G2, C_G3 = p.C_G3;
        const int w_thresh = p.w_thresh;

        const u32 a1  = (u32)(start_a1 + tid);
        const u32 e1  = a1 + C_e1;
        const u32 sm1 = S0_dev(a1) + MAJ_dev(a1, a0, am1);
        const u32 G1  = sm1 + S1_dev(e1) + CH_dev(e1, e0, em1) + C_G1;
        const u32 f1  = G1 - a1 - C_W0;

        const u64 K7 = 0x7fff7fff7fff7fffULL;
        const u64 K8 = 0x8000800080008000ULL;
        const u64 K1 = 0x0001000100010001ULL;

        for(int pair_base = 0; pair_base < p.batch_ctx; pair_base += 2){
            const int c0 = pair_base;
            const int n_sub = (pair_base + 1 < p.batch_ctx) ? 2 : 1;
            const u32 T16_0 = p.T16[c0];
            const u32 t16   = T16_0 + f1;

            // Single shared Level-1 lookup for the Exact-T16 collision pair (T16[c0] == T16[c1])
            const u16 *b1 = &d_H_pack6[slide3_off_dev(t16)];
            u32 ulo1_0 = __ldg(&b1[0]);
            u32 ulo1_1 = __ldg(&b1[1]);
            u32 ulo1_2 = __ldg(&b1[2]);

            u32 c1_0 = 0, c1_1 = 0, c1_2 = 0, c1_3 = 0;
            u32 mask1 = 0;
            if(ulo1_0){
                c1_0 = reconstruct_u_dev(ulo1_0, t16);
                if(Qf_dev(c1_0) == t16) mask1 |= 1u;
            }
            if(ulo1_1){
                c1_1 = reconstruct_u_dev(ulo1_1, t16);
                if(Qf_dev(c1_1) == t16) mask1 |= 2u;
            }
            if(ulo1_2){
                c1_2 = reconstruct_u_dev(ulo1_2, t16);
                if(Qf_dev(c1_2) == t16) mask1 |= 4u;
            }
            if(mask1 == 7u){
                u32 ulo1_3 = __ldg(&b1[(t16 & 15u) >= 13u ? -1 : 3]);
                if(ulo1_3){
                    c1_3 = reconstruct_u_dev(ulo1_3, t16);
                    if(Qf_dev(c1_3) == t16) mask1 |= 8u;
                }
            }

            // FUSED LEVEL-1 -> LEVEL-2 BRIDGE: W1, a2, e2, sm2, G2 computed ONCE per pair!
            while(mask1){
                int bit1 = __ffs(mask1) - 1;
                mask1 &= mask1 - 1u;
                u32 W1  = (bit1 == 0) ? c1_0 : ((bit1 == 1) ? c1_1 : ((bit1 == 2) ? c1_2 : c1_3));
                u32 a2  = W1 + G1;
                u32 e2  = a2 - sm1 + am2;
                u32 sm2 = S0_dev(a2) + MAJ_dev(a2, a1, a0);
                u32 G2  = sm2 + S1_dev(e2) + CH_dev(e2, e1, e0) + C_G2;
                u32 g2_minus_w1 = G2 - W1;

                for(int sub = 0; sub < n_sub; sub++){
                    int c = pair_base + sub;
                    const u32 a5  = p.a5[c];
                    const u32 T17 = p.T17[c];
                    const u32 T18 = p.T18[c];
                    const u32 T19 = p.T19[c];
                    const u64 A64 = (u64)(a5 & 0x7fffu) * K1;
                    const u32 t17 = T17 + g2_minus_w1;

                    const u16 *b2 = &d_H_pack6[slide3_off_dev(t17)];
                    u32 ulo2_0 = __ldg(&b2[0]);
                    u32 ulo2_1 = __ldg(&b2[1]);
                    u32 ulo2_2 = __ldg(&b2[2]);

                    u32 c2_0 = 0, c2_1 = 0, c2_2 = 0, c2_3 = 0;
                    u32 mask2 = 0;
                    int n2_raw = 0;
                    if(ulo2_0){
                        c2_0 = reconstruct_u_dev(ulo2_0, t17);
                        if(Qf_dev(c2_0) == t17){
                            n2_raw++;
                            if(__popc(~((c2_0 + G2) ^ a5)) <= w_thresh) mask2 |= 1u;
                        }
                    }
                    if(ulo2_1){
                        c2_1 = reconstruct_u_dev(ulo2_1, t17);
                        if(Qf_dev(c2_1) == t17){
                            n2_raw++;
                            if(__popc(~((c2_1 + G2) ^ a5)) <= w_thresh) mask2 |= 2u;
                        }
                    }
                    if(ulo2_2){
                        c2_2 = reconstruct_u_dev(ulo2_2, t17);
                        if(Qf_dev(c2_2) == t17){
                            n2_raw++;
                            if(__popc(~((c2_2 + G2) ^ a5)) <= w_thresh) mask2 |= 4u;
                        }
                    }
                    if(n2_raw == 3){
                        u32 ulo2_3 = __ldg(&b2[(t17 & 15u) >= 13u ? -1 : 3]);
                        if(ulo2_3){
                            c2_3 = reconstruct_u_dev(ulo2_3, t17);
                            if(Qf_dev(c2_3) == t17){
                                if(__popc(~((c2_3 + G2) ^ a5)) <= w_thresh) mask2 |= 8u;
                            }
                        }
                    }

                    while(mask2){
                        int bit2 = __ffs(mask2) - 1;
                        mask2 &= mask2 - 1u;
                        u32 W2 = (bit2 == 0) ? c2_0 : ((bit2 == 1) ? c2_1 : ((bit2 == 2) ? c2_2 : c2_3));
                        u32 a3 = W2 + G2;
                        u32 M3 = ~(a3 ^ a5);

                        local_l2++;
                        local_l3++;
                        u32 e3  = a3 - sm2 + am1;
                        u32 sm3 = S0_dev(a3) + MAJ_dev(a3, a2, a1);
                        u32 G3  = sm3 + S1_dev(e3) + CH_dev(e3, e2, e1) + C_G3;
                        u32 t18 = T18 + G3 - W2;

                        size_t qoff = (size_t)(t18 >> 4) * 12u;
                        const unsigned long long *q64 = (const unsigned long long *)&d_H_pack6[qoff];
                        u64 w0 = __ldg(&q64[0]);
                        u64 w1 = __ldg(&q64[1]);
                        u64 w2 = __ldg(&q64[2]);
                        u64 g64 = (u64)(G3 & 0x7fffu) * K1;
                        u64 m64 = (u64)(M3 & 0x7fffu) * K1;
                        u64 d0 = ((((w0 & K7) + g64) ^ A64) & m64) + K7;
                        u64 d1 = ((((w1 & K7) + g64) ^ A64) & m64) + K7;
                        u64 d2 = ((((w2 & K7) + g64) ^ A64) & m64) + K7;

                        if(((d0 & d1 & d2) & K8) != K8){
                            u32 M3_lo = M3 & 0xffffu;
                            u64 h0 = ~d0 & K8;
                            while(h0){
                                int s = (__ffsll((long long)h0) - 1) >> 4;
                                h0 &= h0 - 1ULL;
                                u32 ulo = (u32)((w0 >> (s * 16)) & 0xffffULL);
                                if(ulo && (((ulo + G3) ^ a5) & M3_lo) == 0u){
                                    u32 u = reconstruct_u_dev(ulo, t18);
                                    if((((u + G3) ^ a5) & M3) == 0u && Qf_dev(u) == t18){
                                        check_sol19_idea2_dev(u, G3, a1, e1, a2, e2, a3, e3, sm3, a0, a5, T19, c, d_multi_t19_bitmap, &local_sol19, d_stats);
                                    }
                                }
                            }
                            u64 h1 = ~d1 & K8;
                            while(h1){
                                int s = (__ffsll((long long)h1) - 1) >> 4;
                                h1 &= h1 - 1ULL;
                                u32 ulo = (u32)((w1 >> (s * 16)) & 0xffffULL);
                                if(ulo && (((ulo + G3) ^ a5) & M3_lo) == 0u){
                                    u32 u = reconstruct_u_dev(ulo, t18);
                                    if((((u + G3) ^ a5) & M3) == 0u && Qf_dev(u) == t18){
                                        check_sol19_idea2_dev(u, G3, a1, e1, a2, e2, a3, e3, sm3, a0, a5, T19, c, d_multi_t19_bitmap, &local_sol19, d_stats);
                                    }
                                }
                            }
                            u64 h2 = ~d2 & K8;
                            while(h2){
                                int s = (__ffsll((long long)h2) - 1) >> 4;
                                h2 &= h2 - 1ULL;
                                u32 ulo = (u32)((w2 >> (s * 16)) & 0xffffULL);
                                if(ulo && (((ulo + G3) ^ a5) & M3_lo) == 0u){
                                    u32 u = reconstruct_u_dev(ulo, t18);
                                    if((((u + G3) ^ a5) & M3) == 0u && Qf_dev(u) == t18){
                                        check_sol19_idea2_dev(u, G3, a1, e1, a2, e2, a3, e3, sm3, a0, a5, T19, c, d_multi_t19_bitmap, &local_sol19, d_stats);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    unsigned int mask = 0xffffffffu;
    for(int offset = 16; offset > 0; offset >>= 1){
        local_l2    += __shfl_down_sync(mask, local_l2, offset);
        local_l3    += __shfl_down_sync(mask, local_l3, offset);
        local_sol19 += __shfl_down_sync(mask, local_sol19, offset);
    }
    if((threadIdx.x & 31) == 0){
        if(local_l2)    atomicAdd(&d_stats->l2_nodes, (unsigned long long)local_l2);
        if(local_l3)    atomicAdd(&d_stats->l3_lookups, (unsigned long long)local_l3);
        if(local_sol19) atomicAdd(&d_stats->sol19_count, (unsigned long long)local_sol19);
    }
}

// --- IDEA 3: GPU-Native 16 GB Direct-Indexed 2-Way Exact Table + 4 GB 3rd-Root Overflow Table ---
// Built directly in L4 VRAM in ~2 seconds! Zero bucket collisions & zero Qf_dev checks on slots 0 & 1!
__global__ void build_exact16_gpu_kernel(
    u32 * __restrict__ d_H_exact16,
    u32 * __restrict__ d_H_ovf4_u32,
    u64 start_u,
    u64 count_u)
{
    u64 tid = (u64)blockIdx.x * (u64)blockDim.x + (u64)threadIdx.x;
    if(tid < count_u){
        u32 u = (u32)(start_u + tid);
        u32 ulo = u & 0xffffu;
        if(ulo != 0u){
            u32 y = Qf_dev(u);
            u32 old = atomicCAS(&d_H_exact16[y], 0u, ulo);
            if(old != 0u){
                if((old >> 16) == 0u && old != ulo){
                    u32 prev = atomicCAS(&d_H_exact16[y], old, old | (ulo << 16));
                    if(prev == old) return;
                }
                // 3rd root overflow slot (shared by 2 adjacent y values in d_H_ovf4)
                if(d_H_ovf4_u32){
                    u32 idx32 = y >> 2;
                    u32 shift = ((y >> 1) & 1u) * 16u;
                    u32 cur32 = d_H_ovf4_u32[idx32];
                    while(((cur32 >> shift) & 0xffffu) == 0u){
                        u32 prev32 = atomicCAS(&d_H_ovf4_u32[idx32], cur32, cur32 | (ulo << shift));
                        if(prev32 == cur32) break;
                        cur32 = prev32;
                    }
                }
            }
        }
    }
}

__global__ void finalize_exact16_order_bit_kernel(
    u32 * __restrict__ d_H_exact16,
    const u16 * __restrict__ d_H_ovf4,
    u64 start_y,
    u64 count_y)
{
    u64 tid = (u64)blockIdx.x * (u64)blockDim.x + (u64)threadIdx.x;
    if(tid < count_y){
        u32 y  = (u32)(start_y + tid);
        u32 p  = d_H_exact16[y];
        u32 u0 = p & 0xffffu;
        u32 u1 = p >> 16;
        if(u1 != 0u){
            u32 lo = (u0 < u1) ? u0 : u1;
            u32 hi = (u0 < u1) ? u1 : u0;
            int has_ovf = 0;
            if(d_H_ovf4){
                u32 u2 = d_H_ovf4[y >> 1];
                if(u2 != 0u && u2 != u0 && u2 != u1){
                    u32 full2 = reconstruct_u_dev(u2, y);
                    if(Qf_dev(full2) == y) has_ovf = 1;
                }
            }
            // Encode whether d_H_ovf4[y >> 1] holds a valid 3rd root of y into the ordering of (u0, u1):
            //   has_ovf == 0  ==>  ulo_0 < ulo_1  (74.2% of 2-root entries: skip d_H_ovf4 read completely!)
            //   has_ovf == 1  ==>  ulo_0 > ulo_1  (25.8% of 2-root entries: d_H_ovf4[y>>1] is GUARANTEED exact 3rd root!)
            d_H_exact16[y] = has_ovf ? (hi | (lo << 16)) : (lo | (hi << 16));
        }
    }
}

__global__ void __launch_bounds__(256, 4)
sweep_exact16_cuda_kernel(
    const u32 * __restrict__ d_H_exact16,
    const u16 * __restrict__ d_H_ovf4,
    const gpu_params_t p,
    const u32 * __restrict__ d_multi_t19_bitmap,
    u64 start_a1,
    u64 count_a1,
    gpu_stats_t * __restrict__ d_stats)
{
    u64 tid = (u64)blockIdx.x * (u64)blockDim.x + (u64)threadIdx.x;
    u32 local_l2 = 0, local_l3 = 0, local_sol19 = 0;

    if(tid < count_a1){
        const u32 a0 = p.a0, am1 = p.am1, am2 = p.am2;
        const u32 e0 = p.e0, em1 = p.em1;
        const u32 C_e1 = p.C_e1, C_W0 = p.C_W0;
        const u32 C_G1 = p.C_G1, C_G2 = p.C_G2, C_G3 = p.C_G3;
        const int w_thresh = p.w_thresh;

        const u32 a1  = (u32)(start_a1 + tid);
        const u32 e1  = a1 + C_e1;
        const u32 sm1 = S0_dev(a1) + MAJ_dev(a1, a0, am1);
        const u32 G1  = sm1 + S1_dev(e1) + CH_dev(e1, e0, em1) + C_G1;
        const u32 f1  = G1 - a1 - C_W0;

        int pair_base = 0;
        while(pair_base < p.batch_ctx){
            int group4 = (pair_base + 7 < p.batch_ctx) &&
                         (p.T16[pair_base + 0] == p.T16[pair_base + 1]) &&
                         (p.T16[pair_base + 2] == p.T16[pair_base + 3]) &&
                         (p.T16[pair_base + 4] == p.T16[pair_base + 5]) &&
                         (p.T16[pair_base + 6] == p.T16[pair_base + 7]);
            int single4 = (!group4) && (pair_base + 3 < p.batch_ctx) &&
                          (p.T16[pair_base + 0] != p.T16[pair_base + 1]) &&
                          (p.T16[pair_base + 2] != p.T16[pair_base + 3]);
            int n_pairs_grp = group4 ? 4 : (single4 ? 2 : 1);
            u32 p1_pref0 = 0, p1_pref1 = 0, p1_pref2 = 0, p1_pref3 = 0;
            if(group4){
                p1_pref0 = __ldg(&d_H_exact16[p.T16[pair_base + 0] + f1]);
                p1_pref1 = __ldg(&d_H_exact16[p.T16[pair_base + 2] + f1]);
                p1_pref2 = __ldg(&d_H_exact16[p.T16[pair_base + 4] + f1]);
                p1_pref3 = __ldg(&d_H_exact16[p.T16[pair_base + 6] + f1]);
            } else if(single4){
                p1_pref0 = __ldg(&d_H_exact16[p.T16[pair_base + 0] + f1]);
                p1_pref1 = __ldg(&d_H_exact16[p.T16[pair_base + 1] + f1]);
                p1_pref2 = __ldg(&d_H_exact16[p.T16[pair_base + 2] + f1]);
                p1_pref3 = __ldg(&d_H_exact16[p.T16[pair_base + 3] + f1]);
            }

            for(int gp = 0; gp < n_pairs_grp; gp++){
                const int cur_pb = pair_base + gp * 2;
                const int c0 = cur_pb;
                const int c1 = (cur_pb + 1 < p.batch_ctx) ? (cur_pb + 1) : c0;
                const int is_exact_twin = group4 ? 1 : ((cur_pb + 1 < p.batch_ctx) && (p.T16[c0] == p.T16[c1]));
                const int outer_steps = is_exact_twin ? 1 : ((cur_pb + 1 < p.batch_ctx) ? 2 : 1);

                for(int ostep = 0; ostep < outer_steps; ostep++){
                    const int base_c = cur_pb + ostep;
                    const int n_sub  = is_exact_twin ? 2 : 1;
                    const u32 t16    = p.T16[base_c] + f1;

                    // Level 1: Single 32-bit exact lookup (4-way ILP prefetched in registers for both group4 and single4!)
                    int pref_idx = group4 ? gp : (gp * 2 + ostep);
                    u32 p1 = (group4 || single4)
                        ? ((pref_idx < 2) ? ((pref_idx == 0) ? p1_pref0 : p1_pref1) : ((pref_idx == 2) ? p1_pref2 : p1_pref3))
                        : __ldg(&d_H_exact16[t16]);
                    if(!p1) continue;

                    // Fused Level-1 -> Level-2 bridge for Exact-T16 Twins
                    for(int r1 = 0; r1 < 3; r1++){
                        u32 ulo1;
                        if(r1 == 0){
                            ulo1 = p1 & 0xffffu;
                        } else if(r1 == 1){
                            ulo1 = p1 >> 16;
                            if(!ulo1) break;
                        } else {
                            if((p1 & 0xffffu) <= (p1 >> 16)) break;
                            ulo1 = __ldg(&d_H_ovf4[t16 >> 1]);
                        }
                        u32 W1  = reconstruct_u_dev(ulo1, t16);
                        u32 a2  = W1 + G1;
                        u32 e2  = a2 - sm1 + am2;
                        u32 sm2 = S0_dev(a2) + MAJ_dev(a2, a1, a0);
                        u32 G2  = sm2 + S1_dev(e2) + CH_dev(e2, e1, e0) + C_G2;
                        u32 g2_minus_w1 = G2 - W1;

                        // 2-way ILP prefetch of Level-2 lookups for Exact-T16 Twin pair
                        u32 t17_0 = p.T17[base_c] + g2_minus_w1;
                        u32 p2_0  = __ldg(&d_H_exact16[t17_0]);
                        u32 t17_1 = (n_sub == 2) ? (p.T17[base_c + 1] + g2_minus_w1) : 0u;
                        u32 p2_1  = (n_sub == 2) ? __ldg(&d_H_exact16[t17_1]) : 0u;

                        for(int sub = 0; sub < n_sub; sub++){
                            u32 p2 = (sub == 0) ? p2_0 : p2_1;
                            if(!p2) continue;
                            u32 t17 = (sub == 0) ? t17_0 : t17_1;
                            int c = base_c + sub;
                            const u32 a5  = p.a5[c];
                            const u32 T18 = p.T18[c];
                            const u32 T19 = p.T19[c];

                            for(int r2 = 0; r2 < 3; r2++){
                                u32 ulo2;
                                if(r2 == 0){
                                    ulo2 = p2 & 0xffffu;
                                } else if(r2 == 1){
                                    ulo2 = p2 >> 16;
                                    if(!ulo2) break;
                                } else {
                                    if((p2 & 0xffffu) <= (p2 >> 16)) break;
                                    ulo2 = __ldg(&d_H_ovf4[t17 >> 1]);
                                }
                                u32 W2 = reconstruct_u_dev(ulo2, t17);
                                u32 a3 = W2 + G2;
                                u32 M3 = ~(a3 ^ a5);
                                if(__popc(M3) > w_thresh) continue;

                                local_l2++;
                                local_l3++;
                                u32 e3  = a3 - sm2 + am1;
                                u32 sm3 = S0_dev(a3) + MAJ_dev(a3, a2, a1);
                                u32 G3  = sm3 + S1_dev(e3) + CH_dev(e3, e2, e1) + C_G3;
                                u32 t18 = T18 + G3 - W2;

                                // Level 3: Single 32-bit exact lookup (4 bytes instead of 24 bytes, zero Qf_dev checks!)
                                u32 p3 = __ldg(&d_H_exact16[t18]);
                                if(!p3) continue;

                                u32 M3_lo = M3 & 0xffffu;
                                for(int r3 = 0; r3 < 3; r3++){
                                    u32 ulo3;
                                    if(r3 == 0){
                                        ulo3 = p3 & 0xffffu;
                                    } else if(r3 == 1){
                                        ulo3 = p3 >> 16;
                                        if(!ulo3) break;
                                    } else {
                                        if((p3 & 0xffffu) <= (p3 >> 16)) break;
                                        ulo3 = __ldg(&d_H_ovf4[t18 >> 1]);
                                    }
                                    if((((ulo3 + G3) ^ a5) & M3_lo) == 0u){
                                        u32 u = reconstruct_u_dev(ulo3, t18);
                                        if((((u + G3) ^ a5) & M3) == 0u){
                                            check_sol19_idea2_dev(u, G3, a1, e1, a2, e2, a3, e3, sm3, a0, a5, T19, c, d_multi_t19_bitmap, &local_sol19, d_stats);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            pair_base += n_pairs_grp * 2;
        }
    }

    unsigned int mask = 0xffffffffu;
    for(int offset = 16; offset > 0; offset >>= 1){
        local_l2    += __shfl_down_sync(mask, local_l2, offset);
        local_l3    += __shfl_down_sync(mask, local_l3, offset);
        local_sol19 += __shfl_down_sync(mask, local_sol19, offset);
    }
    if((threadIdx.x & 31) == 0){
        if(local_l2)    atomicAdd(&d_stats->l2_nodes, (unsigned long long)local_l2);
        if(local_l3)    atomicAdd(&d_stats->l3_lookups, (unsigned long long)local_l3);
        if(local_sol19) atomicAdd(&d_stats->sol19_count, (unsigned long long)local_sol19);
    }
}

static void build_exact16_table_on_gpu(int dev_id, u32 **out_exact16, u16 **out_ovf4, int use_ovf4){
    CUDA_CHECK(cudaSetDevice(dev_id));
    double tb0 = now_sec();
    size_t exact_bytes = (1ULL << 32) * sizeof(u32); // 16.00 GB
    size_t ovf_bytes   = (1ULL << 31) * sizeof(u16); //  4.00 GB
    u32 *d_exact16 = NULL;
    u16 *d_ovf4    = NULL;
    CUDA_CHECK(cudaMalloc(&d_exact16, exact_bytes));
    CUDA_CHECK(cudaMemset(d_exact16, 0, exact_bytes));
    if(use_ovf4){
        CUDA_CHECK(cudaMalloc(&d_ovf4, ovf_bytes));
        CUDA_CHECK(cudaMemset(d_ovf4, 0, ovf_bytes));
    }
    const u64 chunk = 1ULL << 28;
    for(u64 start = 0; start < (1ULL << 32); start += chunk){
        int threads = 256;
        int blocks  = (int)(chunk / threads);
        build_exact16_gpu_kernel<<<blocks, threads>>>(d_exact16, (u32 *)d_ovf4, start, chunk);
    }
    for(u64 start = 0; start < (1ULL << 32); start += chunk){
        int threads = 256;
        int blocks  = (int)(chunk / threads);
        finalize_exact16_order_bit_kernel<<<blocks, threads>>>(d_exact16, d_ovf4, start, chunk);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    fprintf(stderr, "[GPU %d] Built %s Direct-Indexed Exact Q0^-1 table (with Order-Bit Ovf Tag) natively in VRAM in %.2fs!\n",
            dev_id, use_ovf4 ? "20.00 GB (2-Way Exact + 3rd-Root Ovf)" : "16.00 GB (2-Way Exact)",
            now_sec() - tb0);
    *out_exact16 = d_exact16;
    if(out_ovf4) *out_ovf4 = d_ovf4;
}

// ---------- Multi-GPU Worker Dispatch ----------
#define GPU_CHUNK_A1 (1ULL << 24)

static volatile u64 g_next_a1_chunk = 0;
static volatile u64 g_chains_done   = 0;
static volatile u64 g_l2_nodes      = 0;
static volatile u64 g_l3_lookups    = 0;
static volatile u64 g_sol19_count   = 0;
static volatile int g_solved20      = 0;
static pthread_mutex_t g_print_mu   = PTHREAD_MUTEX_INITIALIZER;

typedef struct {
    int dev_id;
    int table_mode; // 6 = 6GB H_pack6, 16 = 16GB exact16, 20 = 16GB exact16 + 4GB ovf4
    u16 *d_H_pack6;
    u16 *d_REC_HI16;
    u32 *d_H_exact16;
    u16 *d_H_ovf4;
    gpu_stats_t *d_stats;
    gpu_params_t params;
} gpu_dev_ctx_t;

static void *gpu_sweep_worker(void *arg){
    gpu_dev_ctx_t *gctx = (gpu_dev_ctx_t *)arg;
    CUDA_CHECK(cudaSetDevice(gctx->dev_id));

    gpu_stats_t zero_stats;
    memset(&zero_stats, 0, sizeof(zero_stats));
    CUDA_CHECK(cudaMemcpy(gctx->d_stats, &zero_stats, sizeof(gpu_stats_t), cudaMemcpyHostToDevice));

    while(!__atomic_load_n(&g_solved20, __ATOMIC_RELAXED)){
        u64 start = __atomic_fetch_add(&g_next_a1_chunk, GPU_CHUNK_A1, __ATOMIC_RELAXED);
        if(start >= (1ULL << 32)) break;
        u64 end = start + GPU_CHUNK_A1;
        if(end > (1ULL << 32)) end = (1ULL << 32);
        u64 count = end - start;

        int threads = 256;
        int blocks  = (int)((count + (u64)threads - 1ULL) / (u64)threads);
        if(gctx->table_mode >= 16 && gctx->d_H_exact16){
            sweep_exact16_cuda_kernel<<<blocks, threads>>>(
                gctx->d_H_exact16, gctx->d_H_ovf4, gctx->params, NULL, start, count, gctx->d_stats);
        } else if(gctx->params.batch_ctx >= 2 && gctx->params.T16[0] == gctx->params.T16[1]){
            sweep_idea1_fused_6gb_kernel<<<blocks, threads>>>(
                gctx->d_H_pack6, gctx->params, NULL, start, count, gctx->d_stats);
        } else {
            sweep_6gb_cuda_kernel<<<blocks, threads>>>(
                gctx->d_H_pack6, gctx->d_REC_HI16, gctx->params, start, count, gctx->d_stats);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        __atomic_fetch_add(&g_chains_done, count, __ATOMIC_RELAXED);

        gpu_stats_t cur_stats;
        CUDA_CHECK(cudaMemcpy(&cur_stats, gctx->d_stats, sizeof(gpu_stats_t), cudaMemcpyDeviceToHost));
        if(cur_stats.solved20){
            pthread_mutex_lock(&g_print_mu);
            if(!__atomic_load_n(&g_solved20, __ATOMIC_RELAXED)){
                int cid = cur_stats.sol_ctx_id;
                int tid = G_STATE.rctx[cid].target_id;
                u32 A_sol[R_STEPS + OFF + 2];
                memcpy(A_sol, G_STATE.rctx[cid].A_base, sizeof(A_sol));
                A_sol[1+OFF] = cur_stats.sol_a1;
                A_sol[2+OFF] = cur_stats.sol_a2;
                A_sol[3+OFF] = cur_stats.sol_a3;
                A_sol[4+OFF] = cur_stats.sol_a4;
                u32 Wm[16], out[8];
                for(int t=0;t<16;t++) Wm[t] = Wt_arr(A_sol, t);
                compress_r(Wm, R_STEPS, out);
                const u32 *exp_dig = g_all_14_mode ? G_STATES_14[tid].dig : G_STATE.dig;
                int ok = 1;
                for(int k=0;k<8;k++) ok &= (out[k] == exp_dig[k]);
                ok &= ((Wm[13] & 0xffu) == 0x80u) && (Wm[14] == 0u) && (Wm[15] == (u32)PADLEN_BITS);
                const char *lbl = g_all_14_mode ? G_TARGETS_14[tid].label : g_single_label;

                if(!g_all_14_mode || !g_target14_solved[tid]){
                    printf("\n*** [GPU %d] FOUND 20-STEP PADDED PREIMAGE (ok=%d, ctx=%d, tid=%d, label=\"%s\") ***\n",
                           gctx->dev_id, ok, cid, tid, lbl);
                    printf("[padded] label=\"%s\"\n", lbl);
                    printf("target = ");
                    for(int k=0;k<8;k++) printf("%08x", exp_dig[k]);
                    printf("\nmessage_hex = ");
                    for(int t=0;t<13;t++) printf("%08x", Wm[t]);
                    printf("%06x\n", Wm[13] >> 8);
                    printf("W =");
                    for(int t=0;t<16;t++) printf(" %08x", Wm[t]);
                    printf("\nsha256_20 = ");
                    for(int k=0;k<8;k++) printf("%08x", out[k]);
                    printf("\n[SOLVED_TARGET tid=%d label=\"%s\"]\n\n", tid, lbl);
                    fflush(stdout);
                }

                if(ok){
                    if(g_all_14_mode){
                        if(!g_target14_solved[tid]){
                            g_target14_solved[tid] = 1;
                            g_targets14_solved_count++;
                        }
                        if(g_targets14_solved_count >= 14){
                            __atomic_store_n(&g_solved20, 1, __ATOMIC_RELAXED);
                        } else {
                            u32 zero = 0;
                            CUDA_CHECK(cudaMemcpy(&gctx->d_stats->solved20, &zero, sizeof(u32), cudaMemcpyHostToDevice));
                        }
                    } else {
                        __atomic_store_n(&g_solved20, 1, __ATOMIC_RELAXED);
                    }
                } else {
                    u32 zero = 0;
                    CUDA_CHECK(cudaMemcpy(&gctx->d_stats->solved20, &zero, sizeof(u32), cudaMemcpyHostToDevice));
                }
            }
            pthread_mutex_unlock(&g_print_mu);
        }
    }

    gpu_stats_t final_stats;
    CUDA_CHECK(cudaMemcpy(&final_stats, gctx->d_stats, sizeof(gpu_stats_t), cudaMemcpyDeviceToHost));
    __atomic_fetch_add(&g_l2_nodes,    (u64)final_stats.l2_nodes,    __ATOMIC_RELAXED);
    __atomic_fetch_add(&g_l3_lookups,  (u64)final_stats.l3_lookups,  __ATOMIC_RELAXED);
    __atomic_fetch_add(&g_sol19_count, (u64)final_stats.sol19_count, __ATOMIC_RELAXED);
    return NULL;
}


// --- [IDEA 8] Fast GPU Right-Context Generator (Hoisted 8-Way s15 Filter + 64-Way Parallel s19 x s22) ---
// Eliminates the 8x redundant evaluation of (a9_5, S_5, S1(e13_base), MAJ, S0, delta0) across s19,
// and replaces the 8-step sequential s22 loop with a 2-step 32-lane warp-parallel evaluation.
__global__ void __launch_bounds__(256, 4)
ctx_gen_fast_cuda_kernel(
    u32 a13, u32 a14, u32 a15, u32 a16, u32 e17,
    u32 a12_perm_key,
    int target_count,
    int max_cap,
    u32 * __restrict__ d_out_a12,
    u32 * __restrict__ d_out_a9,
    u32 * __restrict__ d_out_T16,
    int * __restrict__ d_ctx_count,
    unsigned long long * __restrict__ d_a12_counter,
    unsigned long long * __restrict__ d_trials)
{
    u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    u32 lane = tid & 31u;
    unsigned long long local_a12_tried = 0;

    while(1){
        int cur = 0;
        unsigned long long my_base_idx = 0;
        if(lane == 0){
            cur = atomicAdd(d_ctx_count, 0);
            if(cur < target_count){
                my_base_idx = atomicAdd(d_a12_counter, 8ULL); // Batch 8 a12s per atomic!
            }
        }
        cur = __shfl_sync(0xffffffffu, cur, 0);
        if(cur >= target_count) break;
        my_base_idx = __shfl_sync(0xffffffffu, my_base_idx, 0);
        if(my_base_idx >= (1ULL << 32)) break;

        for(u32 a12_step = 0; a12_step < 8u; a12_step++){
            u32 a12 = perm32_dev((u32)(my_base_idx + a12_step) + a12_perm_key);
            local_a12_tried++;

            const u32 e15 = 0xffffffffu;
            u32 a11 = e15 - a15 + S0_dev(a14) + MAJ_dev(a14, a13, a12);
            u32 a10 = a11;
            u32 e14 = a14 - S0_dev(a13) - MAJ_dev(a13, a12, a11) + a10;
            u32 e16 = a16 - S0_dev(a15) - MAJ_dev(a15, a14, a13) + a12;

            u32 c_e13 = a13 - S0_dev(a12) - a11;
            u32 c_e12 = e16 - a12 - 0xffffffffu - 0xc19bf174u - (u32)PADLEN_BITS;
            u32 c_e11 = e15 - a11 - S1_dev(e14) - 0x9bdc06a7u;
            u32 c_W13 = e14 - a10 - 0xffffffffu - 0x80deb1feu;
            u32 c_W16 = e17 - a13 - S1_dev(e16) - CH_dev(e16, e15, e14) - 0xe49b69c1u;

            u32 c_a8 = S0_dev(a11) - a12;
            u32 c_a7 = S0_dev(a10) - a11;
            u32 c_a6 = 0xffffffffu - a10;

            u32 e12 = c_e12 - e14;
            u32 a8  = e12 + c_a8 + a11;
            u32 c_a5_s0a8  = 0xffffffffu + S0_dev(a8);
            u32 m_10_8     = a10 ^ a8;
            u32 a10_and_a8 = a10 & a8;
            u32 c_e11_base = c_e11 - (~e14 & e12);
            u32 c_a7_full  = c_e11_base + c_a7 + a10_and_a8;

            for(u32 base_lo = 0; base_lo < 256u; base_lo += 32u){
                u32 my_e13_lo = base_lo + lane;
                u32 s0 = my_e13_lo & 0xfu;
                u32 s4 = my_e13_lo & 0xf0u;

                u32 a9_0a  = s0 - c_e13;
                u32 X_0a   = c_a5_s0a8 - a9_0a;
                u32 a7_b0a = c_a7_full - (e14 & s0);
                u32 a7_0a  = a7_b0a + (m_10_8 & a9_0a);
                u32 D_0a   = a8 ^ a7_0a;
                int ok4    = ((((X_0a + CH_dev(D_0a, X_0a, a8)) ^ X_0a) & D_0a & 0xfu) == 0u);

                u32 a9_0  = a9_0a + s4;
                u32 X_0   = X_0a  - s4;
                u32 a7_b0 = a7_b0a - (e14 & s4);
                u32 a7_0  = a7_b0 + (m_10_8 & a9_0);
                u32 D_0   = a8 ^ a7_0;
                int ok8   = ok4 && ((((X_0 + CH_dev(D_0, X_0, a8)) ^ X_0) & D_0 & 0xffu) == 0u);

                u32 mask = __ballot_sync(0xffffffffu, ok8);
                while(mask){
                    int src = __ffs(mask) - 1;
                    mask &= mask - 1u;

                    u32 e13_lo  = __shfl_sync(0xffffffffu, my_e13_lo, src);
                    u32 a9_0_w  = __shfl_sync(0xffffffffu, a9_0, src);
                    u32 X_0_w   = __shfl_sync(0xffffffffu, X_0, src);
                    u32 a7_b0_w = __shfl_sync(0xffffffffu, a7_b0, src);

                    u32 e11_lo = (c_e11_base - (e14 & e13_lo)) & 0xffu;
                    u32 req_S1_lo = (c_W13 - CH_dev(e13_lo, e12, e11_lo) - 0x80u) & 0xffu;
                    u32 req_b7_xor_lo0 = ((req_S1_lo >> 7) ^ e13_lo) & 1u;

                    for(u32 k128 = 0; k128 < 128u; k128 += 32u){
                        u32 idx128  = k128 + lane;
                        u32 s8      = (idx128 & 15u) << 8;
                        u32 s12_idx = idx128 >> 4;

                        u32 a9_1  = a9_0_w + s8;
                        u32 X_1   = X_0_w  - s8;
                        u32 a7_b1 = a7_b0_w - (e14 & s8);
                        u32 a7_1  = a7_b1 + (m_10_8 & a9_1);
                        u32 D_1   = a8 ^ a7_1;
                        int ok12  = ((((X_1 + CH_dev(D_1, X_1, a8)) ^ X_1) & D_1 & 0xfffu) == 0u);

                        u32 s12   = s12_idx << 12;
                        u32 b18   = ((req_b7_xor_lo0 ^ (s12 >> 13)) & 1u) << 18;
                        u32 s12_b = s12 | b18;
                        u32 a9_2  = a9_1 + s12_b;
                        u32 X_2   = X_1  - s12_b;
                        u32 a7_b2 = a7_b1 - (e14 & s12_b);
                        u32 a7_2  = a7_b2 + (m_10_8 & a9_2);
                        u32 D_2   = a8 ^ a7_2;
                        int ok15  = ok12 && ((((X_2 + CH_dev(D_2, X_2, a8)) ^ X_2) & D_2 & 0x7fffu) == 0u);
                        u32 e13_14_lane = e13_lo | s8 | s12;

                        // Process up to 4 ok15 survivors simultaneously (4 survivors x 8 s15 values = 32 lanes!)
                        u32 mask15 = __ballot_sync(0xffffffffu, ok15);
                        while(mask15){
                            int s_idx[4];
                            int n_surv = 0;
                            #pragma unroll
                            for(int g = 0; g < 4; g++){
                                if(mask15){
                                    s_idx[g] = __ffs(mask15) - 1;
                                    mask15 &= mask15 - 1u;
                                    n_surv++;
                                } else {
                                    s_idx[g] = 0;
                                }
                            }
                            int grp = lane >> 3;
                            int src15 = (grp == 0) ? s_idx[0] : ((grp == 1) ? s_idx[1] : ((grp == 2) ? s_idx[2] : s_idx[3]));

                            u32 a9_2_w   = __shfl_sync(0xffffffffu, a9_2, src15);
                            u32 X_2_w    = __shfl_sync(0xffffffffu, X_2, src15);
                            u32 a7_b2_w  = __shfl_sync(0xffffffffu, a7_b2, src15);
                            u32 e13_14_w = __shfl_sync(0xffffffffu, e13_14_lane, src15);

                            u32 s15   = (lane & 7u) << 15;
                            u32 a9_5  = a9_2_w + s15;
                            u32 X_5   = X_2_w  - s15;
                            u32 a7_b5 = a7_b2_w - (e14 & s15);
                            u32 a7_5  = a7_b5 + (m_10_8 & a9_5);
                            u32 D_5   = a8 ^ a7_5;
                            u32 S_5   = X_5 + CH_dev(D_5, X_5, a8);
                            int ok19  = (grp < n_surv) && (((S_5 ^ X_5) & D_5 & 0x7ffffu) == 0u);

                            u32 e13_base = e13_14_w | s15;
                            u32 diff     = (req_S1_lo ^ S1_dev(e13_base)) & 0x7fu;
                            u32 hi25     = diff << 25;
                            u32 a9_base  = a9_5 + hi25;
                            u32 X0       = X_5  - hi25;
                            u32 a7_base  = a7_b5 - (e14 & hi25);

                            // Only the surviving s15 candidates expand the 64 (s19 x s22) combinations across all 32 lanes!
                            u32 mask19 = __ballot_sync(0xffffffffu, ok19);
                            while(mask19){
                                int src19 = __ffs(mask19) - 1;
                                mask19 &= mask19 - 1u;

                                u32 a9_b_w = __shfl_sync(0xffffffffu, a9_base, src19);
                                u32 X0_w   = __shfl_sync(0xffffffffu, X0, src19);
                                u32 a7_b_w = __shfl_sync(0xffffffffu, a7_base, src19);

                                // 64 combinations of (s19 in 0..7, s22 in 0..7) evaluated in 2 lockstep warp steps!
                                for(u32 k64 = 0; k64 < 64u; k64 += 32u){
                                    u32 idx64 = k64 + lane;
                                    u32 s19_22 = idx64 << 19; // bits 19..24 = s19 | s22!
                                    u32 a9 = a9_b_w + s19_22;
                                    u32 X  = X0_w   - s19_22;
                                    u32 a7 = a7_b_w - (e14 & s19_22) + (m_10_8 & a9);
                                    u32 D  = a8 ^ a7;
                                    u32 S  = X + CH_dev(D, X, a8);
                                    if(((S ^ X) & D) != 0u) continue;
                                    u32 a6 = c_a6 + S0_dev(a9) + MAJ_dev(a9, a8, a7);
                                    if(((a6 ^ S) & ~D) == 0u){
                                        int idx = atomicAdd(d_ctx_count, 1);
                                        if(idx < max_cap){
                                            u32 W16 = c_W16 - (a9 + c_e13);
                                            u32 c9  = S0_dev(a6) + S0_dev(a7) - a6 - a8 - 0x12835b01u;
                                            d_out_a12[idx] = a12;
                                            d_out_a9[idx]  = a9;
                                            d_out_T16[idx] = W16 - c9;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if(lane == 0 && local_a12_tried){
        atomicAdd(d_trials, local_a12_tried << 24);
    }
}


static int parse_hex32(const char *s, u32 out[8]){
    if(strlen(s) != 64) return -1;
    for(int i=0;i<8;i++){
        char buf[9]; memcpy(buf, s + i*8, 8); buf[8] = 0;
        char *end = NULL;
        unsigned long v = strtoul(buf, &end, 16);
        if(!end || *end != 0) return -1;
        out[i] = (u32)v;
    }
    return 0;
}

int main(int argc, char **argv){
    const char *target_hex = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    const char *iv_hex = NULL;
    int cpu_threads = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if(cpu_threads < 1) cpu_threads = 8;
    int w_thresh = 12;
    int batch_ctx = 128;
    int table_gb = 0; // 0 = auto (20 GB on L4 >= 22 GB VRAM, else 6 GB)
    u64 max_sweeps = 0;

    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i], "--target") && i+1<argc) target_hex = argv[++i];
        else if(!strcmp(argv[i], "--iv") && i+1<argc) iv_hex = argv[++i];
        else if(!strcmp(argv[i], "--padlen") && i+1<argc) g_padlen_bits = (u32)strtoul(argv[++i], NULL, 10);
        else if(!strcmp(argv[i], "--label") && i+1<argc) g_single_label = argv[++i];
        else if(!strcmp(argv[i], "--all-14-1block")) g_all_14_mode = 1;
        else if(!strcmp(argv[i], "--target-range") && i+1<argc){
            if(sscanf(argv[++i], "%d:%d", &g_target_first, &g_target_last) != 2 ||
               g_target_first < 0 || g_target_last > 13 || g_target_first > g_target_last){
                fprintf(stderr, "Invalid --target-range (expected first:last in 0..13)\n");
                return 1;
            }
        }
        else if(!strcmp(argv[i], "--threads") && i+1<argc){
            cpu_threads = atoi(argv[++i]);
            if(cpu_threads < 1 || cpu_threads > 1024){ fprintf(stderr, "Invalid --threads (must be 1..1024)\n"); return 1; }
        }
        else if(!strcmp(argv[i], "--w-thresh") && i+1<argc){
            w_thresh = atoi(argv[++i]);
            if(w_thresh < 0 || w_thresh > 32){ fprintf(stderr, "Invalid --w-thresh (must be 0..32)\n"); return 1; }
        }
        else if(!strcmp(argv[i], "--batch-ctx") && i+1<argc){
            batch_ctx = atoi(argv[++i]);
            if(batch_ctx < 1 || batch_ctx > MAX_BATCH_CTX){ fprintf(stderr, "Invalid --batch-ctx (must be 1..%d)\n", MAX_BATCH_CTX); return 1; }
        }
        else if(!strcmp(argv[i], "--table-gb") && i+1<argc){
            table_gb = atoi(argv[++i]);
            if(table_gb != 0 && table_gb != 6 && table_gb != 16 && table_gb != 20){ fprintf(stderr, "Invalid --table-gb (must be 0, 6, 16, or 20)\n"); return 1; }
        }
        else if(!strcmp(argv[i], "--max-sweeps") && i+1<argc) max_sweeps = strtoull(argv[++i], NULL, 10);
        else {
            fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]);
            return 1;
        }
    }

    if(iv_hex){
        if(parse_hex32(iv_hex, IV) != 0){
            fprintf(stderr, "Invalid --iv hex\n");
            return 1;
        }
    }

    u32 dig[8];
    if(parse_hex32(target_hex, dig) != 0){
        fprintf(stderr, "Invalid --target hex\n");
        return 1;
    }

    int n_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n_gpus));
    if(n_gpus < 1){
        fprintf(stderr, "No CUDA GPUs detected!\n");
        return 1;
    }

    for(int d = 0; d < n_gpus; d++){
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaMemcpyToSymbol(d_padlen_bits, &g_padlen_bits, sizeof(u32)));
    }

    if(g_all_14_mode){
        // Pre-mark already solved 1-block targets:
        // tid=2 ("padded test 1"), tid=3 ("padded test 2"), and tid=12 ("padded test 11")
        g_target14_solved[2]  = 1;
        g_target14_solved[3]  = 1;
        g_target14_solved[12] = 1;
        g_targets14_solved_count = 3;
    }

    init_inv_s0_lut();
    for(int d = 0; d < n_gpus; d++){
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaMemcpyToSymbol(d_inv_s0_lut, h_inv_s0_lut, sizeof(h_inv_s0_lut)));
    }

    cudaDeviceProp prop0;
    CUDA_CHECK(cudaGetDeviceProperties(&prop0, 0));
    if(table_gb == 0){
        double vram_gb = (double)prop0.totalGlobalMem / (1024.0 * 1024.0 * 1024.0);
        table_gb = (vram_gb >= 21.5) ? 20 : ((vram_gb >= 17.5) ? 16 : 6);
    }

    fprintf(stderr, "=== 20-Step Padded SHA-256 Heterogeneous CPU + CUDA Solver ===\n");
    if(g_all_14_mode){
        fprintf(stderr, "Mode: Multi-Target All-14 1-Block (primary range %d..%d) | padlen_bits=%u\n",
                g_target_first, g_target_last, g_padlen_bits);
    } else {
        fprintf(stderr, "Target: %s (label=\"%s\", padlen_bits=%u)\n", target_hex, g_single_label, g_padlen_bits);
    }
    fprintf(stderr, "Host CPU Threads: %d | GPUs: %d | table_gb: %d | w_thresh: %d | batch_ctx: %d\n",
            cpu_threads, n_gpus, table_gb, w_thresh, batch_ctx);

    double t0 = now_sec();
    if(table_gb < 16){
        init_q0_table_6gb(cpu_threads);
    }
    if(g_all_14_mode){
        for(int t = 0; t < 14; t++){
            init_iv_and_target(&G_STATES_14[t], G_TARGETS_14[t].dig);
            G_STATES_14[t].w_thresh  = w_thresh;
            G_STATES_14[t].batch_ctx = batch_ctx;
        }
        init_iv_and_target(&G_STATE, G_STATES_14[g_target_first].dig);
    } else {
        init_iv_and_target(&G_STATE, dig);
    }
    G_STATE.w_thresh = w_thresh;
    G_STATE.batch_ctx = batch_ctx;

    u64 table_bytes = (3ULL << 30) * sizeof(u16);
    gpu_dev_ctx_t *gctxs = (gpu_dev_ctx_t *)calloc(n_gpus, sizeof(gpu_dev_ctx_t));
    for(int d = 0; d < n_gpus; d++){
        CUDA_CHECK(cudaSetDevice(d));
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, d));
        gctxs[d].dev_id = d;
        gctxs[d].table_mode = table_gb;
        CUDA_CHECK(cudaMalloc(&gctxs[d].d_stats, sizeof(gpu_stats_t)));
        if(table_gb >= 16){
            build_exact16_table_on_gpu(d, &gctxs[d].d_H_exact16, &gctxs[d].d_H_ovf4, (table_gb >= 20) ? 1 : 0);
        } else {
            fprintf(stderr, "[GPU %d] %s (%.2f GB VRAM, %d SMs) - uploading 6.00 GB Q0^-1 table...\n",
                    d, prop.name, (double)prop.totalGlobalMem / (1024.0*1024.0*1024.0), prop.multiProcessorCount);
            CUDA_CHECK(cudaMalloc(&gctxs[d].d_H_pack6, table_bytes));
            CUDA_CHECK(cudaMalloc(&gctxs[d].d_REC_HI16, sizeof(REC_HI16)));
            CUDA_CHECK(cudaMemcpy(gctxs[d].d_H_pack6, H_pack6, table_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(gctxs[d].d_REC_HI16, REC_HI16, sizeof(REC_HI16), cudaMemcpyHostToDevice));
        }
    }
    fprintf(stderr, "[init] Ready in %.2fs\n", now_sec() - t0);

    // Seed RNG from /dev/urandom so multiple parallel CPU/GPU cloud VMs never overlap
    u64 urand_seed = 0;
    int ufd = open("/dev/urandom", O_RDONLY);
    if(ufd >= 0){
        if(read(ufd, &urand_seed, sizeof(urand_seed)) != sizeof(urand_seed)) urand_seed = 0;
        close(ufd);
    }
    if(!urand_seed) urand_seed = ((u64)getpid() << 32) ^ (u64)(now_sec() * 1e6);
    __atomic_store_n(&g_ctx_seed_gen, urand_seed, __ATOMIC_RELAXED);

    u64 sweep_idx = 0;
    u64 grand_sol19 = 0;
    double solve_start = now_sec();

    while(!__atomic_load_n(&g_solved20, __ATOMIC_RELAXED)){
        if(max_sweeps > 0 && sweep_idx >= max_sweeps) break;
        sweep_idx++;

        double tc0 = now_sec();
        u64 trials = generate_right_contexts_parallel(cpu_threads);
        double tc1 = now_sec();

        u32 dt0 = (batch_ctx >= 2) ? (G_STATE.rctx[1].T16 - G_STATE.rctx[0].T16) : 0u;
        u32 dt1 = (batch_ctx >= 4) ? (G_STATE.rctx[3].T16 - G_STATE.rctx[2].T16) : 0u;
        u32 dt2 = (batch_ctx >= 6) ? (G_STATE.rctx[5].T16 - G_STATE.rctx[4].T16) : 0u;
        u32 dt3 = (batch_ctx >= 8) ? (G_STATE.rctx[7].T16 - G_STATE.rctx[6].T16) : 0u;
        fprintf(stderr, "\n[sweep %llu] Selected %d Exact-T16 paired contexts in %.2fs (%llu trials, pool=%d, pairs_left=%d) | dT16=(%u,%u,%u,%u) | a5=(%08x,%08x)\n",
                (unsigned long long)sweep_idx, batch_ctx, tc1 - tc0, (unsigned long long)trials,
                g_ctx_pool_len, g_exact_pairs_avail,
                dt0, dt1, dt2, dt3, G_STATE.rctx[0].a5, (batch_ctx >= 2) ? G_STATE.rctx[1].a5 : 0u);

        gpu_params_t params;
        memset(&params, 0, sizeof(params));
        params.a0 = G_STATE.a0; params.am1 = G_STATE.am1; params.am2 = G_STATE.am2;
        params.e0 = G_STATE.e0; params.em1 = G_STATE.em1;
        params.C_e1 = G_STATE.C_e1; params.C_W0 = G_STATE.C_W0;
        params.C_G1 = G_STATE.C_G1; params.C_G2 = G_STATE.C_G2; params.C_G3 = G_STATE.C_G3;
        params.w_thresh  = G_STATE.w_thresh;
        params.batch_ctx = G_STATE.batch_ctx;
        for(int c = 0; c < batch_ctx; c++){
            params.a5[c]  = G_STATE.rctx[c].a5;
            params.T16[c] = G_STATE.rctx[c].T16;
            params.T17[c] = G_STATE.rctx[c].T17;
            params.T18[c] = G_STATE.rctx[c].T18;
            params.T19[c] = G_STATE.rctx[c].T19;
        }
        for(int d = 0; d < n_gpus; d++) gctxs[d].params = params;

        __atomic_store_n(&g_next_a1_chunk, 0, __ATOMIC_RELAXED);
        __atomic_store_n(&g_chains_done, 0, __ATOMIC_RELAXED);
        __atomic_store_n(&g_l2_nodes, 0, __ATOMIC_RELAXED);
        __atomic_store_n(&g_l3_lookups, 0, __ATOMIC_RELAXED);
        __atomic_store_n(&g_sol19_count, 0, __ATOMIC_RELAXED);

        double ts0 = now_sec();
        pthread_t *gth = (pthread_t *)malloc(n_gpus * sizeof(pthread_t));
        for(int d = 0; d < n_gpus; d++){
            pthread_create(&gth[d], NULL, gpu_sweep_worker, &gctxs[d]);
        }

        while(!__atomic_load_n(&g_solved20, __ATOMIC_RELAXED)){
            usleep(250000);
            u64 done = __atomic_load_n(&g_chains_done, __ATOMIC_RELAXED);
            if(done >= (1ULL << 32)) break;
            double el = now_sec() - ts0;
            fprintf(stderr, "\r  a1=%lluM/4096M (%.1f%%) | %.2f M/s | elapsed=%.1fs   ",
                    (unsigned long long)(done >> 20), 100.0 * (double)done / (double)(1ULL << 32),
                    (double)done / el / 1e6, el);
        }

        for(int d = 0; d < n_gpus; d++) pthread_join(gth[d], NULL);
        free(gth);

        double ts1 = now_sec();
        u64 s19 = __atomic_load_n(&g_sol19_count, __ATOMIC_RELAXED);
        u64 l2  = __atomic_load_n(&g_l2_nodes, __ATOMIC_RELAXED);
        u64 l3  = __atomic_load_n(&g_l3_lookups, __ATOMIC_RELAXED);
        grand_sol19 += s19;
        double tot_el = ts1 - solve_start;
        fprintf(stderr, "\n[sweep %llu done] time=%.2fs | L2=%llu | L3=%llu | sol19=%llu (%.1f/s) | grand_sol19=%llu (net %.1f/s)\n",
                (unsigned long long)sweep_idx, ts1 - ts0,
                (unsigned long long)l2, (unsigned long long)l3,
                (unsigned long long)s19, (double)s19 / (ts1 - ts0),
                (unsigned long long)grand_sol19, (double)grand_sol19 / tot_el);
    }

    return 0;
}
