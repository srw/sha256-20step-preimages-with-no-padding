// Ultra-Fast Multi-Threaded Preimage Solver for 20-Step Padded SHA-256
//
// Major Algorithmic & Architectural Breakthroughs over solve20_padded.c:
//   1. Exact O(1) Algebraic Inversion of (W13 & 0xff) == 0x80 in ctx_worker (256x Right Speedup):
//      - Exploits the exact bit-disjointness of S1(e13) & 0xff: bits 25..31 and bit 18 of e13
//        bijectively control S1(e13) & 0xff without affecting e13 & 0xff, e12 & 0xff, or e11 & 0xff.
//      - Bits 19..24 of e13 do not affect (W13 & 0xff) at all, enabling a branchless 64-way
//        inner loop where 100% of iterations satisfy W15=440, W14=0, (W13 & 0xff)==0x80.
//   2. Exact kmax=0 (D = a5 ^ a6 = 0) Enforcement in ~0.15s:
//      - Because ctx_worker is 256x faster, we enforce exact a5 == a6 (D = 0, D5 = 0) on every
//        context, doubling sol19 yield (2.0x) and eliminating all D / D5 masks in Level 2/3.
//   3. 32.00 GB Direct-Indexed 4-Way Exact Q0^-1 Table (--table-gb 32 on 240 GB VM):
//      - Indexes H_exact4[y] directly by 32-bit y (2^32 x u64 = 32 GB), storing up to 4 x 16-bit
//        preimages per y (99.57% coverage of all 2^32 preimages -> 98.71% 3-level chain survival,
//        12.32x higher sol19 yield than the 4 GB table).
//      - Eliminates 100% of false-bucket lookups and eliminates Qf(u) == y verification checks.
//      - Also supports --table-gb 4 fallback for local 16 GB machines.
//   4. Multi-Context Batching (--batch-ctx B) & Lazy State Materialization:
//      - Evaluates B Right contexts simultaneously in a single a1 pass, amortizing Level-1 ALU work.
//      - Defers loading (a1, e1, a2, e2, sm2) in Level 2 until after the 81% popcount(M3) filter,
//        and recomputes (e1, e2, e3, sm3) in Level 3 only on the 0.05% of candidates that survive
//        ((a4 ^ a5) & M3) == 0.
//
// Compile:
//   gcc -O3 -march=native -pthread solve20_padded_fast.c -o solve20_padded_fast

#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
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
static const u32 IV[8] = {
    0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
    0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19
};

#define PADLEN_BITS 440ULL
#define R_STEPS 20
#define OFF 7
#define MAX_BATCH_CTX 32

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

static inline __attribute__((unused)) u64 splitmix64(u64 *state){
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

// ---------- Q_0^{-1} Lookup Tables (Supports 32 GB Exact 4-Way, 6 GB Hierarchical 12-Slot Quad H_hier6, & 4 GB 2-Slot Pack2) ----------
#define CACHE_FILE_4GB     "/tmp/q0_u16_pack2.bin"
#define CACHE_FILE_6GB     "/tmp/q0_u16_pack3.bin"
#define CACHE_FILE_HIER6   "/tmp/q0_u16_hier6.bin"
#define CACHE_FILE_SUDOKU6 "/tmp/q0_u16_sudoku6.bin"

#define CHUNK_SIZE (1ULL << 18)
#define N_F1_CHUNKS ((1ULL << 32) / CHUNK_SIZE)

static int g_table_gb = 6;
static u32 *H_pack4 = NULL;           // 4 GB mode: 2^30 x u32 (2 x u16 per y>>2)
static u16 *H_pack6 = NULL;           // 6 GB mode: 2^30 x 3 x u16 (3 x u16 contiguous per y>>2, single cache line!)
static _Atomic u64 *H_exact32 = NULL; // 32 GB exact: 2^32 x u64 (4 x u16 per exact y)
static _Atomic u8  *F1_count = NULL;  // 4 GB exact:  2^32 x u8  count of a1 per F1(a1)=f
static u32 *A1_perm = NULL;           // 16 GB exact: 2^32 x u32 a1 sorted by F1(a1)=f
static u64 F1_chunk_off[N_F1_CHUNKS + 1];
static u16 REC_HI16[65536];

static void init_rec_tables(void){
    for(u32 uhi = 0; uhi < 65536; uhi++){
        u32 v17 = (s0(uhi << 16) & 0x1ffffu) ^ ((uhi & 1u) << 16);
        REC_HI16[(v17 >> 1) & 0xffffu] = (u16)uhi;
    }
}

// Breakthrough 8 (Phase 7): 5-Instruction Zero-Rotate REC_HI16 Index Identity!
// Balanced-tree 4-cycle REC_HI16 index: keeping >>1 & 0xffff on the outside allows clang/gcc
// to fuse >>1 & 0xffff into a single ubfx instruction and parallelize s0(ulo) across 2 ALU pipes!
static inline u32 rec_idx(u32 ulo, u32 y){
    return (((ulo + y) ^ s0(ulo)) >> 1) & 0xffffu;
}

// Breakthrough 9 (Phase 7): Optimal Sliding 3-Slot Quad Schedule (slide3_b) + 2-Tier Nested 4-Slot Schedule (S_SLIDE4_NEST)!
// Maps remainder r = (y & 15) to starting slot s_slide3_b[r] = {0,0,0,1, 2,3,3,4, 5,6,6,7, 8,9,9,9}
// inside each 12-slot quad, boosting 3-slot L1/L2 coverage from 66.43% to 69.17% (+8.53% sol19/sweep!).
// Computed in 5 single-cycle ALU ops (4-cycle latency, zero multiplies, zero masks) via the identity:
//   12*(y >> 4) + s_slide3_b[y & 15] == (y >> 1) + (y >> 2) - (ROR(0x6AAA6AAAu, y & 31u) >> 31)
static const int S_SLIDE3[16]      = {0,0,0,1, 2,3,3,4, 5,6,6,7, 8,9,9,9};
static const int S_SLIDE4_NEST[16] = {0,0,0,1, 2,3,3,4, 5,6,6,7, 8,8,8,8};

static inline size_t slide3_off(u32 y){
    u32 d = ROR(0x6AAA6AAAu, y & 31u) >> 31;
    return (size_t)((y >> 1) + (y >> 2) - d);
}

static inline __attribute__((unused)) size_t slide4_off(u32 y){
    return slide3_off(y) - (size_t)((y & 15u) >= 13u);
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

static void *build_worker_32gb(void *arg){
    build_arg_t *ba = (build_arg_t *)arg;
    u64 span = ba->end_u - ba->start_u;
    u64 start = ba->start_u + ((u64)ba->tid * span) / (u64)ba->nthreads;
    u64 end   = ba->start_u + ((u64)(ba->tid + 1) * span) / (u64)ba->nthreads;
    start &= ~(u64)(BUILD_PF - 1);
    if(ba->tid < ba->nthreads - 1) end &= ~(u64)(BUILD_PF - 1);
    u32 b_y[BUILD_PF];
    u16 b_ulo[BUILD_PF];
    u64 local_stored = 0;

    for(u64 base = start; base + BUILD_PF <= end; base += BUILD_PF){
        for(int i=0; i<BUILD_PF; i++){
            u32 u = (u32)(base + (u64)i);
            u32 y = Qf(u);
            b_y[i] = y;
            b_ulo[i] = (u16)(u & 0xffffu);
            __builtin_prefetch(&H_exact32[y], 1, 0);
        }
        for(int i=0; i<BUILD_PF; i++){
            u16 ulo = b_ulo[i];
            if(!ulo) continue;
            u32 y = b_y[i];
            u64 cur = atomic_load_explicit(&H_exact32[y], memory_order_relaxed);
            for(;;){
                int slot = -1;
                if((cur & 0xffffULL) == 0ULL) slot = 0;
                else if(((cur >> 16) & 0xffffULL) == 0ULL) slot = 1;
                else if(((cur >> 32) & 0xffffULL) == 0ULL) slot = 2;
                else if(((cur >> 48) & 0xffffULL) == 0ULL) slot = 3;
                if(slot < 0) break;
                u64 next = cur | ((u64)ulo << (slot * 16));
                if(atomic_compare_exchange_weak_explicit(&H_exact32[y], &cur, next,
                        memory_order_relaxed, memory_order_relaxed)){
                    local_stored++;
                    break;
                }
            }
        }
    }
    ba->stored = local_stored;
    return NULL;
}

static void *prefault_worker_6gb(void *arg){
    build_arg_t *ba = (build_arg_t *)arg;
    u64 total_u16 = 3ULL << 30;
    u64 start = ((u64)ba->tid * total_u16) / (u64)ba->nthreads;
    u64 end   = ((u64)(ba->tid + 1) * total_u16) / (u64)ba->nthreads;
    volatile u64 sum = 0;
    for(u64 idx = start; idx < end; idx += 8192){
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
    _Atomic u16 *H_at = (_Atomic u16 *)H_pack6;
    u64 local_stored = 0;

    for(u64 base = start; base + BUILD_PF <= end; base += BUILD_PF){
        for(int i=0; i<BUILD_PF; i++){
            u32 u = (u32)(base + (u64)i);
            u32 y = Qf(u);
            u32 idx = y >> 2;
            b_idx[i] = idx;
            b_ulo[i] = (u16)(u & 0xffffu);
            __builtin_prefetch(&H_at[(size_t)idx * 3], 1, 0);
        }
        for(int i=0; i<BUILD_PF; i++){
            u16 ulo = b_ulo[i];
            if(!ulo) continue;
            size_t off = (size_t)b_idx[i] * 3;
            for(int s=0; s<3; s++){
                u16 cur = atomic_load_explicit(&H_at[off + s], memory_order_relaxed);
                if(cur == ulo) break;
                if(cur == 0u){
                    if(atomic_compare_exchange_strong_explicit(&H_at[off + s], &cur, ulo,
                            memory_order_relaxed, memory_order_relaxed)){
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

static void *build_worker_4gb(void *arg){
    build_arg_t *ba = (build_arg_t *)arg;
    u64 span = ba->end_u - ba->start_u;
    u64 start = ba->start_u + ((u64)ba->tid * span) / (u64)ba->nthreads;
    u64 end   = ba->start_u + ((u64)(ba->tid + 1) * span) / (u64)ba->nthreads;
    u32 b_idx[BUILD_PF];
    u16 b_ulo[BUILD_PF];

    for(u64 base = start; base + BUILD_PF <= end; base += BUILD_PF){
        for(int i=0; i<BUILD_PF; i++){
            u32 u = (u32)((base + (u64)i) * 0x9E3779B1u + 1u);
            u32 y = Qf(u);
            u32 idx = y >> 2;
            b_idx[i] = idx;
            b_ulo[i] = (u16)(u & 0xffffu);
            __builtin_prefetch(&H_pack4[idx], 1, 0);
        }
        _Atomic u32 *H_at4 = (_Atomic u32 *)H_pack4;
        for(int i=0; i<BUILD_PF; i++){
            u16 ulo = b_ulo[i];
            if(!ulo) continue;
            u32 idx = b_idx[i];
            u32 cur = atomic_load_explicit(&H_at4[idx], memory_order_relaxed);
            for(;;){
                u32 next = cur;
                if((cur & 0xffffu) == 0u){
                    next = cur | (u32)ulo;
                } else if((cur >> 16) == 0u && (cur & 0xffffu) != (u32)ulo){
                    next = cur | ((u32)ulo << 16);
                } else {
                    break;
                }
                if(atomic_compare_exchange_weak_explicit(&H_at4[idx], &cur, next,
                        memory_order_relaxed, memory_order_relaxed)){
                    break;
                }
            }
        }
    }
    return NULL;
}

static void *prefault_worker_4gb(void *arg){
    build_arg_t *ba = (build_arg_t *)arg;
    u64 total_words = (1ULL << 30);
    u64 start = ((u64)ba->tid * total_words) / (u64)ba->nthreads;
    u64 end   = ((u64)(ba->tid + 1) * total_words) / (u64)ba->nthreads;
    volatile u64 sum = 0;
    for(u64 idx = start; idx < end; idx += 16){
        sum += H_pack4[idx];
    }
    (void)sum;
    return NULL;
}

static void *prefault_cow_worker_4gb(void *arg){
    build_arg_t *ba = (build_arg_t *)arg;
    u64 total_words = (1ULL << 30);
    u64 start = ((u64)ba->tid * total_words) / (u64)ba->nthreads;
    u64 end   = ((u64)(ba->tid + 1) * total_words) / (u64)ba->nthreads;
    volatile u32 *p = (volatile u32 *)H_pack4;
    for(u64 idx = start; idx < end; idx += 1024){
        u32 v = p[idx];
        p[idx] = v;
    }
    return NULL;
}

static void *topup_pass1_worker_4gb(void *arg){
    build_arg_t *ba = (build_arg_t *)arg;
    u64 total_u = (1ULL << 32);
    u64 start = ((u64)ba->tid * total_u) / (u64)ba->nthreads;
    u64 end   = ((u64)(ba->tid + 1) * total_u) / (u64)ba->nthreads;
    start &= ~63ULL;
    u32 b_idx[64];
    u16 b_ulo[64];
    for(u64 base = start; base < end; base += 64){
        for(int i=0; i<64; i++){
            u32 u = (u32)(base + (u64)i);
            u32 y = Qf(u);
            u32 idx = y >> 2;
            b_idx[i] = idx;
            b_ulo[i] = (u16)(u & 0xffffu);
            __builtin_prefetch(&H_pack4[idx], 0, 0);
        }
        _Atomic u32 *H_at4 = (_Atomic u32 *)H_pack4;
        for(int i=0; i<64; i++){
            u16 ulo = b_ulo[i];
            if(!ulo) continue;
            u32 idx = b_idx[i];
            u32 cur = atomic_load_explicit(&H_at4[idx], memory_order_relaxed);
            for(;;){
                if((cur >> 16) != 0u) break;
                u32 s0_lo = cur & 0xffffu;
                u32 next = cur;
                if(s0_lo == 0u) next = (u32)ulo;
                else if(s0_lo != ulo) next = cur | ((u32)ulo << 16);
                else break;
                if(atomic_compare_exchange_weak_explicit(&H_at4[idx], &cur, next,
                        memory_order_relaxed, memory_order_relaxed)){
                    break;
                }
            }
        }
    }
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
    _Atomic u16 *P_at = (_Atomic u16 *)H_pack6;
    u64 ovf = 0;

    for(u64 base = start; base + 64 <= end; base += 64){
        for(int i=0; i<64; i++){
            u32 u = (u32)(base + (u64)i);
            u32 y = Qf(u);
            u32 idx = y >> 2;
            b_idx[i] = idx;
            b_ulo[i] = (u16)(u & 0xffffu);
            __builtin_prefetch(&P_at[(size_t)(idx >> 1) * 6u], 1, 0);
        }
        for(int i=0; i<64; i++){
            u16 ulo = b_ulo[i];
            if(!ulo) continue;
            u32 idx = b_idx[i];
            size_t poff = (size_t)(idx >> 1) * 6u;
            if((idx & 1u) == 0u){
                u16 s2 = atomic_load_explicit(&P_at[poff + 2], memory_order_relaxed);
                if(s2 == 0u) continue;
                u16 s0 = atomic_load_explicit(&P_at[poff + 0], memory_order_relaxed);
                u16 s1 = atomic_load_explicit(&P_at[poff + 1], memory_order_relaxed);
                if(s0 == ulo || s1 == ulo || s2 == ulo) continue;
                u16 cur = atomic_load_explicit(&P_at[poff + 3], memory_order_relaxed);
                if(cur == 0u){
                    if(atomic_compare_exchange_strong_explicit(&P_at[poff + 3], &cur, ulo,
                            memory_order_relaxed, memory_order_relaxed)){
                        ovf++;
                    }
                }
            } else {
                u16 s3 = atomic_load_explicit(&P_at[poff + 3], memory_order_relaxed);
                if(s3 == 0u) continue;
                u16 s5 = atomic_load_explicit(&P_at[poff + 5], memory_order_relaxed);
                u16 s4 = atomic_load_explicit(&P_at[poff + 4], memory_order_relaxed);
                if(s5 == ulo || s4 == ulo || s3 == ulo) continue;
                u16 cur = atomic_load_explicit(&P_at[poff + 2], memory_order_relaxed);
                if(cur == 0u){
                    if(atomic_compare_exchange_strong_explicit(&P_at[poff + 2], &cur, ulo,
                            memory_order_relaxed, memory_order_relaxed)){
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
    _Atomic u16 *P_at = (_Atomic u16 *)H_pack6;
    u64 ovf = 0;

    for(u64 base = start; base + 64 <= end; base += 64){
        for(int i=0; i<64; i++){
            u32 u = (u32)(base + (u64)i);
            u32 y = Qf(u);
            u32 idx = y >> 2;
            b_idx[i] = idx;
            b_ulo[i] = (u16)(u & 0xffffu);
            __builtin_prefetch(&P_at[(size_t)(idx >> 1) * 6u], 1, 0);
        }
        for(int i=0; i<64; i++){
            u16 ulo = b_ulo[i];
            if(!ulo) continue;
            u32 idx = b_idx[i];
            size_t poff = (size_t)(idx >> 1) * 6u;
            u16 s0 = atomic_load_explicit(&P_at[poff + 0], memory_order_relaxed);
            u16 s1 = atomic_load_explicit(&P_at[poff + 1], memory_order_relaxed);
            u16 s2 = atomic_load_explicit(&P_at[poff + 2], memory_order_relaxed);
            u16 s3 = atomic_load_explicit(&P_at[poff + 3], memory_order_relaxed);
            u16 s4 = atomic_load_explicit(&P_at[poff + 4], memory_order_relaxed);
            u16 s5 = atomic_load_explicit(&P_at[poff + 5], memory_order_relaxed);
            if(s0 == ulo || s1 == ulo || s2 == ulo || s3 == ulo || s4 == ulo || s5 == ulo) continue;
            if((s0 != 0u) & (s1 != 0u) & (s2 != 0u) & (s3 != 0u) & (s4 != 0u) & (s5 != 0u)) continue;
            for(int s=0; s<6; s++){
                u16 cur = atomic_load_explicit(&P_at[poff + (size_t)s], memory_order_relaxed);
                if(cur == ulo) break;
                if(cur == 0u){
                    if(atomic_compare_exchange_strong_explicit(&P_at[poff + (size_t)s], &cur, ulo,
                            memory_order_relaxed, memory_order_relaxed)){
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
    _Atomic u16 *P_at = (_Atomic u16 *)H_pack6;
    u64 ovf = 0;

    for(u64 base = start; base + 64 <= end; base += 64){
        for(int i=0; i<64; i++){
            u32 u = (u32)(base + (u64)i);
            u32 y = Qf(u);
            u32 idx = y >> 2;
            b_idx[i] = idx;
            b_ulo[i] = (u16)(u & 0xffffu);
            __builtin_prefetch(&P_at[(size_t)(idx >> 2) * 12u], 1, 0);
        }
        for(int i=0; i<64; i++){
            u16 ulo = b_ulo[i];
            if(!ulo) continue;
            u32 idx = b_idx[i];
            size_t qoff = (size_t)(idx >> 2) * 12u;
            int found = 0, has_zero = 0;
            for(int s=0; s<12; s++){
                u16 v = atomic_load_explicit(&P_at[qoff + (size_t)s], memory_order_relaxed);
                if(v == ulo){ found = 1; break; }
                if(v == 0u) has_zero = 1;
            }
            if(found || !has_zero) continue;
            for(int s=0; s<12; s++){
                u16 cur = atomic_load_explicit(&P_at[qoff + (size_t)s], memory_order_relaxed);
                if(cur == ulo) break;
                if(cur == 0u){
                    if(atomic_compare_exchange_strong_explicit(&P_at[qoff + (size_t)s], &cur, ulo,
                            memory_order_relaxed, memory_order_relaxed)){
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
static atomic_flag g_quad_locks[65536];

static inline void quad_lock(u32 quad_idx){
    u32 h = (quad_idx ^ (quad_idx >> 16)) & 0xffffu;
    while(atomic_flag_test_and_set_explicit(&g_quad_locks[h], memory_order_acquire)){
        // spin
    }
}

static inline void quad_unlock(u32 quad_idx){
    u32 h = (quad_idx ^ (quad_idx >> 16)) & 0xffffu;
    atomic_flag_clear_explicit(&g_quad_locks[h], memory_order_release);
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

            // Fast lockless pre-check on the 3 window slots:
            // If all 3 slots in S_SLIDE3[r]..S_SLIDE3[r]+2 already have matched remainder == r,
            // no augmenting path can ever move them because their windows are identical.
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

static void init_q0_table(int nthreads, int table_gb){
    init_rec_tables();
    g_table_gb = table_gb;
    if(table_gb == 32){
        u64 bytes = (1ULL << 32) * sizeof(u64); // 32.00 GB
        fprintf(stderr, "[init] Allocating & building 32.00 GB exact 4-way Q0^-1 table across %d threads...\n", nthreads);
        void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if(p == MAP_FAILED){
            perror("mmap 32GB");
            exit(1);
        }
#ifdef MADV_HUGEPAGE
        madvise(p, bytes, MADV_HUGEPAGE);
#endif
        H_exact32 = (_Atomic u64 *)p;
        pthread_t *th = malloc(nthreads * sizeof(pthread_t));
        build_arg_t *args = calloc(nthreads, sizeof(build_arg_t));
        for(int i=0;i<nthreads;i++){
            args[i].tid = i;
            args[i].nthreads = nthreads;
            args[i].start_u = 0;
            args[i].end_u   = (1ULL << 32);
            pthread_create(&th[i], NULL, build_worker_32gb, &args[i]);
        }
        u64 total_stored = 0;
        for(int i=0;i<nthreads;i++){
            pthread_join(th[i], NULL);
            total_stored += args[i].stored;
        }
        free(th);
        free(args);
        double pct = 100.0 * (double)total_stored / (double)(1ULL << 32);
        fprintf(stderr, "[init] 32.00 GB exact Q0^-1 table built: %llu / 4294967296 preimages stored (%.2f%% coverage, %.2f%% 3-level chain survival)\n",
                (unsigned long long)total_stored, pct, (pct/100.0)*(pct/100.0)*(pct/100.0)*100.0);
        return;
    }

    if(table_gb == 6){
        u64 bytes = (3ULL << 30) * sizeof(u16); // 6.00 GB
        // 1. Check for already-built Sudoku-augmented 6 GB cache file first
        int sfd = open(CACHE_FILE_SUDOKU6, O_RDONLY);
        if(sfd >= 0){
            struct stat st;
            if(fstat(sfd, &st) == 0 && (u64)st.st_size == bytes){
                void *p = mmap(NULL, bytes, PROT_READ, MAP_SHARED, sfd, 0);
                if(p != MAP_FAILED){
#ifdef MADV_WILLNEED
                    madvise(p, bytes, MADV_WILLNEED);
#endif
                    H_pack6 = (u16 *)p;
                    close(sfd);
                    pthread_t *th = malloc(nthreads * sizeof(pthread_t));
                    build_arg_t *args = calloc(nthreads, sizeof(build_arg_t));
                    for(int i=0;i<nthreads;i++){
                        args[i].tid = i;
                        args[i].nthreads = nthreads;
                        pthread_create(&th[i], NULL, prefault_worker_6gb, &args[i]);
                    }
                    for(int i=0;i<nthreads;i++) pthread_join(th[i], NULL);

                    int sample_t2 = 0;
                    for(u32 qi = 0; qi < 16384u; qi++){
                        sample_t2 += tier2_reorder_single_quad(&H_pack6[(size_t)qi * 12u], qi, 0);
                    }
                    if(sample_t2 > 0){
                        fprintf(stderr, "[init] Upgrading cached %s in-place with 2-Tier Nested 4-Slot Sudoku (sample_t2=%d)...\n",
                                CACHE_FILE_SUDOKU6, sample_t2);
                        munmap(p, bytes);
                        int rfd = open(CACHE_FILE_SUDOKU6, O_RDWR);
                        p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, rfd, 0);
                        close(rfd);
                        if(p == MAP_FAILED){ perror("mmap shared sudoku6"); exit(1); }
                        H_pack6 = (u16 *)p;
                        double t_t2 = now_sec();
                        for(int i=0;i<nthreads;i++){
                            args[i].tid = i;
                            args[i].nthreads = nthreads;
                            args[i].stored = 0;
                            pthread_create(&th[i], NULL, tier2_nest4_quad_worker, &args[i]);
                        }
                        u64 total_t2 = 0;
                        for(int i=0;i<nthreads;i++){
                            pthread_join(th[i], NULL);
                            total_t2 += args[i].stored;
                        }
                        mprotect(p, bytes, PROT_READ);
                        fprintf(stderr, "[init] 2-Tier Nested 4-Slot Sudoku placed +%llu Tier-2 preimages in %.2fs\n",
                                (unsigned long long)total_t2, now_sec() - t_t2);
                    }

                    free(th);
                    free(args);
                    madvise(p, bytes, MADV_WILLNEED);
                    fprintf(stderr, "[init] Mapped & pre-faulted cached 6.00 GB 2-Tier Sudoku-Augmented Q0^-1 table from %s\n",
                            CACHE_FILE_SUDOKU6);
                    return;
                }
            }
            close(sfd);
        }

        // 2. Otherwise, load read-only base table into MAP_PRIVATE RAM (never modifying CACHE_FILE_HIER6)
        //    and run the Zone 3 Iterative Sudoku Augmenting-Path Slot Solver (Tier 1 + Tier 2)!
        int fd = open(CACHE_FILE_HIER6, O_RDONLY);
        if(fd >= 0){
            struct stat st;
            if(fstat(fd, &st) == 0 && (u64)st.st_size == bytes){
                void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
                close(fd);
                if(p != MAP_FAILED){
#ifdef MADV_WILLNEED
                    madvise(p, bytes, MADV_WILLNEED);
#endif
                    H_pack6 = (u16 *)p;
                    pthread_t *th = malloc(nthreads * sizeof(pthread_t));
                    build_arg_t *args = calloc(nthreads, sizeof(build_arg_t));
                    for(int i=0;i<nthreads;i++){
                        args[i].tid = i;
                        args[i].nthreads = nthreads;
                        pthread_create(&th[i], NULL, prefault_worker_6gb, &args[i]);
                    }
                    for(int i=0;i<nthreads;i++) pthread_join(th[i], NULL);

                    double t_up0 = now_sec();
                    for(int i=0;i<nthreads;i++){
                        args[i].tid = i;
                        args[i].nthreads = nthreads;
                        args[i].start_u = 0;
                        args[i].end_u   = (1ULL << 32);
                        args[i].stored  = 0;
                        pthread_create(&th[i], NULL, sudoku_upgrade_worker_6gb, &args[i]);
                    }
                    u64 total_upgraded = 0;
                    for(int i=0;i<nthreads;i++){
                        pthread_join(th[i], NULL);
                        total_upgraded += args[i].stored;
                    }
                    double t_t2 = now_sec();
                    for(int i=0;i<nthreads;i++){
                        args[i].tid = i;
                        args[i].nthreads = nthreads;
                        args[i].stored = 0;
                        pthread_create(&th[i], NULL, tier2_nest4_quad_worker, &args[i]);
                    }
                    u64 total_t2 = 0;
                    for(int i=0;i<nthreads;i++){
                        pthread_join(th[i], NULL);
                        total_t2 += args[i].stored;
                    }
                    free(th);
                    free(args);
                    fprintf(stderr, "[init] Sudoku Slot Solver upgraded +%llu Tier-1 (%.2fs) & +%llu Tier-2 (%.2fs) preimages\n",
                            (unsigned long long)total_upgraded, t_t2 - t_up0,
                            (unsigned long long)total_t2, now_sec() - t_t2);

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
                        fprintf(stderr, "[init] Cached upgraded 6.00 GB Sudoku table to %s\n", CACHE_FILE_SUDOKU6);
                    }
                    return;
                }
            } else {
                close(fd);
            }
        }
        fprintf(stderr, "[init] Building 6.00 GB 4-Pass Hierarchical Overflow + EDF Sliding Q0^-1 table (%s) across %d threads...\n",
                CACHE_FILE_HIER6, nthreads);
        void *p = mmap(NULL, bytes + 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if(p == MAP_FAILED){ perror("mmap 6GB"); exit(1); }
        H_pack6 = (u16 *)p;
        pthread_t *th = malloc(nthreads * sizeof(pthread_t));
        build_arg_t *args = calloc(nthreads, sizeof(build_arg_t));
        for(int i=0;i<nthreads;i++){
            args[i].tid = i; args[i].nthreads = nthreads;
            args[i].start_u = 0; args[i].end_u = (1ULL << 32);
            pthread_create(&th[i], NULL, build_worker_6gb, &args[i]);
        }
        u64 total_stored = 0;
        for(int i=0;i<nthreads;i++){ pthread_join(th[i], NULL); total_stored += args[i].stored; }
        for(int i=0;i<nthreads;i++) pthread_create(&th[i], NULL, convert_pack6_to_pair6_pass1_inplace, &args[i]);
        for(int i=0;i<nthreads;i++) pthread_join(th[i], NULL);
        for(int i=0;i<nthreads;i++){ args[i].stored = 0; pthread_create(&th[i], NULL, build_pair6_pass2_ovf, &args[i]); }
        for(int i=0;i<nthreads;i++){ pthread_join(th[i], NULL); total_stored += args[i].stored; }
        for(int i=0;i<nthreads;i++){ args[i].stored = 0; pthread_create(&th[i], NULL, build_hier6_pass3_pair_ovf, &args[i]); }
        for(int i=0;i<nthreads;i++){ pthread_join(th[i], NULL); total_stored += args[i].stored; }
        for(int i=0;i<nthreads;i++){ args[i].stored = 0; pthread_create(&th[i], NULL, build_hier6_pass4_quad_ovf, &args[i]); }
        for(int i=0;i<nthreads;i++){ pthread_join(th[i], NULL); total_stored += args[i].stored; }
        for(int i=0;i<nthreads;i++) pthread_create(&th[i], NULL, edf_slide3_worker, &args[i]);
        for(int i=0;i<nthreads;i++) pthread_join(th[i], NULL);
        for(int i=0;i<nthreads;i++){ args[i].stored = 0; pthread_create(&th[i], NULL, sudoku_upgrade_worker_6gb, &args[i]); }
        for(int i=0;i<nthreads;i++){ pthread_join(th[i], NULL); total_stored += args[i].stored; }
        for(int i=0;i<nthreads;i++){ args[i].stored = 0; pthread_create(&th[i], NULL, tier2_nest4_quad_worker, &args[i]); }
        for(int i=0;i<nthreads;i++){ pthread_join(th[i], NULL); total_stored += args[i].stored; }
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
        }
        double pct = 100.0 * (double)total_stored / (double)(1ULL << 32);
        fprintf(stderr, "[init] Built & cached 6.00 GB 2-Tier Sudoku-Augmented table (%s): %llu / 4294967296 preimages (%.2f%% coverage)\n",
                CACHE_FILE_SUDOKU6, (unsigned long long)total_stored, pct);
        return;
    }

    // 4 GB mode with MADV_WILLNEED, 64B cache-line pre-fault, and automatic 94.75% occupancy verification
    u64 bytes = (1ULL << 30) * sizeof(u32);
    int fd = open(CACHE_FILE_4GB, O_RDONLY);
    if(fd >= 0){
        struct stat st;
        if(fstat(fd, &st) == 0 && (u64)st.st_size == bytes){
            void *p = mmap(NULL, bytes, PROT_READ, MAP_SHARED, fd, 0);
            if(p != MAP_FAILED){
#ifdef MADV_WILLNEED
                madvise(p, bytes, MADV_WILLNEED);
#endif
                H_pack4 = (u32 *)p;
                close(fd);
                pthread_t *th = malloc(nthreads * sizeof(pthread_t));
                build_arg_t *args = calloc(nthreads, sizeof(build_arg_t));
                for(int i=0;i<nthreads;i++){
                    args[i].tid = i;
                    args[i].nthreads = nthreads;
                    pthread_create(&th[i], NULL, prefault_worker_4gb, &args[i]);
                }
                for(int i=0;i<nthreads;i++) pthread_join(th[i], NULL);

                u64 occ_slots = 0;
                for(u64 i = 0; i < (1ULL << 20); i++){
                    u32 v = H_pack4[i * 1024ULL];
                    if(v & 0xffffu) occ_slots++;
                    if(v >> 16) occ_slots++;
                }
                double occ = (double)occ_slots / (double)(2ULL << 20);
                if(occ < 0.935){
                    fprintf(stderr, "[init] Table occupancy %.2f%% < 93.5%%; topping up in private RAM...\n", occ * 100.0);
                    munmap(p, bytes);
                    int rfd = open(CACHE_FILE_4GB, O_RDONLY);
                    p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE, rfd, 0);
                    close(rfd);
                    if(p == MAP_FAILED){ perror("mmap private"); exit(1); }
                    H_pack4 = (u32 *)p;
                    for(int i=0;i<nthreads;i++) pthread_create(&th[i], NULL, prefault_cow_worker_4gb, &args[i]);
                    for(int i=0;i<nthreads;i++) pthread_join(th[i], NULL);
                    for(int i=0;i<nthreads;i++) pthread_create(&th[i], NULL, topup_pass1_worker_4gb, &args[i]);
                    for(int i=0;i<nthreads;i++) pthread_join(th[i], NULL);
                    int wfd = open(CACHE_FILE_4GB, O_WRONLY);
                    if(wfd >= 0){
                        u64 off = 0;
                        const u8 *buf = (const u8 *)H_pack4;
                        while(off < bytes){
                            size_t chunk = (bytes - off > (1ULL << 26)) ? (1ULL << 26) : (size_t)(bytes - off);
                            ssize_t nw = write(wfd, buf + off, chunk);
                            if(nw <= 0) break;
                            off += (u64)nw;
                        }
                        close(wfd);
                    }
                    munmap(p, bytes);
                    rfd = open(CACHE_FILE_4GB, O_RDONLY);
                    p = mmap(NULL, bytes, PROT_READ, MAP_SHARED, rfd, 0);
                    close(rfd);
                    if(p == MAP_FAILED){ perror("mmap shared"); exit(1); }
#ifdef MADV_WILLNEED
                    madvise(p, bytes, MADV_WILLNEED);
#endif
                    H_pack4 = (u32 *)p;
                    for(int i=0;i<nthreads;i++) pthread_create(&th[i], NULL, prefault_worker_4gb, &args[i]);
                    for(int i=0;i<nthreads;i++) pthread_join(th[i], NULL);
                    occ_slots = 0;
                    for(u64 i = 0; i < (1ULL << 20); i++){
                        u32 v = H_pack4[i * 1024ULL];
                        if(v & 0xffffu) occ_slots++;
                        if(v >> 16) occ_slots++;
                    }
                    occ = (double)occ_slots / (double)(2ULL << 20);
                }
                free(th);
                free(args);
                fprintf(stderr, "[init] Mapped & pre-faulted cached 4.00 GB Q0^-1 table from %s (slot_occ=%.2f%%)\n",
                        CACHE_FILE_4GB, occ * 100.0);
                return;
            }
        }
        close(fd);
    }
    H_pack4 = calloc(1ULL << 30, sizeof(u32));
    if(!H_pack4){ fprintf(stderr, "calloc 4GB failed\n"); exit(1); }
    pthread_t *th = malloc(nthreads * sizeof(pthread_t));
    build_arg_t *args = calloc(nthreads, sizeof(build_arg_t));
    for(int i=0;i<nthreads;i++){
        args[i].tid = i;
        args[i].nthreads = nthreads;
        args[i].start_u = 0;
        args[i].end_u   = (1ULL << 32);
        pthread_create(&th[i], NULL, build_worker_4gb, &args[i]);
    }
    for(int i=0;i<nthreads;i++) pthread_join(th[i], NULL);
    free(th);
    free(args);
}

// Exact 32 GB lookup: 0 false positives, NO Qf(u) == y check needed!
// Uses 5-instruction zero-rotate rec_idx(ulo, y) + 128 KB L1-cache-resident REC_HI16
static inline int lookup_exact32(u32 y, u32 *out){
    u64 pack = atomic_load_explicit(&H_exact32[y], memory_order_relaxed);
    if(!pack) return 0;
    int k = 0;
    do {
        u32 ulo = (u32)(pack & 0xffffULL);
        u32 idx = rec_idx(ulo, y);
        out[k++] = ulo | ((u32)REC_HI16[idx] << 16);
        pack >>= 16;
    } while(pack);
    return k;
}

// Dual-Slot Branchless 4 GB lookup (MLP=2 parallel L1 REC_HI16 loads, 0 internal branches)
static inline int lookup_pack4(u32 y, u32 *out){
    u32 pack = H_pack4[y >> 2];
    if(!pack) return 0;
    u32 ulo0 = pack & 0xffffu;
    u32 ulo1 = pack >> 16;
    u32 idx0 = rec_idx(ulo0, y);
    u32 idx1 = rec_idx(ulo1, y);
    u32 u0   = ulo0 | ((u32)REC_HI16[idx0] << 16);
    u32 u1   = ulo1 | ((u32)REC_HI16[idx1] << 16);
    int k = 0;
    out[k] = u0; k += (Qf(u0) == y);
    out[k] = u1; k += (Qf(u1) == y) & (ulo1 != 0u);
    return k;
}

// Sliding 3-Slot Branchless 6 GB lookup (EDF-ordered 12-slot quad, 69.17% 3-slot coverage)
static inline int lookup_pack6(u32 y, u32 *out){
    const u16 *b = &H_pack6[slide3_off(y)];
    u32 ulo0 = b[0];
    u32 ulo1 = b[1];
    u32 ulo2 = b[2];
    if(!(ulo0 | ulo1 | ulo2)) return 0;
    u32 idx0 = rec_idx(ulo0, y);
    u32 idx1 = rec_idx(ulo1, y);
    u32 idx2 = rec_idx(ulo2, y);
    u32 u0   = ulo0 | ((u32)REC_HI16[idx0] << 16);
    u32 u1   = ulo1 | ((u32)REC_HI16[idx1] << 16);
    u32 u2   = ulo2 | ((u32)REC_HI16[idx2] << 16);
    int k = 0;
    out[k] = u0; k += (Qf(u0) == y) & (ulo0 != 0u);
    out[k] = u1; k += (Qf(u1) == y) & (ulo1 != 0u);
    out[k] = u2; k += (Qf(u2) == y) & (ulo2 != 0u);
    return k;
}

// ---------- Right Context Structures & O(1) Padded Generator ----------
typedef struct {
    u32 A_base[R_STEPS + OFF + 2];
    u32 a5, a6;
    u32 T16, T17, T18, T19;
} right_ctx_t;

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

static solver_state_t G_STATE;
static atomic_int g_ctx_count;
static atomic_uint_fast64_t g_ctx_trials;
static atomic_uint_fast64_t g_ctx_seed_gen = 0;
static atomic_uint_fast64_t g_a12_counter  = 0;
static u32                  g_a12_perm_key = 0;
static pthread_mutex_t g_ctx_mu = PTHREAD_MUTEX_INITIALIZER;

static inline u32 perm32(u32 x){
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static void init_iv_and_target(solver_state_t *st_ctx, const u32 dig[8]){
    memset(st_ctx->A_template, 0, sizeof(st_ctx->A_template));
    memcpy(st_ctx->dig, dig, 32);
    u32 wiv[8]; window_from_state(IV, wiv);
    for(int i=0;i<8;i++) st_ctx->A_template[i] = wiv[i]; // a_{-7..0}

    u32 st[8];
    for(int i=0;i<8;i++) st[i] = dig[i] - IV[i];
    u32 wt[8]; window_from_state(st, wt);
    for(int i=0;i<8;i++) st_ctx->A_template[(R_STEPS - 7 + i) + OFF] = wt[i]; // a_{13..20}

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

typedef struct {
    u32 a12, a9, a5;
    u32 T16, T17, T18, T19;
} compact_ctx_t;

#define CTX_POOL_CAP         262144
#define CTX_RESERVOIR_INIT   16384
#define CTX_REPLENISH_BATCH  512
static compact_ctx_t g_ctx_pool[CTX_POOL_CAP];
static u8            g_ctx_pool_used[CTX_POOL_CAP];
static int           g_ctx_pool_len = 0;
static int           g_ctx_target_gen = 1;
static int           g_ctx_init_pool  = 4096;

static void expand_compact_ctx(const compact_ctx_t *cc, right_ctx_t *rc){
    u32 A[R_STEPS + OFF + 2];
    memcpy(A, G_STATE.A_template, sizeof(A));
    const u32 a13 = A[13+OFF], a14 = A[14+OFF], a15 = A[15+OFF], a16 = A[16+OFF];
    const u32 a12 = cc->a12, a9 = cc->a9;
    const u32 e15 = 0xffffffffu;
    u32 a11 = e15 - a15 + S0(a14) + MAJ(a14, a13, a12);
    u32 a10 = a11;
    u32 e14 = a14 - S0(a13) - MAJ(a13, a12, a11) + a10;
    u32 e16 = a16 - S0(a15) - MAJ(a15, a14, a13) + a12;
    u32 e13 = a13 - S0(a12) - a11 + a9;
    u32 c_e12 = e16 - a12 - 0xffffffffu - K[15] - (u32)PADLEN_BITS;
    u32 c_e11 = e15 - a11 - S1(e14) - K[14];
    u32 e12 = c_e12 - e14;
    u32 e11 = c_e11 - CH(e14, e13, e12);
    u32 a8  = e12 + S0(a11) - a12 + a11;
    u32 a7  = e11 + S0(a10) - a11 + MAJ(a10, a9, a8);
    u32 a6  = 0xffffffffu - a10 + S0(a9) + MAJ(a9, a8, a7);
    u32 a5  = a6;

    A[12+OFF] = a12; A[11+OFF] = a11; A[10+OFF] = a10; A[9+OFF] = a9;
    A[8+OFF]  = a8;  A[7+OFF]  = a7;  A[6+OFF]  = a6;  A[5+OFF] = a5;
    A[4+OFF]  = 0;   A[3+OFF]  = 0;   A[2+OFF]  = 0;   A[1+OFF] = 0;

    memcpy(rc->A_base, A, sizeof(A));
    rc->a5  = cc->a5;
    rc->a6  = cc->a5;
    rc->T16 = cc->T16;
    rc->T17 = cc->T17;
    rc->T18 = cc->T18;
    rc->T19 = cc->T19;
}

// Breakthrough 1, 2, 3 & 9 (Phase 8):
// - Exact O(1) (W13 & 0xff) == 0x80 inversion with Fused Bit-18 (b18) Early Inversion at Bit 13
// - Full Bit-0..24 7-Stage Causal Carry-Sudoku Cascade (e15 = 0xffffffffu, a10 = a11) -> 54.7x fewer Stage-1 evals, ~16x faster context generation!
// - Exact Carry-Sudoku Equivalence Theorem: ((a6 ^ S) & ~D) == 0 replaces a5 = X + MAJ(a8, a7, a6)
static void *ctx_worker(void *arg){
    (void)arg;
    u32 A[R_STEPS + OFF + 2];
    memcpy(A, G_STATE.A_template, sizeof(A));
    const u32 a13 = A[13+OFF], a14 = A[14+OFF], a15 = A[15+OFF], a16 = A[16+OFF];
    const int target_count = g_ctx_target_gen;
    u64 local_tried = 0;

    while(atomic_load_explicit(&g_ctx_count, memory_order_relaxed) < target_count){
        u64 my_idx = atomic_fetch_add_explicit(&g_a12_counter, 1ULL, memory_order_relaxed);
        u32 a12 = perm32((u32)my_idx + g_a12_perm_key);
        // Set e15 = 0xffffffffu so CH(e15, e14, e13) == e14 is 100% independent of e13 across all 2^24 trials!
        const u32 e15 = 0xffffffffu;
        u32 a11 = e15 - a15 + S0(a14) + MAJ(a14, a13, a12);
        // Set a10 = a11 so MAJ(a11, a10, a9) = a11 (making a8 and S0(a8) 100% constant across all 2^24 trials!)
        u32 a10 = a11;
        u32 e14 = a14 - S0(a13) - MAJ(a13, a12, a11) + a10;
        u32 e16 = a16 - S0(a15) - MAJ(a15, a14, a13) + a12;

        u32 c_e13 = a13 - S0(a12) - a11;
        u32 c_e12 = e16 - a12 - 0xffffffffu - K[15] - (u32)PADLEN_BITS;
        u32 c_e11 = e15 - a11 - S1(e14) - K[14];
        u32 c_W13 = e14 - a10 - 0xffffffffu - K[13];

        u32 c_a8 = S0(a11) - a12;
        u32 c_a7 = S0(a10) - a11;
        u32 c_a6 = 0xffffffffu - a10;

        u32 e12 = c_e12 - e14;
        u32 a8  = e12 + c_a8 + a11;
        u32 c_a5_s0a8  = 0xffffffffu + S0(a8);
        u32 m_10_8     = a10 ^ a8;
        u32 a10_and_a8 = a10 & a8;
        u32 c_e11_base = c_e11 - (~e14 & e12);
        u32 c_a7_full  = c_e11_base + c_a7 + a10_and_a8;

        local_tried += (1ULL << 24);

        // Stage 0a: bits 0..3 of e13_lo (16 values, 4-bit Carry-Sudoku check & 0xfu)
        for(u32 s0 = 0; s0 < 16u; s0++){
            u32 a9_0a  = s0 - c_e13;
            u32 X_0a   = c_a5_s0a8 - a9_0a;
            u32 a7_b0a = c_a7_full - (e14 & s0);
            u32 a7_0a  = a7_b0a + (m_10_8 & a9_0a);
            u32 D_0a   = a8 ^ a7_0a;
            if((((X_0a + CH(D_0a, X_0a, a8)) ^ X_0a) & D_0a & 0xfu) != 0u) continue;

            // Stage 0b: bits 4..7 of e13_lo (16 values, 8-bit Carry-Sudoku check & 0xffu)
            for(u32 s4 = 0; s4 < (16u << 4); s4 += (1u << 4)){
                u32 a9_0  = a9_0a + s4;
                u32 X_0   = X_0a  - s4;
                u32 a7_b0 = a7_b0a - (e14 & s4);
                u32 a7_0  = a7_b0 + (m_10_8 & a9_0);
                u32 D_0   = a8 ^ a7_0;
                if((((X_0 + CH(D_0, X_0, a8)) ^ X_0) & D_0 & 0xffu) != 0u) continue;

                u32 e13_lo = s0 | s4;
                u32 e11_lo = (c_e11_base - (e14 & e13_lo)) & 0xffu;
                u32 req_S1_lo = (c_W13 - CH(e13_lo, e12, e11_lo) - 0x80u) & 0xffu;
                u32 req_b7_xor_lo0 = ((req_S1_lo >> 7) ^ e13_lo) & 1u;

                // Stage 1a: bits 8..11 (16 values, 12-bit Carry-Sudoku check & 0xfffu)
                for(u32 s8 = 0; s8 < (16u << 8); s8 += (1u << 8)){
                    u32 a9_1  = a9_0 + s8;
                    u32 X_1   = X_0  - s8;
                    u32 a7_b1 = a7_b0 - (e14 & s8);
                    u32 a7_1  = a7_b1 + (m_10_8 & a9_1);
                    u32 D_1   = a8 ^ a7_1;
                    if((((X_1 + CH(D_1, X_1, a8)) ^ X_1) & D_1 & 0xfffu) != 0u) continue;

                    // Stage 1b: bits 12..14 (8 values) + exact bit 18 fused! (15-bit Carry-Sudoku check & 0x7fffu)
                    for(u32 s12 = 0; s12 < (8u << 12); s12 += (1u << 12)){
                        u32 b18   = ((req_b7_xor_lo0 ^ (s12 >> 13)) & 1u) << 18;
                        u32 s12_b = s12 | b18;
                        u32 a9_2  = a9_1 + s12_b;
                        u32 X_2   = X_1  - s12_b;
                        u32 a7_b2 = a7_b1 - (e14 & s12_b);
                        u32 a7_2  = a7_b2 + (m_10_8 & a9_2);
                        u32 D_2   = a8 ^ a7_2;
                        if((((X_2 + CH(D_2, X_2, a8)) ^ X_2) & D_2 & 0x7fffu) != 0u) continue;

                        // Stage 1c: bits 15..17 (8 values) + checks bits 15..18 (& 0x7ffffu) BEFORE S1(e13_base)!
                        u32 e13_14 = e13_lo | s8 | s12;
                        for(u32 s15 = 0; s15 < (8u << 15); s15 += (1u << 15)){
                            u32 a9_5  = a9_2 + s15;
                            u32 X_5   = X_2  - s15;
                            u32 a7_b5 = a7_b2 - (e14 & s15);
                            u32 a7_5  = a7_b5 + (m_10_8 & a9_5);
                            u32 D_5   = a8 ^ a7_5;
                            u32 S_5 = X_5 + CH(D_5, X_5, a8);
                            if(((S_5 ^ X_5) & D_5 & 0x7ffffu) != 0u) continue;

                            u32 e13_base = e13_14 | s15;
                            u32 diff = (req_S1_lo ^ S1(e13_base)) & 0x7fu;
                            u32 hi25 = diff << 25;
                            u32 a9_base = a9_5 + hi25;
                            u32 X0      = X_5  - hi25;
                            u32 a7_base = a7_b5 - (e14 & hi25);

                            // Stage 2a: bits 19..21 (8 values, 22-bit Carry-Sudoku check & 0x3fffffu)
                            for(u32 s19 = 0; s19 < (8u << 19); s19 += (1u << 19)){
                                u32 a9_i1  = a9_base + s19;
                                u32 X_i1   = X0 - s19;
                                u32 a7_bi1 = a7_base - (e14 & s19);
                                u32 a7_i1  = a7_bi1 + (m_10_8 & a9_i1);
                                u32 D_i1   = a8 ^ a7_i1;
                                if((((X_i1 + CH(D_i1, X_i1, a8)) ^ X_i1) & D_i1 & 0x3fffffu) != 0u) continue;

                                // Stage 3b: bits 22..24 (8 values, full 32-bit Carry-Sudoku + Exact Equivalence check)
                                for(u32 s22 = 0; s22 < (8u << 22); s22 += (1u << 22)){
                                    u32 a9 = a9_i1 + s22;
                                    u32 X  = X_i1  - s22;
                                    u32 a7 = a7_bi1 - (e14 & s22) + (m_10_8 & a9);
                                    u32 D  = a8 ^ a7;
                                    u32 S  = X + CH(D, X, a8);
                                    if(__builtin_expect(((S ^ X) & D) != 0u, 1)) continue;
                                    u32 a6 = c_a6 + S0(a9) + MAJ(a9, a8, a7);
                                    if(__builtin_expect(((a6 ^ S) & ~D) == 0u, 0)){
                                        u32 a5 = a6;
                                        A[12+OFF] = a12; A[11+OFF] = a11; A[10+OFF] = a10; A[9+OFF] = a9;
                                        A[8+OFF]  = a8;  A[7+OFF]  = a7;  A[6+OFF]  = a6;  A[5+OFF] = a5;
                                        A[4+OFF]  = 0;   A[3+OFF]  = 0;   A[2+OFF]  = 0;   A[1+OFF] = 0;

                                        compact_ctx_t local_cc;
                                        local_cc.a12 = a12;
                                        local_cc.a9  = a9;
                                        local_cc.a5  = a5;

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

                                        local_cc.T16 = W16 - s1(W14) - c9;
                                        local_cc.T17 = W17 - s1(W15) - c10;
                                        local_cc.T18 = W18 - s1(W16) - c11;
                                        local_cc.T19 = W19 - s1(W17) - W12;

                                        pthread_mutex_lock(&g_ctx_mu);
                                        int idx = atomic_load_explicit(&g_ctx_count, memory_order_relaxed);
                                        if(idx < target_count && g_ctx_pool_len < CTX_POOL_CAP){
                                            g_ctx_pool[g_ctx_pool_len] = local_cc;
                                            g_ctx_pool_used[g_ctx_pool_len] = 0;
                                            g_ctx_pool_len++;
                                            atomic_store_explicit(&g_ctx_count, idx + 1, memory_order_relaxed);
                                        }
                                        pthread_mutex_unlock(&g_ctx_mu);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    atomic_fetch_add(&g_ctx_trials, local_tried);
    return NULL;
}

static int cmp_pool_idx_by_t16(const void *a, const void *b){
    int ia = *(const int *)a;
    int ib = *(const int *)b;
    u32 ta = g_ctx_pool[ia].T16;
    u32 tb = g_ctx_pool[ib].T16;
    if(ta != tb) return (ta > tb) - (ta < tb);
    u32 a12a = g_ctx_pool[ia].a12, a12b = g_ctx_pool[ib].a12;
    if(a12a != a12b) return (a12a > a12b) - (a12a < a12b);
    u32 a9a = g_ctx_pool[ia].a9, a9b = g_ctx_pool[ib].a9;
    return (a9a > a9b) - (a9a < a9b);
}

static int g_avail_idx[CTX_POOL_CAP];

#define CACHE_FILE_CTX_POOL "/tmp/ctx_pool_2a.bin"

typedef struct {
    u32 magic;
    u32 count;
    u32 dig[8];
} ctx_pool_hdr_t;

static u64 generate_right_contexts_parallel(int nthreads){
    int B = G_STATE.batch_ctx;
    if(g_ctx_pool_len == 0){
        int cfd = open(CACHE_FILE_CTX_POOL, O_RDONLY);
        if(cfd >= 0){
            ctx_pool_hdr_t hdr;
            if(read(cfd, &hdr, sizeof(hdr)) == (ssize_t)sizeof(hdr) &&
               hdr.magic == 0x43545832u && hdr.count > 0 && hdr.count <= CTX_POOL_CAP &&
               memcmp(hdr.dig, G_STATE.dig, 32) == 0){
                ssize_t need = (ssize_t)(hdr.count * sizeof(compact_ctx_t));
                if(read(cfd, g_ctx_pool, (size_t)need) == need){
                    memset(g_ctx_pool_used, 0, hdr.count);
                    g_ctx_pool_len = (int)hdr.count;
                }
            }
            close(cfd);
        }
    }

    int unused = 0;
    for(int i=0; i<g_ctx_pool_len; i++){
        if(!g_ctx_pool_used[i]) unused++;
    }

    int gen_target;
    if(B == 1){
        gen_target = (unused >= 1) ? 0 : 32;
    } else if(g_ctx_pool_len == 0){
        gen_target = g_ctx_init_pool;
    } else if(unused < B){
        gen_target = CTX_REPLENISH_BATCH;
    } else {
        gen_target = 0;
    }
    if(unused + gen_target < B) gen_target = B - unused;

    if(g_ctx_pool_len + gen_target > CTX_POOL_CAP){
        int dst = 0;
        for(int i=0; i<g_ctx_pool_len; i++){
            if(!g_ctx_pool_used[i]){
                g_ctx_pool[dst] = g_ctx_pool[i];
                g_ctx_pool_used[dst] = 0;
                dst++;
            }
        }
        g_ctx_pool_len = dst;
        if(g_ctx_pool_len + gen_target > CTX_POOL_CAP){
            gen_target = CTX_POOL_CAP - g_ctx_pool_len;
        }
    }

    atomic_store(&g_ctx_count, 0);
    atomic_store(&g_ctx_trials, 0);
    g_ctx_target_gen = gen_target;

    if(gen_target > 0){
        pthread_t *th = malloc(nthreads * sizeof(pthread_t));
        for(int i=0;i<nthreads;i++) pthread_create(&th[i], NULL, ctx_worker, (void *)(intptr_t)i);
        for(int i=0;i<nthreads;i++) pthread_join(th[i], NULL);
        free(th);

        int cfd = open(CACHE_FILE_CTX_POOL, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if(cfd >= 0){
            ctx_pool_hdr_t hdr;
            hdr.magic = 0x43545832u;
            hdr.count = (u32)g_ctx_pool_len;
            memcpy(hdr.dig, G_STATE.dig, 32);
            (void)write(cfd, &hdr, sizeof(hdr));
            (void)write(cfd, g_ctx_pool, (size_t)g_ctx_pool_len * sizeof(compact_ctx_t));
            close(cfd);
        }
    }

    int n_avail = 0;
    for(int i=0; i<g_ctx_pool_len; i++){
        if(!g_ctx_pool_used[i]) g_avail_idx[n_avail++] = i;
    }

    if(B == 1 || n_avail <= B){
        for(int c=0; c<B && c<n_avail; c++){
            int idx = g_avail_idx[c];
            expand_compact_ctx(&g_ctx_pool[idx], &G_STATE.rctx[c]);
            g_ctx_pool_used[idx] = 1;
        }
    } else {
        qsort(g_avail_idx, (size_t)n_avail, sizeof(int), cmp_pool_idx_by_t16);
        int w = 0;
        for(int i=0; i<n_avail; i++){
            int idx = g_avail_idx[i];
            if(w > 0){
                int prev = g_avail_idx[w - 1];
                if(g_ctx_pool[idx].a12 == g_ctx_pool[prev].a12 &&
                   g_ctx_pool[idx].a9  == g_ctx_pool[prev].a9){
                    g_ctx_pool_used[idx] = 1;
                    continue;
                }
            }
            g_avail_idx[w++] = idx;
        }
        n_avail = w;

        if((B & 1) == 0 && B >= 2){
            int n_pairs = B >> 1;
            for(int p = 0; p < n_pairs; p++){
                u32 best_span = 0xffffffffu;
                int best_pos  = -1;
                for(int i = 0; i <= n_avail - 2; i++){
                    int idx0 = g_avail_idx[i], idx1 = g_avail_idx[i + 1];
                    if(g_ctx_pool_used[idx0] || g_ctx_pool_used[idx1]) continue;
                    u32 span = g_ctx_pool[idx1].T16 - g_ctx_pool[idx0].T16;
                    if(span < best_span){
                        best_span = span;
                        best_pos  = i;
                    }
                }
                if(best_pos >= 0){
                    int idx0 = g_avail_idx[best_pos], idx1 = g_avail_idx[best_pos + 1];
                    expand_compact_ctx(&g_ctx_pool[idx0], &G_STATE.rctx[2 * p]);
                    expand_compact_ctx(&g_ctx_pool[idx1], &G_STATE.rctx[2 * p + 1]);
                    g_ctx_pool_used[idx0] = 1;
                    g_ctx_pool_used[idx1] = 1;
                }
            }
        } else {
            u32 best_span = 0xffffffffu;
            int best_pos  = 0;
            for(int i=0; i <= n_avail - B; i++){
                u32 span = g_ctx_pool[g_avail_idx[i + B - 1]].T16 - g_ctx_pool[g_avail_idx[i]].T16;
                if(span < best_span){
                    best_span = span;
                    best_pos  = i;
                }
            }
            for(int c=0; c<B; c++){
                int idx = g_avail_idx[best_pos + c];
                expand_compact_ctx(&g_ctx_pool[idx], &G_STATE.rctx[c]);
                g_ctx_pool_used[idx] = 1;
            }
        }
    }
    return atomic_load(&g_ctx_trials);
}

// ---------- 3-Stage Pipelined Left-to-Right Tree Search ----------
static atomic_uint_fast64_t g_next_a1_chunk;
static atomic_uint_fast64_t g_chains_done;
static atomic_uint_fast64_t g_l2_nodes;
static atomic_uint_fast64_t g_l3_lookups;
static atomic_uint_fast64_t g_sol19_count;
static atomic_int g_solved20;

#define CHUNK_SIZE (1ULL << 18)
#define PF_BATCH 64

typedef struct {
    u32 t17, G2, a1, e1, a2, e2, sm2, ctx_id;
} q2_item_t;

// Compact 28-byte Level-3 queue item (recomputes e1,e2,e3,sm3 only on the 0.05% sol19 hits!)
typedef struct {
    u32 t18, G3, M3, a1, a2, a3, ctx_id;
} q3_item_t;

static inline void prefetch_q0(u32 y){
    if(g_table_gb == 32){
        __builtin_prefetch(&H_exact32[y], 0, 0);
    } else if(g_table_gb == 6){
        __builtin_prefetch(&H_pack6[((size_t)y * 3u) >> 2], 0, 0);
    } else {
        __builtin_prefetch(&H_pack4[y >> 2], 0, 0);
    }
}

static inline int lookup_q0(u32 y, u32 *out){
    if(g_table_gb == 32){
        return lookup_exact32(y, out);
    } else if(g_table_gb == 6){
        return lookup_pack6(y, out);
    } else {
        return lookup_pack4(y, out);
    }
}

// Breakthrough 5: Fused Low-16-Bit Algebraic Early Rejection for Level-3 (2.10x faster!)
// Since (a4)_lo16 = (ulo + G3) & 0xffff depends ONLY on ulo (0 dependence on REC_HI16),
// checking (((ulo + G3) ^ a5) & (M3 & 0xffff)) == 0 rejects ~97.8% of candidates in 3 ALU ops
// BEFORE computing rec_idx(ulo, t18), loading REC_HI16[idx], or checking Qf(u) == t18!
static inline int lookup_q0_q3_pack4(u32 t18, u32 G3, u32 a5, u32 M3, u32 *out_W3){
    u32 pack = H_pack4[t18 >> 2];
    if(!pack) return 0;
    u32 M3_lo = M3 & 0xffffu;
    int k = 0;
    u32 ulo0 = pack & 0xffffu;
    if(__builtin_expect((((ulo0 + G3) ^ a5) & M3_lo) == 0u, 0)){
        u32 idx0 = rec_idx(ulo0, t18);
        u32 u0 = ulo0 | ((u32)REC_HI16[idx0] << 16);
        if((((u0 + G3) ^ a5) & M3) == 0u && Qf(u0) == t18) out_W3[k++] = u0;
    }
    u32 ulo1 = pack >> 16;
    if(__builtin_expect((((ulo1 + G3) ^ a5) & M3_lo) == 0u, 0) && ulo1){
        u32 idx1 = rec_idx(ulo1, t18);
        u32 u1 = ulo1 | ((u32)REC_HI16[idx1] << 16);
        if((((u1 + G3) ^ a5) & M3) == 0u && Qf(u1) == t18) out_W3[k++] = u1;
    }
    return k;
}

static inline int lookup_q0_q3_pack6(u32 t18, u32 G3, u32 a5, u32 M3, u32 *out_W3){
    const u64 K7 = 0x7fff7fff7fff7fffULL;
    const u64 K8 = 0x8000800080008000ULL;
    const u64 K1 = 0x0001000100010001ULL;
    size_t off = (size_t)(t18 >> 4) * 12u;
    const u64 *q64 = (const u64 *)&H_pack6[off];
    u64 w0 = q64[0], w1 = q64[1], w2 = q64[2];
    u64 g64 = (u64)(G3 & 0x7fffu) * K1;
    u64 a64 = (u64)(a5 & 0x7fffu) * K1;
    u64 m64 = (u64)(M3 & 0x7fffu) * K1;
    u64 d0 = ((((w0 & K7) + g64) ^ a64) & m64) + K7;
    u64 d1 = ((((w1 & K7) + g64) ^ a64) & m64) + K7;
    u64 d2 = ((((w2 & K7) + g64) ^ a64) & m64) + K7;
    if(__builtin_expect(((d0 & d1 & d2) & K8) == K8, 1)) return 0;
    const u16 *q = (const u16 *)q64;
    u32 M3_lo = M3 & 0xffffu;
    int k = 0;
    u64 h0 = ~d0 & K8;
    while(h0){
        int s = __builtin_ctzll(h0) >> 4;
        h0 &= h0 - 1ULL;
        u32 ulo = q[s];
        if(ulo && (((ulo + G3) ^ a5) & M3_lo) == 0u){
            u32 idx = rec_idx(ulo, t18);
            u32 u = ulo | ((u32)REC_HI16[idx] << 16);
            if((((u + G3) ^ a5) & M3) == 0u && Qf(u) == t18) out_W3[k++] = u;
        }
    }
    u64 h1 = ~d1 & K8;
    while(h1){
        int s = 4 + (__builtin_ctzll(h1) >> 4);
        h1 &= h1 - 1ULL;
        u32 ulo = q[s];
        if(ulo && (((ulo + G3) ^ a5) & M3_lo) == 0u){
            u32 idx = rec_idx(ulo, t18);
            u32 u = ulo | ((u32)REC_HI16[idx] << 16);
            if((((u + G3) ^ a5) & M3) == 0u && Qf(u) == t18) out_W3[k++] = u;
        }
    }
    u64 h2 = ~d2 & K8;
    while(h2){
        int s = 8 + (__builtin_ctzll(h2) >> 4);
        h2 &= h2 - 1ULL;
        u32 ulo = q[s];
        if(ulo && (((ulo + G3) ^ a5) & M3_lo) == 0u){
            u32 idx = rec_idx(ulo, t18);
            u32 u = ulo | ((u32)REC_HI16[idx] << 16);
            if((((u + G3) ^ a5) & M3) == 0u && Qf(u) == t18) out_W3[k++] = u;
        }
    }
    return k;
}

static inline int lookup_q0_q3_fused(u32 t18, u32 G3, u32 a5, u32 M3, u32 *out_W3){
    if(g_table_gb == 32){
        u32 M3_lo = M3 & 0xffffu;
        int k = 0;
        u64 pack = atomic_load_explicit(&H_exact32[t18], memory_order_relaxed);
        if(!pack) return 0;
        do {
            u32 ulo = (u32)(pack & 0xffffULL);
            if(__builtin_expect((((ulo + G3) ^ a5) & M3_lo) == 0u, 0)){
                u32 idx = rec_idx(ulo, t18);
                u32 u = ulo | ((u32)REC_HI16[idx] << 16);
                if((((u + G3) ^ a5) & M3) == 0u) out_W3[k++] = u;
            }
            pack >>= 16;
        } while(pack);
        return k;
    } else if(g_table_gb == 6){
        return lookup_q0_q3_pack6(t18, G3, a5, M3, out_W3);
    } else {
        return lookup_q0_q3_pack4(t18, G3, a5, M3, out_W3);
    }
}

// Breakthrough 6 (Phase 3): Dual-Slot Branchless Level-2 Fused Lookup!
// Evaluates both slots (u0, u1) in parallel across dual L1 load pipelines with 0 internal branches!
static inline int lookup_q0_q2_pack4_fast(u32 t17, u32 G2, u32 a5, int w_thresh, u32 *out_W2, u32 *out_M3){
    u32 pack = H_pack4[t17 >> 2];
    if(!pack) return 0;
    u32 ulo0 = pack & 0xffffu;
    u32 ulo1 = pack >> 16;
    u32 idx0 = rec_idx(ulo0, t17);
    u32 idx1 = rec_idx(ulo1, t17);
    u32 u0   = ulo0 | ((u32)REC_HI16[idx0] << 16);
    u32 u1   = ulo1 | ((u32)REC_HI16[idx1] << 16);
    u32 m3_0 = ~((u0 + G2) ^ a5);
    u32 m3_1 = ~((u1 + G2) ^ a5);
    int k = 0;
    out_W2[k] = u0; out_M3[k] = m3_0;
    k += (Qf(u0) == t17) & (__builtin_popcount(m3_0) <= w_thresh);
    out_W2[k] = u1; out_M3[k] = m3_1;
    k += (Qf(u1) == t17) & (__builtin_popcount(m3_1) <= w_thresh) & (ulo1 != 0u);
    return k;
}

static inline int lookup_q0_q2_pack6_fast(u32 t17, u32 G2, u32 a5, int w_thresh, u32 *out_W2, u32 *out_M3){
    const u16 *b = &H_pack6[slide3_off(t17)];
    u32 ulo0 = b[0];
    u32 ulo1 = b[1];
    u32 ulo2 = b[2];
    if(!(ulo0 | ulo1 | ulo2)) return 0;
    u32 idx0 = rec_idx(ulo0, t17);
    u32 idx1 = rec_idx(ulo1, t17);
    u32 idx2 = rec_idx(ulo2, t17);
    u32 u0   = ulo0 | ((u32)REC_HI16[idx0] << 16);
    u32 u1   = ulo1 | ((u32)REC_HI16[idx1] << 16);
    u32 u2   = ulo2 | ((u32)REC_HI16[idx2] << 16);
    u32 m3_0 = ~((u0 + G2) ^ a5);
    u32 m3_1 = ~((u1 + G2) ^ a5);
    u32 m3_2 = ~((u2 + G2) ^ a5);
    int k = 0;
    out_W2[k] = u0; out_M3[k] = m3_0;
    k += (Qf(u0) == t17) & (__builtin_popcount(m3_0) <= w_thresh) & (ulo0 != 0u);
    out_W2[k] = u1; out_M3[k] = m3_1;
    k += (Qf(u1) == t17) & (__builtin_popcount(m3_1) <= w_thresh) & (ulo1 != 0u);
    out_W2[k] = u2; out_M3[k] = m3_2;
    k += (Qf(u2) == t17) & (__builtin_popcount(m3_2) <= w_thresh) & (ulo2 != 0u);
    return k;
}

static inline int lookup_q0_q2_fused(u32 t17, u32 G2, u32 a5, int w_thresh, u32 *out_W2, u32 *out_M3){
    if(g_table_gb == 32){
        int k = 0;
        u64 pack = atomic_load_explicit(&H_exact32[t17], memory_order_relaxed);
        if(!pack) return 0;
        do {
            u32 ulo = (u32)(pack & 0xffffULL);
            u32 idx = rec_idx(ulo, t17);
            u32 u = ulo | ((u32)REC_HI16[idx] << 16);
            u32 m3 = ~((u + G2) ^ a5);
            out_W2[k] = u;
            out_M3[k] = m3;
            k += (__builtin_popcount(m3) <= w_thresh);
            pack >>= 16;
        } while(pack);
        return k;
    } else if(g_table_gb == 6){
        return lookup_q0_q2_pack6_fast(t17, G2, a5, w_thresh, out_W2, out_M3);
    } else {
        return lookup_q0_q2_pack4_fast(t17, G2, a5, w_thresh, out_W2, out_M3);
    }
}

typedef struct {
    u32 t17, G2;
} q2_hot_t;

typedef struct {
    u32 a1, e1, a2, e2, sm2;
} q2_cold_t;

static inline void drain_q3(q3_item_t *q3, int n_q3, const solver_state_t *st, u64 *local_sol19){
    u32 cand3[4];
    const u32 a0 = st->a0, am1 = st->am1, am2 = st->am2, C_e1 = st->C_e1;

    for(int i=0; i<n_q3; i++){
        u32 t18 = q3[i].t18;
        u32 G3  = q3[i].G3, M3 = q3[i].M3;
        u32 cid = q3[i].ctx_id;
        u32 a5  = st->rctx[cid].a5;

        int n3 = lookup_q0_q3_fused(t18, G3, a5, M3, cand3);
        if(__builtin_expect(!n3, 1)) continue;

        for(int z=0; z<n3; z++){
            u32 W3 = cand3[z];
            u32 a4 = W3 + G3;

            (*local_sol19)++;

            // Reconstruct (e1, e2, e3, sm3) only for genuine 19-round solutions (0.05% of q3)!
            u32 a1 = q3[i].a1, a2 = q3[i].a2, a3 = q3[i].a3;
            u32 e1 = a1 + C_e1;
            u32 sm1 = S0(a1) + MAJ(a1, a0, am1);
            u32 e2 = a2 - sm1 + am2;
            u32 sm2 = S0(a2) + MAJ(a2, a1, a0);
            u32 e3 = a3 - sm2 + am1;
            u32 sm3 = S0(a3) + MAJ(a3, a2, a1);

            u32 e4 = a4 - sm3 + a0;
            u32 sm4 = S0(a4) + MAJ(a4, a3, a2);
            u32 W4 = a5 - sm4 - e1 - S1(e4) - CH(e4, e3, e2) - K[4];
            if(__builtin_expect(s0(W4) + W3 == st->rctx[cid].T19, 0)){
                u32 A_sol[R_STEPS + OFF + 2];
                memcpy(A_sol, st->rctx[cid].A_base, sizeof(A_sol));
                A_sol[1+OFF] = a1; A_sol[2+OFF] = a2;
                A_sol[3+OFF] = a3; A_sol[4+OFF] = a4;
                u32 Wm[16], out[8];
                for(int t=0;t<16;t++) Wm[t] = Wt_arr(A_sol, t);
                compress_r(Wm, R_STEPS, out);
                int ok = 1;
                for(int k=0;k<8;k++) ok &= (out[k] == st->dig[k]);
                ok &= ((Wm[13] & 0xffu) == 0x80u) && (Wm[14] == 0u) && (Wm[15] == (u32)PADLEN_BITS);
                printf("\n*** FOUND 20-STEP PADDED PREIMAGE (ok=%d) ***\n", ok);
                printf("W =");
                for(int t=0;t<16;t++) printf(" %08x", Wm[t]);
                printf("\nH20=");
                for(int k=0;k<8;k++) printf("%08x", out[k]);
                printf("\n");
                fflush(stdout);
                if(ok){
                    atomic_store_explicit(&g_solved20, 1, memory_order_relaxed);
                    return;
                }
            }
        }
    }
}

static inline void drain_q2(q2_item_t *q2, int n_q2, q3_item_t *q3, int *n_q3,
                            const solver_state_t *st, u32 am1, u32 C_G3, int w_thresh,
                            u64 *local_l2, u64 *local_l3, u64 *local_sol19){
    u32 cand2[4], m3_arr[4];
    for(int i=0; i<n_q2; i++){
        u32 t17 = q2[i].t17;
        u32 G2  = q2[i].G2;
        u32 cid = q2[i].ctx_id;
        u32 a5  = st->rctx[cid].a5;
        int n2 = lookup_q0_q2_fused(t17, G2, a5, w_thresh, cand2, m3_arr);
        if(!n2) continue;
        *local_l2 += (u64)n2;

        u32 T18 = st->rctx[cid].T18;

        for(int y=0; y<n2; y++){
            u32 W2 = cand2[y];
            u32 a3 = W2 + G2;
            u32 M3 = m3_arr[y];

            // Load a1, e1, a2, e2, sm2 ONLY for the ~7-15% of nodes surviving popcount(M3) <= w_thresh
            u32 a1 = q2[i].a1, e1 = q2[i].e1;
            u32 a2 = q2[i].a2, e2 = q2[i].e2, sm2 = q2[i].sm2;

            u32 e3 = a3 - sm2 + am1;
            u32 sm3 = S0(a3) + MAJ(a3, a2, a1);
            u32 G3 = sm3 + S1(e3) + CH(e3, e2, e1) + C_G3;
            u32 t18 = T18 + G3 - W2;

            (*local_l3)++;
            prefetch_q0(t18);

            int idx3 = (*n_q3)++;
            q3[idx3].t18 = t18;
            q3[idx3].G3  = G3;
            q3[idx3].M3  = M3;
            q3[idx3].a1  = a1;
            q3[idx3].a2  = a2;
            q3[idx3].a3  = a3;
            q3[idx3].ctx_id = cid;

            if(*n_q3 == PF_BATCH){
                drain_q3(q3, PF_BATCH, st, local_sol19);
                *n_q3 = 0;
            }
        }
    }
}

static inline __attribute__((unused)) void drain_q2_4gb_b1(const q2_hot_t *q2_hot, const q2_cold_t *q2_cold, int n_q2,
                                   q3_item_t *q3, int *n_q3,
                                   u32 a0, u32 am1, u32 C_G3, u32 a5, u32 T18, int w_thresh,
                                   u64 *local_l2, u64 *local_l3){
    u32 cand2[4], m3_arr[4];
    for(int i=0; i<n_q2; i++){
        u32 t17 = q2_hot[i].t17;
        u32 G2  = q2_hot[i].G2;
        int n2 = lookup_q0_q2_pack4_fast(t17, G2, a5, w_thresh, cand2, m3_arr);
        if(!n2) continue;
        *local_l2 += (u64)n2;
        u32 a1 = q2_cold[i].a1, e1 = q2_cold[i].e1;
        u32 a2 = q2_cold[i].a2, e2 = q2_cold[i].e2, sm2 = q2_cold[i].sm2;
        for(int y=0; y<n2; y++){
            u32 W2 = cand2[y];
            u32 a3 = W2 + G2;
            u32 M3 = m3_arr[y];
            u32 e3 = a3 - sm2 + am1;
            u32 sm3 = S0(a3) + MAJ(a3, a2, a1);
            u32 G3 = sm3 + S1(e3) + CH(e3, e2, e1) + C_G3;
            u32 t18 = T18 + G3 - W2;
            (*local_l3)++;
            __builtin_prefetch(&H_pack4[t18 >> 2], 0, 0);
            int idx3 = (*n_q3)++;
            q3[idx3] = (q3_item_t){t18, G3, M3, a1, a2, a3, 0};
        }
    }
    (void)a0;
}

// Phase 4 Fast-Path Worker for 4 GB table, B == 1:
// - 4-Step Cross-Stage Software Pipeline (Step 1: Compute & prefetch b_t16[nxt] -> Step 2: Drain q3 ->
//   Step 3: Branchless compact q2 & prefetch t18 -> Step 4: Branchless compact b_t16[cur] & prefetch t17),
//   doubling DRAM prefetch lead time from ~35ns to ~105ns across all 3 levels!
// - 100% Branchless Level-1 & Level-2 Stream Compaction (eliminates ~5 billion dynamic branches/sweep)
// - 16-byte q2_cold4_t ({a1, a2, e2, sm2}) and 16-byte q3_hot4_t / q3_cold4_t split (single 128-bit stp/ldp)
// - Single-branch Level-3 filter (hit0 | hit1, 98.2% not taken)
typedef struct { u32 a1, a2, e2, sm2; }     q2_cold4_t;
typedef struct { u32 t18, G3, M3, ctx_id; } q3_hot4_t;
typedef struct { u32 a1, a2, a3, pad; }     q3_cold4_t;

static inline void drain_q3_fast_4gb_b1(const q3_hot4_t *q3_hot, const q3_cold4_t *q3_cold, int n_q3,
                                        const solver_state_t *st, u64 *local_sol19){
    const u32 a0 = st->a0, am1 = st->am1, am2 = st->am2, C_e1 = st->C_e1;
    const u32 a5 = st->rctx[0].a5, T19 = st->rctx[0].T19;
    for(int i=0; i<n_q3; i++){
        u32 t18   = q3_hot[i].t18;
        u32 G3    = q3_hot[i].G3;
        u32 M3    = q3_hot[i].M3;
        u32 pack  = H_pack4[t18 >> 2];
        u32 M3_lo = M3 & 0xffffu;
        u32 ulo0  = pack & 0xffffu;
        u32 ulo1  = pack >> 16;
        u32 hit0  = ((((ulo0 + G3) ^ a5) & M3_lo) == 0u);
        u32 hit1  = ((((ulo1 + G3) ^ a5) & M3_lo) == 0u);
        if(__builtin_expect(hit0 | hit1, 0)){
            u32 cand3[2];
            int n3 = 0;
            if(hit0 && ulo0){
                u32 idx0 = rec_idx(ulo0, t18);
                u32 u0 = ulo0 | ((u32)REC_HI16[idx0] << 16);
                if((((u0 + G3) ^ a5) & M3) == 0u && Qf(u0) == t18) cand3[n3++] = u0;
            }
            if(hit1 && ulo1){
                u32 idx1 = rec_idx(ulo1, t18);
                u32 u1 = ulo1 | ((u32)REC_HI16[idx1] << 16);
                if((((u1 + G3) ^ a5) & M3) == 0u && Qf(u1) == t18) cand3[n3++] = u1;
            }
            for(int z=0; z<n3; z++){
                u32 W3 = cand3[z];
                u32 a4 = W3 + G3;
                (*local_sol19)++;
                u32 a1 = q3_cold[i].a1, a2 = q3_cold[i].a2, a3 = q3_cold[i].a3;
                u32 e1 = a1 + C_e1;
                u32 sm1 = S0(a1) + MAJ(a1, a0, am1);
                u32 e2 = a2 - sm1 + am2;
                u32 sm2 = S0(a2) + MAJ(a2, a1, a0);
                u32 e3 = a3 - sm2 + am1;
                u32 sm3 = S0(a3) + MAJ(a3, a2, a1);
                u32 e4 = a4 - sm3 + a0;
                u32 sm4 = S0(a4) + MAJ(a4, a3, a2);
                u32 W4 = a5 - sm4 - e1 - S1(e4) - CH(e4, e3, e2) - K[4];
                if(__builtin_expect(s0(W4) + W3 == T19, 0)){
                    u32 A_sol[R_STEPS + OFF + 2];
                    memcpy(A_sol, st->rctx[0].A_base, sizeof(A_sol));
                    A_sol[1+OFF] = a1; A_sol[2+OFF] = a2;
                    A_sol[3+OFF] = a3; A_sol[4+OFF] = a4;
                    u32 Wm[16], out[8];
                    for(int t=0;t<16;t++) Wm[t] = Wt_arr(A_sol, t);
                    compress_r(Wm, R_STEPS, out);
                    int ok = 1;
                    for(int k=0;k<8;k++) ok &= (out[k] == st->dig[k]);
                    ok &= ((Wm[13] & 0xffu) == 0x80u) && (Wm[14] == 0u) && (Wm[15] == (u32)PADLEN_BITS);
                    printf("\n*** FOUND 20-STEP PADDED PREIMAGE (ok=%d) ***\n", ok);
                    printf("W =");
                    for(int t=0;t<16;t++) printf(" %08x", Wm[t]);
                    printf("\nH20=");
                    for(int k=0;k<8;k++) printf("%08x", out[k]);
                    printf("\n");
                    fflush(stdout);
                    if(ok){
                        atomic_store_explicit(&g_solved20, 1, memory_order_relaxed);
                        return;
                    }
                }
            }
        }
    }
}

static void *worker_thread_4gb_b1(void *arg){
    (void)arg;
    const solver_state_t *st = &G_STATE;
    const u32 a0 = st->a0, am1 = st->am1, am2 = st->am2;
    const u32 e0 = st->e0, em1 = st->em1;
    const u32 C_e1 = st->C_e1, C_W0 = st->C_W0;
    const u32 C_G1 = st->C_G1, C_G2 = st->C_G2, C_G3 = st->C_G3;
    const int w_thresh = st->w_thresh;
    const u32 a5  = st->rctx[0].a5;
    const u32 T16 = st->rctx[0].T16;
    const u32 T17 = st->rctx[0].T17;
    const u32 T18 = st->rctx[0].T18;
    const u32 C_T16_W0 = T16 - C_W0;

    const u32 M_a0_am1 = a0 ^ am1;
    const u32 C_sm1    = a0 & am1;
    const u32 M_e0_em1 = e0 ^ em1;

    u32 b_sm1[2][PF_BATCH], b_G1[2][PF_BATCH], b_t16[2][PF_BATCH];
    u32 l1_W1[256];
    u8  l1_idx[256];

    q2_hot_t   q2_hot[256];
    q2_cold4_t q2_cold[256];

    u32 l2_W2[256];
    u32 l2_M3[256];
    u8  l2_idx[256];

    q3_hot4_t  q3_hot[256];
    q3_cold4_t q3_cold[256];
    int n_q2 = 0, n_q3 = 0;

    while(!atomic_load_explicit(&g_solved20, memory_order_relaxed)){
        u64 start = atomic_fetch_add_explicit(&g_next_a1_chunk, CHUNK_SIZE, memory_order_relaxed);
        if(start >= (1ULL << 32)) break;
        u64 end = start + CHUNK_SIZE;
        if(end > (1ULL << 32)) end = (1ULL << 32);

        u64 local_l2 = 0, local_l3 = 0, local_sol19 = 0;

        // Prime buffer 0 for base = start
        for(int i=0; i<PF_BATCH; i++){
            u32 a1  = (u32)(start + (u64)i);
            u32 e1  = a1 + C_e1;
            u32 sm1 = S0(a1) + (a1 & M_a0_am1) + C_sm1;
            u32 G1  = sm1 + S1(e1) + (em1 ^ (e1 & M_e0_em1)) + C_G1;
            u32 t16 = C_T16_W0 + G1 - a1;
            b_sm1[0][i] = sm1;
            b_G1[0][i]  = G1;
            b_t16[0][i] = t16;
        }
        for(int i=0; i<PF_BATCH; i++) __builtin_prefetch(&H_pack4[b_t16[0][i] >> 2], 0, 0);
        int cur = 0;

        for(u64 base = start; base < end; base += PF_BATCH){
            int nxt = cur ^ 1;
            u64 next_base = base + PF_BATCH;

            // Step 1: Compute & prefetch NEXT batch's b_t16 (provides ~70ns compute shadow for q2 & q3!)
            if(__builtin_expect(next_base < end, 1)){
                for(int i=0; i<PF_BATCH; i++){
                    u32 a1  = (u32)(next_base + (u64)i);
                    u32 e1  = a1 + C_e1;
                    u32 sm1 = S0(a1) + (a1 & M_a0_am1) + C_sm1;
                    u32 G1  = sm1 + S1(e1) + (em1 ^ (e1 & M_e0_em1)) + C_G1;
                    u32 t16 = C_T16_W0 + G1 - a1;
                    b_sm1[nxt][i] = sm1;
                    b_G1[nxt][i]  = G1;
                    b_t16[nxt][i] = t16;
                }
                for(int i=0; i<PF_BATCH; i++) __builtin_prefetch(&H_pack4[b_t16[nxt][i] >> 2], 0, 0);
            }

            // Step 2: Drain q3 (whose t18 prefetches waited through prev Step 4 + cur Step 1!)
            if(n_q3 >= PF_BATCH){
                drain_q3_fast_4gb_b1(q3_hot, q3_cold, n_q3, st, &local_sol19);
                n_q3 = 0;
            }

            // Step 3: Branchless compact q2 (whose t17 prefetches waited through cur Step 1 + cur Step 2!)
            if(n_q2 >= PF_BATCH){
                int n_l2 = 0;
                for(int i=0; i<n_q2; i++){
                    u32 t17  = q2_hot[i].t17, G2 = q2_hot[i].G2;
                    u32 pack = H_pack4[t17 >> 2];
                    u32 ulo0 = pack & 0xffffu;
                    u32 ulo1 = pack >> 16;
                    u32 idx0 = rec_idx(ulo0, t17);
                    u32 idx1 = rec_idx(ulo1, t17);
                    u32 u0   = ulo0 | ((u32)REC_HI16[idx0] << 16);
                    u32 u1   = ulo1 | ((u32)REC_HI16[idx1] << 16);
                    u32 m3_0 = ~((u0 + G2) ^ a5);
                    u32 m3_1 = ~((u1 + G2) ^ a5);
                    l2_W2[n_l2] = u0; l2_M3[n_l2] = m3_0; l2_idx[n_l2] = (u8)i;
                    n_l2 += (Qf(u0) == t17) & (__builtin_popcount(m3_0) <= w_thresh) & (pack != 0u);
                    l2_W2[n_l2] = u1; l2_M3[n_l2] = m3_1; l2_idx[n_l2] = (u8)i;
                    n_l2 += (Qf(u1) == t17) & (__builtin_popcount(m3_1) <= w_thresh) & (ulo1 != 0u);
                }
                local_l2 += (u64)n_l2;
                local_l3 += (u64)n_l2;
                for(int k=0; k<n_l2; k++){
                    int i   = l2_idx[k];
                    u32 W2  = l2_W2[k], M3 = l2_M3[k];
                    u32 G2  = q2_hot[i].G2;
                    u32 a1  = q2_cold[i].a1, a2 = q2_cold[i].a2, e2 = q2_cold[i].e2, sm2 = q2_cold[i].sm2;
                    u32 e1  = a1 + C_e1;
                    u32 a3  = W2 + G2;
                    u32 e3  = a3 - sm2 + am1;
                    u32 sm3 = S0(a3) + MAJ(a3, a2, a1);
                    u32 G3  = sm3 + S1(e3) + CH(e3, e2, e1) + C_G3;
                    u32 t18 = T18 + G3 - W2;
                    __builtin_prefetch(&H_pack4[t18 >> 2], 0, 0);
                    q3_hot[n_q3]  = (q3_hot4_t){t18, G3, M3, 0};
                    q3_cold[n_q3] = (q3_cold4_t){a1, a2, a3, 0};
                    n_q3++;
                    if(__builtin_expect(n_q3 >= 192, 0)){
                        drain_q3_fast_4gb_b1(q3_hot, q3_cold, n_q3, st, &local_sol19);
                        n_q3 = 0;
                    }
                }
                n_q2 = 0;
            }

            // Step 4: Branchless compact b_t16[cur] (prefetched a full iteration ago!) & push/prefetch q2
            int n_l1 = 0;
            for(int i=0; i<PF_BATCH; i++){
                u32 y = b_t16[cur][i];
                u32 pack = H_pack4[y >> 2];
                u32 ulo0 = pack & 0xffffu;
                u32 ulo1 = pack >> 16;
                u32 idx0 = rec_idx(ulo0, y);
                u32 idx1 = rec_idx(ulo1, y);
                u32 u0   = ulo0 | ((u32)REC_HI16[idx0] << 16);
                u32 u1   = ulo1 | ((u32)REC_HI16[idx1] << 16);
                l1_W1[n_l1] = u0; l1_idx[n_l1] = (u8)i;
                n_l1 += (Qf(u0) == y) & (pack != 0u);
                l1_W1[n_l1] = u1; l1_idx[n_l1] = (u8)i;
                n_l1 += (Qf(u1) == y) & (ulo1 != 0u);
            }
            for(int j=0; j<n_l1; j++){
                u32 i   = l1_idx[j];
                u32 W1  = l1_W1[j];
                u32 a1  = (u32)base + i;
                u32 e1  = a1 + C_e1;
                u32 sm1 = b_sm1[cur][i];
                u32 G1  = b_G1[cur][i];
                u32 a2  = W1 + G1;
                u32 e2  = a2 - sm1 + am2;
                u32 sm2 = S0(a2) + MAJ(a2, a1, a0);
                u32 G2  = sm2 + S1(e2) + CH(e2, e1, e0) + C_G2;
                u32 t17 = T17 + G2 - W1;
                __builtin_prefetch(&H_pack4[t17 >> 2], 0, 0);
                q2_hot[n_q2]  = (q2_hot_t){t17, G2};
                q2_cold[n_q2] = (q2_cold4_t){a1, a2, e2, sm2};
                n_q2++;
            }
            cur = nxt;
        }

        if(n_q2 > 0){
            u32 cand2[4], m3_arr[4];
            for(int i=0; i<n_q2; i++){
                int n2 = lookup_q0_q2_pack4_fast(q2_hot[i].t17, q2_hot[i].G2, a5, w_thresh, cand2, m3_arr);
                if(!n2) continue;
                local_l2 += (u64)n2;
                u32 a1 = q2_cold[i].a1, e1 = a1 + C_e1, a2 = q2_cold[i].a2, e2 = q2_cold[i].e2, sm2 = q2_cold[i].sm2;
                for(int y=0; y<n2; y++){
                    u32 W2 = cand2[y], a3 = W2 + q2_hot[i].G2, M3 = m3_arr[y];
                    u32 e3 = a3 - sm2 + am1;
                    u32 sm3 = S0(a3) + MAJ(a3, a2, a1);
                    u32 G3 = sm3 + S1(e3) + CH(e3, e2, e1) + C_G3;
                    u32 t18 = T18 + G3 - W2;
                    local_l3++;
                    q3_hot[n_q3]  = (q3_hot4_t){t18, G3, M3, 0};
                    q3_cold[n_q3] = (q3_cold4_t){a1, a2, a3, 0};
                    n_q3++;
                    if(n_q3 >= PF_BATCH){
                        drain_q3_fast_4gb_b1(q3_hot, q3_cold, n_q3, st, &local_sol19);
                        n_q3 = 0;
                    }
                }
            }
            n_q2 = 0;
        }
        if(n_q3 > 0){
            drain_q3_fast_4gb_b1(q3_hot, q3_cold, n_q3, st, &local_sol19);
            n_q3 = 0;
        }

        atomic_fetch_add_explicit(&g_chains_done, (end - start), memory_order_relaxed);
        atomic_fetch_add_explicit(&g_l2_nodes, local_l2, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_l3_lookups, local_l3, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_sol19_count, local_sol19, memory_order_relaxed);
    }
    return NULL;
}

// Phase 6 Fast-Path Worker for 6 GB 4-Pass Hierarchical Quad Table (H_hier6), B == 1:
// - Stores 73.21% of all 2^32 Q0^-1 preimages in the exact same 6.00 GB footprint via 4 hierarchical passes:
//   Pass 1 (3-slot bucket), Pass 2 (4th-slot paired window), Pass 3 (6-slot pair), Pass 4 (12-slot quad).
// - 12-Slot Aligned Quad SWAR Level-3 Filter: loads the entire 24-byte 8-byte-aligned 12-slot quad (3 x u64)
//   and evaluates all 12 low-16-bit algebraic filters in parallel using portable 64-bit SWAR (+10.74% sol19 yield!).
__attribute__((noinline))
static void verify_sol19_ctx(u32 W3, u32 G3, const q3_cold4_t *cold,
                             const solver_state_t *st, int ctx_id, u64 *local_sol19){
    const u32 a0 = st->a0, am1 = st->am1, am2 = st->am2, C_e1 = st->C_e1;
    const u32 a5 = st->rctx[ctx_id].a5, T19 = st->rctx[ctx_id].T19;
    u32 a4 = W3 + G3;
    (*local_sol19)++;
    u32 a1 = cold->a1, a2 = cold->a2, a3 = cold->a3;
    u32 e1 = a1 + C_e1;
    u32 sm1 = S0(a1) + MAJ(a1, a0, am1);
    u32 e2 = a2 - sm1 + am2;
    u32 sm2 = S0(a2) + MAJ(a2, a1, a0);
    u32 e3 = a3 - sm2 + am1;
    u32 sm3 = S0(a3) + MAJ(a3, a2, a1);
    u32 e4 = a4 - sm3 + a0;
    u32 sm4 = S0(a4) + MAJ(a4, a3, a2);
    u32 W4 = a5 - sm4 - e1 - S1(e4) - CH(e4, e3, e2) - K[4];
    if(__builtin_expect(s0(W4) + W3 == T19, 0)){
        u32 A_sol[R_STEPS + OFF + 2];
        memcpy(A_sol, st->rctx[ctx_id].A_base, sizeof(A_sol));
        A_sol[1+OFF] = a1; A_sol[2+OFF] = a2;
        A_sol[3+OFF] = a3; A_sol[4+OFF] = a4;
        u32 Wm[16], out[8];
        for(int t=0;t<16;t++) Wm[t] = Wt_arr(A_sol, t);
        compress_r(Wm, R_STEPS, out);
        int ok = 1;
        for(int k=0;k<8;k++) ok &= (out[k] == st->dig[k]);
        ok &= ((Wm[13] & 0xffu) == 0x80u) && (Wm[14] == 0u) && (Wm[15] == (u32)PADLEN_BITS);
        printf("\n*** FOUND 20-STEP PADDED PREIMAGE (ok=%d, ctx=%d) ***\n", ok, ctx_id);
        printf("W =");
        for(int t=0;t<16;t++) printf(" %08x", Wm[t]);
        printf("\nH20=");
        for(int k=0;k<8;k++) printf("%08x", out[k]);
        printf("\n");
        fflush(stdout);
        if(ok){
            atomic_store_explicit(&g_solved20, 1, memory_order_relaxed);
        }
    }
}

static inline void drain_q3_fast_6gb_ctx(const q3_hot4_t *q3_hot, const q3_cold4_t *q3_cold, int n_q3,
                                         const solver_state_t *st, int ctx_id, u64 *local_sol19){
    const u32 a5 = st->rctx[ctx_id].a5;
    const u64 K7 = 0x7fff7fff7fff7fffULL;
    const u64 K8 = 0x8000800080008000ULL;
    const u64 K1 = 0x0001000100010001ULL;
    const u64 A64 = (u64)(a5 & 0x7fffu) * K1;

    for(int i=0; i<n_q3; i++){
        u32 t18    = q3_hot[i].t18;
        u32 G3     = q3_hot[i].G3;
        u32 M3     = q3_hot[i].M3;
        size_t off = (size_t)(t18 >> 4) * 12u;
        const u64 *q64 = (const u64 *)&H_pack6[off];
        u64 w0 = q64[0];
        u64 w1 = q64[1];
        u64 w2 = q64[2];
        u64 g64 = (u64)(G3 & 0x7fffu) * K1;
        u64 m64 = (u64)(M3 & 0x7fffu) * K1;
        u64 d0 = ((((w0 & K7) + g64) ^ A64) & m64) + K7;
        u64 d1 = ((((w1 & K7) + g64) ^ A64) & m64) + K7;
        u64 d2 = ((((w2 & K7) + g64) ^ A64) & m64) + K7;
        if(__builtin_expect(((d0 & d1 & d2) & K8) != K8, 0)){
            const u16 *q = (const u16 *)q64;
            u32 M3_lo = M3 & 0xffffu;
            u64 h0 = ~d0 & K8;
            while(h0){
                int s = __builtin_ctzll(h0) >> 4;
                h0 &= h0 - 1ULL;
                u32 ulo = q[s];
                if(ulo && (((ulo + G3) ^ a5) & M3_lo) == 0u){
                    u32 idx = rec_idx(ulo, t18);
                    u32 u = ulo | ((u32)REC_HI16[idx] << 16);
                    if((((u + G3) ^ a5) & M3) == 0u && Qf(u) == t18){
                        verify_sol19_ctx(u, G3, &q3_cold[i], st, ctx_id, local_sol19);
                    }
                }
            }
            u64 h1 = ~d1 & K8;
            while(h1){
                int s = 4 + (__builtin_ctzll(h1) >> 4);
                h1 &= h1 - 1ULL;
                u32 ulo = q[s];
                if(ulo && (((ulo + G3) ^ a5) & M3_lo) == 0u){
                    u32 idx = rec_idx(ulo, t18);
                    u32 u = ulo | ((u32)REC_HI16[idx] << 16);
                    if((((u + G3) ^ a5) & M3) == 0u && Qf(u) == t18){
                        verify_sol19_ctx(u, G3, &q3_cold[i], st, ctx_id, local_sol19);
                    }
                }
            }
            u64 h2 = ~d2 & K8;
            while(h2){
                int s = 8 + (__builtin_ctzll(h2) >> 4);
                h2 &= h2 - 1ULL;
                u32 ulo = q[s];
                if(ulo && (((ulo + G3) ^ a5) & M3_lo) == 0u){
                    u32 idx = rec_idx(ulo, t18);
                    u32 u = ulo | ((u32)REC_HI16[idx] << 16);
                    if((((u + G3) ^ a5) & M3) == 0u && Qf(u) == t18){
                        verify_sol19_ctx(u, G3, &q3_cold[i], st, ctx_id, local_sol19);
                    }
                }
            }
        }
    }
}

static inline void drain_q3_fast_6gb_b1(const q3_hot4_t *q3_hot, const q3_cold4_t *q3_cold, int n_q3,
                                        const solver_state_t *st, u64 *local_sol19){
    drain_q3_fast_6gb_ctx(q3_hot, q3_cold, n_q3, st, 0, local_sol19);
}

#ifndef SLIDE4_MODE
#define SLIDE4_MODE 1
#endif
#ifndef SWEEP_LIMIT
#define SWEEP_LIMIT (1ULL << 32)
#endif
static u64 g_sweep_limit = SWEEP_LIMIT;

static void *worker_thread_6gb_b1(void *arg){
    (void)arg;
    const solver_state_t *st = &G_STATE;
    const u32 a0 = st->a0, am1 = st->am1, am2 = st->am2;
    const u32 e0 = st->e0, em1 = st->em1;
    const u32 C_e1 = st->C_e1, C_W0 = st->C_W0;
    const u32 C_G1 = st->C_G1, C_G2 = st->C_G2, C_G3 = st->C_G3;
    const int w_thresh = st->w_thresh;
    const u32 a5  = st->rctx[0].a5;
    const u32 T16 = st->rctx[0].T16;
    const u32 T17 = st->rctx[0].T17;
    const u32 T18 = st->rctx[0].T18;
    const u32 C_T16_W0 = T16 - C_W0;

    const u32 M_a0_am1 = a0 ^ am1;
    const u32 C_sm1    = a0 & am1;
    const u32 M_e0_em1 = e0 ^ em1;

    u32 b_sm1[2][PF_BATCH], b_G1[2][PF_BATCH], b_t16[2][PF_BATCH];
    u32 l1_W1[320];
    u8  l1_idx[320];

    u32        q2_t17[384];
    u32        q2_G2[384];
    q2_cold4_t q2_cold[384];

    u32 w2_cand[512];
    u8  w2_idx[512];
    u32 l2_W2[512];
    u32 l2_M3[512];
    u8  l2_idx[512];

    q3_hot4_t  q3_hot[256];
    q3_cold4_t q3_cold[256];
    int n_q2 = 0, n_q3 = 0;

    while(!atomic_load_explicit(&g_solved20, memory_order_relaxed)){
        u64 start = atomic_fetch_add_explicit(&g_next_a1_chunk, CHUNK_SIZE, memory_order_relaxed);
        if(start >= g_sweep_limit) break;
        u64 end = start + CHUNK_SIZE;
        if(end > g_sweep_limit) end = g_sweep_limit;

        u64 local_l2 = 0, local_l3 = 0, local_sol19 = 0;

        // Prime buffer 0 for base = start
        for(int i=0; i<PF_BATCH; i++){
            u32 a1  = (u32)(start + (u64)i);
            u32 e1  = a1 + C_e1;
            u32 sm1 = S0(a1) + (a1 & M_a0_am1) + C_sm1;
            u32 G1  = sm1 + S1(e1) + (em1 ^ (e1 & M_e0_em1)) + C_G1;
            u32 t16 = C_T16_W0 + G1 - a1;
            b_sm1[0][i] = sm1;
            b_G1[0][i]  = G1;
            b_t16[0][i] = t16;
        }
        for(int i=0; i<PF_BATCH; i++){
            __builtin_prefetch(&H_pack6[pf12_off(b_t16[0][i])], 0, 0);
        }
        int cur = 0;

        for(u64 base = start; base < end; base += PF_BATCH){
            int nxt = cur ^ 1;
            u64 next_base = base + PF_BATCH;

            // Step 1: Compute & prefetch NEXT batch's b_t16
            if(__builtin_expect(next_base < end, 1)){
                for(int i=0; i<PF_BATCH; i++){
                    u32 a1  = (u32)(next_base + (u64)i);
                    u32 e1  = a1 + C_e1;
                    u32 sm1 = S0(a1) + (a1 & M_a0_am1) + C_sm1;
                    u32 G1  = sm1 + S1(e1) + (em1 ^ (e1 & M_e0_em1)) + C_G1;
                    u32 t16 = C_T16_W0 + G1 - a1;
                    b_sm1[nxt][i] = sm1;
                    b_G1[nxt][i]  = G1;
                    b_t16[nxt][i] = t16;
                }
                for(int i=0; i<PF_BATCH; i++){
                    __builtin_prefetch(&H_pack6[pf12_off(b_t16[nxt][i])], 0, 0);
                }
            }

            // Step 2: Drain q3 (whose t18 prefetches waited through prev Step 4 + cur Step 1!)
            if(n_q3 >= PF_BATCH){
                drain_q3_fast_6gb_b1(q3_hot, q3_cold, n_q3, st, &local_sol19);
                n_q3 = 0;
            }

            // Step 3: Two-stage 100% branchless compact q2 (whose t17 prefetches waited through cur Step 1 + cur Step 2!)
            if(n_q2 >= PF_BATCH){
                int n_w2 = 0;
                for(int i=0; i<n_q2; i++){
                    u32 t17  = q2_t17[i];
#if SLIDE4_MODE == 2
                    const u16 *b = &H_pack6[slide4_off(t17)];
                    u32 ulo0 = b[0], ulo1 = b[1], ulo2 = b[2], ulo3 = b[3];
                    u32 u0 = ulo0 | ((u32)REC_HI16[rec_idx(ulo0, t17)] << 16);
                    u32 u1 = ulo1 | ((u32)REC_HI16[rec_idx(ulo1, t17)] << 16);
                    u32 u2 = ulo2 | ((u32)REC_HI16[rec_idx(ulo2, t17)] << 16);
                    u32 u3 = ulo3 | ((u32)REC_HI16[rec_idx(ulo3, t17)] << 16);
                    w2_cand[n_w2] = u0; w2_idx[n_w2] = (u8)i; n_w2 += (Qf(u0) == t17) & (ulo0 != 0u);
                    w2_cand[n_w2] = u1; w2_idx[n_w2] = (u8)i; n_w2 += (Qf(u1) == t17) & (ulo1 != 0u);
                    w2_cand[n_w2] = u2; w2_idx[n_w2] = (u8)i; n_w2 += (Qf(u2) == t17) & (ulo2 != 0u);
                    w2_cand[n_w2] = u3; w2_idx[n_w2] = (u8)i; n_w2 += (Qf(u3) == t17) & (ulo3 != 0u);
#else
                    const u16 *b = &H_pack6[slide3_off(t17)];
                    u32 ulo0 = b[0];
                    u32 ulo1 = b[1];
                    u32 ulo2 = b[2];
                    u32 idx0 = rec_idx(ulo0, t17);
                    u32 idx1 = rec_idx(ulo1, t17);
                    u32 idx2 = rec_idx(ulo2, t17);
                    u32 u0   = ulo0 | ((u32)REC_HI16[idx0] << 16);
                    u32 u1   = ulo1 | ((u32)REC_HI16[idx1] << 16);
                    u32 u2   = ulo2 | ((u32)REC_HI16[idx2] << 16);
                    u32 h0   = (Qf(u0) == t17) & (ulo0 != 0u);
                    w2_cand[n_w2] = u0; w2_idx[n_w2] = (u8)i; n_w2 += h0;
                    u32 h1   = (Qf(u1) == t17) & (ulo1 != 0u);
                    w2_cand[n_w2] = u1; w2_idx[n_w2] = (u8)i; n_w2 += h1;
                    u32 h2   = (Qf(u2) == t17) & (ulo2 != 0u);
                    w2_cand[n_w2] = u2; w2_idx[n_w2] = (u8)i; n_w2 += h2;
#if SLIDE4_MODE == 1
                    if(__builtin_expect(h0 & h1 & h2, 0)){
                        u32 ulo3 = b[(t17 & 15u) >= 13u ? -1 : 3];
                        u32 u3   = ulo3 | ((u32)REC_HI16[rec_idx(ulo3, t17)] << 16);
                        w2_cand[n_w2] = u3; w2_idx[n_w2] = (u8)i;
                        n_w2 += (Qf(u3) == t17) & (ulo3 != 0u);
                    }
#endif
#endif
                }
                int n_l2 = 0;
                for(int j=0; j<n_w2; j++){
                    u32 i  = w2_idx[j];
                    u32 W2 = w2_cand[j];
                    u32 m3 = ~((W2 + q2_G2[i]) ^ a5);
                    l2_W2[n_l2]  = W2;
                    l2_M3[n_l2]  = m3;
                    l2_idx[n_l2] = (u8)i;
                    n_l2 += (__builtin_popcount(m3) <= w_thresh);
                }
                local_l2 += (u64)n_l2;
                local_l3 += (u64)n_l2;
                for(int k=0; k<n_l2; k++){
                    int i   = l2_idx[k];
                    u32 W2  = l2_W2[k], M3 = l2_M3[k];
                    u32 G2  = q2_G2[i];
                    u32 a1  = q2_cold[i].a1, a2 = q2_cold[i].a2, e2 = q2_cold[i].e2, sm2 = q2_cold[i].sm2;
                    u32 e1  = a1 + C_e1;
                    u32 a3  = W2 + G2;
                    u32 e3  = a3 - sm2 + am1;
                    u32 sm3 = S0(a3) + MAJ(a3, a2, a1);
                    u32 G3  = sm3 + S1(e3) + CH(e3, e2, e1) + C_G3;
                    u32 t18 = T18 + G3 - W2;
                    __builtin_prefetch(&H_pack6[(size_t)(t18 >> 4) * 12u], 0, 0);
                    q3_hot[n_q3]  = (q3_hot4_t){t18, G3, M3, 0};
                    q3_cold[n_q3] = (q3_cold4_t){a1, a2, a3, 0};
                    n_q3++;
                    if(__builtin_expect(n_q3 >= 192, 0)){
                        drain_q3_fast_6gb_b1(q3_hot, q3_cold, n_q3, st, &local_sol19);
                        n_q3 = 0;
                    }
                }
                n_q2 = 0;
            }

            // Step 4: 100% branchless compact b_t16[cur] & push/prefetch q2
            int n_l1 = 0;
            for(int i=0; i<PF_BATCH; i++){
                u32 y = b_t16[cur][i];
                const u16 *b = &H_pack6[slide3_off(y)];
                u32 ulo0 = b[0];
                u32 ulo1 = b[1];
                u32 ulo2 = b[2];
                u32 idx0 = rec_idx(ulo0, y);
                u32 idx1 = rec_idx(ulo1, y);
                u32 idx2 = rec_idx(ulo2, y);
                u32 u0   = ulo0 | ((u32)REC_HI16[idx0] << 16);
                u32 u1   = ulo1 | ((u32)REC_HI16[idx1] << 16);
                u32 u2   = ulo2 | ((u32)REC_HI16[idx2] << 16);
                u32 h0   = (Qf(u0) == y) & (ulo0 != 0u);
                l1_W1[n_l1] = u0; l1_idx[n_l1] = (u8)i; n_l1 += h0;
                u32 h1   = (Qf(u1) == y) & (ulo1 != 0u);
                l1_W1[n_l1] = u1; l1_idx[n_l1] = (u8)i; n_l1 += h1;
                u32 h2   = (Qf(u2) == y) & (ulo2 != 0u);
                l1_W1[n_l1] = u2; l1_idx[n_l1] = (u8)i; n_l1 += h2;
#if SLIDE4_MODE == 1
                if(__builtin_expect(h0 & h1 & h2, 0)){
                    u32 ulo3 = b[(y & 15u) >= 13u ? -1 : 3];
                    u32 u3   = ulo3 | ((u32)REC_HI16[rec_idx(ulo3, y)] << 16);
                    l1_W1[n_l1] = u3; l1_idx[n_l1] = (u8)i;
                    n_l1 += (Qf(u3) == y) & (ulo3 != 0u);
                }
#endif
            }
            for(int j=0; j<n_l1; j++){
                u32 i   = l1_idx[j];
                u32 W1  = l1_W1[j];
                u32 a1  = (u32)base + i;
                u32 e1  = a1 + C_e1;
                u32 sm1 = b_sm1[cur][i];
                u32 G1  = b_G1[cur][i];
                u32 a2  = W1 + G1;
                u32 e2  = a2 - sm1 + am2;
                u32 sm2 = S0(a2) + MAJ(a2, a1, a0);
                u32 G2  = sm2 + S1(e2) + CH(e2, e1, e0) + C_G2;
                u32 t17 = T17 + G2 - W1;
                __builtin_prefetch(&H_pack6[pf12_off(t17)], 0, 0);
                q2_t17[n_q2]  = t17;
                q2_G2[n_q2]   = G2;
                q2_cold[n_q2] = (q2_cold4_t){a1, a2, e2, sm2};
                n_q2++;
            }
            cur = nxt;
        }

        if(n_q2 > 0){
            u32 cand2[4], m3_arr[4];
            for(int i=0; i<n_q2; i++){
                int n2 = lookup_q0_q2_pack6_fast(q2_t17[i], q2_G2[i], a5, w_thresh, cand2, m3_arr);
                if(!n2) continue;
                local_l2 += (u64)n2;
                u32 a1 = q2_cold[i].a1, e1 = a1 + C_e1, a2 = q2_cold[i].a2, e2 = q2_cold[i].e2, sm2 = q2_cold[i].sm2;
                for(int y=0; y<n2; y++){
                    u32 W2 = cand2[y], a3 = W2 + q2_G2[i], M3 = m3_arr[y];
                    u32 e3 = a3 - sm2 + am1;
                    u32 sm3 = S0(a3) + MAJ(a3, a2, a1);
                    u32 G3 = sm3 + S1(e3) + CH(e3, e2, e1) + C_G3;
                    u32 t18 = T18 + G3 - W2;
                    local_l3++;
                    q3_hot[n_q3]  = (q3_hot4_t){t18, G3, M3, 0};
                    q3_cold[n_q3] = (q3_cold4_t){a1, a2, a3, 0};
                    n_q3++;
                    if(n_q3 >= PF_BATCH){
                        drain_q3_fast_6gb_b1(q3_hot, q3_cold, n_q3, st, &local_sol19);
                        n_q3 = 0;
                    }
                }
            }
            n_q2 = 0;
        }
        if(n_q3 > 0){
            drain_q3_fast_6gb_b1(q3_hot, q3_cold, n_q3, st, &local_sol19);
            n_q3 = 0;
        }

        atomic_fetch_add_explicit(&g_chains_done, (end - start), memory_order_relaxed);
        atomic_fetch_add_explicit(&g_l2_nodes, local_l2, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_l3_lookups, local_l3, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_sol19_count, local_sol19, memory_order_relaxed);
    }
    return NULL;
}

#define PF_BATCH_B2 32

#ifndef L2_SUDOKU_MODE
#define L2_SUDOKU_MODE 0
#endif
#ifndef L2_ASYM_MODE
#define L2_ASYM_MODE 0
#endif
#ifndef W_LO_MAX
#define W_LO_MAX 7
#endif

static int g_l2_mode  = 2;
static int g_w_lo_max = W_LO_MAX;
static int g_pair_l1  = 1;

static inline void compact_q2_6gb_ctx(const u32 * __restrict__ q2_t17, const u32 * __restrict__ q2_G2, const q2_cold4_t * __restrict__ q2_cold, int n_q2,
                                      q3_hot4_t * __restrict__ q3_hot, q3_cold4_t * __restrict__ q3_cold, int * __restrict__ p_n_q3,
                                      u32 * __restrict__ w2_cand, u8 * __restrict__ w2_idx, u32 * __restrict__ l2_W2, u32 * __restrict__ l2_M3, u8 * __restrict__ l2_idx,
                                      const solver_state_t * __restrict__ st, int ctx_id,
                                      u64 * __restrict__ local_l2, u64 * __restrict__ local_l3, u64 * __restrict__ local_sol19){
    const u32 am1 = st->am1, C_e1 = st->C_e1, C_G3 = st->C_G3;
    const int w_thresh = st->w_thresh;
    const u32 a5  = st->rctx[ctx_id].a5;
    const u32 T18 = st->rctx[ctx_id].T18;
    int n_l2 = 0;

    if(g_l2_mode == 2){
        (void)w2_cand; (void)w2_idx;
        const u32 inv_a5 = ~a5;
        const u32 inv_a5_lo = inv_a5 & 0xffffu;
        const int w_lo_max = g_w_lo_max;
        for(int i=0; i<n_q2; i++){
            u32 t17 = q2_t17[i];
            u32 G2  = q2_G2[i];
            const u16 *b = &H_pack6[slide3_off(t17)];
            u32 ulo0 = b[0], ulo1 = b[1], ulo2 = b[2];
            u32 m_lo0 = ((ulo0 + G2) ^ inv_a5_lo) & 0xffffu;
            if(__builtin_popcount(m_lo0) <= w_lo_max){
                u32 u0 = ulo0 | ((u32)REC_HI16[rec_idx(ulo0, t17)] << 16);
                u32 m3_0 = (u0 + G2) ^ inv_a5;
                if(__builtin_expect(__builtin_popcount(m3_0) <= w_thresh, 0)){
                    u32 ok0 = (ulo0 != 0u) & (Qf(u0) == t17);
                    l2_W2[n_l2] = u0; l2_M3[n_l2] = m3_0; l2_idx[n_l2] = (u8)i; n_l2 += ok0;
                }
            }
            u32 m_lo1 = ((ulo1 + G2) ^ inv_a5_lo) & 0xffffu;
            if(__builtin_popcount(m_lo1) <= w_lo_max){
                u32 u1 = ulo1 | ((u32)REC_HI16[rec_idx(ulo1, t17)] << 16);
                u32 m3_1 = (u1 + G2) ^ inv_a5;
                if(__builtin_expect(__builtin_popcount(m3_1) <= w_thresh, 0)){
                    u32 ok1 = (ulo1 != 0u) & (Qf(u1) == t17);
                    l2_W2[n_l2] = u1; l2_M3[n_l2] = m3_1; l2_idx[n_l2] = (u8)i; n_l2 += ok1;
                }
            }
            u32 m_lo2 = ((ulo2 + G2) ^ inv_a5_lo) & 0xffffu;
            if(__builtin_popcount(m_lo2) <= w_lo_max){
                u32 u2 = ulo2 | ((u32)REC_HI16[rec_idx(ulo2, t17)] << 16);
                u32 m3_2 = (u2 + G2) ^ inv_a5;
                if(__builtin_expect(__builtin_popcount(m3_2) <= w_thresh, 0)){
                    u32 ok2 = (ulo2 != 0u) & (Qf(u2) == t17);
                    l2_W2[n_l2] = u2; l2_M3[n_l2] = m3_2; l2_idx[n_l2] = (u8)i; n_l2 += ok2;
                }
            }
        }
    } else if(g_l2_mode == 1){
        (void)w2_cand; (void)w2_idx;
        const u32 inv_a5 = ~a5;
        for(int i=0; i<n_q2; i++){
            u32 t17 = q2_t17[i];
            u32 G2  = q2_G2[i];
            const u16 *b = &H_pack6[slide3_off(t17)];
            u32 ulo0 = b[0], ulo1 = b[1], ulo2 = b[2];
            u32 u0 = ulo0 | ((u32)REC_HI16[rec_idx(ulo0, t17)] << 16);
            u32 u1 = ulo1 | ((u32)REC_HI16[rec_idx(ulo1, t17)] << 16);
            u32 u2 = ulo2 | ((u32)REC_HI16[rec_idx(ulo2, t17)] << 16);
            u32 m3_0 = (u0 + G2) ^ inv_a5;
            if(__builtin_expect(__builtin_popcount(m3_0) <= w_thresh, 0)){
                u32 ok0 = (ulo0 != 0u) & (Qf(u0) == t17);
                l2_W2[n_l2] = u0; l2_M3[n_l2] = m3_0; l2_idx[n_l2] = (u8)i; n_l2 += ok0;
            }
            u32 m3_1 = (u1 + G2) ^ inv_a5;
            if(__builtin_expect(__builtin_popcount(m3_1) <= w_thresh, 0)){
                u32 ok1 = (ulo1 != 0u) & (Qf(u1) == t17);
                l2_W2[n_l2] = u1; l2_M3[n_l2] = m3_1; l2_idx[n_l2] = (u8)i; n_l2 += ok1;
            }
            u32 m3_2 = (u2 + G2) ^ inv_a5;
            if(__builtin_expect(__builtin_popcount(m3_2) <= w_thresh, 0)){
                u32 ok2 = (ulo2 != 0u) & (Qf(u2) == t17);
                l2_W2[n_l2] = u2; l2_M3[n_l2] = m3_2; l2_idx[n_l2] = (u8)i; n_l2 += ok2;
            }
        }
    } else {
        int n_w2 = 0;
        for(int i=0; i<n_q2; i++){
            u32 t17  = q2_t17[i];
            const u16 *b = &H_pack6[slide3_off(t17)];
            u32 ulo0 = b[0];
            u32 ulo1 = b[1];
            u32 ulo2 = b[2];
            u32 idx0 = rec_idx(ulo0, t17);
            u32 idx1 = rec_idx(ulo1, t17);
            u32 idx2 = rec_idx(ulo2, t17);
            u32 u0   = ulo0 | ((u32)REC_HI16[idx0] << 16);
            u32 u1   = ulo1 | ((u32)REC_HI16[idx1] << 16);
            u32 u2   = ulo2 | ((u32)REC_HI16[idx2] << 16);
            u32 h0   = (Qf(u0) == t17) & (ulo0 != 0u);
            w2_cand[n_w2] = u0; w2_idx[n_w2] = (u8)i; n_w2 += h0;
            u32 h1   = (Qf(u1) == t17) & (ulo1 != 0u);
            w2_cand[n_w2] = u1; w2_idx[n_w2] = (u8)i; n_w2 += h1;
            u32 h2   = (Qf(u2) == t17) & (ulo2 != 0u);
            w2_cand[n_w2] = u2; w2_idx[n_w2] = (u8)i; n_w2 += h2;
#if SLIDE4_MODE == 1
            if(__builtin_expect(h0 & h1 & h2, 0)){
                u32 ulo3 = b[(t17 & 15u) >= 13u ? -1 : 3];
                u32 u3   = ulo3 | ((u32)REC_HI16[rec_idx(ulo3, t17)] << 16);
                w2_cand[n_w2] = u3; w2_idx[n_w2] = (u8)i;
                n_w2 += (Qf(u3) == t17) & (ulo3 != 0u);
            }
#endif
        }
        for(int j=0; j<n_w2; j++){
            u32 i  = w2_idx[j];
            u32 W2 = w2_cand[j];
            u32 m3 = ~((W2 + q2_G2[i]) ^ a5);
            l2_W2[n_l2]  = W2;
            l2_M3[n_l2]  = m3;
            l2_idx[n_l2] = (u8)i;
            n_l2 += (__builtin_popcount(m3) <= w_thresh);
        }
    }
    *local_l2 += (u64)n_l2;
    *local_l3 += (u64)n_l2;
    int n_q3 = *p_n_q3;
    for(int k=0; k<n_l2; k++){
        int i   = l2_idx[k];
        u32 W2  = l2_W2[k], M3 = l2_M3[k];
        u32 G2  = q2_G2[i];
        u32 a1  = q2_cold[i].a1, a2 = q2_cold[i].a2, e2 = q2_cold[i].e2, sm2 = q2_cold[i].sm2;
        u32 e1  = a1 + C_e1;
        u32 a3  = W2 + G2;
        u32 e3  = a3 - sm2 + am1;
        u32 sm3 = S0(a3) + MAJ(a3, a2, a1);
        u32 G3  = sm3 + S1(e3) + CH(e3, e2, e1) + C_G3;
        u32 t18 = T18 + G3 - W2;
        __builtin_prefetch(&H_pack6[(size_t)(t18 >> 4) * 12u], 0, 0);
        q3_hot[n_q3]  = (q3_hot4_t){t18, G3, M3, 0};
        q3_cold[n_q3] = (q3_cold4_t){a1, a2, a3, 0};
        n_q3++;
        if(__builtin_expect(n_q3 >= 192, 0)){
            drain_q3_fast_6gb_ctx(q3_hot, q3_cold, n_q3, st, ctx_id, local_sol19);
            n_q3 = 0;
        }
    }
    *p_n_q3 = n_q3;
}

// Fused Same-Cache-Line Paired Level-1 Compaction:
// Reads y0 = b_t16[i] and y1 = y0 + dT16 in the exact same loop iteration while the 128B L1D cache line is hot!
static inline void l1_compact_pair_6gb(const u32 *b_t16, u32 dT16,
                                       u32 *l1_W1_0, u8 *l1_idx_0, int *p_n0,
                                       u32 *l1_W1_1, u8 *l1_idx_1, int *p_n1){
    int n0 = 0, n1 = 0;
    for(int i=0; i<PF_BATCH_B2; i++){
        u32 y0 = b_t16[i];
        u32 y1 = y0 + dT16;
        const u16 *b0 = &H_pack6[slide3_off(y0)];
        const u16 *b1 = &H_pack6[slide3_off(y1)];

        u32 ulo0_0 = b0[0], ulo0_1 = b0[1], ulo0_2 = b0[2];
        u32 ulo1_0 = b1[0], ulo1_1 = b1[1], ulo1_2 = b1[2];

        u32 u0_0 = ulo0_0 | ((u32)REC_HI16[rec_idx(ulo0_0, y0)] << 16);
        u32 u0_1 = ulo0_1 | ((u32)REC_HI16[rec_idx(ulo0_1, y0)] << 16);
        u32 u0_2 = ulo0_2 | ((u32)REC_HI16[rec_idx(ulo0_2, y0)] << 16);
        u32 h0_0 = (Qf(u0_0) == y0) & (ulo0_0 != 0u);
        l1_W1_0[n0] = u0_0; l1_idx_0[n0] = (u8)i; n0 += h0_0;
        u32 h0_1 = (Qf(u0_1) == y0) & (ulo0_1 != 0u);
        l1_W1_0[n0] = u0_1; l1_idx_0[n0] = (u8)i; n0 += h0_1;
        u32 h0_2 = (Qf(u0_2) == y0) & (ulo0_2 != 0u);
        l1_W1_0[n0] = u0_2; l1_idx_0[n0] = (u8)i; n0 += h0_2;
#if SLIDE4_MODE == 1
        if(__builtin_expect(h0_0 & h0_1 & h0_2, 0)){
            u32 ulo0_3 = b0[(y0 & 15u) >= 13u ? -1 : 3];
            u32 u0_3   = ulo0_3 | ((u32)REC_HI16[rec_idx(ulo0_3, y0)] << 16);
            l1_W1_0[n0] = u0_3; l1_idx_0[n0] = (u8)i;
            n0 += (Qf(u0_3) == y0) & (ulo0_3 != 0u);
        }
#endif

        u32 u1_0 = ulo1_0 | ((u32)REC_HI16[rec_idx(ulo1_0, y1)] << 16);
        u32 u1_1 = ulo1_1 | ((u32)REC_HI16[rec_idx(ulo1_1, y1)] << 16);
        u32 u1_2 = ulo1_2 | ((u32)REC_HI16[rec_idx(ulo1_2, y1)] << 16);
        u32 h1_0 = (Qf(u1_0) == y1) & (ulo1_0 != 0u);
        l1_W1_1[n1] = u1_0; l1_idx_1[n1] = (u8)i; n1 += h1_0;
        u32 h1_1 = (Qf(u1_1) == y1) & (ulo1_1 != 0u);
        l1_W1_1[n1] = u1_1; l1_idx_1[n1] = (u8)i; n1 += h1_1;
        u32 h1_2 = (Qf(u1_2) == y1) & (ulo1_2 != 0u);
        l1_W1_1[n1] = u1_2; l1_idx_1[n1] = (u8)i; n1 += h1_2;
#if SLIDE4_MODE == 1
        if(__builtin_expect(h1_0 & h1_1 & h1_2, 0)){
            u32 ulo1_3 = b1[(y1 & 15u) >= 13u ? -1 : 3];
            u32 u1_3   = ulo1_3 | ((u32)REC_HI16[rec_idx(ulo1_3, y1)] << 16);
            l1_W1_1[n1] = u1_3; l1_idx_1[n1] = (u8)i;
            n1 += (Qf(u1_3) == y1) & (ulo1_3 != 0u);
        }
#endif
    }
    *p_n0 = n0;
    *p_n1 = n1;
}

static inline void l1_push_q2_6gb(u64 base, const u32 *b_sm1, const u32 *b_G1,
                                  const u32 *l1_W1, const u8 *l1_idx, int n_l1,
                                  u32 T17, const solver_state_t *st,
                                  u32 *q2_t17, u32 *q2_G2, q2_cold4_t *q2_cold, int *p_n_q2){
    const u32 a0 = st->a0, am2 = st->am2, e0 = st->e0, C_e1 = st->C_e1, C_G2 = st->C_G2;
    int n_q2 = *p_n_q2;
    for(int j=0; j<n_l1; j++){
        u32 i   = l1_idx[j];
        u32 W1  = l1_W1[j];
        u32 a1  = (u32)base + i;
        u32 e1  = a1 + C_e1;
        u32 sm1 = b_sm1[i];
        u32 G1  = b_G1[i];
        u32 a2  = W1 + G1;
        u32 e2  = a2 - sm1 + am2;
        u32 sm2 = S0(a2) + MAJ(a2, a1, a0);
        u32 G2  = sm2 + S1(e2) + CH(e2, e1, e0) + C_G2;
        u32 t17 = T17 + G2 - W1;
        __builtin_prefetch(&H_pack6[pf12_off(t17)], 0, 0);
        q2_t17[n_q2]  = t17;
        q2_G2[n_q2]   = G2;
        q2_cold[n_q2] = (q2_cold4_t){a1, a2, e2, sm2};
        n_q2++;
    }
    *p_n_q2 = n_q2;
}

// Pipelined 2-Context Worker for 6 GB Sudoku-Augmented Table (B == 2):
static void *worker_thread_6gb_b2(void *arg){
    (void)arg;
    const solver_state_t *st = &G_STATE;
    const u32 a0 = st->a0, am1 = st->am1, am2 = st->am2;
    const u32 e0 = st->e0, em1 = st->em1;
    const u32 C_e1 = st->C_e1, C_W0 = st->C_W0;
    const u32 C_G1 = st->C_G1, C_G2 = st->C_G2;

    const u32 T16_0 = st->rctx[0].T16, T16_1 = st->rctx[1].T16;
    const u32 T17_0 = st->rctx[0].T17, T17_1 = st->rctx[1].T17;
    const u32 C_T16_0 = T16_0 - C_W0;
    const u32 C_T16_1 = T16_1 - C_W0;

    const u32 M_a0_am1 = a0 ^ am1;
    const u32 C_sm1    = a0 & am1;
    const u32 M_e0_em1 = e0 ^ em1;

    u32 b_sm1[2][PF_BATCH_B2], b_G1[2][PF_BATCH_B2];
    u32 b_t16_0[2][PF_BATCH_B2], b_t16_1[2][PF_BATCH_B2];
    u32 l1_W1[160];
    u8  l1_idx[160];

    u32        q2_t17_0[384], q2_t17_1[384];
    u32        q2_G2_0[384],  q2_G2_1[384];
    q2_cold4_t q2_cold_0[384], q2_cold_1[384];

    u32 w2_cand[512];
    u8  w2_idx[512];
    u32 l2_W2[512];
    u32 l2_M3[512];
    u8  l2_idx[512];

    q3_hot4_t  q3_hot_0[256],  q3_hot_1[256];
    q3_cold4_t q3_cold_0[256], q3_cold_1[256];
    int n_q2_0 = 0, n_q2_1 = 0, n_q3_0 = 0, n_q3_1 = 0;

    while(!atomic_load_explicit(&g_solved20, memory_order_relaxed)){
        u64 start = atomic_fetch_add_explicit(&g_next_a1_chunk, CHUNK_SIZE, memory_order_relaxed);
        if(start >= g_sweep_limit) break;
        u64 end = start + CHUNK_SIZE;
        if(end > g_sweep_limit) end = g_sweep_limit;

        u64 local_l2 = 0, local_l3 = 0, local_sol19 = 0;

        for(int i=0; i<PF_BATCH_B2; i++){
            u32 a1  = (u32)(start + (u64)i);
            u32 e1  = a1 + C_e1;
            u32 sm1 = S0(a1) + (a1 & M_a0_am1) + C_sm1;
            u32 G1  = sm1 + S1(e1) + (em1 ^ (e1 & M_e0_em1)) + C_G1;
            u32 f1  = G1 - a1;
            b_sm1[0][i]   = sm1;
            b_G1[0][i]    = G1;
            b_t16_0[0][i] = C_T16_0 + f1;
            b_t16_1[0][i] = C_T16_1 + f1;
        }
        for(int i=0; i<PF_BATCH_B2; i++){
            __builtin_prefetch(&H_pack6[pf12_off(b_t16_0[0][i])], 0, 0);
            __builtin_prefetch(&H_pack6[pf12_off(b_t16_1[0][i])], 0, 0);
        }
        int cur = 0;

        for(u64 base = start; base < end; base += PF_BATCH_B2){
            int nxt = cur ^ 1;
            u64 next_base = base + PF_BATCH_B2;

            // Step 1: NEON-vectorized Round-1 ALU + separate prefetch loop for NEXT batch
            if(__builtin_expect(next_base < end, 1)){
                for(int i=0; i<PF_BATCH_B2; i++){
                    u32 a1  = (u32)(next_base + (u64)i);
                    u32 e1  = a1 + C_e1;
                    u32 sm1 = S0(a1) + (a1 & M_a0_am1) + C_sm1;
                    u32 G1  = sm1 + S1(e1) + (em1 ^ (e1 & M_e0_em1)) + C_G1;
                    u32 f1  = G1 - a1;
                    b_sm1[nxt][i]   = sm1;
                    b_G1[nxt][i]    = G1;
                    b_t16_0[nxt][i] = C_T16_0 + f1;
                    b_t16_1[nxt][i] = C_T16_1 + f1;
                }
                for(int i=0; i<PF_BATCH_B2; i++){
                    __builtin_prefetch(&H_pack6[pf12_off(b_t16_0[nxt][i])], 0, 0);
                    __builtin_prefetch(&H_pack6[pf12_off(b_t16_1[nxt][i])], 0, 0);
                }
            }

            // Step 2: Drain q3_0 and q3_1
            if(n_q3_0 >= PF_BATCH){
                drain_q3_fast_6gb_ctx(q3_hot_0, q3_cold_0, n_q3_0, st, 0, &local_sol19);
                n_q3_0 = 0;
            }
            if(n_q3_1 >= PF_BATCH){
                drain_q3_fast_6gb_ctx(q3_hot_1, q3_cold_1, n_q3_1, st, 1, &local_sol19);
                n_q3_1 = 0;
            }

            // Step 3: Compact q2_0 and q2_1
            if(n_q2_0 >= PF_BATCH){
                compact_q2_6gb_ctx(q2_t17_0, q2_G2_0, q2_cold_0, n_q2_0,
                                   q3_hot_0, q3_cold_0, &n_q3_0,
                                   w2_cand, w2_idx, l2_W2, l2_M3, l2_idx,
                                   st, 0, &local_l2, &local_l3, &local_sol19);
                n_q2_0 = 0;
            }
            if(n_q2_1 >= PF_BATCH){
                compact_q2_6gb_ctx(q2_t17_1, q2_G2_1, q2_cold_1, n_q2_1,
                                   q3_hot_1, q3_cold_1, &n_q3_1,
                                   w2_cand, w2_idx, l2_W2, l2_M3, l2_idx,
                                   st, 1, &local_l2, &local_l3, &local_sol19);
                n_q2_1 = 0;
            }

            // Step 4a: Branchless compact b_t16_0[cur] -> q2_0
            int n_l1_0 = 0;
            for(int i=0; i<PF_BATCH_B2; i++){
                u32 y = b_t16_0[cur][i];
                const u16 *b = &H_pack6[slide3_off(y)];
                u32 ulo0 = b[0], ulo1 = b[1], ulo2 = b[2];
                u32 u0 = ulo0 | ((u32)REC_HI16[rec_idx(ulo0, y)] << 16);
                u32 u1 = ulo1 | ((u32)REC_HI16[rec_idx(ulo1, y)] << 16);
                u32 u2 = ulo2 | ((u32)REC_HI16[rec_idx(ulo2, y)] << 16);
                u32 h0 = (Qf(u0) == y) & (ulo0 != 0u);
                l1_W1[n_l1_0] = u0; l1_idx[n_l1_0] = (u8)i; n_l1_0 += h0;
                u32 h1 = (Qf(u1) == y) & (ulo1 != 0u);
                l1_W1[n_l1_0] = u1; l1_idx[n_l1_0] = (u8)i; n_l1_0 += h1;
                u32 h2 = (Qf(u2) == y) & (ulo2 != 0u);
                l1_W1[n_l1_0] = u2; l1_idx[n_l1_0] = (u8)i; n_l1_0 += h2;
#if SLIDE4_MODE == 1
                if(__builtin_expect(h0 & h1 & h2, 0)){
                    u32 ulo3 = b[(y & 15u) >= 13u ? -1 : 3];
                    u32 u3   = ulo3 | ((u32)REC_HI16[rec_idx(ulo3, y)] << 16);
                    l1_W1[n_l1_0] = u3; l1_idx[n_l1_0] = (u8)i;
                    n_l1_0 += (Qf(u3) == y) & (ulo3 != 0u);
                }
#endif
            }
            for(int j=0; j<n_l1_0; j++){
                u32 i   = l1_idx[j];
                u32 W1  = l1_W1[j];
                u32 a1  = (u32)base + i;
                u32 e1  = a1 + C_e1;
                u32 sm1 = b_sm1[cur][i];
                u32 G1  = b_G1[cur][i];
                u32 a2  = W1 + G1;
                u32 e2  = a2 - sm1 + am2;
                u32 sm2 = S0(a2) + MAJ(a2, a1, a0);
                u32 G2  = sm2 + S1(e2) + CH(e2, e1, e0) + C_G2;
                u32 t17 = T17_0 + G2 - W1;
                __builtin_prefetch(&H_pack6[pf12_off(t17)], 0, 0);
                q2_t17_0[n_q2_0]  = t17;
                q2_G2_0[n_q2_0]   = G2;
                q2_cold_0[n_q2_0] = (q2_cold4_t){a1, a2, e2, sm2};
                n_q2_0++;
            }

            // Step 4b: Branchless compact b_t16_1[cur] -> q2_1
            int n_l1_1 = 0;
            for(int i=0; i<PF_BATCH_B2; i++){
                u32 y = b_t16_1[cur][i];
                const u16 *b = &H_pack6[slide3_off(y)];
                u32 ulo0 = b[0], ulo1 = b[1], ulo2 = b[2];
                u32 u0 = ulo0 | ((u32)REC_HI16[rec_idx(ulo0, y)] << 16);
                u32 u1 = ulo1 | ((u32)REC_HI16[rec_idx(ulo1, y)] << 16);
                u32 u2 = ulo2 | ((u32)REC_HI16[rec_idx(ulo2, y)] << 16);
                u32 h0 = (Qf(u0) == y) & (ulo0 != 0u);
                l1_W1[n_l1_1] = u0; l1_idx[n_l1_1] = (u8)i; n_l1_1 += h0;
                u32 h1 = (Qf(u1) == y) & (ulo1 != 0u);
                l1_W1[n_l1_1] = u1; l1_idx[n_l1_1] = (u8)i; n_l1_1 += h1;
                u32 h2 = (Qf(u2) == y) & (ulo2 != 0u);
                l1_W1[n_l1_1] = u2; l1_idx[n_l1_1] = (u8)i; n_l1_1 += h2;
#if SLIDE4_MODE == 1
                if(__builtin_expect(h0 & h1 & h2, 0)){
                    u32 ulo3 = b[(y & 15u) >= 13u ? -1 : 3];
                    u32 u3   = ulo3 | ((u32)REC_HI16[rec_idx(ulo3, y)] << 16);
                    l1_W1[n_l1_1] = u3; l1_idx[n_l1_1] = (u8)i;
                    n_l1_1 += (Qf(u3) == y) & (ulo3 != 0u);
                }
#endif
            }
            for(int j=0; j<n_l1_1; j++){
                u32 i   = l1_idx[j];
                u32 W1  = l1_W1[j];
                u32 a1  = (u32)base + i;
                u32 e1  = a1 + C_e1;
                u32 sm1 = b_sm1[cur][i];
                u32 G1  = b_G1[cur][i];
                u32 a2  = W1 + G1;
                u32 e2  = a2 - sm1 + am2;
                u32 sm2 = S0(a2) + MAJ(a2, a1, a0);
                u32 G2  = sm2 + S1(e2) + CH(e2, e1, e0) + C_G2;
                u32 t17 = T17_1 + G2 - W1;
                __builtin_prefetch(&H_pack6[pf12_off(t17)], 0, 0);
                q2_t17_1[n_q2_1]  = t17;
                q2_G2_1[n_q2_1]   = G2;
                q2_cold_1[n_q2_1] = (q2_cold4_t){a1, a2, e2, sm2};
                n_q2_1++;
            }

            cur = nxt;
        }

        if(n_q2_0 > 0){
            compact_q2_6gb_ctx(q2_t17_0, q2_G2_0, q2_cold_0, n_q2_0,
                               q3_hot_0, q3_cold_0, &n_q3_0,
                               w2_cand, w2_idx, l2_W2, l2_M3, l2_idx,
                               st, 0, &local_l2, &local_l3, &local_sol19);
            n_q2_0 = 0;
        }
        if(n_q2_1 > 0){
            compact_q2_6gb_ctx(q2_t17_1, q2_G2_1, q2_cold_1, n_q2_1,
                               q3_hot_1, q3_cold_1, &n_q3_1,
                               w2_cand, w2_idx, l2_W2, l2_M3, l2_idx,
                               st, 1, &local_l2, &local_l3, &local_sol19);
            n_q2_1 = 0;
        }
        if(n_q3_0 > 0){
            drain_q3_fast_6gb_ctx(q3_hot_0, q3_cold_0, n_q3_0, st, 0, &local_sol19);
            n_q3_0 = 0;
        }
        if(n_q3_1 > 0){
            drain_q3_fast_6gb_ctx(q3_hot_1, q3_cold_1, n_q3_1, st, 1, &local_sol19);
            n_q3_1 = 0;
        }

        atomic_fetch_add_explicit(&g_chains_done, (end - start) * 2ULL, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_l2_nodes, local_l2, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_l3_lookups, local_l3, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_sol19_count, local_sol19, memory_order_relaxed);
    }
    return NULL;
}

// Pipelined Multi-Pair Worker for B >= 3 (up to MAX_BATCH_CTX = 32):
#define PF_BATCH_BP 32
static void *worker_thread_6gb_bpairs(void *arg){
    (void)arg;
    const solver_state_t *st = &G_STATE;
    const int B = st->batch_ctx;
    const u32 a0 = st->a0, am1 = st->am1, am2 = st->am2;
    const u32 e0 = st->e0, em1 = st->em1;
    const u32 C_e1 = st->C_e1, C_W0 = st->C_W0, C_G1 = st->C_G1, C_G2 = st->C_G2;

    const u32 M_a0_am1 = a0 ^ am1;
    const u32 C_sm1    = a0 & am1;
    const u32 M_e0_em1 = e0 ^ em1;

    u32 C_T16_c[MAX_BATCH_CTX], T17_c[MAX_BATCH_CTX], dT16_pair[MAX_BATCH_CTX / 2];
    for(int c=0; c<B; c++){
        C_T16_c[c] = st->rctx[c].T16 - C_W0;
        T17_c[c]   = st->rctx[c].T17;
    }
    for(int p=0; p<B/2; p++){
        dT16_pair[p] = st->rctx[2*p+1].T16 - st->rctx[2*p].T16;
    }

    u32 b_sm1[2][PF_BATCH_BP], b_G1[2][PF_BATCH_BP], b_f1[PF_BATCH_BP];
    u32 b_t16_c[2][MAX_BATCH_CTX][PF_BATCH_BP];
    u32 l1_W1[160], l1_W1_b[160];
    u8  l1_idx[160], l1_idx_b[160];

    u32        q2_t17[MAX_BATCH_CTX][384];
    u32        q2_G2[MAX_BATCH_CTX][384];
    q2_cold4_t q2_cold[MAX_BATCH_CTX][384];

    u32 w2_cand[512];
    u8  w2_idx[512];
    u32 l2_W2[512];
    u32 l2_M3[512];
    u8  l2_idx[512];

    q3_hot4_t  q3_hot[MAX_BATCH_CTX][256];
    q3_cold4_t q3_cold[MAX_BATCH_CTX][256];
    int n_q2[MAX_BATCH_CTX] = {0}, n_q3[MAX_BATCH_CTX] = {0};

    while(!atomic_load_explicit(&g_solved20, memory_order_relaxed)){
        u64 start = atomic_fetch_add_explicit(&g_next_a1_chunk, CHUNK_SIZE, memory_order_relaxed);
        if(start >= g_sweep_limit) break;
        u64 end = start + CHUNK_SIZE;
        if(end > g_sweep_limit) end = g_sweep_limit;

        u64 local_l2 = 0, local_l3 = 0, local_sol19 = 0;

        for(int i=0; i<PF_BATCH_BP; i++){
            u32 a1  = (u32)(start + (u64)i);
            u32 e1  = a1 + C_e1;
            u32 sm1 = S0(a1) + (a1 & M_a0_am1) + C_sm1;
            u32 G1  = sm1 + S1(e1) + (em1 ^ (e1 & M_e0_em1)) + C_G1;
            b_sm1[0][i] = sm1;
            b_G1[0][i]  = G1;
            b_f1[i]     = G1 - a1;
        }
        for(int c=0; c<B; c++){
            u32 ct16 = C_T16_c[c];
            for(int i=0; i<PF_BATCH_BP; i++) b_t16_c[0][c][i] = ct16 + b_f1[i];
            if(!g_pair_l1 || (c & 1) == 0){
                for(int i=0; i<PF_BATCH_BP; i++) __builtin_prefetch(&H_pack6[pf12_off(b_t16_c[0][c][i])], 0, 0);
            }
        }
        int cur = 0;

        for(u64 base = start; base < end; base += PF_BATCH_BP){
            int nxt = cur ^ 1;
            u64 next_base = base + PF_BATCH_BP;

            // Step 1: NEON-vectorized Round-1 ALU + separate prefetch loop for NEXT batch
            if(__builtin_expect(next_base < end, 1)){
                for(int i=0; i<PF_BATCH_BP; i++){
                    u32 a1  = (u32)(next_base + (u64)i);
                    u32 e1  = a1 + C_e1;
                    u32 sm1 = S0(a1) + (a1 & M_a0_am1) + C_sm1;
                    u32 G1  = sm1 + S1(e1) + (em1 ^ (e1 & M_e0_em1)) + C_G1;
                    b_sm1[nxt][i] = sm1;
                    b_G1[nxt][i]  = G1;
                    b_f1[i]       = G1 - a1;
                }
                for(int c=0; c<B; c++){
                    u32 ct16 = C_T16_c[c];
                    for(int i=0; i<PF_BATCH_BP; i++) b_t16_c[nxt][c][i] = ct16 + b_f1[i];
                    if(!g_pair_l1 || (c & 1) == 0){
                        for(int i=0; i<PF_BATCH_BP; i++) __builtin_prefetch(&H_pack6[pf12_off(b_t16_c[nxt][c][i])], 0, 0);
                    }
                }
            }

            // Step 2: Drain q3 & Compact q2
            for(int c=0; c<B; c++){
                if(n_q3[c] >= PF_BATCH){
                    drain_q3_fast_6gb_ctx(q3_hot[c], q3_cold[c], n_q3[c], st, c, &local_sol19);
                    n_q3[c] = 0;
                }
            }
            for(int c=0; c<B; c++){
                if(n_q2[c] >= PF_BATCH){
                    compact_q2_6gb_ctx(q2_t17[c], q2_G2[c], q2_cold[c], n_q2[c],
                                       q3_hot[c], q3_cold[c], &n_q3[c],
                                       w2_cand, w2_idx, l2_W2, l2_M3, l2_idx,
                                       st, c, &local_l2, &local_l3, &local_sol19);
                    n_q2[c] = 0;
                }
            }

            // Step 3: Branchless L1 compaction + immediate t17 prefetches
            if(g_pair_l1 && (B & 1) == 0){
                for(int p=0; p<B/2; p++){
                    int c0 = 2*p, c1 = 2*p + 1;
                    int nl0 = 0, nl1 = 0;
                    l1_compact_pair_6gb(b_t16_c[cur][c0], dT16_pair[p],
                                        l1_W1, l1_idx, &nl0, l1_W1_b, l1_idx_b, &nl1);
                    l1_push_q2_6gb(base, b_sm1[cur], b_G1[cur],
                                   l1_W1, l1_idx, nl0, T17_c[c0], st,
                                   q2_t17[c0], q2_G2[c0], q2_cold[c0], &n_q2[c0]);
                    l1_push_q2_6gb(base, b_sm1[cur], b_G1[cur],
                                   l1_W1_b, l1_idx_b, nl1, T17_c[c1], st,
                                   q2_t17[c1], q2_G2[c1], q2_cold[c1], &n_q2[c1]);
                }
            } else {
                for(int c=0; c<B; c++){
                    int nl = 0;
                    for(int i=0; i<PF_BATCH_BP; i++){
                        u32 y = b_t16_c[cur][c][i];
                        const u16 *b = &H_pack6[slide3_off(y)];
                        u32 ulo0 = b[0], ulo1 = b[1], ulo2 = b[2];
                        u32 u0 = ulo0 | ((u32)REC_HI16[rec_idx(ulo0, y)] << 16);
                        u32 u1 = ulo1 | ((u32)REC_HI16[rec_idx(ulo1, y)] << 16);
                        u32 u2 = ulo2 | ((u32)REC_HI16[rec_idx(ulo2, y)] << 16);
                        u32 h0 = (Qf(u0) == y) & (ulo0 != 0u);
                        l1_W1[nl] = u0; l1_idx[nl] = (u8)i; nl += h0;
                        u32 h1 = (Qf(u1) == y) & (ulo1 != 0u);
                        l1_W1[nl] = u1; l1_idx[nl] = (u8)i; nl += h1;
                        u32 h2 = (Qf(u2) == y) & (ulo2 != 0u);
                        l1_W1[nl] = u2; l1_idx[nl] = (u8)i; nl += h2;
#if SLIDE4_MODE == 1
                        if(__builtin_expect(h0 & h1 & h2, 0)){
                            u32 ulo3 = b[(y & 15u) >= 13u ? -1 : 3];
                            u32 u3   = ulo3 | ((u32)REC_HI16[rec_idx(ulo3, y)] << 16);
                            l1_W1[nl] = u3; l1_idx[nl] = (u8)i;
                            nl += (Qf(u3) == y) & (ulo3 != 0u);
                        }
#endif
                    }
                    const u32 T17 = T17_c[c];
                    int nq = n_q2[c];
                    for(int j=0; j<nl; j++){
                        u32 i   = l1_idx[j];
                        u32 W1  = l1_W1[j];
                        u32 a1  = (u32)base + i;
                        u32 e1  = a1 + C_e1;
                        u32 sm1 = b_sm1[cur][i];
                        u32 G1  = b_G1[cur][i];
                        u32 a2  = W1 + G1;
                        u32 e2  = a2 - sm1 + am2;
                        u32 sm2 = S0(a2) + MAJ(a2, a1, a0);
                        u32 G2  = sm2 + S1(e2) + CH(e2, e1, e0) + C_G2;
                        u32 t17 = T17 + G2 - W1;
                        __builtin_prefetch(&H_pack6[pf12_off(t17)], 0, 0);
                        q2_t17[c][nq]  = t17;
                        q2_G2[c][nq]   = G2;
                        q2_cold[c][nq] = (q2_cold4_t){a1, a2, e2, sm2};
                        nq++;
                    }
                    n_q2[c] = nq;
                }
            }

            cur = nxt;
        }

        for(int c=0; c<B; c++){
            if(n_q2[c] > 0){
                compact_q2_6gb_ctx(q2_t17[c], q2_G2[c], q2_cold[c], n_q2[c],
                                   q3_hot[c], q3_cold[c], &n_q3[c],
                                   w2_cand, w2_idx, l2_W2, l2_M3, l2_idx,
                                   st, c, &local_l2, &local_l3, &local_sol19);
                n_q2[c] = 0;
            }
            if(n_q3[c] > 0){
                drain_q3_fast_6gb_ctx(q3_hot[c], q3_cold[c], n_q3[c], st, c, &local_sol19);
                n_q3[c] = 0;
            }
        }

        atomic_fetch_add_explicit(&g_chains_done, (end - start) * (u64)B, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_l2_nodes, local_l2, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_l3_lookups, local_l3, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_sol19_count, local_sol19, memory_order_relaxed);
    }
    return NULL;
}

static int g_f1_stream = 0;

// Sequential Level-1 F1-Streaming Worker for 6 GB Sudoku-Augmented Table (B == 2):
// - Eliminates random DRAM reads at Level 1 by streaming f = F1(a1) = G1(a1) - W0(a1) sequentially!
// - Evaluates H_pack6[slide3_off(f + T16_c)] sequentially (42 consecutive f values share a single 64B L1 cache line!).
// - Skips 36.79% of f values where F1_count[f] == 0, and skips A1_perm + Round-1 ALU when both contexts miss L1.
// - Computes G1 = f + a1 + C_W0 in 1 addition (no S1(e1) or CH(e1, e0, em1) needed at Level 1!).
// - Uses a 64-of-128 sliding FIFO queue so every compacted batch has exactly PF_BATCH (64) items
//   that were prefetched at least 64 q2-pushes (~150 ns) earlier!
static void *worker_thread_sorted6_b2(void *arg){
    (void)arg;
    const solver_state_t *st = &G_STATE;
    const u32 a0 = st->a0, am1 = st->am1, am2 = st->am2;
    const u32 e0 = st->e0;
    const u32 C_e1 = st->C_e1, C_W0 = st->C_W0, C_G2 = st->C_G2;
    const u32 M_a0_am1 = a0 ^ am1;
    const u32 C_sm1    = a0 & am1;

    const u32 T16_0 = st->rctx[0].T16, T16_1 = st->rctx[1].T16;
    const u32 T17_0 = st->rctx[0].T17, T17_1 = st->rctx[1].T17;

    u32        q2_t17_0[256], q2_t17_1[256];
    u32        q2_G2_0[256],  q2_G2_1[256];
    q2_cold4_t q2_cold_0[256], q2_cold_1[256];

    u32 w2_cand[512];
    u8  w2_idx[512];
    u32 l2_W2[512];
    u32 l2_M3[512];
    u8  l2_idx[512];

    q3_hot4_t  q3_hot_0[256],  q3_hot_1[256];
    q3_cold4_t q3_cold_0[256], q3_cold_1[256];
    int n_q2_0 = 0, n_q2_1 = 0, n_q3_0 = 0, n_q3_1 = 0;

    const u8 *cnt_arr = (const u8 *)F1_count;

    while(!atomic_load_explicit(&g_solved20, memory_order_relaxed)){
        u64 start = atomic_fetch_add_explicit(&g_next_a1_chunk, CHUNK_SIZE, memory_order_relaxed);
        if(start >= SWEEP_LIMIT) break;
        u64 end = start + CHUNK_SIZE;
        if(end > SWEEP_LIMIT) end = SWEEP_LIMIT;

        u64 local_l2 = 0, local_l3 = 0, local_sol19 = 0;
        u64 idx = F1_chunk_off[start >> 18];

        for(u64 f = start; f < end; f++){
            u8 cnt = cnt_arr[f];
            if(!cnt) continue;

            u32 y0 = (u32)f + T16_0;
            const u16 *b0 = &H_pack6[slide3_off(y0)];
            u32 ulo0_0 = b0[0], ulo0_1 = b0[1], ulo0_2 = b0[2];
            u32 u0_0 = ulo0_0 | ((u32)REC_HI16[rec_idx(ulo0_0, y0)] << 16);
            u32 u0_1 = ulo0_1 | ((u32)REC_HI16[rec_idx(ulo0_1, y0)] << 16);
            u32 u0_2 = ulo0_2 | ((u32)REC_HI16[rec_idx(ulo0_2, y0)] << 16);
            u32 cand0[4];
            int n0 = 0;
            u32 h0_0 = (Qf(u0_0) == y0) & (ulo0_0 != 0u); cand0[n0] = u0_0; n0 += h0_0;
            u32 h0_1 = (Qf(u0_1) == y0) & (ulo0_1 != 0u); cand0[n0] = u0_1; n0 += h0_1;
            u32 h0_2 = (Qf(u0_2) == y0) & (ulo0_2 != 0u); cand0[n0] = u0_2; n0 += h0_2;
#if SLIDE4_MODE == 1
            if(__builtin_expect(h0_0 & h0_1 & h0_2, 0)){
                u32 ulo0_3 = b0[(y0 & 15u) >= 13u ? -1 : 3];
                u32 u0_3   = ulo0_3 | ((u32)REC_HI16[rec_idx(ulo0_3, y0)] << 16);
                cand0[n0]  = u0_3;
                n0 += (Qf(u0_3) == y0) & (ulo0_3 != 0u);
            }
#endif

            u32 y1 = (u32)f + T16_1;
            const u16 *b1 = &H_pack6[slide3_off(y1)];
            u32 ulo1_0 = b1[0], ulo1_1 = b1[1], ulo1_2 = b1[2];
            u32 u1_0 = ulo1_0 | ((u32)REC_HI16[rec_idx(ulo1_0, y1)] << 16);
            u32 u1_1 = ulo1_1 | ((u32)REC_HI16[rec_idx(ulo1_1, y1)] << 16);
            u32 u1_2 = ulo1_2 | ((u32)REC_HI16[rec_idx(ulo1_2, y1)] << 16);
            u32 cand1[4];
            int n1 = 0;
            u32 h1_0 = (Qf(u1_0) == y1) & (ulo1_0 != 0u); cand1[n1] = u1_0; n1 += h1_0;
            u32 h1_1 = (Qf(u1_1) == y1) & (ulo1_1 != 0u); cand1[n1] = u1_1; n1 += h1_1;
            u32 h1_2 = (Qf(u1_2) == y1) & (ulo1_2 != 0u); cand1[n1] = u1_2; n1 += h1_2;
#if SLIDE4_MODE == 1
            if(__builtin_expect(h1_0 & h1_1 & h1_2, 0)){
                u32 ulo1_3 = b1[(y1 & 15u) >= 13u ? -1 : 3];
                u32 u1_3   = ulo1_3 | ((u32)REC_HI16[rec_idx(ulo1_3, y1)] << 16);
                cand1[n1]  = u1_3;
                n1 += (Qf(u1_3) == y1) & (ulo1_3 != 0u);
            }
#endif

            if(!(n0 | n1)){
                idx += cnt;
                continue;
            }

            u32 f_cw0 = (u32)f + C_W0;
            for(u8 j = 0; j < cnt; j++){
                u32 a1  = A1_perm[idx++];
                u32 e1  = a1 + C_e1;
                u32 sm1 = S0(a1) + (a1 & M_a0_am1) + C_sm1;
                u32 G1  = f_cw0 + a1;

                if(n0){
                    for(int x = 0; x < n0; x++){
                        u32 W1  = cand0[x];
                        u32 a2  = W1 + G1;
                        u32 e2  = a2 - sm1 + am2;
                        u32 sm2 = S0(a2) + MAJ(a2, a1, a0);
                        u32 G2  = sm2 + S1(e2) + CH(e2, e1, e0) + C_G2;
                        u32 t17 = T17_0 + G2 - W1;
                        __builtin_prefetch(&H_pack6[pf12_off(t17)], 0, 0);
                        q2_t17_0[n_q2_0]  = t17;
                        q2_G2_0[n_q2_0]   = G2;
                        q2_cold_0[n_q2_0] = (q2_cold4_t){a1, a2, e2, sm2};
                        n_q2_0++;
                    }
                    if(__builtin_expect(n_q2_0 >= 2 * PF_BATCH, 0)){
                        if(n_q3_0 >= PF_BATCH){
                            drain_q3_fast_6gb_ctx(q3_hot_0, q3_cold_0, n_q3_0, st, 0, &local_sol19);
                            n_q3_0 = 0;
                        }
                        compact_q2_6gb_ctx(q2_t17_0, q2_G2_0, q2_cold_0, PF_BATCH,
                                           q3_hot_0, q3_cold_0, &n_q3_0,
                                           w2_cand, w2_idx, l2_W2, l2_M3, l2_idx,
                                           st, 0, &local_l2, &local_l3, &local_sol19);
                        int rem = n_q2_0 - PF_BATCH;
                        for(int k = 0; k < rem; k++){
                            q2_t17_0[k]  = q2_t17_0[PF_BATCH + k];
                            q2_G2_0[k]   = q2_G2_0[PF_BATCH + k];
                            q2_cold_0[k] = q2_cold_0[PF_BATCH + k];
                        }
                        n_q2_0 = rem;
                    }
                }

                if(n1){
                    for(int x = 0; x < n1; x++){
                        u32 W1  = cand1[x];
                        u32 a2  = W1 + G1;
                        u32 e2  = a2 - sm1 + am2;
                        u32 sm2 = S0(a2) + MAJ(a2, a1, a0);
                        u32 G2  = sm2 + S1(e2) + CH(e2, e1, e0) + C_G2;
                        u32 t17 = T17_1 + G2 - W1;
                        __builtin_prefetch(&H_pack6[pf12_off(t17)], 0, 0);
                        q2_t17_1[n_q2_1]  = t17;
                        q2_G2_1[n_q2_1]   = G2;
                        q2_cold_1[n_q2_1] = (q2_cold4_t){a1, a2, e2, sm2};
                        n_q2_1++;
                    }
                    if(__builtin_expect(n_q2_1 >= 2 * PF_BATCH, 0)){
                        if(n_q3_1 >= PF_BATCH){
                            drain_q3_fast_6gb_ctx(q3_hot_1, q3_cold_1, n_q3_1, st, 1, &local_sol19);
                            n_q3_1 = 0;
                        }
                        compact_q2_6gb_ctx(q2_t17_1, q2_G2_1, q2_cold_1, PF_BATCH,
                                           q3_hot_1, q3_cold_1, &n_q3_1,
                                           w2_cand, w2_idx, l2_W2, l2_M3, l2_idx,
                                           st, 1, &local_l2, &local_l3, &local_sol19);
                        int rem = n_q2_1 - PF_BATCH;
                        for(int k = 0; k < rem; k++){
                            q2_t17_1[k]  = q2_t17_1[PF_BATCH + k];
                            q2_G2_1[k]   = q2_G2_1[PF_BATCH + k];
                            q2_cold_1[k] = q2_cold_1[PF_BATCH + k];
                        }
                        n_q2_1 = rem;
                    }
                }
            }
        }

        while(n_q2_0 > 0){
            int n_proc = (n_q2_0 > PF_BATCH) ? PF_BATCH : n_q2_0;
            compact_q2_6gb_ctx(q2_t17_0, q2_G2_0, q2_cold_0, n_proc,
                               q3_hot_0, q3_cold_0, &n_q3_0,
                               w2_cand, w2_idx, l2_W2, l2_M3, l2_idx,
                               st, 0, &local_l2, &local_l3, &local_sol19);
            int rem = n_q2_0 - n_proc;
            for(int k = 0; k < rem; k++){
                q2_t17_0[k]  = q2_t17_0[n_proc + k];
                q2_G2_0[k]   = q2_G2_0[n_proc + k];
                q2_cold_0[k] = q2_cold_0[n_proc + k];
            }
            n_q2_0 = rem;
        }
        while(n_q2_1 > 0){
            int n_proc = (n_q2_1 > PF_BATCH) ? PF_BATCH : n_q2_1;
            compact_q2_6gb_ctx(q2_t17_1, q2_G2_1, q2_cold_1, n_proc,
                               q3_hot_1, q3_cold_1, &n_q3_1,
                               w2_cand, w2_idx, l2_W2, l2_M3, l2_idx,
                               st, 1, &local_l2, &local_l3, &local_sol19);
            int rem = n_q2_1 - n_proc;
            for(int k = 0; k < rem; k++){
                q2_t17_1[k]  = q2_t17_1[n_proc + k];
                q2_G2_1[k]   = q2_G2_1[n_proc + k];
                q2_cold_1[k] = q2_cold_1[n_proc + k];
            }
            n_q2_1 = rem;
        }
        if(n_q3_0 > 0){
            drain_q3_fast_6gb_ctx(q3_hot_0, q3_cold_0, n_q3_0, st, 0, &local_sol19);
            n_q3_0 = 0;
        }
        if(n_q3_1 > 0){
            drain_q3_fast_6gb_ctx(q3_hot_1, q3_cold_1, n_q3_1, st, 1, &local_sol19);
            n_q3_1 = 0;
        }

        atomic_fetch_add_explicit(&g_chains_done, (end - start) * 2ULL, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_l2_nodes, local_l2, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_l3_lookups, local_l3, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_sol19_count, local_sol19, memory_order_relaxed);
    }
    return NULL;
}

static void *worker_thread(void *arg){
    if(g_table_gb == 6 && G_STATE.batch_ctx == 2 && g_f1_stream){
        return worker_thread_sorted6_b2(arg);
    }
    if(g_table_gb == 6 && G_STATE.batch_ctx == 1){
        return worker_thread_6gb_b1(arg);
    }
    if(g_table_gb == 6 && G_STATE.batch_ctx == 2 && !g_pair_l1){
        return worker_thread_6gb_b2(arg);
    }
    if(g_table_gb == 6 && G_STATE.batch_ctx >= 2){
        return worker_thread_6gb_bpairs(arg);
    }
    if(g_table_gb == 4 && G_STATE.batch_ctx == 1){
        return worker_thread_4gb_b1(arg);
    }
    (void)arg;
    const solver_state_t *st = &G_STATE;
    const u32 a0 = st->a0, am1 = st->am1, am2 = st->am2;
    const u32 e0 = st->e0, em1 = st->em1;
    const u32 C_e1 = st->C_e1, C_W0 = st->C_W0;
    const u32 C_G1 = st->C_G1, C_G2 = st->C_G2, C_G3 = st->C_G3;
    const int w_thresh = st->w_thresh;
    const int B = st->batch_ctx;

    u32 T16_arr[MAX_BATCH_CTX], T17_arr[MAX_BATCH_CTX];
    for(int c=0; c<B; c++){
        T16_arr[c] = st->rctx[c].T16;
        T17_arr[c] = st->rctx[c].T17;
    }

    u32 cand1[4];
    u32 b_a1[PF_BATCH], b_e1[PF_BATCH], b_sm1[PF_BATCH], b_G1[PF_BATCH], b_F1[PF_BATCH];
    q2_item_t q2[PF_BATCH + 8];
    q3_item_t q3[PF_BATCH + 8];
    int n_q2 = 0, n_q3 = 0;

    while(!atomic_load_explicit(&g_solved20, memory_order_relaxed)){
        u64 start = atomic_fetch_add_explicit(&g_next_a1_chunk, CHUNK_SIZE, memory_order_relaxed);
        if(start >= (1ULL << 32)) break;
        u64 end = start + CHUNK_SIZE;
        if(end > (1ULL << 32)) end = (1ULL << 32);

        u64 local_l2 = 0, local_l3 = 0, local_sol19 = 0;

        for(u64 base = start; base < end; base += PF_BATCH){
            for(int i=0; i<PF_BATCH; i++){
                u32 a1 = (u32)(base + (u64)i);
                u32 e1 = a1 + C_e1;
                u32 W0 = a1 + C_W0;
                u32 sm1 = S0(a1) + MAJ(a1, a0, am1);
                u32 G1 = sm1 + S1(e1) + CH(e1, e0, em1) + C_G1;
                u32 F1 = G1 - W0;

                b_a1[i]  = a1;
                b_e1[i]  = e1;
                b_sm1[i] = sm1;
                b_G1[i]  = G1;
                b_F1[i]  = F1;

                for(int c=0; c<B; c++){
                    prefetch_q0(T16_arr[c] + F1);
                }
            }

            for(int i=0; i<PF_BATCH; i++){
                u32 F1  = b_F1[i];
                u32 a1  = b_a1[i];
                u32 e1  = b_e1[i];
                u32 sm1 = b_sm1[i];
                u32 G1  = b_G1[i];

                for(int c=0; c<B; c++){
                    u32 t16 = T16_arr[c] + F1;
                    int n1 = lookup_q0(t16, cand1);
                    if(!n1) continue;

                    u32 T17 = T17_arr[c];
                    for(int x=0; x<n1; x++){
                        u32 W1 = cand1[x];
                        u32 a2 = W1 + G1;
                        u32 e2 = a2 - sm1 + am2;
                        u32 sm2 = S0(a2) + MAJ(a2, a1, a0);
                        u32 G2 = sm2 + S1(e2) + CH(e2, e1, e0) + C_G2;
                        u32 t17 = T17 + G2 - W1;

                        prefetch_q0(t17);
                        q2[n_q2].t17    = t17;
                        q2[n_q2].G2     = G2;
                        q2[n_q2].a1     = a1;
                        q2[n_q2].e1     = e1;
                        q2[n_q2].a2     = a2;
                        q2[n_q2].e2     = e2;
                        q2[n_q2].sm2    = sm2;
                        q2[n_q2].ctx_id = (u32)c;
                        n_q2++;

                        if(n_q2 >= PF_BATCH){
                            drain_q2(q2, n_q2, q3, &n_q3, st, am1, C_G3, w_thresh,
                                     &local_l2, &local_l3, &local_sol19);
                            n_q2 = 0;
                        }
                    }
                }
            }
        }

        if(n_q2 > 0){
            drain_q2(q2, n_q2, q3, &n_q3, st, am1, C_G3, w_thresh,
                     &local_l2, &local_l3, &local_sol19);
            n_q2 = 0;
        }
        if(n_q3 > 0){
            drain_q3(q3, n_q3, st, &local_sol19);
            n_q3 = 0;
        }

        atomic_fetch_add_explicit(&g_chains_done, (end - start) * (u64)B, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_l2_nodes, local_l2, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_l3_lookups, local_l3, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_sol19_count, local_sol19, memory_order_relaxed);
    }
    return NULL;
}

static _Atomic u32 *F1_head_tmp = NULL;

static void *f1_count_worker(void *arg){
    build_arg_t *ba = (build_arg_t *)arg;
    u64 span = ba->end_u - ba->start_u;
    u64 start = ba->start_u + ((u64)ba->tid * span) / (u64)ba->nthreads;
    u64 end   = ba->start_u + ((u64)(ba->tid + 1) * span) / (u64)ba->nthreads;
    start &= ~(u64)(BUILD_PF - 1);
    if(ba->tid < ba->nthreads - 1) end &= ~(u64)(BUILD_PF - 1);
    const solver_state_t *st = &G_STATE;
    const u32 a0 = st->a0, am1 = st->am1, e0 = st->e0, em1 = st->em1;
    const u32 C_e1 = st->C_e1, C_W0 = st->C_W0, C_G1 = st->C_G1;
    u32 b_f[BUILD_PF];

    for(u64 base = start; base + BUILD_PF <= end; base += BUILD_PF){
        for(int i=0; i<BUILD_PF; i++){
            u32 a1 = (u32)(base + (u64)i);
            u32 e1 = a1 + C_e1;
            u32 W0 = a1 + C_W0;
            u32 sm1 = S0(a1) + MAJ(a1, a0, am1);
            u32 G1 = sm1 + S1(e1) + CH(e1, e0, em1) + C_G1;
            u32 f = G1 - W0;
            b_f[i] = f;
            __builtin_prefetch(&F1_count[f], 1, 0);
        }
        for(int i=0; i<BUILD_PF; i++){
            u32 f = b_f[i];
            u8 c = atomic_load_explicit(&F1_count[f], memory_order_relaxed);
            while(c < 255u){
                if(atomic_compare_exchange_weak_explicit(&F1_count[f], &c, (u8)(c + 1u),
                        memory_order_relaxed, memory_order_relaxed))
                    break;
            }
        }
    }
    return NULL;
}

static void *f1_scatter_worker(void *arg){
    build_arg_t *ba = (build_arg_t *)arg;
    u64 span = ba->end_u - ba->start_u;
    u64 start = ba->start_u + ((u64)ba->tid * span) / (u64)ba->nthreads;
    u64 end   = ba->start_u + ((u64)(ba->tid + 1) * span) / (u64)ba->nthreads;
    start &= ~(u64)(BUILD_PF - 1);
    if(ba->tid < ba->nthreads - 1) end &= ~(u64)(BUILD_PF - 1);
    const solver_state_t *st = &G_STATE;
    const u32 a0 = st->a0, am1 = st->am1, e0 = st->e0, em1 = st->em1;
    const u32 C_e1 = st->C_e1, C_W0 = st->C_W0, C_G1 = st->C_G1;
    u32 b_f[BUILD_PF];
    u32 b_a1[BUILD_PF];

    for(u64 base = start; base + BUILD_PF <= end; base += BUILD_PF){
        for(int i=0; i<BUILD_PF; i++){
            u32 a1 = (u32)(base + (u64)i);
            u32 e1 = a1 + C_e1;
            u32 W0 = a1 + C_W0;
            u32 sm1 = S0(a1) + MAJ(a1, a0, am1);
            u32 G1 = sm1 + S1(e1) + CH(e1, e0, em1) + C_G1;
            u32 f = G1 - W0;
            b_f[i] = f;
            b_a1[i] = a1;
            __builtin_prefetch(&F1_head_tmp[f], 1, 0);
        }
        for(int i=0; i<BUILD_PF; i++){
            u32 f = b_f[i];
            u32 pos = atomic_fetch_add_explicit(&F1_head_tmp[f], 1u, memory_order_relaxed);
            A1_perm[pos] = b_a1[i];
        }
    }
    return NULL;
}

typedef struct {
    int tid;
    int nthreads;
    u64 sum;
    u64 offset;
} psum_arg_t;

static void *f1_psum_pass1(void *arg){
    psum_arg_t *pa = (psum_arg_t *)arg;
    u64 start = ((u64)pa->tid * (1ULL << 32)) / (u64)pa->nthreads;
    u64 end   = ((u64)(pa->tid + 1) * (1ULL << 32)) / (u64)pa->nthreads;
    start &= ~(CHUNK_SIZE - 1ULL);
    if(pa->tid == pa->nthreads - 1) end = (1ULL << 32);
    else end &= ~(CHUNK_SIZE - 1ULL);

    u64 s = 0;
    const u8 *cnt_arr = (const u8 *)F1_count;
    for(u64 f = start; f < end; f++) s += cnt_arr[f];
    pa->sum = s;
    return NULL;
}

static void *f1_psum_pass2(void *arg){
    psum_arg_t *pa = (psum_arg_t *)arg;
    u64 start = ((u64)pa->tid * (1ULL << 32)) / (u64)pa->nthreads;
    u64 end   = ((u64)(pa->tid + 1) * (1ULL << 32)) / (u64)pa->nthreads;
    start &= ~(CHUNK_SIZE - 1ULL);
    if(pa->tid == pa->nthreads - 1) end = (1ULL << 32);
    else end &= ~(CHUNK_SIZE - 1ULL);

    u64 cur = pa->offset;
    const u8 *cnt_arr = (const u8 *)F1_count;
    u32 *head_arr = (u32 *)F1_head_tmp;
    for(u64 f = start; f < end; f++){
        if((f & (CHUNK_SIZE - 1ULL)) == 0ULL){
            F1_chunk_off[f >> 18] = cur;
        }
        head_arr[f] = (u32)cur;
        cur += cnt_arr[f];
    }
    return NULL;
}

#define CACHE_FILE_F1_SHM "/dev/shm/f1_stream_v1.bin"

static void init_f1_table(int nthreads){
    double t0 = now_sec();
    size_t sz_cnt  = (1ULL << 32) * sizeof(u8);
    size_t sz_perm = (1ULL << 32) * sizeof(u32);
    size_t sz_off  = sizeof(F1_chunk_off);
    size_t sz_tot  = sz_cnt + sz_perm + sz_off;

    int fd = open(CACHE_FILE_F1_SHM, O_RDONLY);
    if(fd >= 0){
        struct stat st_buf;
        if(fstat(fd, &st_buf) == 0 && (size_t)st_buf.st_size == sz_tot){
            void *p_all = mmap(NULL, sz_tot, PROT_READ, MAP_SHARED, fd, 0);
            close(fd);
            if(p_all != MAP_FAILED){
                F1_count = (_Atomic u8 *)p_all;
                A1_perm  = (u32 *)((u8 *)p_all + sz_cnt);
                memcpy(F1_chunk_off, (const u8 *)p_all + sz_cnt + sz_perm, sz_off);
                fprintf(stderr, "[init] Loaded 20.00 GB Level-1 F1 streaming tables from %s (zero-copy mmap) in %.3f s\n",
                        CACHE_FILE_F1_SHM, now_sec() - t0);
                return;
            }
        } else {
            close(fd);
        }
    }

    fprintf(stderr, "[init] Building 20.00 GB sequential Level-1 (F1_count + A1_perm) streaming tables...\n");
    void *p_cnt = NULL, *p_perm = NULL, *p_all_shm = MAP_FAILED;
    int shm_fd = open(CACHE_FILE_F1_SHM ".tmp", O_RDWR | O_CREAT | O_TRUNC, 0644);
    if(shm_fd >= 0){
        if(ftruncate(shm_fd, (off_t)sz_tot) == 0){
            p_all_shm = mmap(NULL, sz_tot, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
        }
        if(p_all_shm == MAP_FAILED){
            close(shm_fd);
            shm_fd = -1;
            unlink(CACHE_FILE_F1_SHM ".tmp");
        } else {
            p_cnt  = p_all_shm;
            p_perm = (u8 *)p_all_shm + sz_cnt;
        }
    }
    if(p_all_shm == MAP_FAILED){
        p_cnt  = mmap(NULL, sz_cnt,  PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        p_perm = mmap(NULL, sz_perm, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    }
    void *p_head = mmap(NULL, (1ULL << 32) * sizeof(u32), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if(p_cnt == MAP_FAILED || p_perm == MAP_FAILED || p_head == MAP_FAILED){
        perror("mmap F1 tables");
        exit(1);
    }
#ifdef MADV_HUGEPAGE
    madvise(p_cnt,  sz_cnt,  MADV_HUGEPAGE);
    madvise(p_perm, sz_perm, MADV_HUGEPAGE);
    madvise(p_head, (1ULL << 32) * sizeof(u32), MADV_HUGEPAGE);
#endif
    F1_count    = (_Atomic u8 *)p_cnt;
    A1_perm     = (u32 *)p_perm;
    F1_head_tmp = (_Atomic u32 *)p_head;

    pthread_t *th = malloc(nthreads * sizeof(pthread_t));
    build_arg_t *args = calloc(nthreads, sizeof(build_arg_t));
    for(int i=0;i<nthreads;i++){
        args[i].tid = i;
        args[i].nthreads = nthreads;
        args[i].start_u = 0;
        args[i].end_u   = (1ULL << 32);
        pthread_create(&th[i], NULL, f1_count_worker, &args[i]);
    }
    for(int i=0;i<nthreads;i++) pthread_join(th[i], NULL);

    psum_arg_t *pargs = calloc(nthreads, sizeof(psum_arg_t));
    for(int i=0;i<nthreads;i++){
        pargs[i].tid = i;
        pargs[i].nthreads = nthreads;
        pthread_create(&th[i], NULL, f1_psum_pass1, &pargs[i]);
    }
    for(int i=0;i<nthreads;i++) pthread_join(th[i], NULL);
    u64 acc = 0;
    for(int i=0;i<nthreads;i++){
        pargs[i].offset = acc;
        acc += pargs[i].sum;
    }
    F1_chunk_off[N_F1_CHUNKS] = acc;
    for(int i=0;i<nthreads;i++) pthread_create(&th[i], NULL, f1_psum_pass2, &pargs[i]);
    for(int i=0;i<nthreads;i++) pthread_join(th[i], NULL);

    for(int i=0;i<nthreads;i++) pthread_create(&th[i], NULL, f1_scatter_worker, &args[i]);
    for(int i=0;i<nthreads;i++) pthread_join(th[i], NULL);

    munmap(p_head, (1ULL << 32) * sizeof(u32));
    F1_head_tmp = NULL;
    free(th);
    free(args);
    free(pargs);

    if(shm_fd >= 0 && p_all_shm != MAP_FAILED){
        memcpy((u8 *)p_all_shm + sz_cnt + sz_perm, F1_chunk_off, sz_off);
        close(shm_fd);
        rename(CACHE_FILE_F1_SHM ".tmp", CACHE_FILE_F1_SHM);
    }

    fprintf(stderr, "[init] 20.00 GB Level-1 F1 streaming tables built in %.2f s (total indexed a1=%llu)\n",
            now_sec() - t0, (unsigned long long)acc);
}

// Ultra-fast sequential Level-1 worker for 32 GB VM mode (0 random DRAM reads at Level 1!)
static void *worker_thread_sorted32(void *arg){
    (void)arg;
    const solver_state_t *st = &G_STATE;
    const u32 a0 = st->a0, am1 = st->am1, am2 = st->am2;
    const u32 e0 = st->e0;
    const u32 C_e1 = st->C_e1, C_W0 = st->C_W0;
    const u32 C_G2 = st->C_G2, C_G3 = st->C_G3;
    const int w_thresh = st->w_thresh;
    const int B = st->batch_ctx;

    u32 T16_arr[MAX_BATCH_CTX], T17_arr[MAX_BATCH_CTX];
    for(int c=0; c<B; c++){
        T16_arr[c] = st->rctx[c].T16;
        T17_arr[c] = st->rctx[c].T17;
    }

    u32 cand1[4];
    q2_item_t q2[PF_BATCH + 8];
    q3_item_t q3[PF_BATCH + 8];
    int n_q2 = 0, n_q3 = 0;
    const u8 *cnt_arr = (const u8 *)F1_count;

    while(!atomic_load_explicit(&g_solved20, memory_order_relaxed)){
        u64 start = atomic_fetch_add_explicit(&g_next_a1_chunk, CHUNK_SIZE, memory_order_relaxed);
        if(start >= (1ULL << 32)) break;
        u64 end = start + CHUNK_SIZE;
        if(end > (1ULL << 32)) end = (1ULL << 32);

        u64 local_l2 = 0, local_l3 = 0, local_sol19 = 0;
        u64 idx = F1_chunk_off[start >> 18];

        if(B == 1){
            const u32 T16 = T16_arr[0], T17 = T17_arr[0];
            for(u64 f = start; f < end; f++){
                u8 cnt = cnt_arr[f];
                if(!cnt) continue;
                u32 t16 = (u32)f + T16;
                int n1 = lookup_exact32(t16, cand1);
                if(!n1){
                    idx += cnt;
                    continue;
                }
                for(u8 j = 0; j < cnt; j++){
                    u32 a1 = A1_perm[idx++];
                    u32 e1 = a1 + C_e1;
                    u32 sm1 = S0(a1) + MAJ(a1, a0, am1);
                    u32 G1 = (u32)f + a1 + C_W0;

                    for(int x=0; x<n1; x++){
                        u32 W1 = cand1[x];
                        u32 a2 = W1 + G1;
                        u32 e2 = a2 - sm1 + am2;
                        u32 sm2 = S0(a2) + MAJ(a2, a1, a0);
                        u32 G2 = sm2 + S1(e2) + CH(e2, e1, e0) + C_G2;
                        u32 t17 = T17 + G2 - W1;

                        __builtin_prefetch(&H_exact32[t17], 0, 0);
                        q2[n_q2].t17    = t17;
                        q2[n_q2].G2     = G2;
                        q2[n_q2].a1     = a1;
                        q2[n_q2].e1     = e1;
                        q2[n_q2].a2     = a2;
                        q2[n_q2].e2     = e2;
                        q2[n_q2].sm2    = sm2;
                        q2[n_q2].ctx_id = 0;
                        n_q2++;

                        if(n_q2 >= PF_BATCH){
                            drain_q2(q2, n_q2, q3, &n_q3, st, am1, C_G3, w_thresh,
                                     &local_l2, &local_l3, &local_sol19);
                            n_q2 = 0;
                        }
                    }
                }
            }
        } else {
            u32 cand1_b[MAX_BATCH_CTX][4];
            int n1_b[MAX_BATCH_CTX];
            int active_c[MAX_BATCH_CTX];
            for(u64 f = start; f < end; f++){
                u8 cnt = cnt_arr[f];
                if(!cnt) continue;
                int n_act = 0;
                for(int c=0; c<B; c++){
                    int n1 = lookup_exact32((u32)f + T16_arr[c], cand1_b[c]);
                    if(n1){
                        n1_b[c] = n1;
                        active_c[n_act++] = c;
                    }
                }
                if(!n_act){
                    idx += cnt;
                    continue;
                }
                for(u8 j = 0; j < cnt; j++){
                    u32 a1 = A1_perm[idx++];
                    u32 e1 = a1 + C_e1;
                    u32 sm1 = S0(a1) + MAJ(a1, a0, am1);
                    u32 G1 = (u32)f + a1 + C_W0;

                    for(int ac=0; ac<n_act; ac++){
                        int c = active_c[ac];
                        int n1 = n1_b[c];
                        u32 T17 = T17_arr[c];
                        for(int x=0; x<n1; x++){
                            u32 W1 = cand1_b[c][x];
                            u32 a2 = W1 + G1;
                            u32 e2 = a2 - sm1 + am2;
                            u32 sm2 = S0(a2) + MAJ(a2, a1, a0);
                            u32 G2 = sm2 + S1(e2) + CH(e2, e1, e0) + C_G2;
                            u32 t17 = T17 + G2 - W1;

                            __builtin_prefetch(&H_exact32[t17], 0, 0);
                            q2[n_q2].t17    = t17;
                            q2[n_q2].G2     = G2;
                            q2[n_q2].a1     = a1;
                            q2[n_q2].e1     = e1;
                            q2[n_q2].a2     = a2;
                            q2[n_q2].e2     = e2;
                            q2[n_q2].sm2    = sm2;
                            q2[n_q2].ctx_id = (u32)c;
                            n_q2++;

                            if(n_q2 >= PF_BATCH){
                                drain_q2(q2, n_q2, q3, &n_q3, st, am1, C_G3, w_thresh,
                                         &local_l2, &local_l3, &local_sol19);
                                n_q2 = 0;
                            }
                        }
                    }
                }
            }
        }

        if(n_q2 > 0){
            drain_q2(q2, n_q2, q3, &n_q3, st, am1, C_G3, w_thresh,
                     &local_l2, &local_l3, &local_sol19);
            n_q2 = 0;
        }
        if(n_q3 > 0){
            drain_q3(q3, n_q3, st, &local_sol19);
            n_q3 = 0;
        }

        atomic_fetch_add_explicit(&g_chains_done, (end - start) * (u64)B, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_l2_nodes, local_l2, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_l3_lookups, local_l3, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_sol19_count, local_sol19, memory_order_relaxed);
    }
    return NULL;
}

int main(int argc, char **argv){
    // Usage: ./solve20_padded_fast [target_hex] [w_thresh] [nthreads] [table_gb] [batch_ctx] [max_sweeps]
    //    or: ./solve20_padded_fast [--target HEX] [--w-thresh W] [--threads N] [--table-gb G] [--batch-ctx B] [--max-sweeps M] [--f1-stream]
    const char *targ_hex = "0000000000000000000000000000000000000000000000000000000000000000";
    int w_thresh   = 14;
    int nthreads   = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if(nthreads < 1) nthreads = 8;
    int table_gb   = 6;
    int batch_ctx  = 8;
    int max_sweeps = 0; // 0 = unlimited

    if(argc > 1 && argv[1][0] != '-'){
        targ_hex   = argv[1];
        w_thresh   = (argc > 2) ? atoi(argv[2]) : 14;
        nthreads   = (argc > 3) ? atoi(argv[3]) : nthreads;
        table_gb   = (argc > 4) ? atoi(argv[4]) : 6;
        batch_ctx  = (argc > 5) ? atoi(argv[5]) : 2;
        max_sweeps = (argc > 6) ? atoi(argv[6]) : 0;
    } else {
        for(int i=1; i<argc; i++){
            if(!strcmp(argv[i], "--target") && i+1<argc) targ_hex = argv[++i];
            else if(!strcmp(argv[i], "--w-thresh") && i+1<argc) w_thresh = atoi(argv[++i]);
            else if(!strcmp(argv[i], "--threads") && i+1<argc) nthreads = atoi(argv[++i]);
            else if(!strcmp(argv[i], "--table-gb") && i+1<argc) table_gb = atoi(argv[++i]);
            else if(!strcmp(argv[i], "--batch-ctx") && i+1<argc) batch_ctx = atoi(argv[++i]);
            else if(!strcmp(argv[i], "--max-sweeps") && i+1<argc) max_sweeps = atoi(argv[++i]);
            else if(!strcmp(argv[i], "--f1-stream")) g_f1_stream = 1;
            else if(!strcmp(argv[i], "--l2-mode") && i+1<argc) g_l2_mode = atoi(argv[++i]);
            else if(!strcmp(argv[i], "--w-lo-max") && i+1<argc) g_w_lo_max = atoi(argv[++i]);
            else if(!strcmp(argv[i], "--sweep-shift") && i+1<argc){
                int sh = atoi(argv[++i]);
                if(sh < 1) sh = 1;
                if(sh > 32) sh = 32;
                g_sweep_limit = 1ULL << sh;
            }
            else if(!strcmp(argv[i], "--ctx-pool") && i+1<argc) g_ctx_init_pool = atoi(argv[++i]);
            else if(!strcmp(argv[i], "--pair-l1") && i+1<argc) g_pair_l1 = atoi(argv[++i]);
            else {
                fprintf(stderr, "Error: unknown or incomplete option '%s'\n", argv[i]);
                return 1;
            }
        }
    }

    if(nthreads < 1 || nthreads > 4096){
        fprintf(stderr, "Error: --threads must be in [1, 4096] (got %d).\n", nthreads);
        return 1;
    }
    if(w_thresh < 0 || w_thresh > 32){
        fprintf(stderr, "Error: --w-thresh must be in [0, 32] (got %d).\n", w_thresh);
        return 1;
    }
    if(table_gb != 6 && table_gb != 32){
        fprintf(stderr, "Error: --table-gb must be 6 or 32 (got %d).\n", table_gb);
        return 1;
    }
    if(batch_ctx < 1 || batch_ctx > MAX_BATCH_CTX){
        fprintf(stderr, "Error: --batch-ctx must be in [1, %d] (got %d).\n", MAX_BATCH_CTX, batch_ctx);
        return 1;
    }
    if(!targ_hex || strlen(targ_hex) != 64){
        fprintf(stderr, "Error: --target must be an exact 64-character hex string.\n");
        return 1;
    }

    u32 dig[8];
    for(int i=0;i<8;i++){
        char buf[9];
        memcpy(buf, targ_hex + 8*i, 8);
        buf[8] = 0;
        for(int j=0;j<8;j++){
            char c = buf[j];
            if(!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))){
                fprintf(stderr, "Error: --target contains non-hex character '%c'.\n", c);
                return 1;
            }
        }
        dig[i] = (u32)strtoul(buf, NULL, 16);
    }

    G_STATE.w_thresh  = w_thresh;
    G_STATE.batch_ctx = batch_ctx;
    init_iv_and_target(&G_STATE, dig);

    {
        u64 urand_seed = ((u64)getpid() << 32) ^ (u64)(now_sec() * 1e9);
        int ufd = open("/dev/urandom", O_RDONLY);
        if(ufd >= 0){
            (void)read(ufd, &urand_seed, sizeof(urand_seed));
            close(ufd);
        }
        atomic_store(&g_ctx_seed_gen, urand_seed);
        g_a12_perm_key = (u32)(urand_seed ^ (urand_seed >> 32)) | 1u;
    }

    double t0 = now_sec();
    init_q0_table(nthreads, table_gb);
    fprintf(stderr, "[init] %d GB Q0^-1 table ready in %.2f s\n", table_gb, now_sec() - t0);

    if(table_gb == 32 || g_f1_stream){
        init_f1_table(nthreads);
    }

    u64 total_sweeps = 0;
    u64 total_contexts = 0;
    u64 total_sol19 = 0;
    double t_start_all = now_sec();

    while(!atomic_load(&g_solved20)){
        total_sweeps++;
        double t_ctx0 = now_sec();
        u64 ctx_trials = generate_right_contexts_parallel(nthreads);
        double t_ctx = now_sec() - t_ctx0;
        total_contexts += (u64)batch_ctx;
        (void)total_contexts;

        if(batch_ctx >= 2){
            u32 dt16 = G_STATE.rctx[batch_ctx - 1].T16 - G_STATE.rctx[0].T16;
            fprintf(stderr, "[sweep #%llu] Selected %d T16-clustered contexts (pool=%d, +%d new in %.2fs) [T16_0=%08x T16_%d=%08x dT16=%u | a5=(%08x,%08x)]\n",
                    (unsigned long long)total_sweeps, batch_ctx,
                    g_ctx_pool_len, g_ctx_target_gen, t_ctx,
                    G_STATE.rctx[0].T16, batch_ctx - 1, G_STATE.rctx[batch_ctx - 1].T16, dt16,
                    G_STATE.rctx[0].a5, G_STATE.rctx[batch_ctx - 1].a5);
        } else {
            fprintf(stderr, "[sweep #%llu] Generated %d exact D=0 (a5==a6) padded contexts in %llu trials (%.2fs) [a5_0=%08x T16_0=%08x]\n",
                    (unsigned long long)total_sweeps, batch_ctx,
                    (unsigned long long)ctx_trials, t_ctx,
                    G_STATE.rctx[0].a5, G_STATE.rctx[0].T16);
        }

        atomic_store(&g_next_a1_chunk, 0);
        atomic_store(&g_chains_done, 0);
        atomic_store(&g_l2_nodes, 0);
        atomic_store(&g_l3_lookups, 0);
        atomic_store(&g_sol19_count, 0);

        double t_sw0 = now_sec();
        pthread_t *th = malloc(nthreads * sizeof(pthread_t));
        void *(*fn)(void *) = (table_gb == 32) ? worker_thread_sorted32 : worker_thread;
        for(int i=0;i<nthreads;i++) pthread_create(&th[i], NULL, fn, (void *)(intptr_t)i);
        for(int i=0;i<nthreads;i++) pthread_join(th[i], NULL);
        free(th);

        double t_sw = now_sec() - t_sw0;
        u64 sol19 = atomic_load(&g_sol19_count);
        u64 l2    = atomic_load(&g_l2_nodes);
        u64 l3    = atomic_load(&g_l3_lookups);
        total_sol19 += sol19;
        double elapsed_all = now_sec() - t_start_all;

        double scale = (double)(1ULL << 32) / (double)g_sweep_limit;
        fprintf(stderr, " -> Sweep #%llu (%d ctxs, l2m=%d, w=%d/%d, pair=%d) in %.2fs: sol19=%llu (%.1f/ctx, full_eq=%.0f/ctx) | L2=%llu L3=%llu | rate=%.1f sol19/s | avg rate=%.1f sol19/s (total sol19=%llu, %.1fs)\n",
                (unsigned long long)total_sweeps, batch_ctx, g_l2_mode, w_thresh, g_w_lo_max, g_pair_l1, t_sw,
                (unsigned long long)sol19, (double)sol19 / (double)batch_ctx,
                ((double)sol19 / (double)batch_ctx) * scale,
                (unsigned long long)l2, (unsigned long long)l3,
                (double)sol19 / t_sw,
                (double)total_sol19 / elapsed_all,
                (unsigned long long)total_sol19, elapsed_all);

        if(max_sweeps > 0 && (int)total_sweeps >= max_sweeps) break;
    }
    return 0;
}
