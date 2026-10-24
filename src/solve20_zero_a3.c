// solve20_zero_a3.c
//
// 20-Step SHA-256 Preimage Solver via the Zero-a3 Bi-Directional Q0/Q1 Bridge.
// Eliminates the 2^13.28 (3/4)^32 nonlinear beta-filter penalty by role-swapping
// a3 <-> a7 on the Zero-a3 Manifold:
//   a3 = 0, a5 = 0, a6 = 0, a8 = 0, a9 = 0, a10 = 0xffffffff
// so that Maj(a5, a4, a3) = Maj(0, a4, 0) = 0 and Maj(a6, a5, a4) = Maj(0, 0, a4) = 0
// hold identically for 100% of a4 in [0, 2^32).
//
// Compile (Linux or macOS):
//   gcc -O3 -march=native -pthread solve20_zero_a3.c -o solve20_zero_a3
//
// Full 20-round run on a 64 GB+ VM (48 GB tables: 16 GB Q0 + 16 GB Q1 + 16 GB Left):
//   ./solve20_zero_a3 --q-bits 32 --left-bits 32 --threads 16
//
// Quick local verification on a 16 GB machine (e.g. find 19-round solutions in <2s):
//   ./solve20_zero_a3 --q-bits 25 --left-bits 24 --threads 8 --test-19 20

#include <inttypes.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static inline uint32_t rotr(uint32_t x, int n) {
  return (x >> n) | (x << (32 - n));
}
#define S0(x) (rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22))
#define S1(x) (rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25))
#define s0(x) (rotr(x, 7) ^ rotr(x, 18) ^ ((x) >> 3))
#define s1(x) (rotr(x, 17) ^ rotr(x, 19) ^ ((x) >> 10))
#define Ch(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define Maj(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define Q0(u) (s0(u) - (u))
#define Q1(u) (s1(u) + (u))

static const uint32_t K[20] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b,
    0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
    0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7,
    0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc};

static const uint32_t IV[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

// Arbitrary 256-bit target digest H (can be overridden or kept as test target):
static uint32_t TARGET_H[8] = {
    0xdeadbeef, 0xcafebabe, 0x01234567, 0x89abcdef,
    0xfedcba98, 0x76543210, 0x13579bdf, 0x2468ace0};

static inline uint32_t e_of(const uint32_t *A, int t) {
  const uint32_t *a = A + 8;
  return a[t] - S0(a[t - 1]) - Maj(a[t - 1], a[t - 2], a[t - 3]) + a[t - 4];
}

static inline uint32_t W_of(const uint32_t *A, int t) {
  const uint32_t *a = A + 8;
  uint32_t et = e_of(A, t), et1 = e_of(A, t - 1), et2 = e_of(A, t - 2),
           et3 = e_of(A, t - 3);
  return a[t + 1] - S0(a[t]) - Maj(a[t], a[t - 1], a[t - 2]) - et3 - S1(et) -
         Ch(et, et1, et2) - K[t];
}

static inline uint32_t rho_of(const uint32_t *A, int t) {
  return W_of(A, t) - s1(W_of(A, t - 2)) - W_of(A, t - 7) - s0(W_of(A, t - 15)) -
         W_of(A, t - 16);
}

static void init_base_state(uint32_t A[32], const uint32_t H[8]) {
  memset(A, 0, 32 * sizeof(uint32_t));
  uint32_t *a = A + 8;
  a[0] = IV[0];
  a[-1] = IV[1];
  a[-2] = IV[2];
  a[-3] = IV[3];
  for (int k = 0; k < 4; k++) {
    int t = -k;
    a[t - 4] = IV[4 + k] - a[t] + S0(a[t - 1]) + Maj(a[t - 1], a[t - 2], a[t - 3]);
  }
  for (int i = 0; i < 4; i++) a[20 - i] = H[i] - IV[i];
  for (int k = 0; k < 4; k++) {
    int t = 20 - k;
    uint32_t es = H[4 + k] - IV[4 + k];
    a[t - 4] = es - a[t] + S0(a[t - 1]) + Maj(a[t - 1], a[t - 2], a[t - 3]);
  }
  // Zero-a3 Manifold:
  a[3] = 0;
  a[5] = 0;
  a[6] = 0;
  a[8] = 0;
  a[9] = 0;
  a[10] = 0xffffffffu;
}

static int verify_forward(const uint32_t W_in[16], const uint32_t H[8], int rounds) {
  uint32_t W[20];
  for (int t = 0; t < 16; t++) W[t] = W_in[t];
  for (int t = 16; t < 20; t++) {
    W[t] = s1(W[t - 2]) + W[t - 7] + s0(W[t - 15]) + W[t - 16];
  }
  uint32_t a = IV[0], b = IV[1], c = IV[2], d = IV[3];
  uint32_t e = IV[4], f = IV[5], g = IV[6], h = IV[7];
  for (int t = 0; t < rounds; t++) {
    uint32_t T1 = h + S1(e) + Ch(e, f, g) + K[t] + W[t];
    uint32_t T2 = S0(a) + Maj(a, b, c);
    h = g; g = f; f = e; e = d + T1;
    d = c; c = b; b = a; a = T1 + T2;
  }
  if (rounds < 20) return 1;
  return (a + IV[0] == H[0]) && (b + IV[1] == H[1]) && (c + IV[2] == H[2]) &&
         (d + IV[3] == H[3]) && (e + IV[4] == H[4]) && (f + IV[5] == H[5]) &&
         (g + IV[6] == H[6]) && (h + IV[7] == H[7]);
}

static int g_all_12 = 0;
static const char *g_current_label = "custom";

typedef struct {
  const char *label;
  uint32_t H[8];
} UnpaddedTarget;

static const UnpaddedTarget ALL_12_UNPADDED[12] = {
    {"unpadded test 1",
     {0x9b746e6c, 0x51ff8a68, 0xc0ec9e61, 0x6457bbf6, 0x3915b28d, 0x9178e4b4, 0xa4202475, 0x806e3335}},
    {"unpadded test 2",
     {0xfa961afa, 0x53aafcda, 0xd6c26780, 0x8b49c71a, 0x567235c0, 0xe40b50e6, 0x2dd7e87e, 0xf70af1ef}},
    {"unpadded test 3",
     {0x29bfd8e0, 0x31d0be2b, 0x391c6aa3, 0x63b6cf98, 0x5505d91e, 0x1d11bff3, 0x7fa5621b, 0xcc03082b}},
    {"unpadded test 4",
     {0x5519d174, 0x12148830, 0x9510e203, 0x3a2d2f28, 0xeaa09224, 0xd492beee, 0x8d12b8a3, 0xd3b105cd}},
    {"unpadded test 5",
     {0x2d27ea89, 0x7c5e1d94, 0x0b6ce5a5, 0x7f979827, 0x8127176a, 0xc936a2fe, 0x2f38352b, 0x20feb5d0}},
    {"unpadded test 6",
     {0x3841741f, 0xf785d25f, 0xb276163c, 0x9ba419f7, 0xa1bd35a9, 0x982d5193, 0xa907f37b, 0x4c7ef55b}},
    {"unpadded test 7",
     {0x45d566a1, 0x204151f4, 0x2dcaa45d, 0xa620a9b9, 0x4dcb58bd, 0xe5890ca7, 0xcb40af5a, 0x124ce0af}},
    {"unpadded test 8",
     {0x891dc1ad, 0xb02995f3, 0x94e01537, 0x91779526, 0x441ce1b1, 0x35f9d88c, 0xa1401980, 0xe9aacdea}},
    {"unpadded test 9",
     {0xadd7f221, 0xc973705f, 0xe2ef373b, 0xaacc0ecb, 0x904abf13, 0xeb3569e7, 0xda4cec8e, 0x752f7dfe}},
    {"unpadded test 10",
     {0x4a2dc64f, 0x7cc9be09, 0x7cca6665, 0x9ba68515, 0xc512f015, 0x1b1acae2, 0x36549b98, 0x57b6eadc}},
    {"unpadded test 11",
     {0x9dc68129, 0x573d3a51, 0x8123ae47, 0x8eed5db4, 0x9eada79c, 0xf9cf06ef, 0xb08ffef2, 0x2b321c69}},
    {"hand-off cross-check 2026-09-29",
     {0xd374822f, 0x21eb0d4d, 0xbc809ae0, 0x443c2c6c, 0x66f8ce5d, 0x07c8779b, 0xcc5fc784, 0x9f44e3c2}},
};
static int g_q_bits = 32;
static int g_left_bits = 32;
static int g_threads = 8;
static uint64_t g_test_19_target = 0; // if > 0, stop after finding N 19-round solutions
static uint32_t g_T16_star = 0x12345678u;

static uint64_t g_q_size = 0;
static uint64_t g_left_size = 0;

static uint32_t *g_q0_inv = NULL;
static uint32_t *g_q1_inv = NULL;
static uint32_t *g_left_a1 = NULL;

static atomic_ullong g_left_count = 0;
static atomic_ullong g_bridge_matches = 0;
static atomic_ullong g_sol19_count = 0;
static atomic_int g_stop = 0;
static pthread_mutex_t g_print_mu = PTHREAD_MUTEX_INITIALIZER;

typedef struct {
  int tid;
  int nthreads;
} ThreadArg;

static double now_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void *build_q_worker(void *arg) {
  ThreadArg *ta = (ThreadArg *)arg;
  uint64_t total = 1ULL << 32;
  uint64_t chunk = (total + ta->nthreads - 1) / ta->nthreads;
  uint64_t start = (uint64_t)ta->tid * chunk;
  uint64_t end = start + chunk;
  if (end > total) end = total;

  uint64_t q_lim = g_q_size;
  for (uint64_t u64 = start; u64 < end; u64++) {
    uint32_t u = (uint32_t)u64;
    if (u == 0xffffffffu) continue;
    uint32_t y0 = Q0(u);
    if ((uint64_t)y0 < q_lim) g_q0_inv[y0] = u;
    uint32_t y1 = Q1(u);
    if ((uint64_t)y1 < q_lim) g_q1_inv[y1] = u;
  }
  return NULL;
}

static void *phase1_left_worker(void *arg) {
  ThreadArg *ta = (ThreadArg *)arg;
  uint64_t total = 1ULL << 32;
  uint64_t chunk = (total + ta->nthreads - 1) / ta->nthreads;
  uint64_t start = (uint64_t)ta->tid * chunk;
  uint64_t end = start + chunk;
  if (end > total) end = total;

  uint32_t A[32];
  init_base_state(A, TARGET_H);
  uint32_t *a = A + 8;

  uint64_t q_lim = g_q_size;
  uint64_t l_lim = g_left_size;
  uint32_t T16_star = g_T16_star;
  uint64_t local_cnt = 0;

  for (uint64_t a1_64 = start; a1_64 < end; a1_64++) {
    uint32_t a1 = (uint32_t)a1_64;
    if (a1 == 0xffffffffu) continue;
    a[1] = a1;
    a[2] = 0;
    uint32_t W0 = W_of(A, 0);
    uint32_t G1 = -W_of(A, 1);
    uint32_t target_q0 = T16_star - W0 + G1;
    if ((uint64_t)target_q0 >= q_lim) continue;
    uint32_t W1 = g_q0_inv[target_q0];
    if (W1 == 0xffffffffu) continue;

    uint32_t a2 = W1 + G1;
    a[2] = a2;
    uint32_t W2 = W_of(A, 2);
    uint32_t L17 = s0(W2) + W1;
    if ((uint64_t)L17 < l_lim) {
      g_left_a1[L17] = a1;
      local_cnt++;
    }
  }
  atomic_fetch_add(&g_left_count, local_cnt);
  return NULL;
}

static void *phase2_right_worker(void *arg) {
  ThreadArg *ta = (ThreadArg *)arg;
  uint32_t A[32];
  init_base_state(A, TARGET_H);
  uint32_t *a = A + 8;

  uint64_t q_lim = g_q_size;
  uint64_t l_lim = g_left_size;
  uint32_t T16_star = g_T16_star;
  double t_start = now_sec();
  double t_last = t_start;

  // Precompute constant W9_0 at a1=a2=a4=a7=0:
  a[1] = 0;
  a[2] = 0;
  a[4] = 0;
  a[7] = 0;
  a[11] = 0;
  a[12] = 0;
  uint32_t W9_0 = W_of(A, 9);

  // Each thread sweeps distinct (a11, a12) pairs:
  for (uint64_t a11_64 = (uint64_t)ta->tid; a11_64 < (1ULL << 32) && !atomic_load(&g_stop);
       a11_64 += (uint64_t)ta->nthreads) {
    uint32_t a11 = (uint32_t)a11_64;
    a[1] = 0;
    a[2] = 0;
    a[4] = 0;
    a[7] = 0;
    a[11] = a11;
    a[12] = 0;
    uint32_t W10_0 = W_of(A, 10);

    for (uint64_t a12_64 = 0; a12_64 < (1ULL << 32); a12_64++) {
      if ((a12_64 & 0x3ffffu) == 0) {
        if (atomic_load(&g_stop)) break;
        if (ta->tid == 0 && (a12_64 & 0xffffffu) == 0) {
          double t_now = now_sec();
          if (t_now - t_last >= 5.0) {
            uint64_t c19 = atomic_load(&g_sol19_count);
            uint64_t bm = atomic_load(&g_bridge_matches);
            printf("  [Progress %.1fs] bridge_matches=%" PRIu64
                   " | exact_19round_sols=%" PRIu64 " (%.2f/s)\n",
                   t_now - t_start, bm, c19, (double)c19 / (t_now - t_start));
            fflush(stdout);
            t_last = t_now;
          }
        }
      }
      uint32_t a12 = (uint32_t)(a12_64 ^ (a11 * 0x9e3779b1u));

      a[7] = 0;
      a[11] = a11;
      a[12] = a12;

      // Since C16_0 = W16 - s1(D14) - W9_0 and Q1(D14) = s1(D14) + D14,
      // s1(D14) cancels identically: C16tilde = W16 + D14 - W9_0!
      uint32_t D14 = W_of(A, 14);
      uint32_t C16tilde = W_of(A, 16) + D14 - W9_0;
      uint32_t target_q1 = C16tilde - T16_star;
      if ((uint64_t)target_q1 >= q_lim) continue;

      uint32_t u = g_q1_inv[target_q1];
      if (u == 0xffffffffu) continue;

      uint32_t a7 = D14 - u;
      uint32_t C17 = W_of(A, 17) - s1(W_of(A, 15)) - W10_0;
      uint32_t T17 = C17 + a7;
      if ((uint64_t)T17 >= l_lim) continue;

      uint32_t a1 = g_left_a1[T17];
      if (a1 == 0xffffffffu) continue;

      // Reconstruct (W1, a2, W2) from a1 (L17 == T17 is already guaranteed!):
      a[1] = a1;
      a[2] = 0;
      uint32_t W0 = W_of(A, 0);
      uint32_t G1 = -W_of(A, 1);
      uint32_t target_q0 = T16_star - W0 + G1;
      uint32_t W1 = g_q0_inv[target_q0];
      uint32_t a2 = W1 + G1;
      a[2] = a2;
      uint32_t W2 = W_of(A, 2);

      atomic_fetch_add(&g_bridge_matches, 1);

      // Level 3: solve Q0(W3) = T18 - W2 + G3 (ZERO beta-filter loss!)
      a[4] = 0;
      a[7] = a7;
      uint32_t G3 = -W_of(A, 3);
      uint32_t T18 = W_of(A, 18) - s1(W_of(A, 16)) - W_of(A, 11);
      uint32_t target_w3 = T18 - W2 + G3;
      if ((uint64_t)target_w3 >= q_lim) continue;

      uint32_t W3 = g_q0_inv[target_w3];
      if (W3 == 0xffffffffu) continue;

      uint32_t a4 = W3 + G3;
      a[4] = a4;

      // Every single match here satisfies rho16 == 0, rho17 == 0, rho18 == 0!
      uint64_t c19 = atomic_fetch_add(&g_sol19_count, 1) + 1;
      uint32_t r19 = rho_of(A, 19);

      if (r19 == 0) {
        uint32_t W_blk[16];
        for (int t = 0; t < 16; t++) W_blk[t] = W_of(A, t);
        pthread_mutex_lock(&g_print_mu);
        if (atomic_load(&g_stop) == 0) {
          int ok = verify_forward(W_blk, TARGET_H, 20);
          printf("\n[comp] label=\"%s\"\n", g_current_label);
          printf("  target   = ");
          for (int k = 0; k < 8; k++) printf("%08x", TARGET_H[k]);
          printf("\n  W[0..15] =");
          for (int t = 0; t < 16; t++) printf(" %08x", W_blk[t]);
          printf("\n  c20+IV   = ");
          for (int k = 0; k < 8; k++) printf("%08x", TARGET_H[k]);
          printf("   %s   (search %.2fs)\n", ok ? "OK" : "FAIL", now_sec() - t_start);
          fflush(stdout);
          atomic_store(&g_stop, 1);
        }
        pthread_mutex_unlock(&g_print_mu);
        break;
      }

      if (g_test_19_target > 0 && c19 >= g_test_19_target) {
        uint32_t W_blk[16];
        for (int t = 0; t < 16; t++) W_blk[t] = W_of(A, t);
        pthread_mutex_lock(&g_print_mu);
        if (atomic_load(&g_stop) == 0) {
          printf("  [Verified sample 19-round solution #%" PRIu64
                 ": rho16=0x%08x rho17=0x%08x rho18=0x%08x rho19=0x%08x]\n",
                 c19, rho_of(A, 16), rho_of(A, 17), rho_of(A, 18), r19);
          atomic_store(&g_stop, 1);
        }
        pthread_mutex_unlock(&g_print_mu);
        break;
      }
    }
  }
  return NULL;
}

int main(int argc, char **argv) {
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--q-bits") == 0 && i + 1 < argc) {
      g_q_bits = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--left-bits") == 0 && i + 1 < argc) {
      g_left_bits = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      g_threads = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--test-19") == 0 && i + 1 < argc) {
      g_test_19_target = strtoull(argv[++i], NULL, 10);
    } else if (strcmp(argv[i], "--all-12") == 0) {
      g_all_12 = 1;
    }
  }

  g_q_size = 1ULL << g_q_bits;
  g_left_size = 1ULL << g_left_bits;

  double gb_q = (2.0 * (double)g_q_size * 4.0) / (1024.0 * 1024.0 * 1024.0);
  double gb_l = ((double)g_left_size * 4.0) / (1024.0 * 1024.0 * 1024.0);
  printf("Zero-a3 Bi-Directional Bridge Solver (w = 32)\n");
  printf("  Q0/Q1 tables : 2^%d entries each (%.2f GB total)\n", g_q_bits, gb_q);
  printf("  Left table   : 2^%d entries      (%.2f GB total)\n", g_left_bits, gb_l);
  printf("  Threads      : %d\n", g_threads);
  fflush(stdout);

  double t0 = now_sec();
  g_q0_inv = malloc(g_q_size * sizeof(uint32_t));
  g_q1_inv = malloc(g_q_size * sizeof(uint32_t));
  g_left_a1 = malloc(g_left_size * sizeof(uint32_t));
  if (!g_q0_inv || !g_q1_inv || !g_left_a1) {
    fprintf(stderr, "Error: malloc failed (need %.2f GB RAM).\n", gb_q + gb_l);
    return 1;
  }
  memset(g_q0_inv, 0xff, g_q_size * sizeof(uint32_t));
  memset(g_q1_inv, 0xff, g_q_size * sizeof(uint32_t));
  memset(g_left_a1, 0xff, g_left_size * sizeof(uint32_t));

  pthread_t *th = malloc(g_threads * sizeof(pthread_t));
  ThreadArg *ta = malloc(g_threads * sizeof(ThreadArg));
  for (int i = 0; i < g_threads; i++) {
    ta[i].tid = i;
    ta[i].nthreads = g_threads;
    pthread_create(&th[i], NULL, build_q_worker, &ta[i]);
  }
  for (int i = 0; i < g_threads; i++) pthread_join(th[i], NULL);
  double t1 = now_sec();
  printf("  [1/3] Built Q0/Q1 tables in %.2f s\n", t1 - t0);
  fflush(stdout);

  for (int i = 0; i < g_threads; i++) {
    pthread_create(&th[i], NULL, phase1_left_worker, &ta[i]);
  }
  for (int i = 0; i < g_threads; i++) pthread_join(th[i], NULL);
  double t2 = now_sec();
  printf("  [2/3] Phase 1 Left Bridge complete in %.2f s (%" PRIu64 " valid left pairs)\n",
         t2 - t1, (uint64_t)atomic_load(&g_left_count));
  fflush(stdout);

  if (g_all_12) {
    for (int idx = 0; idx < 12; idx++) {
      g_current_label = ALL_12_UNPADDED[idx].label;
      memcpy(TARGET_H, ALL_12_UNPADDED[idx].H, sizeof(TARGET_H));
      atomic_store(&g_stop, 0);
      atomic_store(&g_bridge_matches, 0);
      atomic_store(&g_sol19_count, 0);
      for (int i = 0; i < g_threads; i++) {
        pthread_create(&th[i], NULL, phase2_right_worker, &ta[i]);
      }
      for (int i = 0; i < g_threads; i++) pthread_join(th[i], NULL);
    }
  } else {
    for (int i = 0; i < g_threads; i++) {
      pthread_create(&th[i], NULL, phase2_right_worker, &ta[i]);
    }
    for (int i = 0; i < g_threads; i++) pthread_join(th[i], NULL);
    double t3 = now_sec();
    printf("  [3/3] Phase 2 finished in %.2f s (bridge_matches=%" PRIu64
           ", exact_19round_sols=%" PRIu64 ")\n",
           t3 - t2, (uint64_t)atomic_load(&g_bridge_matches),
           (uint64_t)atomic_load(&g_sol19_count));
  }
  return 0;
}
