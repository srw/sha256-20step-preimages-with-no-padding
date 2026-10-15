import random
import time


def u32(x):
  return x & 0xFFFFFFFF


def rotr(x, n):
  x &= 0xFFFFFFFF
  return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF


def shr(x, n):
  return (x & 0xFFFFFFFF) >> n


def Sigma0(x):
  return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)


def Sigma1(x):
  return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)


def sigma0(x):
  return rotr(x, 7) ^ rotr(x, 18) ^ shr(x, 3)


def sigma1(x):
  return rotr(x, 17) ^ rotr(x, 19) ^ shr(x, 10)


def Ch(x, y, z):
  return ((x & y) ^ ((~x) & z)) & 0xFFFFFFFF


def Maj(x, y, z):
  return ((x & y) ^ (x & z) ^ (y & z)) & 0xFFFFFFFF


def Q0(x):
  return (sigma0(x) - x) & 0xFFFFFFFF


def Q1(x):
  return (sigma1(x) + x) & 0xFFFFFFFF


# Precompute GF(2) inverse of sigma0 (rank 32)
S0_cols = [sigma0(1 << i) for i in range(32)]
aug_s0 = [(S0_cols[i], 1 << i) for i in range(32)]
piv_s0 = {}
for col_val, orig_mask in aug_s0:
  v, m = col_val, orig_mask
  for b in sorted(piv_s0.keys(), reverse=True):
    if (v >> b) & 1:
      v ^= piv_s0[b][0]
      m ^= piv_s0[b][1]
  msb = v.bit_length() - 1
  for b in list(piv_s0.keys()):
    if (piv_s0[b][0] >> msb) & 1:
      piv_s0[b] = (piv_s0[b][0] ^ v, piv_s0[b][1] ^ m)
  piv_s0[msb] = (v, m)

S0_inv_low = [0] * 65536
S0_inv_high = [0] * 65536
for v in range(65536):
  rL, rH = 0, 0
  for b in range(16):
    if (v >> b) & 1:
      rL ^= piv_s0[b][1]
      rH ^= piv_s0[16 + b][1]
  S0_inv_low[v] = rL
  S0_inv_high[v] = rH


def sigma0_inv(y):
  return S0_inv_low[y & 0xFFFF] ^ S0_inv_high[(y >> 16) & 0xFFFF]


# Precompute GF(2) pseudoinverse for L(x) = sigma0(x) ^ x (rank 31)
L_cols = [sigma0(1 << i) ^ (1 << i) for i in range(32)]
aug = [(L_cols[i], 1 << i) for i in range(32)]
pivots = {}
k0 = 0
for col_val, orig_mask in aug:
  v, m = col_val, orig_mask
  for b in sorted(pivots.keys(), reverse=True):
    if (v >> b) & 1:
      v ^= pivots[b][0]
      m ^= pivots[b][1]
  if v == 0:
    k0 = m
  else:
    msb = v.bit_length() - 1
    for b in list(pivots.keys()):
      if (pivots[b][0] >> msb) & 1:
        pivots[b] = (pivots[b][0] ^ v, pivots[b][1] ^ m)
    pivots[msb] = (v, m)

P_cols = [0] * 32
for b, (v, m) in pivots.items():
  P_cols[b] = m

parity_mask = 1 << 2
for b, (v, m) in pivots.items():
  if (v >> 2) & 1:
    parity_mask |= 1 << b

P_low = [0] * 65536
P_high = [0] * 65536
par_low = [0] * 65536
par_high = [0] * 65536
for v in range(65536):
  resL, resH = 0, 0
  for b in range(16):
    if (v >> b) & 1:
      resL ^= P_cols[b]
      resH ^= P_cols[16 + b]
  P_low[v] = resL
  P_high[v] = resH
  par_low[v] = (v & (parity_mask & 0xFFFF)).bit_count() & 1
  par_high[v] = (v & (parity_mask >> 16)).bit_count() & 1


def get_16bit_leaves(y16, c0):
  leaves = [(c0, c0, 0, 0)]
  for i in range(16):
    yi = (y16 >> i) & 1
    next_bit = 1 << (i + 1)
    x_bit = 1 << i
    new_leaves = []
    for c_curr, c_vec, x_mask, x_val in leaves:
      if c_curr == yi:
        if yi:
          new_leaves.append((1, c_vec | next_bit, x_mask, x_val))
        else:
          new_leaves.append((0, c_vec, x_mask, x_val))
      else:
        new_leaves.append((0, c_vec, x_mask | x_bit, x_val))
        new_leaves.append((1, c_vec | next_bit, x_mask | x_bit, x_val | x_bit))
    leaves = new_leaves
  return leaves


def solve_Q0_inv_fast(y):
  yL = y & 0xFFFF
  yH = (y >> 16) & 0xFFFF
  left_leaves = get_16bit_leaves(yL, 0)
  sols = []
  for c16_target in (0, 1):
    left_sub = [lf for lf in left_leaves if lf[0] == c16_target]
    if not left_sub:
      continue
    right_leaves = get_16bit_leaves(yH, c16_target)
    buckets = [[[] for _ in range(256)], [[] for _ in range(256)]]
    for _, cH_full, mH, vH in right_leaves:
      cH = cH_full & 0xFFFF
      idxH = yH ^ cH
      pH = par_high[idxH]
      uH0 = P_high[idxH]
      uH1 = uH0 ^ k0
      buckets[pH][uH0 & 0xFF].append((uH0, mH << 16, vH << 16))
      buckets[pH][uH1 & 0xFF].append((uH1, mH << 16, vH << 16))
    for _, cL_full, mL, vL in left_sub:
      cL = cL_full & 0xFFFF
      idxL = yL ^ cL
      pL = par_low[idxL]
      uL = P_low[idxL]
      mL8 = mL & 0xFF
      base8 = (uL ^ vL) & mL8
      free8 = (~mL8) & 0xFF
      sub = 0
      b_pL = buckets[pL]
      while True:
        for uH, mH32, vH32 in b_pL[base8 | sub]:
          x = uL ^ uH
          if (x & mL) == vL and (x & mH32) == vH32:
            if Q0(x) == y:
              sols.append(x)
        if sub == free8:
          break
        sub = (sub - free8) & free8
  return sols


# Precompute GF(2) inverse/pseudoinverse for L1(x) = sigma1(x) ^ x
L1_cols = [sigma1(1 << i) ^ (1 << i) for i in range(32)]
aug1 = [(L1_cols[i], 1 << i) for i in range(32)]
pivots1 = {}
k1_ker = 0
for col_val, orig_mask in aug1:
  v, m = col_val, orig_mask
  for b in sorted(pivots1.keys(), reverse=True):
    if (v >> b) & 1:
      v ^= pivots1[b][0]
      m ^= pivots1[b][1]
  if v == 0:
    k1_ker = m
  else:
    msb = v.bit_length() - 1
    for b in list(pivots1.keys()):
      if (pivots1[b][0] >> msb) & 1:
        pivots1[b] = (pivots1[b][0] ^ v, pivots1[b][1] ^ m)
    pivots1[msb] = (v, m)

P1_cols = [0] * 32
for b, (v, m) in pivots1.items():
  P1_cols[b] = m

missing_b1 = [b for b in range(32) if b not in pivots1]
parity_mask1 = 0
if missing_b1:
  mb1 = missing_b1[0]
  parity_mask1 = 1 << mb1
  for b, (v, m) in pivots1.items():
    if (v >> mb1) & 1:
      parity_mask1 |= 1 << b

P1_low = [0] * 65536
P1_high = [0] * 65536
par1_low = [0] * 65536
par1_high = [0] * 65536
for v in range(65536):
  resL, resH = 0, 0
  for b in range(16):
    if (v >> b) & 1:
      resL ^= P1_cols[b]
      resH ^= P1_cols[16 + b]
  P1_low[v] = resL
  P1_high[v] = resH
  par1_low[v] = (v & (parity_mask1 & 0xFFFF)).bit_count() & 1
  par1_high[v] = (v & (parity_mask1 >> 16)).bit_count() & 1


def solve_Q1_inv_fast(y):
  yL = y & 0xFFFF
  yH = (y >> 16) & 0xFFFF
  # For addition (sigma1(x) + x = y), carry tree is identical to subtraction with ~y
  left_leaves = get_16bit_leaves((~yL) & 0xFFFF, 0)
  sols = []
  for c16_target in (0, 1):
    left_sub = [lf for lf in left_leaves if lf[0] == c16_target]
    if not left_sub:
      continue
    right_leaves = get_16bit_leaves((~yH) & 0xFFFF, c16_target)
    buckets = [[[] for _ in range(256)], [[] for _ in range(256)]]
    for _, cH_full, mH, vH in right_leaves:
      cH = cH_full & 0xFFFF
      idxH = yH ^ cH
      pH = par1_high[idxH]
      uH0 = P1_high[idxH]
      buckets[pH][uH0 & 0xFF].append((uH0, mH << 16, vH << 16))
      if k1_ker:
        uH1 = uH0 ^ k1_ker
        buckets[pH][uH1 & 0xFF].append((uH1, mH << 16, vH << 16))
    for _, cL_full, mL, vL in left_sub:
      cL = cL_full & 0xFFFF
      idxL = yL ^ cL
      pL = par1_low[idxL]
      uL = P1_low[idxL]
      mL8 = mL & 0xFF
      base8 = (uL ^ vL) & mL8
      free8 = (~mL8) & 0xFF
      sub = 0
      b_pL = buckets[pL]
      while True:
        for uH, mH32, vH32 in b_pL[base8 | sub]:
          x = uL ^ uH
          if (x & mL) == vL and (x & mH32) == vH32:
            if Q1(x) == y:
              sols.append(x)
        if sub == free8:
          break
        sub = (sub - free8) & free8
  return sols


K = [
    0x428A2F98,
    0x71374491,
    0xB5C0FBCF,
    0xE9B5DBA5,
    0x3956C25B,
    0x59F111F1,
    0x923F82A4,
    0xAB1C5ED5,
    0xD807AA98,
    0x12835B01,
    0x243185BE,
    0x550C7DC3,
    0x72BE5D74,
    0x80DEB1FE,
    0x9BDC06A7,
    0xC19BF174,
    0xE49B69C1,
    0xEFBE4786,
    0x0FC19DC6,
    0x240CA1CC,
]
IV = [
    0x6A09E667,
    0xBB67AE85,
    0x3C6EF372,
    0xA54FF53A,
    0x510E527F,
    0x9B05688C,
    0x1F83D9AB,
    0x5BE0CD19,
]
TARGET_H = [
    0x11111111,
    0x22222222,
    0x33333333,
    0x44444444,
    0x55555555,
    0x66666666,
    0x77777777,
    0x88888888,
]


def get_a_from_idx(idx, a_dict):
  if idx >= 1:
    return a_dict[idx]
  return IV[0 - idx]


def get_e_from_idx(idx, e_dict):
  if idx >= 1:
    return e_dict[idx]
  return IV[4 - idx]


def compute_all(a_dict):
  e_dict = {}
  for i in range(1, 17):
    a_im1 = get_a_from_idx(i - 1, a_dict)
    a_im2 = get_a_from_idx(i - 2, a_dict)
    a_im3 = get_a_from_idx(i - 3, a_dict)
    a_im4 = get_a_from_idx(i - 4, a_dict)
    T_i = u32(a_dict[i] - Sigma0(a_im1) - Maj(a_im1, a_im2, a_im3))
    e_dict[i] = u32(a_im4 + T_i)
  for j in range(17, 21):
    e_dict[j] = TARGET_H[4 + (j - 17)]
  W = {}
  for t in range(0, 20):
    e_tp1 = get_e_from_idx(t + 1, e_dict)
    e_t = get_e_from_idx(t, e_dict)
    e_tm1 = get_e_from_idx(t - 1, e_dict)
    e_tm2 = get_e_from_idx(t - 2, e_dict)
    e_tm3 = get_e_from_idx(t - 3, e_dict)
    a_tm3 = get_a_from_idx(t - 3, a_dict)
    W[t] = u32(e_tp1 - e_tm3 - Sigma1(e_t) - Ch(e_t, e_tm1, e_tm2) - K[t] - a_tm3)
  rho = {}
  for t in range(16, 20):
    rho[t] = u32(W[t] - sigma1(W[t - 2]) - W[t - 7] - sigma0(W[t - 15]) - W[t - 16])
  return e_dict, W, rho


def build_state(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12):
  d = {
      1: a1,
      2: a2,
      3: a3,
      4: a4,
      5: a5,
      6: a6,
      7: a7,
      8: a8,
      9: a9,
      10: a10,
      11: a11,
      12: a12,
  }
  for j in range(13, 21):
    d[j] = TARGET_H[j - 13]
  return d


def solve_right_e10_e11(a6, a7, a8, a9, a12, e10_target=0, e11_target=0xFFFFFFFF):
  T10 = u32(e10_target - a6)
  a10 = u32(T10 + Sigma0(a9) + Maj(a9, a8, a7))
  T11 = u32(e11_target - a7)
  a11 = u32(T11 + Sigma0(a10) + Maj(a10, a9, a8))
  return a10, a11


def demo_1_q0_inv_and_sigma0_inv():
  print(
      "=== DEMO 1: Exact O(1)-Amortized Solvers for sigma0^{-1}, Q0^{-1}, and"
      " Q1^{-1} ==="
  )
  rng = random.Random(101)
  t0 = time.time()
  for _ in range(100):
    x = rng.getrandbits(32)
    assert sigma0_inv(sigma0(x)) == x
    y0 = Q0(x)
    assert x in solve_Q0_inv_fast(y0)
    y1 = Q1(x)
    assert x in solve_Q1_inv_fast(y1)
  elapsed = time.time() - t0
  print(
      "  100/100 random 32-bit inversions of sigma0, Q0, and Q1 verified in"
      f" {elapsed:.3f}s!\n"
  )


def demo_2_algebraic_decoupling_and_cascades():
  print(
      "=== DEMO 2: Exact 3-Equation Decoupling & Instantaneous O(1) Cascades"
      " ==="
  )
  rng = random.Random(202)

  # 1. Verify exact decoupling on a6=a7=0, e10=0, e11=0xffffffff across 100 random states
  for _ in range(100):
    a6, a7 = 0, 0
    a8, a9, a12 = (rng.getrandbits(32) for _ in range(3))
    a10, a11 = solve_right_e10_e11(a6, a7, a8, a9, a12)
    d_base = build_state(0, 0, 0, 0, 0, a6, a7, a8, a9, a10, a11, a12)
    _, W_b, _ = compute_all(d_base)
    C17_b = u32(W_b[17] - sigma1(W_b[15]) - W_b[10])
    C18_b = u32(W_b[18] - sigma1(W_b[16]) - W_b[11])
    C19_b = u32(W_b[19] - sigma1(W_b[17]) - W_b[12])

    a1, a2, a3, a4, a5 = (rng.getrandbits(32) for _ in range(5))
    d_full = build_state(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12)
    _, W_f, rhos_f = compute_all(d_full)
    Phi17 = u32(-sigma0(W_f[2]) - W_f[1] + a3)
    Phi18 = u32(-sigma0(W_f[3]) - W_f[2] + a4)
    Phi19 = u32(-sigma0(W_f[4]) - W_f[3] + a5)
    delta17 = a4 & (~a5 & 0xFFFFFFFF)
    assert rhos_f[17] == u32(Phi17 + C17_b + delta17)
    assert rhos_f[18] == u32(Phi18 + C18_b)
    assert rhos_f[19] == u32(Phi19 + C19_b)
  print(
      "  [Verified 100/100] Exact 3-Equation Decoupling on (a6=a7=0, e10=0,"
      " e11=0xffffffff)!"
  )

  # 2. Cascade A: Instantaneous (rho_17 = 0, rho_18 = 0) via a5 = 0xffffffff
  # Since Delta_17(a4, 0xffffffff) = a4 & ~0xffffffff = 0 for ALL a4,
  # rho_17 and rho_18 are strictly triangular in (a3, a4) and solved in 2 calls to Q0^-1!
  t0 = time.time()
  a6, a7 = 0, 0
  a8, a9, a12 = rng.getrandbits(32), rng.getrandbits(32), rng.getrandbits(32)
  a10, a11 = solve_right_e10_e11(a6, a7, a8, a9, a12)
  a5 = 0xFFFFFFFF

  d_right = build_state(0, 0, 0, 0, a5, a6, a7, a8, a9, a10, a11, a12)
  _, W_r, _ = compute_all(d_right)
  C17 = u32(W_r[17] - sigma1(W_r[15]) - W_r[10])
  C18 = u32(W_r[18] - sigma1(W_r[16]) - W_r[11])

  attempts_A = 0
  sol_A = None
  while sol_A is None:
    attempts_A += 1
    a1, a2 = rng.getrandbits(32), rng.getrandbits(32)
    d0 = build_state(a1, a2, 0, 0, a5, a6, a7, a8, a9, a10, a11, a12)
    _, W0, _ = compute_all(d0)
    W1_val = W0[1]
    G2_val = W0[2]  # since a3 = 0
    rhs17 = u32(C17 - W1_val - G2_val)
    for W2_cand in solve_Q0_inv_fast(rhs17):
      a3 = u32(W2_cand - G2_val)
      d1 = build_state(a1, a2, a3, 0, a5, a6, a7, a8, a9, a10, a11, a12)
      _, W1_arr, _ = compute_all(d1)
      G3_val = W1_arr[3]  # since a4 = 0
      rhs18 = u32(C18 - W2_cand - G3_val)
      w3_sols = solve_Q0_inv_fast(rhs18)
      if w3_sols:
        a4 = u32(w3_sols[0] - G3_val)
        sol_A = (a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12)
        break
  dt_A = (time.time() - t0) * 1000.0
  d_solA = build_state(*sol_A)
  _, _, rhos_A = compute_all(d_solA)
  assert rhos_A[17] == 0 and rhos_A[18] == 0
  print(
      f"  [Cascade A: (rho_17=0, rho_18=0) via a5=0xffffffff] Solved in"
      f" {dt_A:.2f} ms ({attempts_A} draws)"
  )
  print(
      f"    Verified residuals: rho_17 = 0x{rhos_A[17]:08x}, rho_18 ="
      f" 0x{rhos_A[18]:08x}"
  )

  # 3. Cascade B: Instantaneous (rho_18 = 0, rho_19 = 0) via +1-monic linearity in a8 (at a9 = 0)
  # followed by a single Q0^-1 inversion for a5!
  t1 = time.time()
  a9_B = 0
  a10_B, a11_B = solve_right_e10_e11(a6, a7, 0, a9_B, a12)
  d_baseB = build_state(0, 0, 0, 0, 0, a6, a7, 0, a9_B, a10_B, a11_B, a12)
  _, W_baseB, _ = compute_all(d_baseB)
  C18_tilde = u32(W_baseB[18] - sigma1(W_baseB[16]) - W_baseB[11])

  attempts_B = 0
  sol_B = None
  while sol_B is None:
    attempts_B += 1
    a1, a2, a3, a4 = (rng.getrandbits(32) for _ in range(4))
    d_left = build_state(a1, a2, a3, a4, 0, a6, a7, 0, a9_B, a10_B, a11_B, a12)
    _, W_left, _ = compute_all(d_left)
    Phi_18 = u32(-sigma0(W_left[3]) - W_left[2] + a4)
    # Exact O(1) monic solve for a8 from rho_18 = 0:
    a8_B = u32(-Phi_18 - C18_tilde)
    # Now that a8_B is known, compute C19(a8_B, 0, a12) and solve for a5 via Q0^-1:
    d_mid = build_state(
        a1, a2, a3, a4, 0, a6, a7, a8_B, a9_B, a10_B, a11_B, a12
    )
    _, W_mid, _ = compute_all(d_mid)
    C19_B = u32(W_mid[19] - sigma1(W_mid[17]) - W_mid[12])
    G4_val = W_mid[4]  # since a5 = 0
    rhs19 = u32(C19_B - W_mid[3] - G4_val)
    w4_sols = solve_Q0_inv_fast(rhs19)
    if w4_sols:
      a5_B = u32(w4_sols[0] - G4_val)
      sol_B = (a1, a2, a3, a4, a5_B, a6, a7, a8_B, a9_B, a10_B, a11_B, a12)
      break
  dt_B = (time.time() - t1) * 1000.0
  d_solB = build_state(*sol_B)
  _, _, rhos_B = compute_all(d_solB)
  assert rhos_B[18] == 0 and rhos_B[19] == 0
  print(
      f"  [Cascade B: (rho_18=0, rho_19=0) via monic a8 + Q0^-1] Solved in"
      f" {dt_B:.2f} ms ({attempts_B} draws)"
  )
  print(
      f"    Verified residuals: rho_18 = 0x{rhos_B[18]:08x}, rho_19 ="
      f" 0x{rhos_B[19]:08x}\n"
  )


def demo_3_bi_directional_a5_isolated_solver():
  print(
      "=== DEMO 3: Bi-Directional Q0^{-1}/Q1^{-1} MITM & a5-Isolated Cascade"
      " ==="
  )
  rng = random.Random(303)
  a4, a6, a7, a9 = 0, 0, 0, 0

  # 1. Verify 100/100 that setting a4 = 0 eliminates Delta_17(0, a5) = (0 & ~a5) = 0
  # identically for ALL a5, and C17(a8, 0, a12) = C17_tilde(a12) - Q1(W15(a8, a12))
  for _ in range(100):
    a1, a2, a3, a8, a12 = (rng.getrandbits(32) for _ in range(5))
    a10, a11 = solve_right_e10_e11(a6, a7, a8, a9, a12)
    d0 = build_state(a1, a2, a3, a4, 0, a6, a7, a8, a9, a10, a11, a12)
    _, W0, rhos0 = compute_all(d0)

    a5_rand = rng.getrandbits(32)
    d5 = build_state(a1, a2, a3, a4, a5_rand, a6, a7, a8, a9, a10, a11, a12)
    _, _, rhos5 = compute_all(d5)
    # Exact invariance of rho_17 and rho_18 under arbitrary a5 when a4 = 0:
    assert rhos5[17] == rhos0[17]
    assert rhos5[18] == rhos0[18]

    # Verify Q1(W15) structure of C17(a8, 0, a12):
    a10_0, a11_0 = solve_right_e10_e11(a6, a7, 0, a9, a12)
    d_r0 = build_state(0, 0, 0, 0, 0, a6, a7, 0, a9, a10_0, a11_0, a12)
    _, W_r0, _ = compute_all(d_r0)
    Q1_W15_0 = Q1(W_r0[15])
    C17_0 = u32(W_r0[17] - sigma1(W_r0[15]) - W_r0[10])
    C17_tilde = u32(C17_0 + Q1_W15_0)

    Q1_W15 = Q1(W0[15])
    # Note: W0[10] = W_r0[10] - a8 - a3, so C17(a8, 0, a12) = W0[17] - sigma1(W0[15]) - (W0[10] + a3)
    C17_actual = u32(W0[17] - sigma1(W0[15]) - W0[10] - a3)
    assert C17_actual == u32(C17_tilde - Q1_W15)

  print(
      "  [Verified 100/100] Exact a5-isolation (Delta_17(0, a5) == 0) & Q1(W15)"
      " structure of C17!"
  )

  # 2. Live Bi-Directional Q0^{-1} / Q1^{-1} MITM + O(1) a5 Cascade (80 constraint bits):
  #    - Fix decoupling constant K17.
  #    - Right List (256 states): for random a12, solve C17(a8, 0, a12) == K17 (mod 2^32)
  #      in O(1) via solve_Q1_inv_fast, bucketed by (-C18(a8, 0, a12)) & 0xFFFF.
  #    - Left List: for random (a1, a2), solve Phi17(a1, a2, a3) == -K17 (mod 2^32)
  #      in O(1) via solve_Q0_inv_fast, match Phi18(a1, a2, a3, 0) & 0xFFFF in bucket.
  #    - Upon match (rho_17 == 0 mod 2^32 AND rho_18 == 0 mod 2^16), solve rho_19 == 0 (mod 2^32)
  #      in 1 call to solve_Q0_inv_fast for a5 (preserving rho_17 and rho_18 identically!).
  t0 = time.time()
  K17 = rng.getrandbits(32)
  right_table = {}
  right_calls = 0
  while len(right_table) < 256:
    right_calls += 1
    a12 = rng.getrandbits(32)
    a10_0, a11_0 = solve_right_e10_e11(a6, a7, 0, a9, a12)
    d_r0 = build_state(0, 0, 0, 0, 0, a6, a7, 0, a9, a10_0, a11_0, a12)
    _, W_r0, _ = compute_all(d_r0)
    C17_0 = u32(W_r0[17] - sigma1(W_r0[15]) - W_r0[10])
    C17_tilde = u32(C17_0 + Q1(W_r0[15]))
    C18_tilde = u32(W_r0[18] - sigma1(W_r0[16]) - W_r0[11])
    W15_0 = W_r0[15]

    for W15_sol in solve_Q1_inv_fast(u32(C17_tilde - K17)):
      a8_sol = u32(W15_0 - W15_sol)
      C18_sol = u32(C18_tilde + a8_sol)
      key16 = u32(-C18_sol) & 0xFFFF
      right_table[key16] = (a8_sol, a12, a10_0, a11_0)

  sol_80 = None
  left_calls = 0
  while sol_80 is None:
    left_calls += 1
    a1, a2 = rng.getrandbits(32), rng.getrandbits(32)
    d_l0 = build_state(a1, a2, 0, a4, 0, a6, a7, 0, a9, 0, 0xFFFFFFFF, 0)
    _, W_l0, _ = compute_all(d_l0)
    W1_val, G2_val = W_l0[1], W_l0[2]
    rhs17 = u32(K17 - W1_val - G2_val)
    for W2_sol in solve_Q0_inv_fast(rhs17):
      a3 = u32(W2_sol - G2_val)
      d_l = build_state(a1, a2, a3, a4, 0, a6, a7, 0, a9, 0, 0xFFFFFFFF, 0)
      _, W_l, _ = compute_all(d_l)
      Phi18 = u32(-sigma0(W_l[3]) - W2_sol)
      key16 = Phi18 & 0xFFFF
      if key16 in right_table:
        a8_s, a12_s, a10_s, a11_s = right_table[key16]
        # Solve rho_19 == 0 (mod 2^32) in O(1) via solve_Q0_inv_fast for a5!
        d_m = build_state(
            a1, a2, a3, a4, 0, a6, a7, a8_s, a9, a10_s, a11_s, a12_s
        )
        _, W_m, _ = compute_all(d_m)
        C19_s = u32(W_m[19] - sigma1(W_m[17]) - W_m[12])
        G4_val = W_m[4]
        rhs19 = u32(C19_s - W_m[3] - G4_val)
        w4_sols = solve_Q0_inv_fast(rhs19)
        if w4_sols:
          a5_s = u32(w4_sols[0] - G4_val)
          sol_80 = (
              a1,
              a2,
              a3,
              a4,
              a5_s,
              a6,
              a7,
              a8_s,
              a9,
              a10_s,
              a11_s,
              a12_s,
          )
          break

  dt_80 = (time.time() - t0) * 1000.0
  d_sol80 = build_state(*sol_80)
  _, _, rhos_80 = compute_all(d_sol80)
  assert rhos_80[17] == 0 and rhos_80[19] == 0 and (rhos_80[18] & 0xFFFF) == 0
  print(
      "  [Live 80-Bit Bi-Directional MITM: rho_17 == 0 (32b), rho_19 == 0"
      " (32b), rho_18 == 0 (mod 2^16)]"
  )
  print(
      f"    Solved in {dt_80:.1f} ms ({right_calls} Right Q1^-1 calls +"
      f" {left_calls} Left Q0^-1 calls)!"
  )
  print(
      f"    Verified residuals: rho_17 = 0x{rhos_80[17]:08x}, rho_18 ="
      f" 0x{rhos_80[18]:08x}, rho_19 = 0x{rhos_80[19]:08x}\n"
  )


def demo_4_full_a6_invariance_and_monic_a9():
  print(
      "=== DEMO 4: Full-Equation a6-Invariance Theorem & Monic a9 Linearity ==="
  )
  rng = random.Random(404)

  # 1. Verify 100/100 Full-Equation a6-Invariance on e11=0, e12=0xffffffff, a5=a7=a8=u
  for _ in range(100):
    a1, a2, a3, a4 = (rng.getrandbits(32) for _ in range(4))
    u = rng.getrandbits(32)
    a5 = a7 = a8 = u
    a9, a10 = rng.getrandbits(32), rng.getrandbits(32)
    a6_1, a6_2 = rng.getrandbits(32), rng.getrandbits(32)

    # Enforce e11 = 0, e12 = 0xffffffff via (a11, a12)
    a11 = u32(0 - a7 + Sigma0(a10) + Maj(a10, a9, a8))
    a12 = u32(0xFFFFFFFF - a8 + Sigma0(a11) + Maj(a11, a10, a9))

    d1 = build_state(a1, a2, a3, a4, a5, a6_1, a7, a8, a9, a10, a11, a12)
    d2 = build_state(a1, a2, a3, a4, a5, a6_2, a7, a8, a9, a10, a11, a12)
    _, _, rhos1 = compute_all(d1)
    _, _, rhos2 = compute_all(d2)
    assert rhos1[18] == rhos2[18]
    assert rhos1[19] == rhos2[19]
  print(
      "  [Verified 100/100] Literal full equations (rho_18, rho_19) are 100%"
      " invariant to a6 on (e11=0, e12=0xffffffff, a5=a7=a8=u)!"
  )

  # 2. Verify 100/100 +1-monic linearity of C19(a9) and Q1-structure of C18(a9)
  # on a5=a7=a8=a10=a11=0, a12=0xffffffff
  d_base = build_state(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xFFFFFFFF)
  _, W_base, _ = compute_all(d_base)
  C19_0 = u32(W_base[19] - sigma1(W_base[17]) - W_base[12])
  C18_0 = u32(W_base[18] - sigma1(W_base[16]) - W_base[11])
  W16_0 = W_base[16]
  C18_const = u32(C18_0 + sigma1(W16_0) + W16_0)

  for _ in range(100):
    a6_val = rng.getrandbits(32)
    a9_val = rng.getrandbits(32)
    d_test = build_state(
        0, 0, 0, 0, 0, a6_val, 0, 0, a9_val, 0, 0, 0xFFFFFFFF
    )
    _, W_t, _ = compute_all(d_test)
    C19_t = u32(W_t[19] - sigma1(W_t[17]) - W_t[12])
    C18_t = u32(W_t[18] - sigma1(W_t[16]) - W_t[11])
    W16_t = u32(W16_0 - a9_val)
    assert C19_t == u32(C19_0 + a9_val)
    assert C18_t == u32(C18_const - sigma1(W16_t) - W16_t)
  print(
      "  [Verified 100/100] Exact +1-monic linearity C19(a9) = C19(0) + a9 &"
      " Q1(W16) structure of C18(a9)!\n"
  )


def solve_right_e9_e10_e11(
    a5,
    a6,
    a7,
    a8,
    e9_target=0,
    e10_target=0xFFFFFFFF,
    e11_target=0xFFFFFFFF,
):
  T9 = u32(e9_target - a5)
  a9 = u32(T9 + Sigma0(a8) + Maj(a8, a7, a6))
  T10 = u32(e10_target - a6)
  a10 = u32(T10 + Sigma0(a9) + Maj(a9, a8, a7))
  T11 = u32(e11_target - a7)
  a11 = u32(T11 + Sigma0(a10) + Maj(a10, a9, a8))
  return a9, a10, a11


def demo_5_20step_4equation_decoupling():
  print(
      "=== DEMO 5: Full 20-Step (4-Equation: rho_16..rho_19) Exact Decoupling"
      " ==="
  )
  rng = random.Random(505)
  a5, a6, a7 = 0, 0, 0

  # Verify 100/100 that on a5=a6=a7=0, e9=0, e10=0xffffffff, e11=0xffffffff,
  # ALL FOUR equations (rho_16, rho_17, rho_18, rho_19) decouple into
  # Left(a1, a2, a3, a4) + Right(a8, a12) with ZERO Left-Right cross-coupling,
  # AND Delta_17 == 0 identically for all (a1, a2, a3, a4)!
  for _ in range(100):
    a8, a12 = rng.getrandbits(32), rng.getrandbits(32)
    a9, a10, a11 = solve_right_e9_e10_e11(a5, a6, a7, a8)
    d_base = build_state(0, 0, 0, 0, a5, a6, a7, a8, a9, a10, a11, a12)
    _, W_b, _ = compute_all(d_base)
    C16_b = u32(W_b[16] - sigma1(W_b[14]) - W_b[9])
    C17_b = u32(W_b[17] - sigma1(W_b[15]) - W_b[10])
    C18_b = u32(W_b[18] - sigma1(W_b[16]) - W_b[11])
    C19_b = u32(W_b[19] - sigma1(W_b[17]) - W_b[12])

    a1, a2, a3, a4 = (rng.getrandbits(32) for _ in range(4))
    d_full = build_state(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12)
    _, W_f, rhos_f = compute_all(d_full)

    Phi16 = u32(-sigma0(W_f[1]) - W_f[0] + a2 + a3 - (a3 & a4))
    Phi17 = u32(-sigma0(W_f[2]) - W_f[1] + a3)
    Phi18 = u32(-sigma0(W_f[3]) - W_f[2] + a4)
    Phi19 = u32(-sigma0(W_f[4]) - W_f[3])

    assert rhos_f[16] == u32(Phi16 + C16_b)
    assert rhos_f[17] == u32(Phi17 + C17_b)
    assert rhos_f[18] == u32(Phi18 + C18_b)
    assert rhos_f[19] == u32(Phi19 + C19_b)

  print(
      "  [Verified 100/100] Exact 4-Equation (20-Step) Left/Right Decoupling on"
      " (a5=a6=a7=0, e9=0, e10=e11=0xffffffff)!\n"
  )


if __name__ == "__main__":
  demo_1_q0_inv_and_sigma0_inv()
  demo_2_algebraic_decoupling_and_cascades()
  demo_3_bi_directional_a5_isolated_solver()
  demo_4_full_a6_invariance_and_monic_a9()
  demo_5_20step_4equation_decoupling()

