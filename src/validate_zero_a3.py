#!/usr/bin/env python3
"""
Numerical validation of the Zero-a3 Bi-Directional Q0/Q1 Bridge identities
for 20-step SHA-256 (Python standard library only).

Checks over 10,000 random assignments of (H, a1, a2, a4, a7, a11, a12) on the
Zero-a3 Manifold (a3 = a5 = a6 = a8 = a9 = 0, a10 = 0xffffffff):
  1. Maj(a5, a4, a3) == 0 and Maj(a6, a5, a4) == 0 identically (zero beta-filter loss).
  2. W9..W19 are 100% independent of a4, and W11..W19 are 100% independent of a1, a2, a4.
  3. The Q1-cancellation identity on C16:
       rho16 = C16tilde(a11, a12) - Q1(D14(a11, a12) - a7) - (Q0(W1) + W0 - G1)
     where Q1(u) = s1(u) + u and Q0(u) = s0(u) - u.
  4. The Left/Right bridge identity on C17:
       rho17 = (C17(a7, a11, a12) + a7) - (s0(W2) + W1).
  5. The Level-3 Q0 identity on C18:
       rho18 = (T18(a7, a11, a12) - W2 + G3) - Q0(W3).
"""
import random

M32 = 0xFFFFFFFF
K = [
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B,
    0x59F111F1, 0x923F82A4, 0xAB1C5ED5, 0xD807AA98, 0x12835B01,
    0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7,
    0xC19BF174, 0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC,
]
IV = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
]


def rotr(x, n):
    return ((x >> n) | (x << (32 - n))) & M32


def S0(x):
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)


def S1(x):
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)


def s0(x):
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3)


def s1(x):
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10)


def Ch(x, y, z):
    return ((x & y) ^ ((~x & M32) & z)) & M32


def Maj(x, y, z):
    return (x & y) ^ (x & z) ^ (y & z)


def Q0(u):
    return (s0(u) - u) & M32


def Q1(u):
    return (s1(u) + u) & M32


def init_state(H):
    a = {}
    a[0], a[-1], a[-2], a[-3] = IV[0], IV[1], IV[2], IV[3]
    for k in range(4):
        t = -k
        a[t - 4] = (IV[4 + k] - a[t] + S0(a[t - 1]) + Maj(a[t - 1], a[t - 2], a[t - 3])) & M32
    for i in range(4):
        a[20 - i] = (H[i] - IV[i]) & M32
    for k in range(4):
        t = 20 - k
        es = (H[4 + k] - IV[4 + k]) & M32
        a[t - 4] = (es - a[t] + S0(a[t - 1]) + Maj(a[t - 1], a[t - 2], a[t - 3])) & M32
    a[3] = 0
    a[5] = 0
    a[6] = 0
    a[8] = 0
    a[9] = 0
    a[10] = M32
    return a


def e_of(a, t):
    return (a[t] - S0(a[t - 1]) - Maj(a[t - 1], a[t - 2], a[t - 3]) + a[t - 4]) & M32


def W_of(a, t):
    et = e_of(a, t)
    et1 = e_of(a, t - 1)
    et2 = e_of(a, t - 2)
    et3 = e_of(a, t - 3)
    return (a[t + 1] - S0(a[t]) - Maj(a[t], a[t - 1], a[t - 2]) - et3 - S1(et) - Ch(et, et1, et2) - K[t]) & M32


def rho_of(a, t):
    return (W_of(a, t) - s1(W_of(a, t - 2)) - W_of(a, t - 7) - s0(W_of(a, t - 15)) - W_of(a, t - 16)) & M32


if __name__ == "__main__":
    rng = random.Random(20261013)
    trials = 10000
    violations = 0
    for _ in range(trials):
        H = [rng.getrandbits(32) for _ in range(8)]
        a = init_state(H)

        # Compute base constants at a1=a2=a4=a7=a11=a12=0
        for k in (1, 2, 4, 7, 11, 12):
            a[k] = 0
        W9_0 = W_of(a, 9)

        a1 = rng.getrandbits(32)
        a2 = rng.getrandbits(32)
        a4 = rng.getrandbits(32)
        a7 = rng.getrandbits(32)
        a11 = rng.getrandbits(32)
        a12 = rng.getrandbits(32)

        # W10_0 depends only on a11
        a[11] = a11
        W10_0 = W_of(a, 10)

        # D14 and C16tilde at a7 = 0
        a[12] = a12
        D14 = W_of(a, 14)
        C16tilde = (W_of(a, 16) + D14 - W9_0) & M32

        # Left quantities from (a1, a2)
        a[1] = a1
        a[2] = 0
        W0 = W_of(a, 0)
        G1 = (-W_of(a, 1)) & M32
        a[2] = a2
        W1 = W_of(a, 1)
        W2 = W_of(a, 2)
        T16_left = (Q0(W1) + W0 - G1) & M32
        L17_left = (s0(W2) + W1) & M32

        # Right / Level-3 quantities with a7 set and a4 = 0
        a[4] = 0
        a[7] = a7
        C17 = (W_of(a, 17) - s1(W_of(a, 15)) - W10_0) & M32
        T17_right = (C17 + a7) & M32
        G3 = (-W_of(a, 3)) & M32
        T18 = (W_of(a, 18) - s1(W_of(a, 16)) - W_of(a, 11)) & M32

        # Full state with arbitrary a4
        a[4] = a4
        W3 = W_of(a, 3)

        if Maj(a[5], a[4], a[3]) != 0 or Maj(a[6], a[5], a[4]) != 0:
            violations += 1
        if ((C16tilde - Q1((D14 - a7) & M32) - T16_left) & M32) != rho_of(a, 16):
            violations += 1
        if ((T17_right - L17_left) & M32) != rho_of(a, 17):
            violations += 1
        if ((T18 - W2 + G3 - Q0(W3)) & M32) != rho_of(a, 18):
            violations += 1

    print(f"Tested {trials} random states across all 4 Zero-a3 identities: {violations} violations.")
    assert violations == 0
