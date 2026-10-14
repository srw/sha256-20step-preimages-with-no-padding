#!/usr/bin/env python3
"""
Standalone verifier for 20-step SHA-256 compression preimage witnesses (Python standard library only).

SHA-256 reduced to R steps = the standard compression function (FIPS 180-4) with the round loop
stopped after R rounds (message schedule computed only as far as needed), followed by the usual
feed-forward H_i = IV_i + state_i.
Sanity check: with R = 64 it reproduces hashlib.sha256.

usage: python3 verify/verify_standalone.py witnesses/witnesses_20step.txt
"""
import hashlib
import os
import sys

K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]
H0 = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]
M32 = 0xFFFFFFFF


def ror(x, n):
    return ((x >> n) | (x << (32 - n))) & M32


def compress(words, rounds):
    W = list(words)
    for t in range(16, rounds):
        s0 = ror(W[t - 15], 7) ^ ror(W[t - 15], 18) ^ (W[t - 15] >> 3)
        s1 = ror(W[t - 2], 17) ^ ror(W[t - 2], 19) ^ (W[t - 2] >> 10)
        W.append((W[t - 16] + s0 + W[t - 7] + s1) & M32)
    a, b, c, d, e, f, g, h = H0
    for t in range(rounds):
        S1 = ror(e, 6) ^ ror(e, 11) ^ ror(e, 25)
        chv = (e & f) ^ (~e & g)
        t1 = (h + S1 + chv + K[t] + W[t]) & M32
        S0 = ror(a, 2) ^ ror(a, 13) ^ ror(a, 22)
        mj = (a & b) ^ (a & c) ^ (b & c)
        t2 = (S0 + mj) & M32
        h, g, f, e, d, c, b, a = g, f, e, (d + t1) & M32, c, b, a, (t1 + t2) & M32
    return b"".join(((x + y) & M32).to_bytes(4, "big") for x, y in zip(H0, (a, b, c, d, e, f, g, h)))


def sha256_steps(msg, rounds):
    L = len(msg)
    data = msg + b"\x80" + b"\x00" * ((55 - L) % 64) + (8 * L).to_bytes(8, "big")
    H = list(H0)
    for off in range(0, len(data), 64):
        words = [int.from_bytes(data[off + i : off + i + 4], "big") for i in range(0, 64, 4)]
        W = list(words)
        for t in range(16, rounds):
            s0 = ror(W[t - 15], 7) ^ ror(W[t - 15], 18) ^ (W[t - 15] >> 3)
            s1 = ror(W[t - 2], 17) ^ ror(W[t - 2], 19) ^ (W[t - 2] >> 10)
            W.append((W[t - 16] + s0 + W[t - 7] + s1) & M32)
        a, b, c, d, e, f, g, h = H
        for t in range(rounds):
            S1 = ror(e, 6) ^ ror(e, 11) ^ ror(e, 25)
            t1 = (h + S1 + ((e & f) ^ (~e & g)) + K[t] + W[t]) & M32
            S0 = ror(a, 2) ^ ror(a, 13) ^ ror(a, 22)
            t2 = (S0 + ((a & b) ^ (a & c) ^ (b & c))) & M32
            h, g, f, e, d, c, b, a = g, f, e, (d + t1) & M32, c, b, a, (t1 + t2) & M32
        H = [(x + y) & M32 for x, y in zip(H, (a, b, c, d, e, f, g, h))]
    return b"".join(x.to_bytes(4, "big") for x in H)


if __name__ == "__main__":
    for n in (0, 3, 55, 56, 64, 119, 200):
        m = os.urandom(n)
        assert sha256_steps(m, 64) == hashlib.sha256(m).digest()

    path = sys.argv[1] if len(sys.argv) > 1 else "witnesses/witnesses_20step.txt"
    ok = bad = 0
    mode = target = None
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            mode = line[1 : line.index("]")]
        elif line.startswith("target"):
            target = line.split("=", 1)[1].strip().replace(" ", "").lower()
        elif line.startswith("W[0..15]") and mode == "comp":
            words = [int(x, 16) for x in line.split("=", 1)[1].split()]
            got19 = compress(words, 19).hex()
            got20 = compress(words, 20).hex()
            got21 = compress(words, 21).hex()
            if got20 == target and got19 != target and got21 != target:
                ok += 1
            else:
                bad += 1
    print(f"verified {ok} 20-step witnesses, {bad} failures (64-step sanity check vs hashlib passed; 19/21-step misses confirmed)")
    sys.exit(0 if (ok > 0 and bad == 0) else 1)
