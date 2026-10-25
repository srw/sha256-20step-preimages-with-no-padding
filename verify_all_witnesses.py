#!/usr/bin/env python3
"""
Standalone Zero-Dependency Verifier for:
  "Practical Preimage Attacks on 19-Step and 20-Step Padded SHA-256
   via Carry-Sudoku and Slot-Sudoku Decomposition"

Verifies:
  1. 20-Step Padded SHA-256 Preimage Witness (Target: All-Zero Digest 0^256)
     + Merkle-Damgard Padding (W13 & 0xff == 0x80, W14 == 0, W15 == 440)
     + Double-Cancel & Carry-Sudoku Invariants (a5 == a6, a10 == a11, e9 == e10 == e15 == 0xffffffff)
  2. 19-Step Padded SHA-256 Preimage Witnesses (Targets: All-Zero Digest 0^256 & SHA-256("abc"))
     + Merkle-Damgard Padding (W13 & 0xff == 0x80, W14 == 0, W15 == 440)
  3. Two-Block (119-Byte, Plain-ASCII Block 1) 19-Step & 20-Step Padded Witnesses (Target: SHA-256(""))
  4. 20-Step Unpadded SHA-256 Preimage Witness (Target: deadbeef... pattern)
  5. All 26 20-Step Benchmark Witnesses in witnesses_20step.txt (14 Padded + 12 Unpadded)
  6. All 14 19-Step Padded Benchmark Witnesses (Table 8.3)
  7. Stage-2 Carry-Sudoku Completeness Regression Witness
"""

import hashlib
import os
import re
import sys

IV = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
]

K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
]

def rotr(x, n):
    return ((x >> n) | (x << (32 - n))) & 0xffffffff

def Sigma0(x):
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)

def Sigma1(x):
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)

def sigma0(x):
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3)

def sigma1(x):
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10)

def Ch(e, f, g):
    return (e & f) ^ ((~e & 0xffffffff) & g)

def Maj(a, b, c):
    return (a & b) ^ (a & c) ^ (b & c)

def sha256_reduced(W16, rounds, init_state=None):
    W = list(W16)
    for t in range(16, rounds):
        w = (sigma1(W[t - 2]) + W[t - 7] + sigma0(W[t - 15]) + W[t - 16]) & 0xffffffff
        W.append(w)

    base = list(IV if init_state is None else init_state)
    state = list(base)
    trace = [tuple(state)]
    for t in range(rounds):
        a, b, c, d, e, f, g, h = state
        T1 = (h + Sigma1(e) + Ch(e, f, g) + K[t] + W[t]) & 0xffffffff
        T2 = (Sigma0(a) + Maj(a, b, c)) & 0xffffffff
        state = [
            (T1 + T2) & 0xffffffff,
            a, b, c,
            (d + T1) & 0xffffffff,
            e, f, g
        ]
        trace.append(tuple(state))

    digest = [(base[i] + state[i]) & 0xffffffff for i in range(8)]
    return digest, trace, W

def fmt_words(words):
    return "".join(f"{w:08x}" for w in words)

def words_from_padded_55b_hex(msg_hex):
    assert len(msg_hex) == 110, f"Expected 110 hex chars (55 bytes), got {len(msg_hex)}"
    raw = bytes.fromhex(msg_hex) + b"\x80" + b"\x00" * 6 + (440).to_bytes(2, "big")
    return [int.from_bytes(raw[4 * i : 4 * i + 4], "big") for i in range(16)]

def main():
    print("=" * 78)
    print("1. VERIFYING 20-STEP PADDED SHA-256 PREIMAGE WITNESS (All-Zero Target 0^256)")
    print("=" * 78)
    W20_padded = [
        0xec531306, 0xc85ad4a4, 0xad211d2e, 0x1a03ece4,
        0xd0ad4a0e, 0x6b4c2b94, 0x36b3babd, 0xcbd52059,
        0x7196f52c, 0x042b8cd9, 0x5f1a39f7, 0x1b58b47b,
        0x0647c78e, 0x65416f80, 0x00000000, 0x000001b8,
    ]
    digest20p, trace20p, W20p_exp = sha256_reduced(W20_padded, 20)
    print(f"Input W[0..15] : {' '.join(f'{w:08x}' for w in W20_padded)}")
    print(f"Expanded W[16..19]: {' '.join(f'{w:08x}' for w in W20p_exp[16:20])}")
    print(f"20-Step Digest : {fmt_words(digest20p)}")

    assert all(x == 0 for x in digest20p), "Digest is not all-zeros!"
    assert (W20_padded[13] & 0xff) == 0x80, "W13 low byte is not 0x80!"
    assert W20_padded[14] == 0x00000000, "W14 is not 0!"
    assert W20_padded[15] == 0x000001b8, "W15 is not 440 (0x1b8)!"

    a3 = trace20p[3][0]
    a4 = trace20p[4][0]
    a5, e5 = trace20p[5][0], trace20p[5][4]
    a6, e6 = trace20p[6][0], trace20p[6][4]
    a9, e9 = trace20p[9][0], trace20p[9][4]
    a10, e10 = trace20p[10][0], trace20p[10][4]
    a11, e11 = trace20p[11][0], trace20p[11][4]
    a15, e15 = trace20p[15][0], trace20p[15][4]
    M3 = (~(a3 ^ a5)) & 0xffffffff
    pop_M3 = bin(M3).count("1")

    print(f"Padding Check  : W13 & 0xff = 0x{W20_padded[13] & 0xff:02x}, W14 = 0x{W20_padded[14]:08x}, W15 = {W20_padded[15]} bits (55 bytes) -> PASS")
    print(f"D=0 Invariant  : a5 = 0x{a5:08x}, a6 = 0x{a6:08x} (D = a6 - a5 = {(a6 - a5) & 0xffffffff}) -> PASS")
    print(f"Maj Absorption : popcount(M3) = {pop_M3} <= 14, ((a4 ^ a5) & M3) = 0x{(a4 ^ a5) & M3:08x} -> PASS")
    print(f"Twin Invariant : a10 = 0x{a10:08x}, a11 = 0x{a11:08x} -> PASS")
    print(f"Ch Mask Check  : e9 = 0x{e9:08x}, e10 = 0x{e10:08x}, e15 = 0x{e15:08x} -> PASS")
    assert a5 == a6 == 0x67978bcd
    assert pop_M3 == 12 and ((a4 ^ a5) & M3) == 0
    assert a10 == a11 == 0xa7b351d7
    assert e9 == e10 == e15 == 0xffffffff

    print("\n" + "=" * 78)
    print("2. VERIFYING 19-STEP PADDED SHA-256 PREIMAGE WITNESSES (0^256 & SHA-256('abc'))")
    print("=" * 78)
    W19_zero_hex = "fc0c9d8eafb2fa329af316940c42a5eb9773cec39e4f7e3961826287ae767638f92d4ce5fc5eb8cabd7606409364dda27a6c1bb5220246"
    W19_zero = words_from_padded_55b_hex(W19_zero_hex)
    digest19z, trace19z, _ = sha256_reduced(W19_zero, 19)
    print(f"0^256 W[0..15] : {' '.join(f'{w:08x}' for w in W19_zero)}")
    print(f"19-Step Digest : {fmt_words(digest19z)}")
    assert all(x == 0 for x in digest19z), "19-step all-zero digest mismatch!"
    assert (W19_zero[13] & 0xff) == 0x80 and W19_zero[14] == 0 and W19_zero[15] == 0x1b8
    assert trace19z[5][0] == trace19z[6][0], "19-step all-zero witness D=0 check failed!"
    assert trace19z[9][4] == trace19z[10][4] == 0xffffffff, "19-step all-zero witness e9=e10 check failed!"
    print(f"0^256 Invariants: a5=a6=0x{trace19z[5][0]:08x}, e9=e10=0xffffffff -> PASS")

    W19_padded = [
        0x8ca254f8, 0xd04dd3fb, 0x04efaad6, 0xb1867588,
        0xffe833de, 0x49a144fe, 0xdbf18394, 0x22892dc9,
        0xf5a9158d, 0x8fc06329, 0x0a14a9a6, 0xc55dfa24,
        0x5642a024, 0x7ec45a80, 0x00000000, 0x000001b8,
    ]
    target_abc = [
        0xba7816bf, 0x8f01cfea, 0x414140de, 0x5dae2223,
        0xb00361a3, 0x96177a9c, 0xb410ff61, 0xf20015ad,
    ]
    digest19p, trace19p, _ = sha256_reduced(W19_padded, 19)
    print(f"'abc' W[0..15] : {' '.join(f'{w:08x}' for w in W19_padded)}")
    print(f"19-Step Digest : {fmt_words(digest19p)}")
    print(f"Target ('abc') : {fmt_words(target_abc)}")
    assert digest19p == target_abc, "19-step digest does not match SHA-256('abc')!"
    assert (W19_padded[13] & 0xff) == 0x80 and W19_padded[14] == 0 and W19_padded[15] == 0x1b8
    print("Padding & Digest Check -> PASS")

    print("\n" + "=" * 78)
    print("3. VERIFYING TWO-BLOCK (119-BYTE, PLAIN-ASCII BLOCK 1) PADDED WITNESS (SHA-256(''))")
    print("=" * 78)
    W19_blk1 = [
        0x54686973, 0x20666972, 0x73742062, 0x6c6f636b,
        0x20697320, 0x706c6169, 0x6e207465, 0x78743b20,
        0x74686520, 0x7365636f, 0x6e64206f, 0x6e652077,
        0x61732063, 0x6f6d7075, 0x7465642e, 0x2e2e2e2e,
    ]
    W19_blk2 = [
        0xfe409cc9, 0x30128a72, 0x0481f86c, 0x0503d7a0,
        0x18519a12, 0x9df44913, 0x9720bb5e, 0xef520895,
        0xf8b51568, 0x5bb40481, 0x7be30188, 0x0f0caabe,
        0x02fc9e5f, 0x1c95c080, 0x00000000, 0x000003b8,
    ]
    target_empty = [
        0xe3b0c442, 0x98fc1c14, 0x9afbf4c8, 0x996fb924,
        0x27ae41e4, 0x649b934c, 0xa495991b, 0x7852b855,
    ]
    chain19, _, _ = sha256_reduced(W19_blk1, 19, IV)
    digest19_2blk, _, _ = sha256_reduced(W19_blk2, 19, chain19)
    ascii_blk1 = b"".join(w.to_bytes(4, "big") for w in W19_blk1).decode("ascii")
    print(f"Block 1 ASCII  : {ascii_blk1!r}")
    print(f"Block 2 W[0..15]: {' '.join(f'{w:08x}' for w in W19_blk2)}")
    print(f"19-Step Digest : {fmt_words(digest19_2blk)}")
    print(f"Target ('')    : {fmt_words(target_empty)}")
    assert digest19_2blk == target_empty, "Two-block 19-step digest mismatch!"
    assert (W19_blk2[13] & 0xff) == 0x80 and W19_blk2[14] == 0 and W19_blk2[15] == 0x3b8
    print("Two-Block Chaining, ASCII Prefix & Padding (952 bits) Check -> PASS")

    print("\n" + "=" * 78)
    print("4. VERIFYING 20-STEP UNPADDED SHA-256 PREIMAGE WITNESS (Target: deadbeef...)")
    print("=" * 78)
    W20_unpadded = [
        0x4334d956, 0x6eccece1, 0x6727a1b2, 0xd85811af,
        0x890c5399, 0x79018a3b, 0xd346b94d, 0x5f26009b,
        0x7562d311, 0x0bfd80d5, 0x1f8c38f4, 0x16d8315d,
        0xe0353aa2, 0x6adf4bd2, 0xa2bf14a2, 0x801ae8f7,
    ]
    target_deadbeef = [
        0xdeadbeef, 0xcafebabe, 0x01234567, 0x89abcdef,
        0xfedcba98, 0x76543210, 0x13579bdf, 0x2468ace0,
    ]
    digest20u, _, _ = sha256_reduced(W20_unpadded, 20)
    print(f"Input W[0..15] : {' '.join(f'{w:08x}' for w in W20_unpadded)}")
    print(f"20-Step Digest : {fmt_words(digest20u)}")
    assert digest20u == target_deadbeef, "20-step unpadded digest mismatch!"
    print("20-Step Unpadded Digest Check -> PASS")

    print("\n" + "=" * 78)
    print("5. VERIFYING 20-STEP FLAGSHIP DEMOS (demos_20step.txt: 'abc' & 2-Block '')")
    print("=" * 78)
    W20_abc = [
        0xedf6e70a, 0x7001ae71, 0xb900f181, 0x19aaa1fd,
        0xf6c1179b, 0x42688912, 0x62a9539b, 0xe04d4123,
        0x6685e0c1, 0x02e0ad23, 0x165e8b71, 0x57669819,
        0x302ddcb3, 0xc6822680, 0x00000000, 0x000001b8,
    ]
    digest20_abc, trace20_abc, _ = sha256_reduced(W20_abc, 20)
    print(f"Demo 1 ('abc') W[0..15]: {' '.join(f'{w:08x}' for w in W20_abc)}")
    print(f"20-Step Digest ('abc') : {fmt_words(digest20_abc)}")
    assert digest20_abc == target_abc, "20-step 'abc' digest mismatch!"
    assert (W20_abc[13] & 0xff) == 0x80 and W20_abc[14] == 0 and W20_abc[15] == 0x1b8
    print(f"Demo 1 ('abc') Invariants: a5=0x{trace20_abc[5][0]:08x}, a6=0x{trace20_abc[6][0]:08x}, "
          f"a10=0x{trace20_abc[10][0]:08x}, a11=0x{trace20_abc[11][0]:08x}, "
          f"e9=0x{trace20_abc[9][4]:08x}, e10=0x{trace20_abc[10][4]:08x}, e15=0x{trace20_abc[15][4]:08x} -> PASS")
    assert trace20_abc[5][0] == trace20_abc[6][0]
    assert trace20_abc[10][0] == trace20_abc[11][0]
    assert trace20_abc[9][4] == trace20_abc[10][4] == trace20_abc[15][4] == 0xffffffff

    W20_blk2 = [
        0xdf28e21b, 0xb6b61202, 0xba87baa7, 0x2096ee0a,
        0x3ac325cf, 0x1f12056e, 0x7cfe2ae1, 0x735b4d10,
        0x6b8f6e0b, 0x22eb9b50, 0x4ed93689, 0x91944b0a,
        0x115a934f, 0x17c2c480, 0x00000000, 0x000003b8,
    ]
    chain20, _, _ = sha256_reduced(W19_blk1, 20, IV)
    digest20_2blk, _, _ = sha256_reduced(W20_blk2, 20, chain20)
    print(f"Block 1 ASCII  : {ascii_blk1!r}")
    print(f"Chain H^(1)    : {fmt_words(chain20)}")
    print(f"Block 2 W[0..15]: {' '.join(f'{w:08x}' for w in W20_blk2)}")
    print(f"20-Step Digest : {fmt_words(digest20_2blk)}")
    print(f"Target ('')    : {fmt_words(target_empty)}")
    assert digest20_2blk == target_empty, "Two-block 20-step digest mismatch!"
    assert (W20_blk2[13] & 0xff) == 0x80 and W20_blk2[14] == 0 and W20_blk2[15] == 0x3b8
    print("20-Step Two-Block Chaining, ASCII Prefix & Padding (952 bits) Check -> PASS")

    print("\n" + "=" * 78)
    print("6. VERIFYING ALL 26 SOLVED 20-STEP BENCHMARK WITNESSES (witnesses_20step.txt)")
    print("=" * 78)
    wfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), "witnesses_20step.txt")
    assert os.path.exists(wfile), "Missing witnesses_20step.txt"
    with open(wfile, "r", encoding="utf-8") as f:
        text = f.read()
    blocks = [b.strip() for b in text.split("\n\n") if b.strip() and not b.strip().startswith("#")]
    n_padded = 0
    n_comp = 0
    for blk in blocks:
        m_hdr = re.search(r'\[(padded|comp)\]\s+label="([^"]*)"', blk)
        assert m_hdr, f"Malformed block header: {blk}"
        kind, label = m_hdr.group(1), m_hdr.group(2)
        m_tgt = re.search(r'target\s*=\s*([0-9a-f]{64})', blk)
        assert m_tgt, f"Missing target in {label}"
        target_hex = m_tgt.group(1)
        expected_target = hashlib.sha256(label.encode("ascii")).hexdigest()
        assert target_hex == expected_target, f"Target mismatch for {label}: {target_hex} != {expected_target}"
        if kind == "padded":
            m_msg = re.search(r'message_hex\s*=\s*([0-9a-f]{110})', blk)
            assert m_msg, f"Missing 55-byte message_hex in {label}"
            words = words_from_padded_55b_hex(m_msg.group(1))
            dig, tr, _ = sha256_reduced(words, 20)
            assert fmt_words(dig) == target_hex, f"20-step padded digest mismatch for {label}"
            assert tr[5][0] == tr[6][0] and tr[10][0] == tr[11][0]
            assert tr[9][4] == tr[10][4] == tr[15][4] == 0xffffffff
            n_padded += 1
            print(f"  [padded20] {label:<32} -> {target_hex[:16]}... PASS")
        else:
            m_w = re.search(r'W\[0\.\.15\]\s*=\s*([0-9a-f ]+)', blk)
            assert m_w, f"Missing W[0..15] in {label}"
            words = [int(x, 16) for x in m_w.group(1).split()]
            assert len(words) == 16
            dig, _, _ = sha256_reduced(words, 20)
            assert fmt_words(dig) == target_hex, f"20-step comp digest mismatch for {label}"
            n_comp += 1
            print(f"  [comp20]   {label:<32} -> {target_hex[:16]}... PASS")
    assert n_padded == 14 and n_comp == 12, f"Expected 14 padded & 12 comp, got {n_padded} & {n_comp}"
    print(f"Verified {n_padded} padded 1-block witnesses and {n_comp} unpadded compression witnesses in witnesses_20step.txt!")

    print("\n" + "=" * 78)
    print("7. VERIFYING ALL 14 SOLVED 19-STEP PADDED BENCHMARK WITNESSES (Table 8.3)")
    print("=" * 78)
    suite19 = [
        ("abc", "8ca254f8d04dd3fb04efaad6b1867588ffe833de49a144fedbf1839422892dc9f5a9158d8fc063290a14a9a6c55dfa245642a0247ec45a"),
        ("hello", "ac055aaa83e1e51a5d74102b7739b7b6e019e31100cd525baf09c642d4b59b8fc4eb8d481df111aee2616325b336b827190103c33584c5"),
        ("padded test 1", "a9cfb3f277e9e66e5a86ea3ff1dac69d3327b726deb9420583f150597ba70fa19cfc18400c1222c58d20911e6162081ebe084915634c3a"),
        ("padded test 2", "c13089d977eea9843460eb14fff02051a5cf91b4ffedf39711c2ec81862939f8adcc0dc25fb391e5c84ed20f8ef02e4a205578ea45e9fd"),
        ("padded test 3", "e24cf94c817e32304c4162886506bdfa5d2766398d36171235b6e3fde064535456052c3022ef488175bbbd6b81085c86401df4e0dae715"),
        ("padded test 4", "3b605fe04bdf7b05dbcae817e1a77d8e7e9ad63db1fe8f5d972828874695dde35bcf2deb7166e78132c9ac7ee9f72d07190b83c398a2ed"),
        ("padded test 5", "502eb30d876c48eaee8a13c08e4e15b5ac74689dfdf5e60cb2fd044d4bc8eaae640c92cf6236d908717a5842304f70d47fc4dddbe6d899"),
        ("padded test 6", "782f45dda1ab000aace616e7c3076339fe8938a6f0a35fc6ff0e2739c3194c08494e5b1a4af0b10e69d87a02352ddd71034511dba58e90"),
        ("padded test 7", "fb676bdeabfc0143be3d2b4c8350083e39b4aedcdfbf2b87c10c23786a901dc14ffc2879516522cbc86235d1a7bf135373355d0cf71888"),
        ("padded test 8", "a1edbba02a9f54cf4083f7fc27156523f3acd5f52339e453c950d9ed841486dff2264d752bea2d27440984bd612c77f69b4479b63cde7f"),
        ("padded test 9", "0736a2716d8ee60644b77f1e85c91f230ed82d7f242ae673baeb3049b858b159e8d90ba74901bda34b0536a82b222f425f9533e25c727c"),
        ("padded test 10", "a80e3ae94cb0c81f3e591bfd4453eba61c2148ab7d32f1e524c220134d9be8e12ee2beebfa02fb6bc994272b720842d68709f1e0023b79"),
        ("padded test 11", "23ae3771d98d69c6925d4ca4b8059f84404eea570db3d014c022af5649565a9a19a0e610fdb9258ce144893cee5d25f260f11614b36253"),
        ("hand-off cross-check 2026-09-29", "6e77d54453883692c4e2fbf20d290ce36ee98f42748f7c618cd07a0d19b36b5a43c25fa3b008d57af11dbc2dabc843dae6117a0f867286"),
    ]
    for label, msg_hex in suite19:
        expected_target = hashlib.sha256(label.encode("ascii")).hexdigest()
        words = words_from_padded_55b_hex(msg_hex)
        dig, tr, _ = sha256_reduced(words, 19)
        assert fmt_words(dig) == expected_target, f"19-step padded digest mismatch for {label}"
        assert tr[9][4] == tr[10][4] == 0xffffffff, f"19-step e9=e10 mismatch for {label}"
        print(f"  [padded19] {label:<32} -> {expected_target[:16]}... PASS")
    print(f"Verified all {len(suite19)} 19-step padded single-block benchmark witnesses!")

    print("\n" + "=" * 78)
    print("8. VERIFYING STAGE-2 CARRY-SUDOKU COMPLETENESS REGRESSION WITNESS")
    print("=" * 78)
    reg_msg_hex = "edf6e70a7001ae71b900f18119aaa1fdf6c1179b4268891262a9539be04d41236685e0c102e0ad23165e8b7157669819308ddcb396e29a"
    reg_target_hex = "a36f9d5c8dd82584bf46d3d86f76f55325f1d90c50846e3b12f7f066f2c015ad"
    reg_words = words_from_padded_55b_hex(reg_msg_hex)
    reg_dig, reg_tr, _ = sha256_reduced(reg_words, 20)
    assert fmt_words(reg_dig) == reg_target_hex
    assert reg_tr[5][0] == reg_tr[6][0] == 0x4ade1afa
    assert reg_tr[10][0] == reg_tr[11][0] == 0xacf3c908
    assert reg_tr[9][4] == reg_tr[10][4] == reg_tr[15][4] == 0xffffffff
    print(f"  [regression20] H20 = {fmt_words(reg_dig)} -> PASS")

    print("\nALL 46 DISTINCT WITNESSES (48 CHECKS INCLUDING 'abc' DEMO+BENCHMARK OVERLAP) VERIFIED 100%!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
