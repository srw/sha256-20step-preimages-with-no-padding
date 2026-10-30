# 20-Step SHA-256 Compression Preimages in $\sim 2^{33}$ via the Zero-$a_3$ Bi-Directional $Q_0/Q_1$ Bridge

*Technical note, 2026-10-13. Nothing here affects full 64-step SHA-256.*

## 1. Summary

We present an algebraic meet-in-the-middle / bi-directional bridge algorithm that finds full 256-bit
preimages for the **20-step SHA-256 compression function** (standard `IV`, steps 0–19, and standard
Davies–Meyer feed-forward, with 16 free message words $W_0 \dots W_{15}$) in **$\sim 2^{33}$
operations**.

In K. S. Basra's 20-round context-shaping attack (*Context Shaping for 19- and 20-Round SHA-256
Compression Preimages*, September 2026), solving the three 19-round schedule equations leaves a
32-bit bitwise residual condition $R = \mathrm{Maj}(a_5, a_4, a_3) - a_4 = 0$ that holds with
probability $(3/4)^{32} \approx 2^{-13.28}$, leading to an overall 20-round cost of $\sim 2^{45.4}$
table lookups (2.45 to 16.16 GPU-hours).

By moving to a **Zero-$a_3$ Manifold** ($a_3 = a_5 = a_6 = a_8 = a_9 = 0,\ a_{10} = \mathtt{0xffffffff}$)
and swapping the role of the $a_3$ Left pivot with the $a_7$ Right variable:
1. Both majority terms involving $a_4$ vanish **identically for all $a_4 \in [0, 2^{32})$**:
   $$\mathrm{Maj}(a_5, a_4, a_3) = 0, \qquad \mathrm{Maj}(a_6, a_5, a_4) = 0,$$
   eliminating the $2^{13.28}$ $\beta$-filter penalty completely.
2. In the first schedule constraint $\rho_{16} = 0$, the variable $a_7$ enters both $\sigma_1(W_{14})$
   and $W_9$ with signs that combine into $Q_1(u) = \sigma_1(u) + u$ at $u = D_{14}(a_{11}, a_{12}) - a_7$,
   allowing $a_7$ to be inverted in $O(1)$ per $(a_{11}, a_{12})$ pair.
3. All $48\text{ GiB}$ of lookup tables (`g_q0_inv`, `g_q1_inv`, and the Left Bridge `g_left_a1`)
   depend **only on `IV`** and are **100% independent of the 256-bit target digest $H^*$**.

On a 16-vCPU x86_64 Linux machine, building the tables took **59.41 s**, after which the solver
sustained **2.16 million exact 19-round solutions per second** and found a verified 20-round
preimage after **`1266.26 s` (21.1 minutes)** (`4,316,790,709` bridge matches $\approx 2^{32.01}$;
`2,735,408,772` exact 19-round solutions $\approx 2^{31.35}$).

---

## 2. Algebraic Formulation on the Zero-$a_3$ Manifold

Using the $a$-sequence representation (see the companion 19- and 20-step padded repository in
[`srw/sha256-19-and-20-step-preimages-with-padding`](https://github.com/srw/sha256-19-and-20-step-preimages-with-padding)):
$$e_t = a_t - \Sigma_0(a_{t-1}) - \mathrm{Maj}(a_{t-1}, a_{t-2}, a_{t-3}) + a_{t-4} \pmod{2^{32}},$$
$$W_t = a_{t+1} - \Sigma_0(a_t) - \mathrm{Maj}(a_t, a_{t-1}, a_{t-2}) - e_{t-3} - \Sigma_1(e_t) - \mathrm{Ch}(e_t, e_{t-1}, e_{t-2}) - K_t \pmod{2^{32}}.$$

For $R = 20$ steps:
- `IV` fixes $a_{-7} \dots a_0$ (and $e_{-3} \dots e_0$).
- The 256-bit target digest $H^*$ fixes $a_{13} \dots a_{20}$ (and $e_{13} \dots e_{16}$).
- The 12 internal state words are $a_1 \dots a_{12}$.
- A 20-step compression preimage is an assignment to $a_1 \dots a_{12}$ satisfying the four message
  schedule equations:
  $$\rho_t := W_t - \sigma_1(W_{t-2}) - W_{t-7} - \sigma_0(W_{t-15}) - W_{t-16} = 0 \pmod{2^{32}}, \qquad t \in \{16, 17, 18, 19\}.$$

### 2.1 Defining the Zero-$a_3$ Manifold

Fix the six internal words:
$$a_3 = 0, \quad a_5 = 0, \quad a_6 = 0, \quad a_8 = 0, \quad a_9 = 0, \quad a_{10} = \mathtt{0xffffffff}.$$
The remaining six active state words are:
- **Left / Bridge variables:** $(a_1, a_2)$
- **Level-3 pivot:** $a_4$
- **Right variables:** $(a_7, a_{11}, a_{12})$

**Lemma 1 (Exact Decoupling of $a_4$ with Zero $\beta$-Filter Loss).**
On the Zero-$a_3$ Manifold, for all $(a_1, a_2, a_4, a_7, a_{11}, a_{12}) \in (\mathbb{Z}/2^{32}\mathbb{Z})^6$:
1. $\mathrm{Maj}(a_5, a_4, a_3) = \mathrm{Maj}(0, a_4, 0) = 0$ and $\mathrm{Maj}(a_6, a_5, a_4) = \mathrm{Maj}(0, 0, a_4) = 0$.
2. $e_6 = a_2$, $e_7 = a_7$, $e_8 = -\Sigma_0(a_7) + a_4$, $e_9 = 0$, and $e_{10} = \mathtt{0xffffffff}$.
3. Because $e_9 = 0$ and $e_{10} = \mathtt{0xffffffff}$:
   - In $W_9$: $\mathrm{Ch}(e_9, e_8, e_7) = \mathrm{Ch}(0, e_8, e_7) = e_7 = a_7$, and $-e_6 = -a_2$. Thus
     $$W_9 = W_{9,0} - a_7 - a_2,$$
     which is **completely independent of $a_4, a_{11}, a_{12}$**.
   - In $W_{10}$: $\mathrm{Ch}(e_{10}, e_9, e_8) = \mathrm{Ch}(\mathtt{0xffffffff}, 0, e_8) = 0$, and $e_{11} - a_7 = a_{11} - \Sigma_0(a_{10})$ depends only on $a_{11}$. Since $-e_7 = -a_7$,
     $$W_{10} = W_{10,0}(a_{11}) - a_7,$$
     which is **completely independent of $a_1, a_2, a_4, a_{12}$**.
   - In $W_{11}$: $\mathrm{Ch}(e_{11}, e_{10}, e_9) = \mathrm{Ch}(e_{11}, \mathtt{0xffffffff}, 0) = e_{11}$, and $-e_8 = \Sigma_0(a_7) - a_4$. Thus
     $$W_{11} = W_{11,0}(a_7, a_{11}, a_{12}) - a_4.$$
   - Words $W_{12} \dots W_{19}$ do not depend on $a_1, a_2, a_4$ at all.

---

## 3. The $Q_1$-Cancellation Identity on $\rho_{16} = 0$

Define the two 32-bit operators:
$$Q_0(u) := \sigma_0(u) - u \pmod{2^{32}}, \qquad Q_1(u) := \sigma_1(u) + u \pmod{2^{32}}.$$

**Lemma 2 ($Q_1$ Inversion for $a_7$).**
On the Zero-$a_3$ Manifold, $W_{14}$ depends on $a_7$ only through $-e_{11} = -(e_{11,0}(a_{11}) + a_7)$:
$$W_{14}(a_7, a_{11}, a_{12}) = D_{14}(a_{11}, a_{12}) - a_7, \qquad D_{14}(a_{11}, a_{12}) := W_{14}(0, a_{11}, a_{12}).$$
Substituting $W_{14} = u := D_{14} - a_7$ (so $a_7 = D_{14} - u$) and $W_9 = W_{9,0} - a_7 - a_2$ into $\rho_{16} = 0$:
$$W_{16} - \sigma_1(u) - (W_{9,0} - (D_{14} - u) - a_2) - \sigma_0(W_1) - W_0 = 0.$$
Grouping all Right-hand terms on one side and all Left-hand terms (using $a_2 = W_1 + G_1(a_1)$) on the other:
$$\underbrace{W_{16}(a_{12}) + D_{14}(a_{11}, a_{12}) - W_{9,0}}_{\widetilde{C}_{16}(a_{11}, a_{12})} - \underbrace{(\sigma_1(u) + u)}_{Q_1(u)} = \underbrace{Q_0(W_1) + W_0(a_1) - G_1(a_1)}_{T_{16}(a_1, a_2)}.$$

Notice that when we evaluate $\widetilde{C}_{16}(a_{11}, a_{12}) = W_{16} + D_{14} - W_{9,0}$ at $a_7 = 0$, we do not even need to compute $\sigma_1(D_{14})$. For any fixed bridge target $T_{16}^*$:
$$Q_1(u) = \widetilde{C}_{16}(a_{11}, a_{12}) - T_{16}^* \implies u = Q_1^{-1}\!\left(\widetilde{C}_{16}(a_{11}, a_{12}) - T_{16}^*\right), \quad a_7 = D_{14}(a_{11}, a_{12}) - u.$$

---

## 4. The Bi-Directional Bridge Algorithm

1. **Phase 0 (Precompute $Q_0^{-1}$ and $Q_1^{-1}$, $32\text{ GiB}$):**
   Sweep $u \in [0, 2^{32})$ and populate `g_q0_inv[Q0(u)] = u` and `g_q1_inv[Q1(u)] = u`. Each map covers $\approx 63.37\%$ of $\mathbb{Z}/2^{32}\mathbb{Z}$.

2. **Phase 1 (Build Left Bridge Table `g_left_a1`, $16\text{ GiB}$):**
   Fix $T_{16}^* = \mathtt{0x12345678}$. For each $a_1 \in [0, 2^{32})$:
   - Look up $W_1 = Q_0^{-1}(T_{16}^* - W_0(a_1) + G_1(a_1))$.
   - If valid, set $a_2 = W_1 + G_1(a_1)$, compute $L_{17} = \sigma_0(W_2(a_1, a_2)) + W_1$, and store `g_left_a1[L17] = a1`.
   - Crucially, Phase 0 and Phase 1 depend **only on `IV` and $T_{16}^*$**, and are **completely independent of $H^*$**.

3. **Phase 2 (Right Sweep & Level-3 Exact Solve):**
   Sweep $(a_{11}, a_{12}) \in [0, 2^{32})^2$ across threads:
   - **Step 2a ($\rho_{16} = 0$):** Compute $\widetilde{C}_{16}(a_{11}, a_{12})$ at $a_7 = 0$, look up `u = g_q1_inv[C16 - T16*]`, and set $a_7 = D_{14} - u$.
   - **Step 2b ($\rho_{17} = 0$):** Compute $T_{17} = W_{17} - \sigma_1(W_{15}(a_7, a_{11}, a_{12})) - W_{10,0}(a_{11}) + a_7$ and look up `a1 = g_left_a1[T17]`. Every hit satisfies both $\rho_{16} = 0$ and $\rho_{17} = 0$.
   - **Step 2c ($\rho_{18} = 0$, zero $\beta$-filter loss):** Reconstruct $(W_1, a_2, W_2)$ from $a_1$, compute $T_{18} = W_{18} - \sigma_1(W_{16}) - W_{11,0}(a_7, a_{11}, a_{12})$, and look up `W3 = g_q0_inv[T18 - W2 + G3]`. Setting $a_4 = W_3 + G_3$ gives an **exact 19-round solution** ($\rho_{16} = \rho_{17} = \rho_{18} = 0$) with probability $1$.
   - **Step 2d ($\rho_{19} = 0$):** Check whether $\rho_{19} = 0$. Each exact 19-round solution satisfies $\rho_{19} = 0$ with probability $2^{-32}$.

---

## 5. Measured Performance and Verification

Executed on a 16-vCPU x86_64 VM (128 GiB RAM, `--q-bits 32 --left-bits 32 --threads 16`):

- **Table build (`Q0`, `Q1`):** `26.36 s`
- **Left Bridge (`g_left_a1`):** `33.05 s` (`2,721,604,389` valid entries $= 0.633673 \times 2^{32}$)
- **Phase 2 search:**
  - Sustained rate: **`2.16 million` exact 19-round solutions/s**
  - Bridge matches ($\rho_{16} = \rho_{17} = 0$): `4,316,790,709` ($\approx 2^{32.01}$)
  - Exact 19-round solutions ($\rho_{16} = \rho_{17} = \rho_{18} = 0$): `2,735,408,772` ($\approx 2^{31.35}$)
  - Empirical survival rate from bridge match to exact 19-round solution: `2,735,408,772 / 4,316,790,709 = 0.633667` (matching the theoretical image density of $Q_0$ and confirming **zero $\beta$-filter loss**)
  - Wall-clock search time to 20-round solution: **`1266.26 s` (`21.1 minutes`)**

---

## 6. Memoryless ($1.06\text{ MB}$) Split Carry-Tree $Q_0^{-1}/Q_1^{-1}$ Solver & $2^{16}$ 19-Step Solver

### 6.1 Eliminating the $32\text{ GiB}$ $Q_0^{-1}/Q_1^{-1}$ Tables and $81.1\%$ Single-Slot Overwrite Loss

In [`src/solve20_zero_a3.c`](../src/solve20_zero_a3.c), `g_q0_inv` and `g_q1_inv` are $16\text{ GiB}$
single-slot arrays storing only the last preimage $u$ per target $y$. Because $Q_b : \mathbb{Z}/2^{32}\mathbb{Z} \to \mathbb{Z}/2^{32}\mathbb{Z}$
is a random-like map with image density $1 - e^{-1} \approx 0.6321$, a single-slot table discards
$1 - (1 - e^{-1}) \approx 36.79\%$ of true preimages at each of the three $Q_0^{-1}/Q_1^{-1}$ steps
($W_1$, $a_7$, $W_3$), retaining only $(1 - e^{-1})^3 \approx 18.85\%$ of true 19-round solutions
(**$81.1\%$ overwrite loss**).

In [`src/carry_tree_bridge.c`](../src/carry_tree_bridge.c), we replace both $16\text{ GiB}$ tables with
an exact $1.06\text{ MB}$ **Split Carry-Tree + $\mathbb{F}_2$-Pseudoinverse Solver**:

1. **Exact Full-Adder / Full-Subtractor Identity:**
   Define the $\mathbb{F}_2$-linear operator $L_b(x) := \sigma_b(x) \oplus x$. Both $L_0$ and $L_1$
   have rank $31$ over $\mathbb{F}_2$ (1-dimensional kernel $\ker(L_b) = \{0, k_b\}$). For both
   $Q_1(x) = (\sigma_1(x) + x) \bmod 2^{32}$ (carries $c_0 = 0$) and $Q_0(x) = (\sigma_0(x) - x) \bmod 2^{32}$
   (borrows $c_0 = 0$), every bit $i \in \{0, \dots, 31\}$ satisfies:
   $$L_b(x) = y \oplus c,$$
   with carry/borrow transition:
   - For $Q_1(x) = y$: if $y_i \oplus c_i = 1$, then $c_{i+1} = c_i$ deterministically (**1 branch**, no constraint on $x_i$); if $y_i \oplus c_i = 0$, then $c_{i+1} \in \{0, 1\}$ with $x_i = c_{i+1}$ (**2 branches**, fixing $x_i$).
   - For $Q_0(x) = y$: if $y_i \oplus c_i = 0$, then $c_{i+1} = c_i$ deterministically (**1 branch**); if $y_i \oplus c_i = 1$, then $c_{i+1} \in \{0, 1\}$ with $x_i = c_{i+1}$ (**2 branches**, fixing $x_i$).

2. **16-Bit Split Carry-Tree + $\mathbb{F}_2$ Pseudoinverse ($1.06\text{ MB}$ Total):**
   By splitting bits $0 \dots 15$ and $16 \dots 31$, each 16-bit half-tree has an average of only
   $1.5^{16} \approx 656.8$ leaves. Using four $262\text{ KB}$ byte-sliced tables for the rank-31
   $\mathbb{F}_2$-pseudoinverse $M_b^+$ (`1.06 MB` total, built in `0.94 ms`), we match the two
   half-trees via a 16-bit bucket array on the compatibility mask of $x_{\text{lo}}$ (`~66 us/call`)
   and recover **all** preimages $\{u : Q_b(u) = y\}$ with **zero overwrite loss**:
   - Measured yield: `0.9923` preimages/query ($Q_0$) and `1.0066` ($Q_1$), versus `0.6298` and
     `0.6406` in single-slot $16\text{ GiB}$ arrays (**$5.31\times$ higher 3-step yield**).

### 6.2 Full 256-Bit 19-Step Compression Preimage in $2^{16}$ Steps & $2.56\text{ MB}$ RAM

At $R = 19$ steps (`Compress_19(IV, W)`), the 256-bit target digest $H^*$ fixes $a_{12} \dots a_{19}$
(and $e_{12} \dots e_{15}$). On the Zero-$a_3$ Manifold ($a_3 = a_5 = a_6 = a_8 = a_9 = 0,\ a_{10} = \mathtt{0xffffffff}$):
- $a_{11} \in [0, 2^{32})$ remains **completely free** on the Right side, and $a_7$ remains the
  $Q_1^{-1}$ pivot for $\rho_{16} = 0$.
- The three 19-step schedule constraints ($\rho_{16} = \rho_{17} = \rho_{18} = 0$) are solved via a
  balanced $2^{16}$ Meet-in-the-Middle cascade:
  1. **Left Table ($2^{16}$ states, $1.50\text{ MB}$ RAM):** Draw $\sim 2^{16}$ values of $a_1$,
     invert $W_1 \in Q_0^{-1}(T_{16}^* - W_0(a_1) + G_1(a_1))$ via the $1.06\text{ MB}$ Carry-Tree
     solver, and index $(a_1, W_1)$ by $L_{17} = (\sigma_0(W_2) + W_1) \bmod 2^{32}$.
  2. **Right Sweep ($\sim 2^{16}$ draws of $a_{11}$):** With $a_{12} \dots a_{18}$ fixed by $H^*$,
     sweep $a_{11}$, invert $u \in Q_1^{-1}(\widetilde{C}_{16}(a_{11}, a_{12}) - T_{16}^*)$, set
     $a_7 = D_{14}(a_{11}, a_{12}) - u$, look up $T_{17}$ in the $2^{16}$-entry Left table, and on a
     collision invert $W_3 \in Q_0^{-1}(T_{18} - W_2 + G_3)$ to obtain $a_4 = W_3 + G_3$.
- Executed on a single CPU core (`clang -O3 src/carry_tree_bridge.c`), this finds a verified full
  256-bit 19-step compression preimage for `TARGET_H` in **`6.81 s` total** (`65,536` Left entries +
  `38,655` Right $a_{11}$ draws $\approx 2^{16.67}$ calls) using **`2.56 MB` RAM**:
  ```text
  W[0..15] = 08dbd665 bcc8d4a9 880dadab 28ec520a 07bceda7 aa640717 911dbf63 a9733e2b
             40671db4 7522899e 94d81eec a784ec0e 6d5a6d20 7e7ad355 773ed868 21733e94
  Compress_19(IV, W) = deadbeefcafebabe0123456789abcdeffedcba987654321013579bdf2468ace0
  ```

### 6.3 The 18 / 19 / 20 / 21-Step Complexity Hierarchy ([`src/validate_sub32_cascades.py`](../src/validate_sub32_cascades.py))

| Steps $R$ | Target Equations | Target Words Fixed by $H^*$ | Active Manifold & Method | Time Complexity | Memory |
| :---: | :---: | :---: | :--- | :---: | :---: |
| **18** | 2 ($\rho_{16}, \rho_{17} = 0$) | $a_{11} \dots a_{18}$ | Fix $a_1 \dots a_6, a_8 \dots a_{10}$; $a_7$ free $\implies$ 1-Step $Q_1^{-1}$ (`[A]` in `validate_sub32_cascades.py`) | **$O(1)$ ($\approx 1.58$ trials)** | $1.06\text{ MB}$ |
| **19** | 3 ($\rho_{16}, \rho_{17}, \rho_{18} = 0$) | $a_{12} \dots a_{19}$ | Zero-$a_3$ Manifold; Left $a_1$ ($2^{16}$) $\bowtie$ Right $a_{11}$ ($2^{16}$) + $Q_0^{-1}$ for $a_4$ (`src/carry_tree_bridge.c`) | **$\sim 2^{16}$** | $2.56\text{ MB}$ |
| **20** | 4 ($\rho_{16} \dots \rho_{19} = 0$) | $a_{13} \dots a_{20}$ | Zero-$a_3$ Manifold; Left $a_1$ ($2^{32}$) $\bowtie$ Right $(a_{11}, a_{12})$ ($2^{32}$) + $Q_0^{-1}$ for $a_4$ (`src/solve20_zero_a3.c`) | **$\sim 2^{32}$** | $16\text{–}48\text{ GiB}$ (or $2^{64}/M$ with Carry-Tree) |
| **21** | 5 ($\rho_{16} \dots \rho_{20} = 0$) | $a_{14} \dots a_{21}$ | Zero-$a_3$ Manifold + $a_{13}$ free on Right ($\rho_{19}=\rho_{20}=0$ is a 64-bit filter) | **$\sim 2^{64}$** | $16\text{–}48\text{ GiB}$ |

---

## 7. Why Single-Block 20-Step Padded SHA-256 Requires a Different Regime

In the companion repository
[`srw/sha256-19-and-20-step-preimages-with-padding`](https://github.com/srw/sha256-19-and-20-step-preimages-with-padding),
we solved 19-step SHA-256 **with full FIPS 180-4 single-block padding** (55-byte messages, where
$W_{14} = 0$, $W_{15} = \mathtt{0x000001b8} = 440$, and $\mathrm{LSB}(W_{13}) = \mathtt{0x80}$) in
$\sim 2^{25.3}$ work on one CPU core. Why does moving from 19 steps to 20 steps change the padded
regime?

1. **Degree-of-freedom counting on the Zero-$a_3$ Manifold:**
   - The Zero-$a_3$ Manifold fixes 6 internal words ($a_3, a_5, a_6, a_8, a_9 = 0$, $a_{10} = \mathtt{0xffffffff}$),
     leaving 3 Left/Bridge words $(a_1, a_2, a_4)$ and 3 Right words $(a_7, a_{11}, a_{12})$.
   - For the **unpadded 20-step compression function**, the 4 schedule constraints
     $\rho_{16} = \rho_{17} = \rho_{18} = \rho_{19} = 0$ (128 bits) are met by the 6 active words
     (192 bits), leaving a $2^{64}$ solution manifold. Here $a_7$ is solved via $Q_1^{-1}$ from
     $\rho_{16} = 0$, while $(a_{11}, a_{12})$ provides the $2^{64}$ Right search space needed to
     hit the 32-bit filter $\rho_{19} = 0$ (and bridge match).
   - If we additionally impose single-block length padding $W_{14} = 0$ and $W_{15} = 440$ on the
     Zero-$a_3$ Manifold:
     - $W_{14}(a_7, a_{11}, a_{12}) = D_{14}(a_{11}, a_{12}) - a_7 = 0$ fixes $a_7 = D_{14}(a_{11}, a_{12})$.
     - Substituting $W_{14} = 0$ into $\rho_{16} = 0$ forces $D_{14}(a_{11}, a_{12}) = T_{16}^* + W_{9,0} - W_{16}(a_{12})$,
       and $W_{15}(a_7, a_{11}, a_{12}) = 440$ imposes a second 32-bit constraint on $(a_{11}, a_{12})$.
     - Together, $W_{14} = 0$ and $W_{15} = 440$ consume 64 bits of Right-side freedom, collapsing
       the Right search space on a fixed Zero-$a_3$ manifold from $2^{64}$ to a discrete set of size
       $O(1)$, whereas satisfying $\rho_{19} = 0$ plus $\mathrm{LSB}(W_{13}) = \mathtt{0x80}$ requires
       40 bits of filtering ($\sim 2^{40}$ trials).

2. **Resulting complexities for 20-step padded SHA-256:**
   - **Single-block 20-step padded ($\sim 2^{64}$):** By relaxing two of the fixed manifold words
     (or using general context shaping where $W_{14} = 0$ and $W_{15} = 440$ linearly absorb $a_7$
     and $a_8$ via $-e_{11}$ and $-e_{12}$), one restores the $2^{64}$ search space, at the cost of
     losing the simultaneous Zero-$a_3$ / $Q_1$ shortcut. This yields an expected complexity of
     $\sim 2^{64}$ (feasible on a GPU cluster, compared to $2^{256}$ brute force).
   - **Two-block 20-step padded ($\sim 2^{39}$):** For a two-block padded message, the first block
     has **no padding constraints** ($W_0 \dots W_{15}$ are 64 free bytes), while the second block
     is padded. Using a bi-directional meet-in-the-middle attack between Forward Block 1 and a
     padded Block 2, a 2-block 20-step padded preimage can be found in $\sim 2^{39}$ work.
