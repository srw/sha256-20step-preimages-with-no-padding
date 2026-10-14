# 20-Step SHA-256 Compression Preimages in $\sim 2^{33}$ via the Zero-$a_3$ Bi-Directional $Q_0/Q_1$ Bridge

*Technical note, 2026-10-13. Nothing here affects full 64-step SHA-256.*

## 1. Summary

We present an algebraic meet-in-the-middle / bi-directional bridge algorithm that finds full 256-bit
preimages for the **20-step SHA-256 compression function** (standard `IV`, steps 0–19, and standard
Davies–Meyer feed-forward, with 16 free message words $W_0 \dots W_{15}$) in **$\sim 2^{33}$
operations**.

In K. S. Basra's 20-round context-shaping attack (*Context Shaping for 19- and 20-Round SHA-256
Compression Preimages*, September 2026), solving the three 19-round schedule equations leaves a
32-bit bitwise residual condition $R = \operatorname{Maj}(a_5, a_4, a_3) - a_4 = 0$ that holds with
probability $(3/4)^{32} \approx 2^{-13.28}$, leading to an overall 20-round cost of $\sim 2^{45.4}$
table lookups (2.45 to 16.16 GPU-hours).

By moving to a **Zero-$a_3$ Manifold** ($a_3 = a_5 = a_6 = a_8 = a_9 = 0,\ a_{10} = \texttt{0xffffffff}$)
and swapping the role of the $a_3$ Left pivot with the $a_7$ Right variable:
1. Both majority terms involving $a_4$ vanish **identically for all $a_4 \in [0, 2^{32})$**:
   $$\operatorname{Maj}(a_5, a_4, a_3) = 0, \qquad \operatorname{Maj}(a_6, a_5, a_4) = 0,$$
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

Using the $a$-sequence representation (see the companion 19-step note in
[`srw/sha256-19step-preimages-with-padding`](https://github.com/srw/sha256-19step-preimages-with-padding)):
$$e_t = a_t - \Sigma_0(a_{t-1}) - \operatorname{Maj}(a_{t-1}, a_{t-2}, a_{t-3}) + a_{t-4} \pmod{2^{32}},$$
$$W_t = a_{t+1} - \Sigma_0(a_t) - \operatorname{Maj}(a_t, a_{t-1}, a_{t-2}) - e_{t-3} - \Sigma_1(e_t) - \operatorname{Ch}(e_t, e_{t-1}, e_{t-2}) - K_t \pmod{2^{32}}.$$

For $R = 20$ steps:
- `IV` fixes $a_{-7} \dots a_0$ (and $e_{-3} \dots e_0$).
- The 256-bit target digest $H^*$ fixes $a_{13} \dots a_{20}$ (and $e_{13} \dots e_{16}$).
- The 12 internal state words are $a_1 \dots a_{12}$.
- A 20-step compression preimage is an assignment to $a_1 \dots a_{12}$ satisfying the four message
  schedule equations:
  $$\rho_t := W_t - \sigma_1(W_{t-2}) - W_{t-7} - \sigma_0(W_{t-15}) - W_{t-16} = 0 \pmod{2^{32}}, \qquad t \in \{16, 17, 18, 19\}.$$

### 2.1 Defining the Zero-$a_3$ Manifold

Fix the six internal words:
$$a_3 = 0, \quad a_5 = 0, \quad a_6 = 0, \quad a_8 = 0, \quad a_9 = 0, \quad a_{10} = \texttt{0xffffffff}.$$
The remaining six active state words are:
- **Left / Bridge variables:** $(a_1, a_2)$
- **Level-3 pivot:** $a_4$
- **Right variables:** $(a_7, a_{11}, a_{12})$

**Lemma 1 (Exact Decoupling of $a_4$ with Zero $\beta$-Filter Loss).**
On the Zero-$a_3$ Manifold, for all $(a_1, a_2, a_4, a_7, a_{11}, a_{12}) \in (\mathbb{Z}/2^{32}\mathbb{Z})^6$:
1. $\operatorname{Maj}(a_5, a_4, a_3) = \operatorname{Maj}(0, a_4, 0) = 0$ and $\operatorname{Maj}(a_6, a_5, a_4) = \operatorname{Maj}(0, 0, a_4) = 0$.
2. $e_6 = a_2$, $e_7 = a_7$, $e_8 = -\Sigma_0(a_7) + a_4$, $e_9 = 0$, and $e_{10} = \texttt{0xffffffff}$.
3. Because $e_9 = 0$ and $e_{10} = \texttt{0xffffffff}$:
   - In $W_9$: $\operatorname{Ch}(e_9, e_8, e_7) = \operatorname{Ch}(0, e_8, e_7) = e_7 = a_7$, and $-e_6 = -a_2$. Thus
     $$W_9 = W_{9,0} - a_7 - a_2,$$
     which is **completely independent of $a_4, a_{11}, a_{12}$**.
   - In $W_{10}$: $\operatorname{Ch}(e_{10}, e_9, e_8) = \operatorname{Ch}(\texttt{0xffffffff}, 0, e_8) = 0$, and $e_{11} - a_7 = a_{11} - \Sigma_0(a_{10})$ depends only on $a_{11}$. Since $-e_7 = -a_7$,
     $$W_{10} = W_{10,0}(a_{11}) - a_7,$$
     which is **completely independent of $a_1, a_2, a_4, a_{12}$**.
   - In $W_{11}$: $\operatorname{Ch}(e_{11}, e_{10}, e_9) = \operatorname{Ch}(e_{11}, \texttt{0xffffffff}, 0) = e_{11}$, and $-e_8 = \Sigma_0(a_7) - a_4$. Thus
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
   Fix $T_{16}^* = \texttt{0x12345678}$. For each $a_1 \in [0, 2^{32})$:
   - Look up $W_1 = Q_0^{-1}(T_{16}^* - W_0(a_1) + G_1(a_1))$.
   - If valid, set $a_2 = W_1 + G_1(a_1)$, compute $L_{17} = \sigma_0(W_2(a_1, a_2)) + W_1$, and store `g_left_a1[L17] = a1`.
   - Crucially, Phase 0 and Phase 1 depend **only on `IV` and $T_{16}^*$**, and are **completely independent of $H^*$**.

3. **Phase 2 (Right Sweep & Level-3 Exact Solve):**
   Sweep $(a_{11}, a_{12}) \in [0, 2^{32})^2$ across threads:
   - **Step 2a ($\rho_{16} = 0$):** Compute $\widetilde{C}_{16}(a_{11}, a_{12})$ at $a_7 = 0$, look up $u = \text{g\_q1\_inv}[\widetilde{C}_{16} - T_{16}^*]$, and set $a_7 = D_{14} - u$.
   - **Step 2b ($\rho_{17} = 0$):** Compute $T_{17} = W_{17} - \sigma_1(W_{15}(a_7, a_{11}, a_{12})) - W_{10,0}(a_{11}) + a_7$ and look up $a_1 = \text{g\_left\_a1}[T_{17}]$. Every hit satisfies both $\rho_{16} = 0$ and $\rho_{17} = 0$.
   - **Step 2c ($\rho_{18} = 0$, zero $\beta$-filter loss):** Reconstruct $(W_1, a_2, W_2)$ from $a_1$, compute $T_{18} = W_{18} - \sigma_1(W_{16}) - W_{11,0}(a_7, a_{11}, a_{12})$, and look up $W_3 = \text{g\_q0\_inv}[T_{18} - W_2 + G_3]$. Setting $a_4 = W_3 + G_3$ gives an **exact 19-round solution** ($\rho_{16} = \rho_{17} = \rho_{18} = 0$) with probability $1$.
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
