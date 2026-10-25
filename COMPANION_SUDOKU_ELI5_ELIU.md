# Carry-Sudoku & Slot-Sudoku in Reduced SHA-256: A Self-Contained ELI5 and ELI-Undergraduate Companion Guide

**Author:** Sebastian Wain (`srw@egont.xyz`)  
**Companion Pedagogical & Reference Document to:**  
*Practical Preimages for 19- and 20-Step Reduced SHA-256: Exact Algebraic Padding Inversion, Context Shaping, and Causal Bit-Slice Pruning* (`PAPER_19_20_PADDED_SHA256.pdf`)

---

## Executive Overview: Why a "Sudoku" Guide to SHA-256?

Cryptographic hash functions like SHA-256 are designed to work like one-way blenders: you drop a message in at the top, turn the crank for 64 rounds of bitwise mixing and 32-bit addition, and get a 256-bit fingerprint at the bottom. Trying to run the blender backward—starting from a target fingerprint and finding a validly padded input message that produces it (*a preimage attack*)—looks impossible by brute force, because there are $2^{256}$ possible target states.

However, when SHA-256 is reduced to **19 steps (padded)** or **20 steps (padded)**, you don't have to guess the whole message at once, and you don't have to fight the blender head-on. Instead, you can treat the internal step-by-register grid of SHA-256 like a **giant interconnected Sudoku puzzle**:

1. **You pre-fill strategic "givens" (fixed cells) in the middle of the board** (`e9 = e10 = e15 = 0xffffffff`, `a5 = a6`, `a10 = a11`) so that nonlinear logic gates ($\operatorname{Ch}$ and $\operatorname{Maj}$) turn into simple pass-through wires or constants (`[E8]`–`[E13]`, `[E22]`).
2. **You solve the padding byte (`0x80`) in $O(1)$ time and solve the arithmetic carry chains bit-slice by bit-slice from right to left (`Carry-Sudoku`)** (`[E18]`–`[E25]`), where knowing the rightmost bits of $e_{13}$ immediately forces the carries and eliminates $99.74\%$ of invalid branches before the heavy rotations are even computed.
3. **You invert the coupled schedule operator $Q_0(u) = \sigma_0(u) - u$ across 3 chained steps using a zero-table closed-form $\mathbb{F}_2$ inverse and a packed hash table (`Slot-Sudoku` / Cuckoo-Sudoku)** (`[E27]`–`[E37]`), eliminating memory bottlenecks on both Apple M3 CPUs and NVIDIA L4 GPUs.

This companion guide is organized so that **anyone**—from a curious reader to a cryptography undergraduate or systems engineer—can understand every single piece of the breakthrough:
- **Section 0 (Master Glossary & Complete `[E1]`–`[E45]` Index):** A 100% self-contained reference explaining every symbol, operator, C/CUDA variable, and numbered equation (`[E1]` through `[E45]`) in the main paper.
- **Section 1 — The Scalar $a$-Sequence Grid (`[E1]`–`[E7]`):** How the 8-register SHA-256 state collapses into a single 1D Sudoku strip $a_{-7}, \dots, a_R$, and why FIPS 180-4 padding (`0x80`, zero fill, length `440` or `952`) locks the board (*ELI5* and *ELI-Undergraduate*).
- **Section 2 — Selector Saturation & Exact Context Shaping $D=0$ (`[E8]`–`[E14]`):** How locking $e_9 = e_{10} = \mathtt{0xffffffff}$ and forcing twin cells $a_5 = a_6$ ($D = a_5 \oplus a_6 = 0$) decouples the Left and Right halves of the cipher (*ELI5* and *ELI-Undergraduate*).
- **Section 3 — Exact $O(1)$ Algebraic Padding Inversion (`[E15]`–`[E22]`):** How to absorb $W_{15}=440$ and $W_{14}=0$ in closed form and solve $(W_{13} \bmod 256) = \mathtt{0x80}$ in $O(1)$ time with zero trial-and-error (*ELI5* and *ELI-Undergraduate*).
- **Section 4 — `Carry-Sudoku`: 7-Stage Causal Bit-Slice Right-Context Solver (`[E23]`–`[E25]`):** How Theorem 1 splits $a_5 = a_6$ into a lower-triangular carry check independent of $a_6$, reducing $2^{24}$ candidates per $a_{12}$ block to an average of $43{,}580$ node evaluations ($385\times$ reduction) with $100\%$ completeness and reaching **`266.8 ctx/s`** on CPU and **`114,507.0 exact $D=0$ contexts/s`** on 1× NVIDIA L4 GPU (*ELI5* and *ELI-Undergraduate*).
- **Section 5 — `Slot-Sudoku` & Zero-Table Closed-Form $\mathbb{F}_2$ Inverse (`[E27]`–`[E37]`):** How Theorem 2 reconstructs the upper 16 bits in closed form (`reconstruct_u`), how 6 GiB / 16 GiB / 32 GiB `Slot-Sudoku` tables work, and how every 19-step bridge candidate (`sol19`) is filtered in $O(1)$ via $\sigma_0^{-1}$ to find the first 20-step padded preimages (*ELI5* and *ELI-Undergraduate*).
- **Section 6 — GPU Warp-Sudoku, Exact-$T_{16}$ Twin Coalescing & The Zero-$a_3$ Unpadded Bridge (`[E38]`–`[E45]`):** How our 16-thread CPU solver reaches **`6,242.8 sol19/s`** (**`191.1 node-hours`** per 20-step padded preimage on a 10-core Apple M4, or **$2^{46.56}$ scalar CPU reference compressions**), how supplementary GPU acceleration in $a_1$-slice sweeps on 1× NVIDIA L4 reaches **`53,847.5–73,560.5 sol19/s`** (`16.22–22.16 GPU-h` projected) on the 6 GiB EDF table and **`132,888.7 sol19/s`** (**`8.98 GPU-hours` projected** on 1× L4, or **`1.12 hours` across 8× L4 GPUs**) on genuine birthday-collided $T_{16}$ context pairs, and how setting $a_3=0$ solves 20-step unpadded compression preimages in 21.1 minutes observed on CPU (*ELI5* and *ELI-Undergraduate*).
- **Section 7 — Scope, "Restricted" Preimages, Two-Block Plain-ASCII Preimages, and 15 Explored Sub-$2^{32}$ Directions:** Why locking Sudoku cells doesn't throw away the solution ($440 \to 184 \to 74.6\text{–}78.8$ free bits), how a **2-block 119-byte message** lets you choose **any 64-byte plain-ASCII first block** in $O(1)$ while **Direction 15 (Two-Block Feed-Forward Locking)** explains why 2 blocks cannot reduce Round 20 below $2^{32}$, and the taxonomy of **15 explored sub-$2^{32}$ directions** (*ELI5* and *ELI-Undergraduate*).

---

## Section 0: Master Self-Contained Glossary & Complete `[E1]`–`[E45]` Index

### 0.1 Core Symbols, Operators, and Code Variables

| Symbol / Code Name | Name | Self-Contained Meaning in Plain English & Mathematics |
| :--- | :--- | :--- |
| $+, -$ (modulo $2^{32}$) | 32-bit Modular Arithmetic | Addition and subtraction on 32-bit unsigned integers (`uint32_t` in C/CUDA), wrapping around modulo $2^{32}$. |
| $\oplus, \wedge, \vee, \neg$ | Bitwise Logic Operators | Bit-by-bit XOR (`^`), AND (`&`), OR (`\|`), and bitwise NOT (`~`) across 32 bits ($b \in \{0,\dots,31\}$). |
| $\operatorname{ROR}(x,n), \operatorname{SHR}(x,n)$ | Right Rotate & Right Shift | Circular right rotation by $n$ bits (`(x >> n) \| (x << (32-n))`) and logical right shift (`x >> n`). |
| $\operatorname{Ch}(e,f,g)$ | Bitwise "Choose" Gate | $(e \wedge f) \oplus (\neg e \wedge g)$. At each bit $b$, if $e_b = 1$, output $f_b$; if $e_b = 0$, output $g_b$. Setting $e = \mathtt{0xffffffff}$ forces $\operatorname{Ch}(e,f,g) = f$, erasing $g$! |
| $\operatorname{Maj}(a,b,c)$ | Bitwise "Majority" Gate | $(a \wedge b) \oplus (a \wedge c) \oplus (b \wedge c)$. Outputs the bit (`0` or `1`) that appears at least twice. If two inputs are equal ($a=b$), $\operatorname{Maj}(a,a,c) = a$, erasing $c$! |
| $\Sigma_0(x), \Sigma_1(x)$ | State Mixing Bijections | Invertible $\mathbb{F}_2$-linear state mixers: $\Sigma_0(x) = \operatorname{ROR}(x,2)\oplus\operatorname{ROR}(x,13)\oplus\operatorname{ROR}(x,22)$ and $\Sigma_1(x) = \operatorname{ROR}(x,6)\oplus\operatorname{ROR}(x,11)\oplus\operatorname{ROR}(x,25)$. |
| $\sigma_0(x), \sigma_1(x)$ | Schedule Mixing Maps | $\mathbb{F}_2$-linear message mixers: $\sigma_0(x) = \operatorname{ROR}(x,7)\oplus\operatorname{ROR}(x,18)\oplus\operatorname{SHR}(x,3)$ (invertible over $\mathbb{F}_2^{32}$!) and $\sigma_1(x) = \operatorname{ROR}(x,17)\oplus\operatorname{ROR}(x,19)\oplus\operatorname{SHR}(x,10)$. |
| $Q_0(u), Q_1(u)$ | Coupled Schedule Operators | $Q_0(u) := \sigma_0(u) - u \pmod{2^{32}}$ and $Q_1(u) := \sigma_1(u) + u \pmod{2^{32}}$. Used to invert the message schedule across steps 16, 17, 18. |
| $a_{-7}, \dots, a_R$ | Scalar $a$-Sequence | Because 6 of the 8 SHA-256 registers simply shift each round, the entire 8-register trajectory is uniquely determined by the single sequence of $a_s$ words (`[E5]`–`[E7]`). |
| $D_{\text{shape}} = a_5 \oplus a_6$ | Context Shaping XOR | Measures how many bits differ between $a_5$ and $a_6$. Exact shaping sets $D_{\text{shape}} = 0 \iff a_5 = a_6$ (`[E11]`). |
| $M_3 = \neg(a_3 \oplus a_5)$ | Level-3 Majority Mask | The bitmask of positions where $a_3$ and $a_5$ agree. On these bits, $a_4$ must equal $a_5$ so that $\operatorname{Maj}(a_5,a_4,a_3) = a_4$ (`[E13]`). |
| $D = a_8 \oplus a_7$ | Carry-Sudoku Difference Mask | Inside Theorem 1 (`[E23]`), $D = a_8 \oplus a_7$ separates bits where $\operatorname{Maj}(a_8,a_7,a_6)$ depends on $a_6$ ($D_b=1$) from bits where it equals $a_{8,b}$ ($D_b=0$). |

---

### 0.2 Complete Cross-Reference of Equations `[E1]`–`[E45]` in `PAPER_19_20_PADDED_SHA256.pdf`

- **`[E1]` (FIPS 180-4 Single-Block 55-Byte Padding):** Fixes $W_{15} = 440$ (`0x000001b8`), $W_{14} = 0$ (`0x00000000`), and $W_{13} \bmod 256 = \mathtt{0x80}$ (72 locked padding bits; or $W_{15} = 952 = \mathtt{0x000003b8}$ for a 2-block 119-byte padded message).
- **`[E2]` (SHA-256 Boolean and Linear Functions):** Defines $\operatorname{Ch}$, $\operatorname{Maj}$, $\Sigma_0, \Sigma_1$, and $\sigma_0, \sigma_1$.
- **`[E3]` (Message Schedule Expansion):** $W_t = \sigma_1(W_{t-2}) + W_{t-7} + \sigma_0(W_{t-15}) + W_{t-16}$ for $t \ge 16$.
- **`[E4]` (Coupled Schedule Operator $Q_0$):** Defines $Q_0(u) = \sigma_0(u) - u \pmod{2^{32}}$.
- **`[E5]` (Initial Boundary Conditions):** Fixes $(a_0, a_{-1}, a_{-2}, a_{-3})$ and $(e_0, e_{-1}, e_{-2}, e_{-3})$ from $\mathrm{IV}$.
- **`[E6]` (Terminal Boundary Conditions):** Fixes $a_{12}\dots a_{19}$ (for $R=19$) or $a_{13}\dots a_{20}$ (for $R=20$) from $H^* - \mathrm{IV}$.
- **`[E7]` (Scalar $a$-Sequence Recovery):** Expresses every $e_s$ as a deterministic function of 5 consecutive $a$-words ($a_{s-4}\dots a_s$) and every $W_t$ via $e_{t+1}, e_t, e_{t-1}, e_{t-2}, e_{t-3}, a_{t-3}$ (spanning 9 consecutive $a$-words $a_{t-7}\dots a_{t+1}$ when fully expanded in $a$), leaving only $a_1 \dots a_{12}$ free for $R=20$ ($a_1 \dots a_{11}$ for $R=19$).
- **`[E8]` (Selector Saturation):** Pins $e_9 = \mathtt{0xffffffff}$ and $e_{10} = \mathtt{0xffffffff}$ to block $e_7, e_8$ from leaking rightward through $\operatorname{Ch}$.
- **`[E9]` (Right-Only Expressions for $a_5, a_6$):** Expresses $a_5$ and $a_6$ purely in terms of the Right words $(a_7, a_8, a_9, a_{10})$.
- **`[E10]` (Collapsed Left Dependence of $W_8 \dots W_{12}$):** Shows how $W_8, W_9, W_{10}, W_{11}, W_{12}$ depend on $(a_1, a_2, a_3, a_4)$: $W_8 = c_8 - a_1 - e_5(a_1,\dots,a_5)$, $W_9 = c_9 - a_2 - \big(a_4 - \operatorname{Maj}(a_5,a_4,a_3)\big)$, $W_{10} = c_{10} - a_3 - \big(a_5 - \operatorname{Maj}(a_6,a_5,a_4)\big)$, $W_{11} = c_{11} - a_4$, and $W_{12} = c_{12}$ (Right-only).
- **`[E11]`–`[E12]` (Exact Context Shaping $a_5 = a_6$):** Setting $D_{\text{shape}} = a_5 \oplus a_6 = 0$ (`[E11]`) forces $\operatorname{Maj}(a_6, a_5, a_4) = a_5$ (`[E12]`), making $W_{10} = c_{10} - a_3$ 100% free of $a_4$ leakage.
- **`[E13]` (Majority Cancellation in $W_9$):** Enforces $\operatorname{Maj}(a_5, a_4, a_3) = a_4 \iff ((a_4 \oplus a_5) \wedge M_3) = 0$ with $M_3 = \neg(a_3 \oplus a_5)$, making $W_9 = c_9 - a_2$ 100% free of $a_3, a_4$ leakage.
- **`[E14]` (Three Chained $Q_0$-Inversions):** Rewrites the schedule equations at $t=16, 17, 18$ as Level 1 ($Q_0(W_1) = T_{16} + F_1(a_1)$), Level 2 ($Q_0(W_2) = T_{17} + G_2(a_1,a_2) - W_1$), and Level 3 ($Q_0(W_3) = T_{18} + G_3(a_1,a_2,a_3) - W_2$).
- **`[E15]`–`[E17]` (Closed-Form Absorption of $W_{15}=440, W_{14}=0$):** Computes $e_{16}, e_{15}, e_{14}$ (`[E15]`), solves $e_{12}, e_{11}$ in closed form from $e_{13}$ (`[E16]`), and recovers $a_8, a_7$ (`[E17]`).
- **`[E18]`–`[E21]` (Exact $O(1)$ Pivot Inversion of $W_{13} \bmod 256 = \mathtt{0x80}$):** Uses the bit-disjointness of $\Sigma_1(e_{13}) \bmod 256$ on bit 18 and bits 25..31 (`[E18]`–`[E20]`) to pin those 8 bits in closed form (`[E21]`) for any choice of bits 0..17 and 19..24.
- **`[E22]` ($\operatorname{Ch}$-Annihilation & Twin Majority Simplification):** Pins $e_{15}=\mathtt{0xffffffff}$ and $a_{10}=a_{11}$, making $e_{12}$ and $a_8$ constant across all $2^{24}$ choices of $e_{13}$.
- **`[E23]` (Theorem 1 — Carry-Sudoku Exact Equivalence):** Proves $a_5 = a_6 \iff (((S \oplus X) \wedge D) = 0) \wedge (((a_6 \oplus S) \wedge \neg D) = 0)$, where $S = X + \operatorname{Ch}(D, X, a_8)$ is 100% independent of $a_6$ and $\Sigma_0(a_9)$.
- **`[E24]`–`[E25]` (7-Stage Causal Bit-Slice Pruning):** Checks $((S \oplus X) \wedge D) = 0$ across 7 LSB-to-MSB bit-slices (`[E24]`–`[E25]`), reducing $2^{24}$ candidates per $a_{12}$ block to an average of $43{,}580$ node evaluations ($385\times$ reduction) with $100\%$ completeness. *(Note: equation index `[E26]` is retired so external references to `[E27]`–`[E45]` remain invariant.)*
- **`[E27]`–`[E29]` (Theorem 2 — Closed-Form $\mathbb{F}_2$ Inverse `reconstruct_u`):** Solves the upper 16 bits $u_{\text{hi}}$ from $(y, u_{\text{lo}})$ in closed form with zero lookup tables.
- **`[E30]`–`[E33]` (`Slot-Sudoku` Table Designs):** Defines the 6.00 GiB 2-Tier Nested 4-Slot EDF + Cuckoo-Sudoku table (`[E30]`–`[E31]`, 82.98% coverage), the 16.00 GiB Direct-Indexed 2-Way Exact GPU table (`[E32]`, storing up to 2 valid roots per $y$, 89.79% unconstrained coverage, zero false checks), and the 32.00 GiB 4-Way Exact CPU table (`[E33]`, 99.59% unconstrained coverage; 99.57% when reserving `u_lo = 0` as an empty-slot sentinel).
- **`[E34]`–`[E35]` (Early Hamming Pruning & 64-Bit SWAR Level-3 Filter):** Prunes $\operatorname{popcount}(M_3) \le w_{\text{thresh}}$ (`[E34]`) and tests 4 Level-3 slots simultaneously in one 64-bit register word (`[E35]`).
- **`[E36]`–`[E37]` (Terminal 20-Step Filter & $\sigma_0^{-1}$ Bijection):** Filters the 19-step padded bridge candidate stream (`sol19`) via $\sigma_0(W_4) + W_3 \equiv T_{19} \pmod{2^{32}}$ (`[E36]`) and proves $\det_{\mathbb{F}_2}(\sigma_0) = 1 \implies W_4 = \sigma_0^{-1}(T_{19}-W_3)$ (`[E37]`).
- **`[E38]`–`[E41]` (Zero-$a_3$ Bi-Directional $Q_0/Q_1$ Bridge for Unpadded 20-Step Compression):** Pins $a_3=a_5=a_6=a_8=a_9=0, a_{10}=\mathtt{0xffffffff}$ (`[E38]`) to annihilate both majority gates (`[E39]`) and solves the 20-step compression function (`[E40]`–`[E41]`) in 21.1 minutes observed on 16 CPU cores ($33.1\text{ min}$ expected).
- **`[E42]` (L2-Tile Range Pre-Filter):** For an L2-resident slice of `Q0_TABLE` covering $y_1 \in [\text{Tile}_{\text{lo}}, \text{Tile}_{\text{lo}} + \Delta_{\text{tile}})$, tests $(T_{16} + F_1(a_1) - \text{Tile}_{\text{lo}}) \bmod 2^{32} < \Delta_{\text{tile}}$ in registers before issuing any global memory read.
- **`[E43]` (Exact-$T_{16}$ Twin Coalescing `twin2` & 4-Way Register-Prefetched Reservoir Sweep `single4`):** Pairs genuine contexts with birthday collisions $T_{16}^{(i)} = T_{16}^{(j)}$ (`twin2`) to share Level-1 lookups, or sweeps 4 distinct contexts simultaneously (`single4`) across the estimated universe of $U \approx 1.68 \times 10^7$ contexts.
- **`[E44]` (Low-16-Bit Hamming & Majority Pre-Filter Before `reconstruct_u`):** Rejects $59.82\%$ (at $w_{\text{lo,thresh}}=7$) or $3.84\%$ (at $w_{\text{lo,thresh}}=11$) of Level-2 candidates via $\operatorname{popcount}(M_3 \wedge \mathtt{0xffff}) \le w_{\text{lo,thresh}}$, and rejects $\approx 93\%$ ($\approx 15\times$ reduction) of Level-3 Slot-1/Slot-2 candidates via $((a_4 \oplus a_5) \wedge M_3 \wedge \mathtt{0xffff}) = 0$ using only the low 16 bits $u_{\text{lo}}$ before running `reconstruct_u`.
- **`[E45]` (4-Round Crossed Carry-Sudoku Spine $a_2 = a_3 = a_4 = a_5$):** Evaluates 4 consecutive majority-collapsed rounds (`a2 = a3 = a4 = a5`) directly in GPU registers with zero DRAM tables (`9,617.2M` specialized crossed-spine evaluations/s on 1× L4 GPU).

---

## Section 1: The Scalar $a$-Sequence Grid (`[E1]`–`[E7]`)

### 1.1 Explain Like I'm 5 (ELI5): Turning an 8-Lane Highway into a Single Strip of Paper

If you look at a textbook diagram of SHA-256, it looks terrifying: there are **8 parallel lanes of numbers** ($a, b, c, d, e, f, g, h$) scrambling into each other for 20 steps, plus 16 message slots ($W_0 \dots W_{15}$) that expand into 4 recycled slots ($W_{16} \dots W_{19}$).

How do we turn that 8-lane mess into a clean Sudoku strip?
Look closely at what happens to the 8 lanes in each step:
- Lane $b$ is just a copy of lane $a$ from 1 step ago! Lane $c$ is lane $a$ from 2 steps ago! Lane $d$ is lane $a$ from 3 steps ago!
- Similarly, lanes $f, g, h$ are just copies of lane $e$ from 1, 2, and 3 steps ago!
- Even better: lane $e$ itself is completely determined by lane $a$ (`[E7]`)!

That means **there aren't really 8 independent lanes at all**—there is only **one single strip of numbers**: $a_{-7}, a_{-6}, \dots, a_{20}$!
- The top 8 cells of the strip ($a_{-7} \dots a_0$) are pre-filled "givens" printed on the board by the SHA-256 starting value $\mathrm{IV}$ (`[E5]`).
- The bottom 8 cells of the strip ($a_{13} \dots a_{20}$ for 20 rounds, or $a_{12} \dots a_{19}$ for 19 rounds) are pre-filled "givens" locked by the target hash $H^*$ (`[E6]`).
- **The entire 20-round SHA-256 puzzle comes down to filling in just 12 empty cells in the middle of the strip: $a_1, a_2, \dots, a_{12}$** (or 11 cells $a_1 \dots a_{11}$ for 19 rounds) so that the padding rules (`[E1]`) and the 4 recycled message rules (`[E3]`) hold!

---

### 1.2 Explain Like I'm an Undergraduate (ELIU): Derivation of the Scalar $a$-Sequence (`[E5]`–`[E7]`)

In SHA-256, the step update at round $t \in \{0, \dots, R-1\}$ computes two new words:

$$T_{1,t} = h_t + \Sigma_1(e_t) + \operatorname{Ch}(e_t, f_t, g_t) + K_t + W_t, \qquad T_{2,t} = \Sigma_0(a_t) + \operatorname{Maj}(a_t, b_t, c_t),$$

and sets $a_{t+1} = T_{1,t} + T_{2,t}$ and $e_{t+1} = d_t + T_{1,t}$, while shifting $b_{t+1}=a_t, c_{t+1}=b_t, d_{t+1}=c_t$ and $f_{t+1}=e_t, g_{t+1}=f_t, h_{t+1}=g_t$.
Subtracting $T_{1,t} = a_{t+1} - T_{2,t}$ from $e_{t+1} = d_t + T_{1,t}$ eliminates $T_{1,t}$ and $W_t$ completely, giving the first equation of `[E7]` (with index $s = t+1$ and $d_t = a_{s-4}$):

$$e_s = a_s - \Sigma_0(a_{s-1}) - \operatorname{Maj}(a_{s-1}, a_{s-2}, a_{s-3}) + a_{s-4}.$$

Once $e_s$ is known for all $s$, solving $e_{t+1} = d_t + h_t + \Sigma_1(e_t) + \operatorname{Ch}(e_t, e_{t-1}, e_{t-2}) + K_t + W_t$ for $W_t$ gives the second equation of `[E7]` (where $d_t = a_{t-3}$ and $h_t = e_{t-3}$):

$$W_t = e_{t+1} - a_{t-3} - e_{t-3} - \Sigma_1(e_t) - \operatorname{Ch}(e_t, e_{t-1}, e_{t-2}) - K_t.$$

Therefore, once the boundary cells $a_{-7}\dots a_0$ (`[E5]`) and $a_{R-7}\dots a_R$ (`[E6]`) are fixed, **any** choice of the 12 interior words $(a_1, \dots, a_{12})$ defines a valid 20-step SHA-256 computation hashing to $H^*$. We only need $(a_1, \dots, a_{12})$ to satisfy:
1. The **72-bit FIPS 180-4 padding constraint (`[E1]`)**: $W_{15} = 440$, $W_{14} = 0$, and $W_{13} \bmod 256 = \mathtt{0x80}$ (72 bit constraints).
2. The **4 message schedule recurrence equations (`[E3]`)** for $t = 16, 17, 18, 19$ ($4 \times 32 = 128$ bit constraints).

---

## Section 2: Selector Saturation & Exact Context Shaping $D=0$ (`[E8]`–`[E14]`)

### 2.1 Explain Like I'm 5 (ELI5): Building a Soundproof Wall in the Middle of the Board

We have 12 empty cells ($a_1 \dots a_{12}$) to fill in:
- Think of the first 4 cells **$(a_1, a_2, a_3, a_4)$** as **Team Left**.
- Think of the last 8 cells **$(a_5, a_6, a_7, a_8, a_9, a_{10}, a_{11}, a_{12})$** as **Team Right**.

Why can't Team Right just solve the padding slots ($W_{13}, W_{14}, W_{15}$) first, and then let Team Left solve the recycled slots ($W_{16}, W_{17}, W_{18}, W_{19}$) afterward?
Because normally, whenever Team Left changes $(a_1, a_2, a_3, a_4)$, that change ripples rightward through $e_5, e_6, e_7, e_8$ and scrambles the middle slots $W_9, W_{10}, W_{11}, W_{12}$ and the padding slots $W_{13}, W_{14}, W_{15}$!

How do we build a **soundproof wall** between Team Left and Team Right?
1. **Lock the Railroad Switches (`[E8]`):** We set $e_9 = \mathtt{0xffffffff}$ and $e_{10} = \mathtt{0xffffffff}$. Remember how the Choose gate $\operatorname{Ch}(e, f, g)$ works: when $e$ is all `1`s, it picks $f$ and **disconnects the track from $g$**! Locking $e_9$ and $e_{10}$ to all `1`s disconnects $e_7$ and $e_8$, stopping Team Left's ripples dead in their tracks so $W_{12}\dots W_{15}$ depend only on Team Right and $W_{11} = c_{11} - a_4$!
2. **Make $a_5$ and $a_6$ Identical Twins (`[E11]`–`[E12]`):** Even with $e_9 = e_{10} = \mathtt{0xffffffff}$, cell $a_4$ from Team Left tries to sneak into slot $W_{10}$ through the Majority Vote gate $\operatorname{Maj}(a_6, a_5, a_4)$ (`[E10]`). How do we silence voter $a_4$? **We force voters $a_5$ and $a_6$ to cast the exact same ballot ($a_5 = a_6$, so $D_{\text{shape}} = a_5 \oplus a_6 = 0$)!** Since $a_5$ and $a_6$ agree on every bit, they win the 2-out-of-3 majority vote automatically ($\operatorname{Maj}(a_5, a_5, a_4) = a_5$), and $a_4$ is **100% locked out of $W_{10} = c_{10} - a_3$**!
3. **Cancel $a_4$ and $a_3$ Out of $W_9$ (`[E13]`):** In slot $W_9$ (`[E10]`), $a_4$ appears twice with opposite signs: $-a_4 + \operatorname{Maj}(a_5, a_4, a_3)$. Whenever $a_4$ wins the vote $\operatorname{Maj}(a_5, a_4, a_3) = a_4$ (`[E13]`), $-a_4 + a_4 = 0$ cancels out completely, leaving $W_9 = c_9 - a_2$!

---

### 2.2 Explain Like I'm an Undergraduate (ELIU): Formal Derivation of `[E8]`–`[E14]`

Let us trace equations `[E8]` through `[E14]` in detail:

1. **Selector Saturation (`[E8]`–`[E9]`):**
   Setting $e_9 = e_{10} = \mathtt{0xffffffff}$ (`[E8]`) gives $\operatorname{Ch}(e_9, e_8, e_7) = e_8$, $\operatorname{Ch}(e_{10}, e_9, e_8) = \mathtt{0xffffffff}$, and $\operatorname{Ch}(e_{11}, e_{10}, e_9) = \mathtt{0xffffffff}$.
   Substituting $e_9 = e_{10} = \mathtt{0xffffffff}$ into `[E7]` at $s = 9$ and $s = 10$ isolates $a_5$ and $a_6$ purely as functions of the Right variables $(a_7, a_8, a_9, a_{10})$ (`[E9]`):

   $$\begin{aligned}
   a_5 &= \mathtt{0xffffffff} - a_9 + \Sigma_0(a_8) + \operatorname{Maj}(a_8, a_7, a_6), \\
   a_6 &= \mathtt{0xffffffff} - a_{10} + \Sigma_0(a_9) + \operatorname{Maj}(a_9, a_8, a_7).
   \end{aligned}$$

2. **Exact Context Shaping (`[E10]`–`[E13]`):**
   Expanding $W_8, W_9, W_{10}, W_{11}, W_{12}$ via `[E7]` under `[E8]` yields `[E10]`:

   $$\begin{aligned}
   W_8 &= c_8 - a_1 - e_5(a_1,\dots,a_5), & W_{11} &= c_{11} - a_4, \qquad W_{12} = c_{12}, \\
   W_9 &= c_9 - a_2 - \big(a_4 - \operatorname{Maj}(a_5, a_4, a_3)\big), & W_{10} &= c_{10} - a_3 - \big(a_5 - \operatorname{Maj}(a_6, a_5, a_4)\big).
   \end{aligned}$$

   - When we enforce **exact context shaping** $D_{\text{shape}} = a_5 \oplus a_6 = 0 \iff a_5 = a_6$ (`[E11]`), the identity $\operatorname{Maj}(a_5, a_5, a_4) = a_5$ simplifies $W_{10}$ to $W_{10} = c_{10} - a_3$ (`[E12]`) for **every** $a_4 \in \mathbb{Z}_{2^{32}}$.
   - Simultaneously, in $W_9$, when does $\operatorname{Maj}(a_5, a_4, a_3) = a_4$? On any bit position where $a_3$ and $a_5$ are *opposite* ($a_{3,b} \ne a_{5,b}$), the majority of $(a_{5,b}, a_{4,b}, a_{3,b})$ is automatically $a_{4,b}$! On bit positions where $a_{3,b} = a_{5,b}$ (marked by the mask $M_3 := \neg(a_3 \oplus a_5)$), we simply require $a_{4,b} = a_{5,b}$, i.e., $((a_4 \oplus a_5) \wedge M_3) = 0$ (`[E13]`). Then $W_9 = c_9 - a_2$ (`[E13]`), so $W_9$ depends only on $a_2$, $W_{10}$ only on $a_3$, $W_{11}$ only on $a_4$, and $W_{12}, \dots, W_{19}$ are **100% independent of $(a_1, a_2, a_3, a_4)$**!

3. **The Three Chained $Q_0$-Inversions (`[E14]`):**
   Substituting $W_9 = c_9 - a_2$, $W_{10} = c_{10} - a_3$, $W_{11} = c_{11} - a_4$ and $W_t = a_{t+1} - G_t(a_1,\dots,a_t)$ into `[E3]` for $t = 16, 17, 18$ produces the 3-level tree search (`[E14]`):

   $$\begin{aligned}
   \text{Level 1 } (t=16): \quad Q_0(W_1) &= T_{16} + F_1(a_1), \\
   \text{Level 2 } (t=17): \quad Q_0(W_2) &= T_{17} + G_2(a_1, a_2) - W_1, \\
   \text{Level 3 } (t=18): \quad Q_0(W_3) &= T_{18} + G_3(a_1, a_2, a_3) - W_2.
   \end{aligned}$$

---

## Section 3: Exact $O(1)$ Algebraic Padding Inversion (`[E15]`–`[E22]`)

### 3.1 Explain Like I'm 5 (ELI5): Picking the 8 Secret Pins in the Padding Lock

Now Team Right $(a_5 \dots a_{12})$ has to satisfy the postal inspector's three locked padding rules (`[E1]`):
1. $W_{15} = 440$ (message length),
2. $W_{14} = 0$ (zero block), and
3. The last byte of $W_{13}$ must be the red STOP bead `0x80` ($W_{13} \bmod 256 = \mathtt{0x80}$).

Here is how we unlock all three without guessing:
- First, once we pick $a_{10}, a_{11}, a_{12}$ and a dial $e_{13}$, the equations for $W_{15} = 440$ and $W_{14} = 0$ (`[E16]`) immediately tell us the exact values of $e_{12}$ and $e_{11}$!
- Second, what about the red STOP bead $W_{13} \bmod 256 = \mathtt{0x80}$? Normally, random choices of $e_{13}$ only hit `0x80` 1 time out of 256 ($1/256$).
- **The $O(1)$ Lockpick (`[E18]`–`[E21]`):** Look at how $e_{13}$ is mixed inside $W_{13}$: it passes through $\Sigma_1(e_{13})$, which rotates $e_{13}$ right by 6, 11, and 25 bits (`[E20]`). When we look at the lowest 8 bits of $\Sigma_1(e_{13})$, we discover a miracle: **bits 25..31 and bit 18 of $e_{13}$ each touch exactly one bit of the low byte of $W_{13}$, and never touch the low byte of $e_{13}$ itself!**
- So we can pick the other 24 bits of $e_{13}$ however we want, and then **flip those 8 secret pins (bit 18 and bits 25..31) in a single step (`[E21]`)** so that $W_{13} \bmod 256 = \mathtt{0x80}$ is guaranteed **100% of the time ($256\times$ speedup)**!

---

### 3.2 Explain Like I'm an Undergraduate (ELIU): Derivation of `[E15]`–`[E22]`

1. **Absorbing $W_{15}=440$ and $W_{14}=0$ (`[E15]`–`[E17]`):**
   Given $(a_{10}, a_{11}, a_{12})$ and the terminal boundary $a_{13}\dots a_{20}$, `[E7]` immediately determines $e_{16}, e_{15}, e_{14}$ (`[E15]`).
   Setting $W_{15} = 440$ and $W_{14} = 0$ in `[E7]` at $t = 15, 14$ yields `[E16]`:

   $$\begin{aligned}
   e_{12} &= (e_{16} - a_{12} - \Sigma_1(e_{15}) - K_{15} - 440) - \operatorname{Ch}(e_{15}, e_{14}, e_{13}), \\
   e_{11} &= (e_{15} - a_{11} - \Sigma_1(e_{14}) - K_{14}) - \operatorname{Ch}(e_{14}, e_{13}, e_{12}),
   \end{aligned}$$

   and `[E7]` at $s = 12, 11$ recovers $a_8, a_7$ (`[E17]`).

2. **Exact $O(1)$ Pivot Inversion of $W_{13} \bmod 256 = \mathtt{0x80}$ (`[E18]`–`[E21]`):**
   Reducing $W_{13}$ (`[E18]`) modulo 256 gives `[E19]`:

   $$\Sigma_1(e_{13}) \equiv R_{\text{lo}} := \big(c_{W_{13}} - \operatorname{Ch}(e_{13,\text{lo}}, e_{12,\text{lo}}, e_{11,\text{lo}}) - \mathtt{0x80}\big) \pmod{256}.$$

   Notice that $R_{\text{lo}}$ depends **only on the low 8 bits $e_{13,\text{lo}} = e_{13} \bmod 256$**!
   Meanwhile, for each bit $b \in \{0, \dots, 7\}$, the $b$-th bit of $\Sigma_1(e_{13})$ is (`[E20]`):

   $$\big(\Sigma_1(e_{13})\big)_b = (e_{13})_{b+6} \oplus (e_{13})_{b+11} \oplus (e_{13})_{(b+25)\bmod 32}.$$

   - For $b \in \{0,\dots,6\}$, bit $(e_{13})_{b+25} \in \{25,\dots,31\}$ appears uniquely in row $b$ and nowhere in bits $0..17$.
   - For $b = 7$, bit $(e_{13})_{7+11} = (e_{13})_{18}$ appears uniquely in row $7$ and nowhere in bits $0..17$.
   Hence, for **any** assignment to bits $0..17$ of $e_{13}$ ($e_{13,\text{base}}$) and any assignment to bits $19..24$, setting bit 18 and bits $25..31$ via `[E21]`:

   $$\Delta := \big(R_{\text{lo}} \oplus \Sigma_1(e_{13,\text{base}})\big) \bmod 256, \qquad e_{13,\text{fixed}} := e_{13,\text{base}} \;\vert\; \big((\Delta \wedge \mathtt{0x7f}) \ll 25\big) \;\vert\; \big(((\Delta \gg 7) \wedge 1) \ll 18\big),$$

   satisfies $(W_{13} \bmod 256) = \mathtt{0x80}$ with **probability $1$**!

3. **$\operatorname{Ch}$-Annihilation & Twin Majority (`[E22]`):**
   By choosing $a_{12}$ so that $e_{15} = \mathtt{0xffffffff}$ and setting $a_{10} = a_{11}$ (`[E22]`), we get $\operatorname{Ch}(e_{15}, e_{14}, e_{13}) = e_{14}$ and $\operatorname{Maj}(a_{11}, a_{10}, a_9) = a_{11}$. Thus **$e_{12}$ and $a_8$ become 100% constant across all $2^{24}$ choices of $e_{13}$**!

---

## Section 4: `Carry-Sudoku` — 7-Stage Causal Bit-Slice Right-Context Solver (`[E23]`–`[E25]`)

### 4.1 Explain Like I'm 5 (ELI5): Solving Addition from Right to Left Without Knowing $a_6$

To build a valid Right Context, we need to find values of $e_{13}$ where $a_5$ and $a_6$ are identical twins ($a_5 = a_6$, `[E11]`).
Look at the formula for $a_5$ (`[E9]`):

$$a_5 = X + \operatorname{Maj}(a_8, a_7, a_6), \qquad \text{where } X = \mathtt{0xffffffff} - a_9 + \Sigma_0(a_8).$$

Normally, to check if $a_5 = a_6$, you would have to compute all 32 bits of $a_9$, rotate $a_9$ through $\Sigma_0(a_9)$ to get $a_6$, and then add $X + \operatorname{Maj}(a_8, a_7, a_6)$ to get $a_5$. That means you can't check *anything* until all 32 bits of $e_{13}$ are chosen!

**The `Carry-Sudoku` Breakthrough (Theorem 1, `[E23]`):**
What if we **already assume** $a_6$ is going to equal $a_5$, and plug $X$ in place of $a_6$ inside the Majority gate?
Call that test sum $S = X + \operatorname{Ch}(D, X, a_8)$, where $D = a_8 \oplus a_7$ (`[E23]`).
Notice two mind-blowing facts about $S$:
1. **$S$ does NOT use $a_6$ or $\Sigma_0(a_9)$ at all!**
2. **Every bit where $a_8$ and $a_7$ differ ($D_b = 1$), $S_b$ MUST equal $X_b$ (`((S ^ X) & D) == 0`, `[E24]`)!** Why? Because on any bit where $a_8 \ne a_7$, $\operatorname{Maj}(a_8, a_7, a_6)$ equals $a_6$, so $a_5 = X + a_6 + \text{carry}$ can only equal $a_6$ if $X + \text{carry} = 0 \pmod 2$, which means $S$ and $X$ must match on bit $b$!
3. Since addition carries only flow from **right to left (bit 0 $\to$ bit 31)**, the lowest $k$ bits of $S$, $X$, and $D$ depend **only on the lowest $k$ bits of $e_{13}$**!

So just like Sudoku, we fill in $e_{13}$ **3 to 4 bits at a time from right to left across 7 causal stages (`[E25]`)**, throwing away any chunk where `((S ^ X) & D)` is non-zero! Across the $2^{24}$ candidate values of $e_{13}$ per $a_{12}$ block, the 7 stages evaluate an average of only **$43{,}580$ tree nodes ($385\times$ reduction, pruning $99.74\%$ of the search space)** before ever computing $\Sigma_0(a_9)$—with **zero false negatives ($100\%$ completeness)**!

---

### 4.2 Explain Like I'm an Undergraduate (ELIU): Proof of Theorem 1 & The 7-Stage Pipeline (`[E23]`–`[E25]`)

Let us review **Theorem 1 (`[E23]`)**:
Define $X := \mathtt{0xffffffff} - a_9 + \Sigma_0(a_8)$, $D := a_8 \oplus a_7$, and $S := X + \operatorname{Ch}(D, X, a_8) \pmod{2^{32}}$. Then:

$$a_5 = a_6 \iff \Big(((S \oplus X) \wedge D) = 0\Big) \;\wedge\; \Big(((a_6 \oplus S) \wedge \neg D) = 0\Big).$$

- **Why does `((S ^ X) & D) == 0` prune causally (`[E24]`–`[E25]`)?**
   Because $a_8$ is constant (`[E22]`), $a_9 = e_{13} - c_{e_{13}}$, and $a_7 = c_{a_7,\text{full}} - (e_{14} \wedge e_{13}) + ((a_{10} \oplus a_8) \wedge a_9)$, every operation in $X, a_7, D$, and $S = X + \operatorname{Ch}(D, X, a_8)$ is **strictly causal (lower-triangular)**: bits $0..k$ of $S, X, D$ depend exclusively on bits $0..k$ of $e_{13}$!
   Thus `solve20_padded_fast.c` and `solve20_padded_cuda.cu` assign the 24 free bits of $e_{13}$ across 7 causal stages (`[E25]`):
   - **Stage 1 (bits 0..3):** checks mask `0xf` ($16.0$ nodes).
   - **Stage 2 (bits 4..7):** checks mask `0xff` ($80.4$ nodes), fixing $R_{\text{lo}}$ and $(e_{13})_{18} \oplus (e_{13})_{13}$.
   - **Stage 3 (bits 8..11):** checks mask `0xfff` ($412.9$ nodes).
   - **Stage 4 (bits 12..14 + exact bit 18):** once bit 13 is known, bit 18 is known! Checks mask `0x7fff` ($1{,}043.8$ nodes).
   - **Stage 5 (bits 15..17, checked through bit 18):** checks 19-bit mask `0x7ffff` ($3{,}507.8$ nodes) and pins bits $25..31$ via `[E21]`.
   - **Stage 6 (bits 19..21):** checks 22-bit mask `0x3fffff` ($8{,}845.4$ nodes).
   - **Stage 7 (bits 22..24 + full 32-bit verification):** checks full 32-bit `((S ^ X) & D) == 0` ($29{,}673.5$ leaves; only $\approx 2{,}149$ survivors per $a_{12}$ block evaluate $\Sigma_0(a_9)$ and $a_6 == a_5$).
   Across $2^{24} = 16{,}777{,}216$ candidates per $a_{12}$ block, the 7 stages visit an average of **$43{,}579.9$ nodes ($385\times$ reduction)** with $100\%$ completeness, reaching **`266.8 ctx/s`** on 16 CPU threads and **`114,507.0 ctx/s`** on 1× NVIDIA L4 GPU!

---

## Section 5: `Slot-Sudoku` & Zero-Table Closed-Form $\mathbb{F}_2$ Inverse (`[E27]`–`[E37]`)

### 5.1 Explain Like I'm 5 (ELI5): The Closed-Form Origami Unfolder and the 4-Slot Pigeonholes

Now that Team Right can generate thousands of exact $D=0$ padded contexts per second, Team Left sweeps $a_1 \in [0, 2^{32})$ and solves the three chained equations (`[E14]`):
- **Level 1:** Find $W_1$ such that $Q_0(W_1) = T_{16} + F_1(a_1)$.
- **Level 2:** Find $W_2$ such that $Q_0(W_2) = T_{17} + G_2(a_1, a_2) - W_1$.
- **Level 3:** Find $W_3$ such that $Q_0(W_3) = T_{18} + G_3(a_1, a_2, a_3) - W_2$.

Here we hit two huge engineering challenges—and solve both with **origami math (`Theorem 2`)** and **`Slot-Sudoku`**:
1. **Cut Memory in Half with the Closed-Form Origami Unfolder (`[E27]`–`[E29]`):**
   Storing a 32-bit number $u$ in every table slot takes 4 bytes per slot. What if we only store the **bottom 16 bits ($u_{\text{lo}}$)** in each slot (cutting table size in half to 2 bytes per slot!)?
   Previously, recovering the top 16 bits ($u_{\text{hi}}$) required looking up a 128 KiB table in memory 200 billion times per sweep, clogging the GPU's cache.
   In **Theorem 2 (`[E28]`–`[E29]`)**, we prove that you don't need any lookup table at all: because $\sigma_0(u_{\text{hi}} \ll 16)$ shifts bits in a neat triangle, **simple bit-shift/XOR operations (`reconstruct_u`) unfold the exact top 16 bits directly inside CPU/GPU registers**!
2. **Pack the Pigeonholes with `Slot-Sudoku` (`[E30]`–`[E33]`):**
   Some target values $y$ have 0 solutions to $Q_0(u) = y$, while others have 1, 2, 3, or 4 solutions.
   - In our **6.00 GiB `Slot-Sudoku` table (`[E30]`–`[E31]`)**, every group of 16 adjacent $y$'s shares 12 slots arranged by Earliest-Deadline-First (EDF) sliding windows plus a Cuckoo-Sudoku 4th overflow slot, capturing **82.98%** of all preimages in just 6 GiB!
   - In our **16.00 GiB Direct-Indexed 2-Way Exact GPU table (`[E32]`)**, every 32-bit $y$ directly indexes its own `uint32_t` holding up to two exact 16-bit roots $(u_{\text{lo},1}, u_{\text{lo},2})$—giving **zero false-positive checks** across all 3 levels!
   - And every survivor of Level 3 is a **19-step padded bridge candidate (`sol19`)**, which we test in 1 step (`[E36]`–`[E37]`) to find our **20-step padded preimage**!

---

### 5.2 Explain Like I'm an Undergraduate (ELIU): Theorem 2 (`reconstruct_u`), SWAR Filtering, & $\sigma_0^{-1}$ (`[E27]`–`[E37]`)

1. **Proof of Theorem 2 — Closed-Form Upper-Half Inverse (`[E27]`–`[E29]`):**
   Given $y = Q_0(u) = \sigma_0(u) - u \pmod{2^{32}}$ and the low 16 bits $u_{\text{lo}} = u \bmod 2^{16}$, let $U = u_{\text{hi}} \ll 16$. Since $(u + y) \bmod 2^{17} = \sigma_0(u) \bmod 2^{17}$, bit 16 of $u = u_{\text{lo}} + U$ equals $U_{16}$ (with no carry from $u_{\text{lo}}$ into bit 16), and $\sigma_0(u) = \sigma_0(U) \oplus \sigma_0(u_{\text{lo}})$, we know bits $0..16$ of $\sigma_0(U) \oplus U$ (`[E27]`):

   $$\sigma_0(U) \oplus U \equiv v := (u_{\text{lo}} + y) \oplus \sigma_0(u_{\text{lo}}) \pmod{2^{17}}.$$

   Now write $U = L(T) := T \oplus (T \ll 11) \oplus (T \ll 15) \pmod{2^{32}}$ for an upper-half word $T$. Why this specific operator $L(T)$?
   Because inside $\sigma_0(U) = \operatorname{ROR}(U,7) \oplus \operatorname{ROR}(U,18) \oplus (U \gg 3)$, the term $\operatorname{ROR}(U,18) = (U \gg 18) \oplus (U \ll 14)$ interacts with $(U \gg 7)$ and $(U \gg 3)$ at shifts $+11$ and $+15$! Substituting $U = T \oplus (T \ll 11) \oplus (T \ll 15)$ cancels the two upper bands across bits $0..15$ and yields $v_{16} = T_{23} \oplus T_{19} \oplus T_{16} = v_5 \oplus v_1 \oplus T_{16}$ at bit 16 (since $U_{16} = T_{16}, U_{23} = T_{23} = v_5$, and $U_{19} = T_{19} = v_1$), leaving a direct closed-form expression for $T$ (`[E28]`):

   $$\begin{aligned}
   b_0 &= \big((v \ll 15) \oplus (v \ll 11) \oplus v\big) \wedge \mathtt{0x00010000}, \\
   b_{12} &= \big((v \ll 14) \oplus (v \ll 3)\big) \wedge \mathtt{0x00060000}, \\
   T &= b_0 \;\vert\; b_{12} \;\vert\; \big((v \ll 18) \wedge \mathtt{0xfff80000}\big),
   \end{aligned}$$

   followed by $u = u_{\text{lo}} \;\vert\; (T \oplus (T \ll 11) \oplus (T \ll 15))$ (`[E29]`).

2. **64-Bit SWAR Pre-Reconstruction Filter (`[E34]`–`[E35]`):**
   At Level 3, before reconstructing $u_{\text{hi}}$, notice that the majority condition $((a_4 \oplus a_5) \wedge M_3) = 0$ (`[E13]`) on bits $0..14$ depends only on $u_{\text{lo}}$! Using SIMD-Within-A-Register (SWAR, `[E35]`), we test 4 candidate 16-bit slots packed in a 64-bit word `w` in 3 instructions, rejecting $>99\%$ of quads without unpacking a single slot!

3. **Terminal 20-Step Bijection (`[E36]`–`[E37]`):**
   Every candidate passing Level 3 satisfies steps $0..18$ and all 72 padding bits—so it is a **schedule-consistent 19-step padded bridge candidate (`sol19`)**!
   To extend to **20 steps**, we only need step 19's schedule equation (`[E36]`):

   $$\sigma_0(W_4) + W_3 \equiv T_{19} := W_{19} - \sigma_1(W_{17}) - W_{12} \pmod{2^{32}}.$$

   Since $\sigma_0$ is a bijection over $\mathbb{F}_2^{32}$ ($\det_{\mathbb{F}_2}(\sigma_0) = 1$, `[E37]`), $W_4 = \sigma_0^{-1}((T_{19} - W_3) \bmod 2^{32})$ is uniquely determined by $W_3$ and $T_{19}$, yielding a $2^{-32}$ filter per `sol19` bridge candidate!

---

## Section 6: GPU Warp-Sudoku, Exact-$T_{16}$ Twin Coalescing & The Zero-$a_3$ Unpadded Bridge (`[E38]`–`[E45]`)

### 6.1 Explain Like I'm 5 (ELI5): Keeping All 32 GPU Lanes Busy + The Unpadded Shortcut

1. **Why GPUs Need `Warp-Sudoku` & Exact-$T_{16}$ Twin Coalescing (`[E42]`–`[E44]`):**
   On an NVIDIA L4 GPU, threads work in locked teams of 32 (**Warps**). At Level 2, only a few threads in a warp find a valid $a_3$ with small Hamming weight (`[E34]`). If those few threads immediately run Level 3 while the other threads wait, most of the GPU sits idle!
   - **Direct 2-Way Table & Warp-Aggregated Level-3 Work Queue (`[E32]`):** Every thread uses `__ballot_sync` to pack surviving Level-2 candidates into a shared queue until at least 32 candidates are ready, so all 32 lanes execute Level 3 at $100\%$ SIMD utilization!
   - **1.0M-Context GPU Reservoir & Exact-$T_{16}$ Twin Coalescing (`twin2`) / 4-Way Sweep (`single4`) (`[E43]`):** By generating a large reservoir of exact $D=0$ contexts at `114,507.0 ctx/s` and sorting by $T_{16}$, birthday pairs with $T_{16}^{(2m)} = T_{16}^{(2m+1)}$ (`twin2`) share their Level-1 table lookup with zero synthetic modification, while `single4` sweeps 4 distinct contexts simultaneously from the full reservoir $U \approx 1.68 \times 10^7$.
   - **Pre-`ulo2` 16-Bit Lower-Triangular Pruning (`[E44]`):** At Level 3, checking the lower 16 bits of the majority condition (`((a4 ^ a5) & M3 & 0xffff) == 0`) rejects **$\approx 93\%$ ($\approx 15\times$ reduction)** of invalid Level-3 Slot-1 and Slot-2 candidates *before* `reconstruct_u` (`[E29]`), reaching **`53,847.5–73,560.5` genuine `sol19/s`** (`16.22–22.16 GPU-h` projected) on the 6 GiB EDF table and **`132,888.7` genuine `sol19/s`** (**`8.98 GPU-hours` projected** per 20-step padded preimage on 1× L4, or **`1.12 hours` across 8× L4 GPUs**) in $a_1$-slice sweeps on genuine birthday-collided $T_{16}$ context pairs!
2. **The Unpadded 20-Round Shortcut (`Zero-a3 Bridge`, `[E38]`–`[E41]`):**
   What if the postal inspector doesn't require padding (the unpadded 20-round compression function)? Then we can set cell **$a_3 = 0$** along with $a_5 = a_6 = a_8 = a_9 = 0$ (`[E38]`). Since two zeros always win a 2-out-of-3 vote (`Maj(0, a4, 0) = 0`, `[E39]`), voter $a_4$ is defeated **100% of the time without any Hamming weight filter loss**, solving 20-step unpadded SHA-256 in **21.1 minutes observed on 16 CPU threads** ($33.1\text{ min}$ expected, `[E41]`)!

---

### 6.2 Explain Like I'm an Undergraduate (ELIU): Summary of Genuine Unmutated Benchmarks

Following standard reduced-round SHA-256 complexity accounting, we report complexity in **scalar CPU reference 20-step compression equivalents (`CPU-ref SHA20-eq`)** anchored to the 16-thread scalar CPU reference implementation on our 10-core Apple M4 ($1.5125 \times 10^8\text{ SHA20-eq/s}$, where $1\text{ node-h} = 2^{38.99}\text{ CPU-ref SHA20-eq}$), and report GPU acceleration in direct GPU wall-clock hours (`GPU-h`) projected from $a_1$-slice sweep telemetry alongside `CPU-ref SHA20-eq` equivalents:

| Solver / Architecture | Table Structure | Right-Context Speed ($D=0$) | 19-Step Bridge Rate (`sol19/s`) | `CPU-ref SHA20-eq` per `sol19` | Expected 20-Step Padded Time & `CPU-ref` Complexity (1 Device) | Expected Time on 8× L4 GPUs |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Primary CPU (`solve20_padded_fast.c`, 16 threads)** | **6.0 GiB 2-Tier Nested 4-Slot Sudoku** | **`266.8 ctx/s`** | **`6,242.8 sol19/s`** | **`24,228` ($2^{14.56}$)** | **`191.1 node-h` ($2^{46.56}\text{ CPU-ref}$)** | — |
| **Cycle-0 CUDA Baseline (1× L4 GPU, $a_1$-slice)** | 6.0 GiB 2-Tier Sudoku ($B=2$) | `7,580 ctx/s` | `53,847.5 sol19/s` | `2,808.9` ($2^{11.46}$) | `22.16 GPU-h` ($2^{43.46}\text{ CPU-ref}$) | `2.77 hours` |
| **Upgraded 6 GiB CUDA (`solve20_padded_cuda.cu`, 1× L4 GPU, $a_1$-slice)** | 6.0 GiB + Zero-Table `[E29]` + $B=8$ | `7,580 ctx/s` | `73,560.5 sol19/s` | `2,056.1` ($2^{11.01}$) | `16.22 GPU-h` ($2^{43.01}\text{ CPU-ref}$) | `2.03 hours` |
| **Exact-$T_{16}$ Twin CUDA (`16 GiB Direct 2-Way`, 1× L4 GPU, $a_1$-slice)** | **16.0 GiB Direct 2-Way Exact (`[E32]`)** | **`29,310–114,507 ctx/s`** | **`132,888.7 sol19/s`** | **`1,138.2` ($2^{10.15}$)** | **`8.98 GPU-h` ($2^{42.15}\text{ CPU-ref}$)** | **`1.12 hours`** |
| **Zero-$a_3$ Bi-Directional Bridge CPU (`solve20_zero_a3.c`, 16 threads)** | 48.0 GiB Unpadded $Q_0^{-1}/Q_1^{-1}$ (`[E38]`–`[E41]`) | Precomputed | `2,160,000 sol19/s` (unpadded) | `70.0` ($2^{6.13}$) | `33.1 min` exp. ($2^{38.13}\text{ CPU-ref}$); `21.1 min` obs. ($2^{37.48}$) | — |

---

## Section 7: Scope, "Restricted" Preimages, Two-Block Plain-ASCII Preimages, and 15 Explored Sub-$2^{32}$ Directions

### 7.1 Explain Like I'm 5 (ELI5): The 440-Coin Budget, Plain-ASCII First Blocks, and Why Round 20 Takes $2^{32}$ Bridge Candidates

1. **Wait—By Pre-Filling Sudoku Cells ($e_9, e_{10}, a_5=a_6, e_{15}, a_{10}=a_{11}$), Aren't We Throwing Away the Winning Answers?**
   Imagine you are looking for a winning lottery ticket in a giant stadium, and **a target hash has an average of $2^{184}$ ($\approx 2.4 \times 10^{55}$) different winning 55-byte messages**! Why so many? Because a 55-byte message has **440 free bits ("coins")**, and matching a 256-bit hash only costs **256 coins**—leaving you with **184 extra coins** to spend however you like!
   - Locking the soundproof wall ($e_9 = e_{10} = \mathtt{0xffffffff}$) costs **64 coins** ($120$ coins left).
   - Making $a_5 = a_6$ identical twins costs **32 coins** ($88$ coins left).
   - Canceling $a_4$ in the majority vote costs **$13.40$ coins** ($13.28$ coins from $(3/4)^{32}$ plus $0.12$ coins from the $w \le 14$ cutoff, leaving **$74.60$ coins $\implies 2^{74.60}$ winning 20-round messages still left!**).
   - Finally, turning on the Right-context turbocharger ($e_{15} = \mathtt{0xffffffff}$, $a_{10} = a_{11}$) costs **64 coins**, leaving **$2^{10.60} \approx 1{,}552$ unfiltered 20-step padded preimages** across an estimated universe of **$U \approx 1.68 \times 10^7 \approx 2^{24.00}$ valid Right contexts per target** (of which **$2^{8.69} \approx 413$** are reachable in our 2-way table configuration across $U$, or **$250 = 2^{7.97}$** when restricted to $U_{\text{filt}} \approx 1.014 \times 10^7$ with optional heuristic filtering since $U_{\text{filt}}/U \approx 60.5\%$)!
   - Pairing contexts by exact 32-bit birthday collisions $T_{16}^{(i)} = T_{16}^{(j)}$ (`twin2`) or sweeping uniformly across the full reservoir $U$ (`single4`) solves **all 16 padded 20-step targets (`100%` solve rate)**!

2. **How a Two-Block ($119\text{-Byte}$) Message Gives You a 64-Byte Plain-English First Block for Free!**
   In a 1-block ($55\text{-byte}$) message, our solver picks all 55 bytes algebraically, so the message looks like random computer bytes (`ec531306...`).
   What if we use a **2-block ($119\text{-byte}$) message** $M = B_1 \parallel B_2$?
   - **Block 1 ($B_1$, bytes `0..63`) has ZERO padding rules!** You can pick **any 64-byte plain-English sentence you want** (for example: `"This first block is plain text; the second one was computed....."`) and run 20-step SHA-256 forward in $O(1)$ time to get the middle chaining state $V = H^{(1)}$.
   - **Block 2 ($B_2$, bytes `64..118` + `0x80` + zeros + length `952 bits = 0x3b8`)** is then solved by our exact same GPU solver starting from $\mathrm{IV}' = V$ instead of $\mathrm{IV}$!
   - Better yet: even if someone gives a trick target like $H^* = \mathrm{IV}$ (where $H^* - \mathrm{IV} = 0$ in a 1-block attack), choosing $B_1$ makes $H^* - V \ne 0$ completely non-degenerate!

3. **Can Two Blocks Reduce Round 20 Below $2^{32}$? (Direction 15: The Two-Block Feed-Forward Trap!)**
   At first glance, you might think: *"Wait! In a 2-block message, Block 1 gives us 512 brand-new free coins! Can't we use Block 1 to pay for Round 20 in Block 2 so Round 20 becomes $O(1)$?"*
   When we trace how Block 1 connects to Block 2, we discover **The Two-Block Feed-Forward Trap (Direction 15)**:
   - Because of SHA-256's **Davies–Meyer feed-forward addition** at the end of Block 2 ($H_{20}^* = V + S_{20}^{(2)} \pmod{2^{32}}$), Block 1's output $V$ is wired to **BOTH ends of Block 2 at the same time**:
     - The **top** of Block 2 starts at $\mathrm{IV}' = V$, AND
     - The **bottom** of Block 2 must end at $S_{20}^{(2)} = H_{20}^* - V \pmod{2^{32}}$!
   - So if you change Block 1 to tweak $V$, **both the top and bottom of Block 2 move simultaneously!** And if you solve Block 2's Right context first (fixing $S_{20}^{(2)}$), that locks **all 256 bits** of $V = H_{20}^* - S_{20}^{(2)}$, which would force you to invert a 256-bit 20-step hash on Block 1 ($\Theta(2^{32})$) anyway!

---

### 7.2 Explain Like I'm an Undergraduate (ELIU): Formal Degree-of-Freedom Accounting & Taxonomy of 15 Explored Sub-$2^{32}$ Directions

1. **Formal Degree-of-Freedom Accounting ($440 \to 184 \to 74.60 \to 10.60$ Bits):**

   | Algebraic / Structural Constraint | Equations | Degrees of Freedom Spent | Expected Preimages per 256-Bit Target ($R=19$ or $R=20$) | 19-Step Bridge Candidates (`sol19`) per 20-Step Target |
   | :--- | :---: | :---: | :---: | :---: |
   | 55-byte FIPS 180-4 padded block ($L=440$) | `[E1]` | $72.0\text{ bits}$ (of 512) | $2^{184.00}$ | $2^{216.00}$ |
   | Selector saturation ($e_9 = e_{10} = \mathtt{0xffffffff}$) | `[E8]` | $64.0\text{ bits}$ | $2^{120.00}$ | $2^{152.00}$ |
   | Exact context shaping ($a_5 = a_6 \iff D_{\text{shape}}=0$) | `[E11]` | $32.0\text{ bits}$ | $2^{88.00}$ | $2^{120.00}$ |
   | Majority cancellation ($\operatorname{Maj}(a_5,a_4,a_3)=a_4, w\le 14$) | `[E13]` | $13.40\text{ bits}$ ($13.28 + 0.12$) | **$2^{74.60}$** | $2^{106.60}$ |
   | Right-context turbocharger ($e_{15}=\mathtt{0xffffffff}, a_{10}=a_{11}$) | `[E22]` | $64.0\text{ bits}$ | **$2^{10.60} \approx 1{,}552$** ($2^{8.69} \approx 413$ in 2-way table) | $2^{42.60}$ ($2^{40.69}$ in 2-way table) |

2. **Construction-Specific Design Explorations of 15 Candidate Sub-$2^{32}$ Directions (`test_all_12_theories.c`, Cycle 10, and Direction 15):**
   To document why we retained our 3-level Left-tree architecture under realistic memory ($\le 64\text{ GiB}$), we evaluated 15 candidate modifications (verified via model checks and surrogate probes in `test_all_12_theories.c`, Cycle 10, and Section 9.5; these represent construction-specific design analyses rather than universal impossibility proofs):

   | # | Candidate Sub-$2^{32}$ Mechanism | Mathematical / Empirical Outcome Within Our Left-Tree Architecture | Observed Bottleneck |
   | :---: | :--- | :--- | :---: |
   | **1** | Shift saturation wall right ($e_{10}=e_{11}=\mathtt{0xffffffff}$) | $W_9$ depends on $a_5$; Level 1 ($W_{16}$) becomes circular in $a_5$ | Infeasible in our tree |
   | **2** | Backward $Q_0^{-1}$ tree from $W_{19} \to W_{16}$ | $W_4(a_1..a_5)$ couples all 4 Left words simultaneously | $\Theta(2^{32})$ |
   | **3** | Cross-context MitM on $T_{19} = \sigma_0(W_4)+W_3$ | Left tree path depends on 96-bit tuple $(T_{16}, T_{17}, T_{18})$ | $\Theta(2^{48})$ birthday |
   | **4** | Partial bit-slice / LSB-first causal solver for $(a_1..a_4)$ | $\sigma_0(u)$ rotates by $7, 18$ bits, coupling LSBs to MSBs at every level | $\Theta(2^{32})$ |
   | **5** | Differential / rotational invariant trails across $Q_0^{-1}$ (surrogate 3-level probe) | Addition carries in $Q_0(u) = \sigma_0(u)-u$ destroy linear invariants in 3-level surrogate map | No bias observed |
   | **6** | Linearizing $Q_0$ over $\mathbb{F}_2^{32}$ or $\mathbb{Z}_{2^{32}}$ (surrogate 3-level probe) | Empirical Jacobian across 3-level surrogate map has full rank ($32/32$ bits) | Full rank ($32/32$) |
   | **7** | Colliding $T_{16}$ and $T_{17}$ simultaneously across Right contexts | Colliding 64 bits of $(T_{16}, T_{17})$ requires $\ge 2^{32}$ Right contexts | $\Theta(2^{32})$ |
   | **8** | Relaxed context shaping ($D_{\text{shape}} = a_5 \oplus a_6 \ne 0$, low weight) | Restores $a_4$ leakage into $W_{10}$, coupling Level 2 to Level 3 | Slower than $D=0$ |
   | **9** | Higher-order multi-slot $Q_0^{-1}$ precomputation ($W_1, W_2$ joint table) | Joint table over $(a_1, T_{16}, T_{17})$ requires $2^{96}$ entries | $\gg 64\text{ GiB}$ |
   | **10** | Symbolic Gröbner basis / ANF elimination on $(a_1..a_4)$ | Degree-31 carry polynomials across 3 coupled 32-bit additions | $\ge 2^{32}$ |
   | **11** | SAT / SMT CDCL & CryptoMiniSat bit-blasting on $(a_1..a_4 \mid \text{Right})$ | XOR-addition chains induce exponential unit-propagation bottlenecks | $\gg 10^4\times$ slower |
   | **12** | Neural / RL / heuristic priority-queue guidance on $a_1$ | $\sigma_0^{-1}(T_{19}-W_3)$ exhibits avalanche uniformity ($r^2 < 10^{-6}$) | $\Theta(2^{32})$ |
   | **13** | **4-Round Crossed Carry-Sudoku Spine ($a_2 = a_3 = a_4 = a_5$, `[E45]`)** | Collapses 4 majority gates (`0 GiB DRAM`, `9.62B evals/s` on L4), but spends 96 Left bits turning $T_{16}, T_{17}, T_{18}$ into a 96-bit Right filter | Overconstrained ($2^{-64}$/ctx) |
   | **14** | **5-Step Crossed Carry-Sudoku Spine ($a_1 = a_2 = a_3 = a_4 = a_5$)** | Spends 128 Left bits (`1 trial/ctx`); under constraint-count model, 128-bit schedule match needs $2^{128}$ Right contexts | Overconstrained ($2^{-128}$/ctx) |
   | **15** | **Two-Block ($119\text{-Byte}$) Feed-Forward Attack ($M = B_1 \parallel B_2$)** | Davies–Meyer feed-forward $S_{20}^{(2)} = H_{20}^* - V \pmod{2^{32}}$ couples $V = \mathrm{Compress}_{20}(\mathrm{IV}, B_1)$ to both ends of Block 2 simultaneously | $\Theta(2^{32})$ (`8.98–16.22 GPU-h` projected on 1× L4 + plain-ASCII $B_1$) |

---

### Standalone Verification (`verify_all_witnesses.py`)

Run the zero-dependency Python verifier to check **all 46 distinct 20-step and 19-step witnesses across 48 checks (including the `'abc'` demo+benchmark overlap)** (17 20-step padded checks across 16 distinct witnesses, 17 19-step padded checks across 16 distinct witnesses including the $0^{256}$ exact $a_5 = a_6$ witness and all 14 benchmark targets, 13 distinct 20-step unpadded witnesses, and 1 Stage-2 completeness regression witness) and all internal Carry-Sudoku invariants:

```bash
python3 verify_all_witnesses.py
```

### Disclosure & Attribution

Building upon K. S. Basra's selector-saturation and backward padding-absorption framework (`https://github.com/kamb-code/sha256-r19-preimage`), the algebraic discoveries (closed-form $O(1)$ $W_{13}$ padding pivot inversion, Theorem 1 / Carry-Sudoku, Theorem 2 / closed-form $\mathbb{F}_2$ upper-half reconstruction, Slot-Sudoku tables, and the Zero-$a_3$ bi-directional bridge), C/CUDA solvers, preimage witnesses, and manuscript in this work were discovered, developed, and written using **Google DeepMind Gemini 4 Argon**.
