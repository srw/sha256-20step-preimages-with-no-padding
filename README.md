# Preimages for 20-step SHA-256 (compression function, no padding)

This repository gives a working solver and verified witness for **the SHA-256 compression function
reduced to its first 20 of 64 steps**: standard initial value (`IV`), standard message schedule and
round constants, steps 0–19, and the usual Davies–Meyer feed-forward, with all 256 output bits
matching a prescribed target digest.

Using a **Zero-$a_3$ Bi-Directional $Q_0/Q_1$ Bridge**, the solver eliminates the
$(3/4)^{32} \approx 2^{-13.28}$ residual $\beta$-filter loss of earlier context-shaping approaches
and solves the 20-step compression preimage problem in **$\sim 2^{33}$ SHA-256-equivalent operations**
(**21.1 minutes** of search on a 16-vCPU CPU VM after a **59.4 s** one-time target-independent table
build of 48 GiB).

For padded 19-step SHA-256 (full hash function with standard FIPS 180-4 padding), see the companion
repository [srw/sha256-19step-preimages-with-padding](https://github.com/srw/sha256-19step-preimages-with-padding).
**This builds on K. S. Basra's context-shaping framework**
([kamb-code/sha256-r19-preimage](https://github.com/kamb-code/sha256-r19-preimage)).

Nothing here affects full 64-step SHA-256.

## Example (Verified 20-Step Witness)

```text
target   = deadbeefcafebabe0123456789abcdeffedcba987654321013579bdf2468ace0

W[0..15] = 4334d956 6eccece1 6727a1b2 d85811af
           890c5399 79018a3b d346b94d 5f26009b
           7562d311 0bfd80d5 1f8c38f4 16d8315d
           e0353aa2 6adf4bd2 a2bf14a2 801ae8f7

Compress_20(IV, W[0..15]) = deadbeefcafebabe0123456789abcdeffedcba987654321013579bdf2468ace0
```

## Comparison with Prior Work

| steps | object | work | cost / timing |
|---|---|---|---|
| 16 | hash function, with padding | Homsirikamol et al., CISIM 2012 ([ePrint 2012/421](https://eprint.iacr.org/2012/421)) | SAT, ~1 h on Core i7 |
| 17 | hash function, with padding | McDair, *cryptography* mailing list, March 2024 | 3 witnesses posted, no method |
| 17–18 | compression function | Davydov et al., 2025 | SAT, seconds (all-zero target) |
| **18** | **compression function** | **this repository (`src/validate_sub32_cascades.py`)** | **$O(1)$ ($\approx 1.58$ trials, $\sim 3.5\text{ ms}$ on 1 CPU core, $1.06\text{ MB}$ RAM)** |
| 19 | compression function | Zaikin, *Constraints* 2026; Basra, Sept 2026 | SAT (18.5 h on 192 cores); algebraic ($\sim 2^{29}$ lookups) |
| **19** | **compression function** | **this repository (`src/carry_tree_bridge.c`)** | **$2^{16}$ MITM steps in $2.56\text{ MB}$ RAM ($\sim 6.8\text{ s}$ on 1 CPU core, $<1\text{ ms}$ init)** |
| 19 | hash function, with padding | [srw/sha256-19step-preimages-with-padding](https://github.com/srw/sha256-19step-preimages-with-padding) | $\sim 2^{25.3}$ chain starts (~7–22 s median on 1 core) |
| 20 | compression function | Basra, Sept 2026 ([kamb-code/sha256-r19-preimage](https://github.com/kamb-code/sha256-r19-preimage)) | $\sim 2^{45.4}$ table lookups (2.45 and 16.16 GPU-hours) |
| **20** | **compression function** | **this repository (`src/solve20_zero_a3.c`, `src/carry_tree_bridge.c`)** | **$\sim 2^{32.0}$ bridge matches / $\sim 2^{33}$ work (21.1 min on 16 CPU vCPUs; $1.06\text{ MB}$ Carry-Tree replaces $32\text{ GiB}$ $Q_0^{-1}/Q_1^{-1}$ tables with $5.31\times$ higher yield)** |

## Verify (Python 3, standard library only)

Verify the archived 20-step witness (and check the 64-step implementation against `hashlib`):

```bash
python3 verify/verify_standalone.py witnesses/witnesses_20step.txt
```

Verify the algebraic identities of the Zero-$a_3$ Bi-Directional $Q_0/Q_1$ Bridge over 10,000 random
states:

```bash
python3 src/validate_zero_a3.py
```

Verify the memoryless ($1.06\text{ MB}$) Carry-Tree $Q_0^{-1}/Q_1^{-1}$ solvers and the 18-, 19-, and
20-step algebraic decoupling cascades ($O(1)$ at 18 steps, $2^{16}$ at 19 steps, and $\sim 2^{32}$ at
20 steps):

```bash
python3 src/validate_sub32_cascades.py
```

## Reproduce

### 1. Memoryless ($2.56\text{ MB}$ RAM) $O(1)$ Carry-Tree & $2^{16}$-Step 19-Round Preimage Solver

Compile and run [`src/carry_tree_bridge.c`](src/carry_tree_bridge.c), which replaces the $32\text{ GiB}$
`g_q0_inv` / `g_q1_inv` arrays with a **$1.06\text{ MB}$ $\mathbb{F}_2$-pseudoinverse + 16-bit split
carry-tree solver** (built in $<1\text{ ms}$, recovering 100% of colliding preimages with zero
single-slot overwrite loss) and solves a full 256-bit 19-step compression preimage in **$2^{16}$
steps ($\sim 6.8\text{ s}$ on 1 CPU core) and $2.56\text{ MB}$ of RAM**:

```bash
gcc -O3 -march=native src/carry_tree_bridge.c -o carry_tree_bridge
./carry_tree_bridge
```

### 2. Multi-Threaded 20-Step Zero-$a_3$ Bridge Solver (`src/solve20_zero_a3.c`)

Compile the solver:

```bash
gcc -O3 -march=native -pthread src/solve20_zero_a3.c -o solve20_zero_a3
```

#### Quick local check (19-round solutions in < 2 seconds on a laptop)

With reduced table sizes (`--q-bits 25 --left-bits 24`, ~320 MiB RAM), the solver verifies the
bridge and generates exact 19-round solutions ($\rho_{16} = \rho_{17} = \rho_{18} = 0$) in under 2
seconds:

```bash
./solve20_zero_a3 --q-bits 25 --left-bits 24 --threads 8 --test-19 20
```

#### Full 20-step solve (requires 48 GiB RAM for full $2^{32}$ tables)

On a machine with $\ge 64\text{ GiB}$ RAM (tested on a 16-vCPU x86_64 VM with 128 GiB RAM):

```bash
./solve20_zero_a3 --q-bits 32 --left-bits 32 --threads 16
```

You can also pass any 64-hex-character 256-bit target digest via `--target`:

```bash
./solve20_zero_a3 --q-bits 32 --left-bits 32 --threads 16 \
  --target deadbeefcafebabe0123456789abcdeffedcba987654321013579bdf2468ace0
```

## How It Works (Short)

Full mathematical details are in [`docs/NOTE.md`](docs/NOTE.md).

1. **Why earlier 20-step context shaping cost $\sim 2^{45.4}$.**
   In standard context shaping ($a_5 = a_6$, $e_9 = e_{10} = \mathtt{0xffffffff}$), solving the three
   19-step constraints $\rho_{16} = \rho_{17} = \rho_{18} = 0$ leaves a nonlinear residual condition
   $$R = \mathrm{Maj}(a_5, a_4, a_3) - a_4 = 0,$$
   which holds with probability $(3/4)^{32} \approx 2^{-13.28}$. Adding the 20th-step constraint
   $\rho_{19} = 0$ ($2^{-32}$) gives an expected complexity of $2^{32 + 13.28} = 2^{45.28}$ (plus
   $Q^{-1}$ table-coverage factors).

2. **The Zero-$a_3$ Manifold (zero $\beta$-filter loss).**
   We place the internal state on the **Zero-$a_3$ Manifold**:
   $$a_3 = 0, \quad a_5 = 0, \quad a_6 = 0, \quad a_8 = 0, \quad a_9 = 0, \quad a_{10} = \mathtt{0xffffffff}.$$
   Because $a_3 = a_5 = a_6 = 0$, both majority terms involving $a_4$ vanish **identically for 100%
   of $a_4 \in [0, 2^{32})$**:
   $$\mathrm{Maj}(a_5, a_4, a_3) = \mathrm{Maj}(0, a_4, 0) = 0, \qquad \mathrm{Maj}(a_6, a_5, a_4) = \mathrm{Maj}(0, 0, a_4) = 0.$$
   Furthermore, $e_9 = 0$ and $e_{10} = \mathtt{0xffffffff}$ hold automatically, decoupling $a_4$ from
   $W_9 \dots W_{19}$.

3. **Role-Swapping $a_3 \leftrightarrow a_7$ and the $\sigma_1(D_{14})$ Cancellation.**
   Fixing $a_3 = 0$ removes $a_3$ as a Left pivot, so we promote $a_7$ to the active pivot on the
   Right side. In $W_{14}$, $e_{11}$ enters as $-e_{11} = -(e_{11,0} + a_7)$, so
   $W_{14} = D_{14}(a_{11}, a_{12}) - a_7$. Meanwhile, in $W_9$, $e_9 = 0$ gives
   $\mathrm{Ch}(0, e_8, e_7) = e_7 = a_7$, so $W_9 = W_{9,0} - a_7 - a_2$.
   Substituting both into $\rho_{16} = W_{16} - \sigma_1(W_{14}) - W_9 - \sigma_0(W_1) - W_0 = 0$,
   the linear $+a_7$ from $-W_9$ and the $-\sigma_1(D_{14} - a_7)$ combine into
   $$Q_1(u) := \sigma_1(u) + u \quad \text{at } u = D_{14}(a_{11}, a_{12}) - a_7,$$
   allowing $a_7$ to be solved in $O(1)$ via $Q_1^{-1}$!

4. **Bi-Directional Bridge and Target-Independent $48\text{ GiB}$ Tables (`src/solve20_zero_a3.c`).**
   - **Phase 0 (26.36 s):** Build $16\text{ GiB}$ inverse tables for $Q_0(u) = \sigma_0(u) - u$ and
     $Q_1(u) = \sigma_1(u) + u$.
   - **Phase 1 (33.05 s):** Fix an arbitrary bridge constant $T_{16}^*$. Sweep $a_1 \in [0, 2^{32})$,
     solve $Q_0(W_1) = T_{16}^* - W_0(a_1) + G_1(a_1)$ for $a_2$, and store $a_1$ indexed by
     $L_{17} = \sigma_0(W_2) + W_1$ in a $16\text{ GiB}$ table `g_left_a1` (`2,721,604,389` valid
     Left pairs $= 0.633673 \times 2^{32}$).
   - **Universal target independence:** `g_q0_inv`, `g_q1_inv`, and `g_left_a1` depend **only on
     `IV`** and $T_{16}^*$ — they do not depend on the 256-bit target $H^*$ at all.
   - **Phase 2 (1266.26 s on 16 vCPUs):** Sweep $(a_{11}, a_{12})$, invert $Q_1$ to get $a_7$, look
     up `a1 = g_left_a1[T17]` to satisfy $\rho_{16} = \rho_{17} = 0$ simultaneously,
     invert $Q_0(W_3)$ to get $a_4$ (satisfying $\rho_{18} = 0$ with **zero $\beta$-filter loss**,
     yielding **2.16 million exact 19-round solutions/sec**), and check $\rho_{19} = 0$.

5. **Memoryless ($1.06\text{ MB}$) Carry-Tree Solvers & 18/19/20/21-Step Complexity Hierarchy (`src/carry_tree_bridge.c`, `src/validate_sub32_cascades.py`).**
   - **Eliminating $32\text{ GiB}$ of $Q_0^{-1}/Q_1^{-1}$ tables and $81.1\%$ overwrite loss:**
     In $y = (\sigma_b(x) \pm x) \bmod 2^{32}$, the $\mathbb{F}_2$-linear operator
     $L_b(x) = \sigma_b(x) \oplus x$ satisfies $L_b(x) = y \oplus c$, where the carry/borrow bit
     $c_{i+1}$ is deterministic ($c_{i+1} = c_i$) on half of all bits and branches with $x_i = c_{i+1}$
     fixed on the other half. Splitting the carry tree into two 16-bit halves coupled by $1.06\text{ MB}$
     of $\mathbb{F}_2$-pseudoinverse tables (`P0_low/high`, `P1_low/high`, built in $<1\text{ ms}$)
     recovers **100% of preimages** (average $1.000$ preimage per random target vs. $0.6337$ in a
     single-slot $16\text{ GiB}$ array, boosting 3-step yield by **$5.31\times$**).
   - **18-, 19-, 20-, and 21-step complexity hierarchy (compression function, $R - 16$ schedule equations):**
     - **18 steps (2 equations, 64 bits):** Solved in **$O(1)$ ($\approx 1.58$ trials, $\sim 3.5\text{ ms}$)**
       in $1.06\text{ MB}$ RAM via direct $Q_0^{-1}/Q_1^{-1}$ inversion with zero search loop.
     - **19 steps (3 equations, 96 bits):** Solved in **$2^{16}$ MITM steps ($\sim 6.8\text{ s}$ on 1
       core) and $2.56\text{ MB}$ RAM** (`src/carry_tree_bridge.c`), matching all 256 output bits of
       `Compress_19(IV, W[0..15]) == target`.
     - **20 steps (4 equations, 128 bits):** Solved in **$\sim 2^{32}$ bridge matches / $\sim 2^{33}$
       work** (`21.1 min` on 16 vCPUs in `src/solve20_zero_a3.c`). Using the $1.06\text{ MB}$
       Carry-Tree solver (`src/carry_tree_bridge.c`) eliminates the $32\text{ GiB}$ `g_q0_inv` /
       `g_q1_inv` tables (reducing full-table RAM from $48\text{ GiB}$ to $16\text{ GiB}$, or allowing
       any $2^{64}/M$ time-memory tradeoff down to $2.56\text{ MB}$ RAM) and verifies the 20-step
       witness in **`0.21 ms`** (`3` Carry-Tree calls).
     - **21 steps (5 equations, 160 bits):** Solved in **$\sim 2^{64}$ work** on the Zero-$a_3$
       Manifold (with $a_{13}$ additionally free on the Right and $\rho_{19} = \rho_{20} = 0$ acting
       as a 64-bit filter).

## Scope: Compression Function vs. Padded Hash Function at 20 Steps

- **19 steps with padding ($\sim 2^{25.3}$, solved on 1 CPU core):**
  In [srw/sha256-19step-preimages-with-padding](https://github.com/srw/sha256-19step-preimages-with-padding),
  we solved single-block 55-byte padded messages ($W_{14} = 0$, $W_{15} = \mathtt{0x000001b8}$,
  $\mathrm{LSB}(W_{13}) = \mathtt{0x80}$). At 19 steps ($a_{12} \dots a_{19}$ fixed by the target),
  $W_{14}$ and $W_{15}$ linearly absorb $a_7$ and $a_8$ via $-e_{11}$ and $-e_{12}$, leaving enough
  Right degrees of freedom alongside the three Left pivots $(a_2, a_3, a_4)$.
- **19 and 20 steps without padding ($2^{16}$ at 19 steps; $\sim 2^{32}$ bridge matches / $\sim 2^{33}$ work at 20 steps, solved here):**
  For the 19- and 20-step compression functions ($W_0 \dots W_{15}$ unconstrained), the Zero-$a_3$
  bridge solves 19 steps in $2^{16}$ operations ($2.56\text{ MB}$ RAM) by sweeping $a_{11}$ against a
  $2^{16}$-entry Left table, and solves 20 steps in $\sim 2^{32}$ bridge matches by sweeping
  $(a_{11}, a_{12})$ with $a_7$ as the $Q_1^{-1}$ pivot.
- **Why single-block 20 steps *with* padding is harder ($\sim 2^{64}$ single-block, $\sim 2^{39}$ two-block):**
  Imposing $W_{14} = 0$ and $W_{15} = 440$ at 20 steps pins two of the three Right variables
  $(a_7, a_{11}, a_{12})$, collapsing the $2^{64}$ Right search space on the fixed Zero-$a_3$
  manifold to a single curve of $2^{32}$ states (whereas 20-step + 8-bit padding requires $\sim 2^{40}$
  trials). Overcoming this for a single-block 20-step padded message requires either $\sim 2^{64}$
  work across unshaped contexts (where $W_{14}=0, W_{15}=440$ absorb two variables in $O(1)$) or a
  2-block meet-in-the-middle construction in $\sim 2^{39}$.

## Layout

| path | contents |
|---|---|
| `src/solve20_zero_a3.c` | Multi-threaded C solver implementing the Zero-$a_3$ Bi-Directional $Q_0/Q_1$ Bridge ($48\text{ GiB}$ RAM) |
| `src/carry_tree_bridge.c` | Memoryless ($1.06\text{ MB}$) $O(1)$ Carry-Tree $Q_0^{-1}/Q_1^{-1}$ solver and $2^{16}$-step ($2.56\text{ MB}$ RAM) full 256-bit 19-step preimage solver in C |
| `src/validate_zero_a3.py` | Numerical verification of the Zero-$a_3$ identities over 10,000 random states |
| `src/validate_sub32_cascades.py` | Python verification suite for the $O(1)$ Carry-Tree solvers and 18/19/20-step sub-$2^{32}$ algebraic decoupling cascades |
| `verify/verify_standalone.py` | Zero-dependency Python verifier for 20-step witnesses |
| `witnesses/` | Verified 20-step witness (`witnesses_20step.txt`) and raw solver log (`comp20_2026-10-13.log`) |
| `docs/NOTE.md` | Technical note with full algebraic derivation and measured telemetry |

## Credit and Licence

Context shaping is due to K. S. Basra ([kamb-code/sha256-r19-preimage](https://github.com/kamb-code/sha256-r19-preimage));
please cite his work alongside this repository (see `CITATION.cff`). The Zero-$a_3$ Bi-Directional
$Q_0/Q_1$ Bridge, Carry-Sudoku / Slot-Sudoku solvers, and preimage witnesses in this repository and
[`PAPER_19_20_PADDED_SHA256_final-v1.pdf`](PAPER_19_20_PADDED_SHA256_final-v1.pdf) were discovered and
developed using **Google DeepMind Gemini 4 Argon** under the direction and review of the repository owner (`srw@egont.xyz`).
Released under the MIT licence.
