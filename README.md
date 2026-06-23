# GNNv2 — a weights-free graph neural substrate

A graph neural network built **without weights and without a trainer**. Computation
is carried by **phase, interference and superposition** on a unitary graph-wave
substrate; meaning lives in the **gauge-invariant flux**, not in stored numbers.

```
complex state z  →  Hermitian graph Hamiltonian H (topology + flux)  →  unitary evolution
```

## The three substrates

```
   FLOW                     DECORRELATION                 SETTLING / MEMORY
   unitary propagation  →   c = G^-1 ⟨k,q⟩           →    dissipative relaxation,
   (message passing,        (separate correlated          holographic recall,
    flux is the signal)      patterns; the glue)          amplitude amplification
   mix / spread             make content separable        sharpen / decide
```

One layer reduces to a single phase-weighted multiply-accumulate over edges,
`z_i ← Σ_j e^{i·δ_ij} z_j`; depth is `U^k`. No weights — the edge flux `δ_ij` is the
structure. Full description in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Headline results

| | |
|---|---|
| substrate physics | unitarity 1.2e-15, gauge/Wilson 4.4e-16, destructive interference 5.6e-31 |
| engine | **1,000,000 nodes**, unitary 2.2e-16, linear time; propagator exact at any t (1e-12) |
| GNN | classification **100%**, weights-free learning **99.5%** on unseen (shuffle 52%) |
| decorrelation glue | routing **1.000** where naive collapses to 0.48 |
| **real data (Cora)** | **77.4%** weights-free — beats label-prop (~68%), nears trained GCN (~81.5%) |

Full table in [`docs/RESULTS.md`](docs/RESULTS.md). Every number was run live;
nothing is fitted.

## Repository layout

| path | contents |
|---|---|
| [`tools/`](tools/README.md) | substrate core (`graph_wave_substrate.hpp`) + 59 contract tests (physics, GNN grammar, the working GNN, memory, decision). 32 green at machine precision. |
| [`research/`](research/README.md) | production studies (scaling engine, decorrelation glue, the Cora benchmark) and honest exploratory probes, including negative results. |
| [`docs/`](docs/) | `ARCHITECTURE.md`, `RESULTS.md`. |
| `scripts/` | `run_tests.ps1` — build and run every contract, print a pass/fail summary. |

## Build & run (Windows / MSVC)

Each file is standalone C++20. From a Developer prompt:

```bat
cl /O2 /EHsc /std:c++20 /I tools tools\graph_wave_unitarity_test.cpp && graph_wave_unitarity_test.exe
cl /O2 /EHsc /std:c++20 /I tools research\probe_sparse_scale.cpp       && probe_sparse_scale.exe
```

Run the whole contract suite:

```
powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1
```

CMake (builds every contract and probe as a target) is also provided:

```
cmake -B build -S . && cmake --build build
```

## Principle

Everything is **weights-free** and verified by **exact finite-N identities**
(unitarity, gauge invariance, exact interference) at machine precision — not by
fitted exponents or asymptotic claims. Where a result is uncertain or a probe
failed, that is recorded honestly rather than hidden.
