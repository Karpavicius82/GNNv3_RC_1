# GNNv2 — a weights-free graph neural substrate

A graph neural network built **without weights and without a trainer**. Computation
is carried by **phase, interference and superposition** on a unitary graph-wave
substrate; meaning lives in the **gauge-invariant flux**, not in stored numbers.

This is an independent system. It has nothing in common with any prior control
stack — it is substrate physics first, then a GNN grown on top of it.

```
complex state z  →  Hermitian graph Hamiltonian H (topology + flux)  →  unitary evolution
```

## The three substrates (the closed system)

```
   SRAUTAS / FLOW            DECORRELATION              SETTLING / MEMORY
   unitary propagation   →   G^-1 (CG / lateral    →    resonator cleanup,
   (message passing,         inhibition): separates     holographic recall,
   flux is the signal)       correlated content         amplitude amplification
        mix / spread              the glue                   sharpen / decide
```

- **FLOW** — one GNN layer is one unitary step `U = e^{-iHt}`; depth is `U^k`; the
  irreducible op is a phase-weighted multiply-accumulate over edges
  `z_i ← Σ_j e^{iδ_ij} z_j`. No weights — `δ_ij` is the gauge flux.
- **DECORRELATION** — raw overlaps `⟨k_i, q⟩` are contaminated when patterns are
  correlated (off-diagonal Gram). The dual coordinates `c = G⁻¹⟨k,q⟩` decontaminate.
  Native realization = recurrent lateral inhibition / conjugate gradient. This is
  the glue that lets FLOW connect to MEMORY on real (correlated) data.
- **SETTLING / MEMORY** — dissipative relaxation onto the nearest attractor;
  holographic bind/unbind; flat-band persistence; amplitude amplification (decision).

## Verified numbers (all run live)

### Substrate physics — machine precision
| | number |
|---|---|
| unitarity (1000 steps) | norm drift 1.68e-13, operator unitarity 1.2e-15 |
| destructive interference | 5.6e-31 (constructive/destructive ratio 3.8e11×) |
| gauge / Wilson invariance | Wilson-loop diff 4.4e-16, Landau flux err 0 |
| interferometer routing | flux=0 → out+ 0.99999, flux=π → out− 0.99999 |

### Production engine — sparse unitary propagation
| | number |
|---|---|
| scale | **N = 1,000,000 nodes**, ~1 s, **linear time** (Chebyshev, 23 terms) |
| unitarity at scale | norm drift **2.22e-16** |
| propagator exact at ANY t (vs exact eigendecomposition) | 1.13e-12 @t1, 1.26e-12 @t5, 1.8e-12 @t20 |
| gauge invariance of the engine | \|ψ\|² unchanged under local gauge = 6.9e-17 |
| exact flux=π decoupling through the engine | 4.95e-16 |

### The GNN
| | number |
|---|---|
| node classification (gnn_task) | **100%**; message-passing necessary (own-feature 69% ≈ chance) |
| weights-free learning | **99.5%** on unseen features; shuffled-label control 52% (genuinely learned) |
| depth (2-hop) | 1-round 57% → 2-round 93% |
| deep attention | 2 layers route (overlap 1.000); 1 layer / linear cannot (0.048) |

### Decorrelation glue
| | number |
|---|---|
| whitened 2-hop routing (correlated keys) | **1.000** at correlation 0.0–0.98 |
| same task, naive readout (Born) | 0.48–0.65 (collapses) |
| native CG ≡ exact G⁻¹ | match to 1e-16…1e-10 |

### Memory / cleanup / decision
| | number |
|---|---|
| holographic unbind | sim 1.000000000000 |
| flat-band memory | recall 100%, fidelity 1.0 to t=30 |
| resonator cleanup | corrupted → 1.000000, recall 24/24 |
| amplitude amplification (decision) | 7.56× over uniform, winner = target |

**32 substrate/GNN contracts — all green.**

### Real data — Cora citation graph (2708 nodes, 5429 edges, 7 classes), weights-free
| method | naive | + decorrelation |
|---|---|---|
| own features (no message passing) | 58.3% | 58.6% |
| FLOW 1-hop | 74.6% | 74.8% |
| FLOW 2-hop | **77.4%** | 77.9% |

20 labels/class, test = 2568 nodes. No weights, no backprop (prototypes = class
means, G⁻¹ = the prototype Gram, FLOW = graph aggregation). The own-features 58.3%
matches the known raw-feature reference (~59%); message passing lifts it to **77.4%**,
beating label-propagation (~68%) and approaching the **trained** GCN (~81.5%) — with
no trained parameters. (Decorrelation is marginal on Cora because its classes are
only mildly correlated; it is decisive in the binding/high-correlation regime where
naive routing collapses to 0.48, see `research/probe_whiten.cpp`.) Code: `research/probe_cora.cpp`.

## Layout

- `tools/` — the substrate core (`graph_wave_substrate.hpp`) and the contract tests
  (substrate physics, GNN grammar, the working GNN, memory, decision). Each is a
  self-contained executable that prints its own invariants and a PASS/FAIL.
- `research/` — session research probes: the sparse scaling engine
  (`probe_sparse_scale`, `probe_physics`, `probe_crosscheck`), the decorrelation
  glue (`probe_whiten`, `probe_lateral`), and exploratory studies (some carry
  honest negative results — kept as the research record).
- `docs/` — the foundation checkpoint, the accepted GNN grammar, and the scope note.

## Build (Windows / MSVC)

Each file is standalone C++20. From a Developer prompt:

```bat
cl /O2 /EHsc /std:c++20 tools\graph_wave_unitarity_test.cpp && graph_wave_unitarity_test.exe
cl /O2 /EHsc /std:c++20 research\probe_sparse_scale.cpp     && probe_sparse_scale.exe
```

Files in `research/` that include the substrate core expect it one directory up
(`#include "../tools/graph_wave_substrate.hpp"` after relocation, or build from a
checkout that keeps `tools/` and `research/` siblings — adjust the include path).

## Discipline

Everything is **weights-free** and verified by **exact finite-N identities**
(unitarity, gauge invariance, exact interference) at machine precision — not by
fitted exponents or asymptotic claims. Where a result is uncertain or a probe
failed, that is recorded honestly rather than hidden.
