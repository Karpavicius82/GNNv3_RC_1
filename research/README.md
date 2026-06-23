# research/ — production studies and exploratory probes

Standalone C++20 programs from the build-out of the system. Some include the
substrate core via `../tools/graph_wave_substrate.hpp`; build from a checkout that
keeps `tools/` and `research/` siblings.

The first three groups are the **production-verified** results that back the engine,
the glue, and the real-data benchmark. The last group is the **honest exploratory
record** — including probes that failed or were superseded. Negative results are
kept on purpose; they mark the dead-ends so they are not re-walked.

## Production engine (verified)
| file | result |
|---|---|
| `probe_sparse_scale` | sparse unitary propagation, 1,000,000 nodes, norm drift 2.22e-16, linear time |
| `probe_physics` | propagator exact at any t (1e-12 vs exact eig), gauge-invariant 6.9e-17, flux=π 4.95e-16 |
| `probe_crosscheck` | sparse Chebyshev engine vs exact eigendecomposition: 1.13e-12 |

## Decorrelation glue (verified)
| file | result |
|---|---|
| `probe_whiten` | whitened routing 1.000 at correlation 0.0–0.98 where naive collapses to 0.48 |
| `probe_lateral` | native conjugate gradient ≡ exact G⁻¹ (1e-16…1e-10); naive Jacobi only holds to ~0.8 |

## Real-data benchmark
| file | result |
|---|---|
| `probe_cora` | weights-free GNN on the Cora citation graph: **77.4%** (see `CORA_DATA.md` to fetch data) |

## Exploratory record (honest — includes negatives)
| file | what it showed |
|---|---|
| `probe_a4_abcage` | Aharonov–Bohm cage: exact flux-gated confinement, length-invariant |
| `probe_a5_compose` | composed flux gates: exact, independent, conserving |
| `probe_steer` | phase-ramp beamforming: phase alone steers the field, linearly |
| `probe_diffract` | the field as a diffraction map; holographic refocus 7.3× (phase carries it) |
| `probe_gnn1` | weights-free GNN exceeds the 1-WL ceiling (distinguishes C6 vs 2·C3) |
| `probe_a1_levelspacing` | **failed** — level-spacing collapses to 0 on degenerate lattice spectra |
| `probe_a3_diffusion`, `probe_a3b_robust` | diffusion exponent **drifts** with box size — not a converged result |
| `probe_a2_phasedepth` | phase survives depth (chirality), with a snapshot caveat |
| `probe_holoclf` | diffractive matched-filter classifier — **focus-limited**, principle exact / realization weak |
| `probe_depth_close`, `probe_settle_route` | single Born / settling **do not** close the correlated-content gap |
| `probe_gnn_prod`, `probe_pipeline`, `probe_pipeline2` | scaled synthetic pipelines — message passing helps (13%→100%) but the synthetic tasks were either trivial or regime-mismatched; superseded by `probe_cora` on real data |
