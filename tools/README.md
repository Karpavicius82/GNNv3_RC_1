# tools/ — substrate core and contract tests

`graph_wave_substrate.hpp` is the single source of truth for the physics (Hermitian
graph families, the unitary Cayley stepper, the orthonormal-row kernel, the
eigensolver). Every other file is a **self-contained contract test**: it builds to a
standalone executable that prints its own invariants and a `RESULT : n / n verified`
line. The repo currently has **62 C++ contract tests** registered through CTest; the
GNNv3 RC1 carrier-field tests are labeled `gnnv3`. A contract proves one property
exactly, at machine precision.

Build any one (Windows / MSVC, Developer prompt):

```bat
cl /O2 /EHsc /std:c++20 /I . graph_wave_unitarity_test.cpp && graph_wave_unitarity_test.exe
```

## Substrate physics (foundation)
| file | proves |
|---|---|
| `graph_wave_unitarity_test` | unitary evolution, time reversibility, interference, diffraction |
| `complex_graph_wave_minimal_test` | superposition preserved, relative phase controls interference |
| `gauge_wilson_invariance_test` | edge phases are physical Wilson holonomies, not coordinates |
| `graph_wave_observable_contract_test` | non-invasive observables (probability, Wilson phase, edge current) |
| `graph_wave_interferometer_cell_contract_test` | flux routing: 0 → out+, π → out− |
| `graph_wave_substrate_port_contract_test` | the clean packet → medium → observable boundary |
| `absorbing_screen` | measurement as a separate non-Hermitian layer (detector accounting) |

## GNN grammar (message-passing primitives)
| file | proves |
|---|---|
| `graph_wave_node_update_contract_test` | one node update `h_v,{h_u},edges → h'_v` with residual modes |
| `graph_wave_fanin_aggregation_contract_test` | neighbourhood aggregation, no false compression |
| `graph_wave_fanout_contract_test` | the dual of fan-in (enables unitary depth) |
| `graph_wave_message_round_contract_test` | **one working unitary message-passing layer** |
| `graph_wave_message_materialization_contract_test` | degree-d hub aggregation by interference |
| `graph_wave_graph_layer_materialization_contract_test` | a full small-graph layer in one unitary |
| `graph_wave_materialization_physical_contract_test` | the kernel as real Hermitian propagation |
| `graph_wave_flux_gnn_contract_test` | message passing where the **flux** computes |
| `graph_wave_deep_round_contract_test` | depth via chainable field propagation (honest: edge-rails don't chain) |
| `graph_wave_port_transfer_composition_contract_test` | composition grammar (and the linear-collapse boundary) |
| `graph_wave_physical_port_wiring_contract_test` | algebraic composition as one Hermitian body |
| `graph_wave_port_wiring_robustness_contract_test` | tolerance of the wiring under defects |
| `graph_wave_physical_depth_chain_contract_test` | depth needs engineered coupling, not naive stacking |
| `graph_wave_shared_edge_consistency_contract_test` | two node updates sharing one reciprocal edge |
| `graph_wave_degree2_path_guardrail_contract_test` | broadcast is invalid; edge-message rails required |

## GNN computation (the working GNN)
| file | proves |
|---|---|
| `graph_wave_gnn_message_passing_contract_test` | one live superposition: features + propagation = aggregation |
| `graph_wave_attention_contract_test` | content-dependent message passing; composed layers don't collapse |
| `graph_wave_deep_attention_contract_test` | depth advantage: 2 layers route, 1 layer / linear cannot |
| `graph_wave_gnn_task_contract_test` | **end-to-end node classification, 100%** |
| `graph_wave_learning_contract_test` | **weights-free learning, 99.5% on unseen, shuffle-controlled** |

## Binding / VSA / compute
`graph_wave_dynamic_bind`, `graph_wave_general_graph_bind`,
`graph_wave_superposition_compute`, `graph_wave_fractional_encoding`,
`graph_wave_entanglement`, `graph_wave_entanglement_dynamics`, `graph_wave_settling`.

## Memory / cleanup / decision
`graph_wave_holographic_memory`, `graph_wave_vsa_structure`,
`graph_wave_flatband_memory`, `graph_wave_resonator_cleanup`,
`graph_wave_dissipative_cleanup`, `graph_wave_resonator_factorization`,
`graph_wave_amplitude_amplification`.

## Spectral diagnostics (substrate characterisation)
`substrate_spectral_comparison`, `substrate_flux_scan`, `substrate_scaling_diagnostic`,
`substrate_multifractal_diagnostic`, `substrate_criticality_diagnostic`,
`graph_wave_flux_plaquette_diagnostic`, `graph_wave_lattice_transport_diagnostic`.
These characterise the golden-flux substrate's spectrum; the strong, decisive
results are unitarity / gauge / flux above — spectral criticality is a weaker,
size-dependent signal and is treated as characterisation, not a load-bearing claim.

## Exploratory — self-organising layer (kept, but FRAGILE)
`graph_wave_metabolism`, `graph_wave_plasticity`, `graph_wave_autopoiesis`,
`graph_wave_closure` and their noise diagnostics
(`graph_wave_autopoiesis_noise_diagnostic`, `graph_wave_closure_noise_diagnostic`,
`graph_wave_living_scale_diagnostic`, `graph_wave_bistable_drive_diagnostic`).
These explore self-sustaining dynamics. The diagnostics show the self-organisation
is **fragile** — it collapses under noise (σ ≈ 0.02). Kept as an honest record; not
part of the working GNN.

Other: `graph_wave_end_to_end`, `graph_wave_int16_quantization`,
`graph_wave_physics_necessity`.

## GNNv3 RC1 carrier-field contracts

| file | proves |
|---|---|
| `graph_wave_v3_feeling_gate_contract_test` | `chi/tau/aperture` narrow the active local physics window without trigonometric gate/readout; 1M stream passes 8/8 with active pairs 0.631 |
| `graph_wave_v3_self_sensing_medium_contract_test` | fixed SoA carrier medium can carry its own active heavy-flow window; 1M stream passes 8/8 with active edges 0.572, `psi` support separation 5.22, `chi` support separation 9.24 |
