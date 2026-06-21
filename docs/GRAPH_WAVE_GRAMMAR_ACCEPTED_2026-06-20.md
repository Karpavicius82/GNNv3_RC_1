# Graph-Wave GNN-Analog Grammar — Accepted as Isolated Substrate Work (2026-06-20)

Per the promotion discipline in `GRAPH_WAVE_FOUNDATION_CHECKPOINT_2026-06-16.md`
("Before adding another primitive, this expansion should be either accepted as
isolated substrate work or split into a smaller checkpoint"), the GNN-analog
substrate grammar is hereby **accepted as a stable, isolated substrate grammar**.
No further grammar primitives are added until/unless a new question requires one.

## The accepted grammar (all isolated `EXCLUDE_FROM_ALL`, no V26, no decision/learning)

Core primitives (degree 1-2):
- `graph_wave_node_update_contract_test`        — self-state + neighbour messages -> updated centre
- `graph_wave_fanin_aggregation_contract_test`  — coherent aggregate port + residual modes
- `graph_wave_port_transfer_composition_contract_test` — composable layer; collapses linearly
- `graph_wave_physical_port_wiring_contract_test`
- `graph_wave_port_wiring_robustness_contract_test`
- `graph_wave_physical_depth_chain_contract_test` — ordered multi-cell transfer
- `graph_wave_shared_edge_consistency_contract_test`   — reciprocal two-node consistency
- `graph_wave_degree2_path_guardrail_contract_test`    — rejects one-rail broadcast; edge rails

General-degree materialization (2026-06-20, this checkpoint):
- `graph_wave_message_materialization_contract_test`   — hub of arbitrary degree d
- `graph_wave_graph_layer_materialization_contract_test` — full small-graph layer, locality
- `graph_wave_materialization_physical_contract_test`  — kernel = real Hermitian propagation (T = -i W)

## Invariants the grammar holds (machine precision)

- Hermitian medium; unitary propagation (`U^dag U = I`, gauge/Wilson invariant).
- Message passing = unitary propagation; aggregation = phase-controlled
  interference of a SUPERPOSITION; the meaning lives in the gauge-invariant
  (flux) phases, not absolute phase.
- Explicit oriented edge-rails (NOT classical broadcast — broadcast rows overlap
  and are rejected); residual modes conserve the superposition (no false
  unitary compression).
- No optimizer, trainer, decision loop, learning, readout, or V26 dependency.

## What is NOT yet built (deliberately, separate tracks, out of scope here)

Measurement, decision, learning, integration. Depth/composition at full
graph-layer scale is a possible future grammar question but is paused under the
promotion discipline.

## Next sanctioned track

Substrate-candidate selection by spectra: `substrate_spectral_comparison_test`
and `substrate_flux_scan_test` compare plain / random-phase / golden-flux
substrates (fractal resonance through Fibonacci spectral gaps). The open
foundational question is which substrate candidate to choose by its spectral
signature — this is "refine direct graph-wave substrate candidates; compare
spectra against controls", the first allowed-work item.
