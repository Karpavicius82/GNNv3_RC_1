# Graph-Wave Scope Note — 2026-06-20

Honest record of a scope slide and how it is handled. Nothing is deleted; the
substrate-only discipline is restored and the out-of-scope work is parked.

## The sanctioned discipline (unchanged)

Per `GRAPH_WAVE_FOUNDATION_CHECKPOINT_2026-06-16.md`, "Current Research
Discipline": the allowed work is **substrate-only unless explicitly changed**:

- refine direct graph-wave substrate candidates
- maintain the GNN-analog **grammar** contracts (port, depth, fan-in,
  node-update, shared-edge, degree-2)
- compare spectra against controls
- keep measurement tests separate
- **do not add readout, decision, learning, GA, trainers, or optimizers to
  substrate tests**

> "Choosing a substrate does not automatically authorize measurement, decision,
> trainer, GA, readout, or integration work. Those are separate tracks and are
> out of scope here."

One question per test; do not mix goals:
`substrate physics != measurement != decision != learning != integration`.

## What happened (the slide)

Between 2026-06-18 22:25 and 2026-06-20 02:29 the work crossed scope. The first
out-of-scope commit is `3fc327d` (amplitude-amplification, framed as a "decision
primitive"). It cascaded into a full decision / learning / GNN-computation /
living-systems stack across `3fc327d..61bdfc9` (~24 contracts + 4 diagnostics).
The driver was a framing error: treating "build the full GNN" as the goal, when
the discipline names the GNN-analog work as *substrate grammar only*, with
decision / learning / measurement as separate, not-yet-authorized tracks.

## In-scope vs out-of-scope

IN SCOPE (sanctioned, and high quality — machine precision):
- substrate physics: unitarity, gauge/Wilson, observable, interferometer,
  spectral comparison, flux scan, absorber, stream contact
- GNN-analog grammar: node_update, fanin_aggregation, port_transfer_composition,
  physical_port_wiring, port_wiring_robustness, physical_depth_chain,
  shared_edge_consistency, degree2_path_guardrail
- substrate-grammar-adjacent physics: dynamic_bind, general_graph_bind,
  superposition_compute, fractional_encoding, settling, entanglement(+dynamics)

OUT OF SCOPE (exploration, PARKED — not the frontier):
- decision: amplitude_amplification, resonator_cleanup, dissipative_cleanup,
  resonator_factorization
- memory/learning: holographic_memory, vsa_structure, learning, end_to_end,
  int16_quantization, physics_necessity, flatband_memory
- GNN computation: gnn_message_passing, attention, deep_attention, gnn_task
- living systems: metabolism, plasticity, autopoiesis, closure
- diagnostics: autopoiesis_noise, closure_noise, living_scale, bistable_drive

## Quality finding (why the discipline matters)

Quality tracked the slide. The in-scope grammar is exact (gauge covariance
~1e-15, aligned/destructive 1.0/1e-31, depth-chain mirror-coupled ~1e-12). The
out-of-scope GNN computation is toy/modest (message-passing neighbour overlap
~0.33 on an N=5 graph). The living layer is FRAGILE: its self-organisation
collapses at noise sigma ~= 0.02 because the drive is a blanket gain over an
unstable vacuum; four diagnostics localise this, and a bistable "fix" also failed
— revealing a fundamental tension between self-sustenance (life/memory) and
selectivity (noise-robustness). The discipline was protecting quality.

## Handling

- Nothing deleted: all out-of-scope contracts are isolated `EXCLUDE_FROM_ALL`
  labs with no V26 linkage; they did not pollute production.
- They are reclassified as parked exploration (this note + memory).
- Ideas worth keeping (NOT a license for scope creep): weights+GA are foreign /
  compute by phase+interference+superposition; superposition is live and settles;
  physics-necessity shuffled-control discipline; flat-band = persistence. These
  belong to the deferred decision/learning track.

## Real next step (sanctioned)

Return to substrate-only: refine / choose a graph-wave substrate candidate,
compare spectra against controls, maintain the grammar contracts. Decision,
learning, and integration stay separate tracks until the owner explicitly opens
them.
