# GNNv3 RC1 Validation - 2026-07-01

This validation records the local GNNv3 RC1 state after the self-sensing medium
checkpoint. No source code changes were made for this run.

Repository state:

```text
branch: main
validated source head: 5d009146ee15e1a7e17c9c07747092d03289a7ea
validated source commit: codex: add GNNv3 self-sensing medium contract
remote: https://github.com/Karpavicius82/GNNv3_RC_1.git
```

## GNNv3 RC1 Tests

Command:

```bat
ctest --test-dir build -C Release -L gnnv3 --output-on-failure
```

Result:

```text
2 / 2 passed
```

Registered GNNv3 tests:

```text
graph_wave_v3_feeling_gate_contract_test
graph_wave_v3_self_sensing_medium_contract_test
```

## 1M Stream Results

Command:

```bat
build\Release\graph_wave_v3_feeling_gate_contract_test.exe 1000000
```

Result:

```text
CONTRACT RESULT : 8 / 8 verified
RESULT : PASS
```

Key metrics:

```text
active_pairs=0.631
psi compression=2.05x
chi compression=3.56x
tau_mean=21.558313
aperture_mean=21.291863
max_rho=47.371
skipped current bias=0.008
skipped stress bias=0.080
gated_sync_error=0
```

Command:

```bat
build\Release\graph_wave_v3_self_sensing_medium_contract_test.exe 1000000
```

Result:

```text
CONTRACT RESULT: 8 / 8 PASS
```

Key metrics:

```text
active_edges=0.572
psi compression=1.72x
chi compression=2.03x
internal support separation: psi=5.22 chi=9.24
skipped current bias=0.006
skipped stress bias=0.161
reverse-order sync_error=0
max_rho=4.339
max_tau=0.686
```

## Broader Contract Checks

Recognition / learning / semantic checks:

```text
graph_wave_gnn_task_contract_test                  4 / 4 PASS
graph_wave_learning_contract_test                  3 / 3 PASS
graph_wave_attention_contract_test                 4 / 4 PASS
graph_wave_deep_attention_contract_test            4 / 4 PASS
graph_wave_semantic_word_substrate_contract_test   6 / 6 PASS
graph_wave_streaming_word_plasticity_contract_test 8 / 8 PASS
graph_wave_vsa_structure_contract_test             3 / 3 PASS
```

Physics / nonlinear / bridge checks:

```text
graph_wave_nonlinear_compute_contract_test         4 / 4 PASS
graph_wave_horizon_densification_contract_test     8 / 8 PASS
graph_wave_superposition_compute_contract_test     4 / 4 PASS
graph_wave_stream_order_wilson_flux_contract_test  4 / 4 PASS
graph_wave_overlap_bridge_contract_test            6 / 6 PASS
graph_wave_resonance_bridge_contract_test          7 / 7 PASS
graph_wave_physics_necessity_contract_test         2 / 2 PASS
```

Full regression:

```bat
ctest --test-dir build -C Release --output-on-failure -j 8
```

Result:

```text
62 / 62 passed
```

## Interpretation

The RC1 self-sensing substrate is locally closed at the contract level:

- `psi/chi/tau/aperture` carry the active physics window without label/oracle input.
- The feeling gate reduces heavy flow while keeping the field readable.
- `chi` remains denser than `psi` and separates support more strongly.
- Reverse-order synchronization stays exact.
- The skipped phase load is not one-sidedly clipped.
- The broader contract set confirms recognition, learning, semantic structure,
  nonlinear computation, densification, Wilson-flux order and bridge/resonance
  behavior remain intact.

This validation does not add a new mechanism. It records that the existing GNNv3
RC1 substrate behaves as a self-sensing physical substrate under the local test
matrix.
