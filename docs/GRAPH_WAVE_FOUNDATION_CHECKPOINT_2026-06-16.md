# Graph-Wave Substrate Foundation Checkpoint - 2026-06-16

This checkpoint preserves the new graph-wave direction without treating the
existing V26 control system as part of the new foundation.

## Status

The foundation is substrate-only:

```text
complex state z -> Hermitian graph Hamiltonian H -> unitary/gauge evolution
```

The intended direction is still GNN-analog, but only at the substrate contract
level:

```text
graph adjacency        -> Hermitian graph topology
message passing        -> unitary propagation over H
edge/phase parameters  -> physical flux/coupling fields
neighborhood fan-in    -> coherent aggregate port + residual modes
node update            -> self-state + neighbor messages -> updated center port
shared edge            -> reciprocal two-node update consistency
degree-2 path          -> explicit edge-message rails, not one-rail broadcast
depth                  -> port-wired physical composition
```

This GNN framing is allowed. V26 is not prohibited; it remains the existing
system context and a possible future boundary. What is not allowed in this
checkpoint is letting the existing V26 control stack define the substrate, or
treating substrate tests as a production decision/training path.

The substrate work in this checkpoint is not wired into `src/control`, runtime
decisions, learning JSON files, or production service paths.

Anything that links `StateEncoder`, `RunState`, Living Silicon, GA/readout, or
V26 runtime code is boundary/contact work, not substrate foundation. Those tests
may audit where a V26 boundary would need to exist, but must not define the new
graph-wave substrate.

The CMake targets are intentionally `EXCLUDE_FROM_ALL`, so normal builds do not
pull this direction in by accident.

## 2026-06-18 History Review Gate

Git history shows two separate layers that must not be blurred:

- committed graph-wave foundation baseline: `8814e73`
- local GNN-analog substrate expansion: port transfer, physical wiring,
  robustness, physical depth, fan-in, node update, shared-edge consistency, and
  degree-2 path guardrails

Older V26 continuation/foundation notes may still describe a `GNN+GA` native
paradigm for the previous control path. That is historical V26 context, not an
instruction to reintroduce GA into this graph-wave substrate checkpoint.

Promotion discipline after this review:

- Core substrate extension candidates: port transfer, physical wiring,
  robustness, physical depth, fan-in, and node update.
- Guardrail/experimental candidates: shared-edge consistency and degree-2 path.
  They are useful because they reject unsafe graph semantics, but they are not a
  general graph layer.
- No optimizer, trainer, decision loop, legacy runtime adapter, production
  integration, or V26 control-path dependency is opened by any of these tests.
- Before adding another primitive, this expansion should be either accepted as
  isolated substrate work or split into a smaller checkpoint.

Build the committed independent substrate baseline explicitly:

```powershell
cmake --build build --config Release --target graph_wave_foundation_contract_tests
.\build\Release\graph_wave_unitarity_test.exe
.\build\Release\complex_graph_wave_minimal_test.exe
.\build\Release\gauge_wilson_invariance_test.exe
.\build\Release\graph_wave_substrate_port_contract_test.exe
.\build\Release\graph_wave_observable_contract_test.exe
.\build\Release\graph_wave_interferometer_cell_contract_test.exe
.\build\Release\substrate_spectral_comparison_test.exe
.\build\Release\substrate_flux_scan_test.exe
```

Build the local GNN-analog core expansion explicitly:

```powershell
cmake --build build --config Release --target graph_wave_gnn_core_contract_tests
.\build\Release\graph_wave_port_transfer_composition_contract_test.exe
.\build\Release\graph_wave_physical_port_wiring_contract_test.exe
.\build\Release\graph_wave_port_wiring_robustness_contract_test.exe
.\build\Release\graph_wave_physical_depth_chain_contract_test.exe
.\build\Release\graph_wave_fanin_aggregation_contract_test.exe
.\build\Release\graph_wave_node_update_contract_test.exe
```

Build the local guardrail/experimental expansion explicitly:

```powershell
cmake --build build --config Release --target graph_wave_gnn_guardrail_contract_tests
.\build\Release\graph_wave_shared_edge_consistency_contract_test.exe
.\build\Release\graph_wave_degree2_path_guardrail_contract_test.exe
```

Build measurement separately:

```powershell
cmake --build build --config Release --target absorbing_screen_test
.\build\Release\absorbing_screen_test.exe
```

Legacy V26 contact probes are documented separately in
`docs/GRAPH_WAVE_LEGACY_CONTACT_AUDIT_2026-06-17.md`.

## Files

- `tools/graph_wave_unitarity_test.cpp`
  Proves the unitary graph-wave substrate.
- `tools/complex_graph_wave_minimal_test.cpp`
  Minimal test for unitary evolution, superposition, phase-controlled
  interference, and global phase invariance.
- `tools/gauge_wilson_invariance_test.cpp`
  Proves that graph phases are gauge/Wilson holonomies, not coordinate
  artifacts.
- `tools/graph_wave_substrate_port_contract_test.cpp`
  Defines the pure new-engine substrate port:
  `complex packet + Hermitian medium -> observables`. No V26 types, no
  measurement, no decision, no learning.
- `tools/graph_wave_observable_contract_test.cpp`
  Defines the non-invasive observable layer: probability, port power,
  contrast, Wilson phase, edge current, participation ratio, and entropy.
  No absorber, no decision, no learning.
- `tools/graph_wave_interferometer_cell_contract_test.cpp`
  Defines a substrate-grown unitary interferometer cell. One input port, two
  arms, two output ports, Wilson-flux routing. No absorber, no measurement
  layer, no decision, no learning.
- `tools/graph_wave_port_transfer_composition_contract_test.cpp`
  Defines the first composable port-transfer contract: two input rails, two
  output rails, transfer matrix algebra, gauge covariance across port frames,
  and explicit confirmation that unitary cell stacks still collapse linearly
  without a separate measurement/nonlinearity boundary.
- `tools/graph_wave_physical_port_wiring_contract_test.cpp`
  Tests whether algebraic port composition can be realized as one larger
  Hermitian graph-wave body: input rails -> shared bus rails -> output rails.
  Confirms the physical wiring is clocked by a calibrated time-of-flight
  boundary and remains linear without measurement/re-injection.
- `tools/graph_wave_port_wiring_robustness_contract_test.cpp`
  Falsification-style local tolerance check around the physical wiring contract.
  Measures deterministic coupling, phase, and combined perturbation ladders, and
  verifies that uniform coupling scale behaves as a retimable clock detuning.
- `tools/graph_wave_physical_depth_chain_contract_test.cpp`
  Tests a three-cell / four-frame physical depth chain. It deliberately records
  that uniform inter-frame coupling is not a full-transfer primitive, then shows
  that mirror-coupled gains `sqrt(3), 2, sqrt(3)` recover ordered transfer and
  bounded local disorder response.
- `tools/graph_wave_fanin_aggregation_contract_test.cpp`
  Defines a graph-topology fan-in primitive: three input rails transfer into
  one aggregate output mode plus two orthogonal residual modes. This tests
  aggregation without making a false unitary-compression claim.
- `tools/graph_wave_node_update_contract_test.cpp`
  Defines the first GNN-analog node-update primitive: center self-state plus
  three neighbor messages transfer into one updated-center output mode plus
  three residual modes. This tests self contribution, edge-controlled neighbor
  messages, neighbor relabeling equivariance, gauge covariance, and explicit
  no-compression accounting.
- `tools/graph_wave_shared_edge_consistency_contract_test.cpp`
  Defines the first two-node shared-edge consistency primitive. It tests whether
  two local node-update rows can share one reciprocal physical edge without
  losing orthogonality, gauge covariance, edge-phase control, or residual-mode
  accounting.
- `tools/graph_wave_degree2_path_guardrail_contract_test.cpp`
  Defines the first degree-2 path guardrail. It proves that naive one-rail
  broadcast from a center node into two leaf updates is invalid in a unitary
  substrate, then validates an edge-lifted three-node path using explicit
  oriented message rails.
- `tools/substrate_spectral_comparison_test.cpp`
  Pure spectral comparison of plain, random-phase, and golden-flux graph-wave
  substrates. No measurement, no decision, no learning.
- `tools/substrate_flux_scan_test.cpp`
  Pure spectral flux-family sanity check for direct graph-wave Hamiltonians. No
  measurement, no decision, no learning.
- `tools/absorbing_screen.cpp`
  Proves a separated measurement layer over the substrate. This is not part of
  the substrate foundation itself.
- `CMakeLists.txt`
  Adds lab targets only; they are excluded from default builds.

V26-contact probes quarantined from the substrate foundation:

- `tools/stream_to_wave_injection_contract_test.cpp`
- `tools/stream_wave_gauge_contact_test.cpp`

These two tests deliberately link old V26 types. They are useful only for
auditing how legacy contact behaves and where a boundary would need to be
enforced. They are not a migration plan and are not evidence that the new engine
should be integrated into V26 as-is.

Older exploratory polar tests remain separate lab artifacts:

- `tools/polar_interference_lab.cpp`
- `tools/polar_physics_invariants_test.cpp`
- `tools/polar_measurement_loop_test.cpp`

## Proven So Far

### 1. Unitary Substrate

State:

```text
z in C^N
||z||^2 = sum_i |z_i|^2
```

Graph law:

```text
H_ij = A_ij * exp(i * delta_ij)
H_ji = conj(H_ij)
H = H^dagger
```

Evolution uses the Cayley / Crank-Nicolson step:

```text
K = H * dt / 2
U = (I + iK)^-1 (I - iK)
```

This gives unitary evolution to solver precision.

Validated invariants:

- norm conservation over 1000 steps
- time reversibility
- operator check `U^dagger U ~= I`
- interference through a gauge-flux diamond
- diffraction through aperture geometry
- fractal resonance through Fibonacci spectral gaps

Latest local result:

```text
graph_wave_unitarity_test: 5 / 5 PASS
max norm drift: 1.683e-13
return error: 1.370e-14
operator unitarity error: 1.221e-15
diamond destructive transfer at phi=pi: 5.628e-31
diffraction L1(single,double): 0.466
Fibonacci significant gaps: 4, uniform gaps: 0
```

### 2. Minimal Complex Graph-Wave Substrate

This test asks only whether the complex graph-wave substrate has the minimum
physics needed for later investigation. It does not test measurement, decision,
learning, or V26 integration.

Validated facts:

- Hermitian graph Hamiltonian gives unitary evolution.
- Superposition is preserved by propagation.
- Relative loop phase controls interference.
- Global phase is not observable.

Latest local result:

```text
complex_graph_wave_minimal_test: 4 / 4 PASS
max norm drift: 1.55e-14
linearity error: 2.86e-16
flux=0 target transfer: 0.381926
flux=pi target transfer: 5.627628e-31
global phase probability error: 1.67e-15
```

### 3. Gauge / Wilson Holonomy

This test asks only whether edge phases are physical loop holonomies rather than
a coordinate artifact.

Validated facts:

- Wilson loop phase is invariant under local gauge changes.
- Evolution is gauge-covariant.
- Interference depends on loop flux, not on which edge carries the phase.
- Landau-gauge plaquettes carry the expected Wilson flux.

Latest local result:

```text
gauge_wilson_invariance_test: 4 / 4 PASS
Wilson loop diff after gauge transform: 4.44e-16
gauge-covariant state error: 1.41e-15
probability difference after gauge transform: 1.72e-15
same-flux phase placement transfer diff: 1.39e-16
Landau plaquette flux max error: 0.00e+00
```

### 4. Pure Substrate Port Contract

This test defines the clean new-engine boundary:

```text
normalized complex packet + Hermitian medium -> observable probabilities
```

It does not include V26 types, `StateEncoder`, measurement, decision, or
learning.

Validated facts:

- The port accepts a normalized complex packet and Hermitian medium.
- The output observable keeps total probability at 1.
- Gauge-equivalent ports give the same physical observables.
- Global phase does not cross the observable boundary.
- Loop holonomy changes physical observables.

Latest local result:

```text
graph_wave_substrate_port_contract_test: 4 / 4 PASS
hermiticity error: 0.00e+00
input norm: 1.000000000000
output norm: 1.000000000000
norm drift: 3.77e-15
gauge-covariant state error: 1.07e-15
observable probability diff after gauge transform: 4.37e-16
global phase probability diff: 2.64e-16
flux=0 target transfer: 0.381926
flux=pi/2 target transfer: 0.254251
flux=pi target transfer: 5.627628e-31
```

Interpretation:

- This is the first clean port contract for the new substrate itself.
- It is independent of V26 and must be preferred over legacy-contact tests when
  designing the replacement engine.
- It still does not prove measurement, decision, learning, or real-world
  usefulness.

### 5. Observable Layer Contract

This test defines the non-invasive read boundary for the substrate. Observable
reads are not measurement: they do not alter `z`, do not alter `H`, and do not
add an absorber.

Observable core:

```text
probability[i]      = |z_i|^2
port_power(group)   = sum |z_i|^2 over group
contrast(A,B)       = (P_A - P_B) / (P_A + P_B)
wilson_phase(loop)  = arg product H_ij/|H_ij|
edge_current(i,j)   = 2 Im(conj(z_i) H_ij z_j)
participation_ratio = (sum p)^2 / sum p^2
entropy             = -sum q_i log(q_i)
```

Validated facts:

- Observable reads are non-invasive.
- Probability/port/contrast observables are global-phase safe.
- Gauge-equivalent states and media give the same observable values.
- Wilson phase is gauge-invariant.
- Edge current is gauge-invariant and antisymmetric.
- Edge current satisfies the Schrodinger continuity equation.
- Interferometer routing is read by port power and contrast, not by absorber
  measurement.

Latest local result:

```text
graph_wave_observable_contract_test: 4 / 4 PASS
state gauge-transform error: 0.00e+00
medium gauge-transform error: 0.00e+00
norm: 1.000000000000
P(out+): 0.499999100959
P(out-): 0.499999100959
contrast: -5.551e-16
participation: 2.000007
entropy: 0.693173
probability diff gauge: 9.21e-15
probability diff global: 4.39e-15
Wilson phase diff gauge: 0.00e+00
edge current diff gauge: 4.64e-16
antisymmetry max |J_ij + J_ji|: 1.11e-16
continuity max |sum_j J_ij - dp_i/dt|: 1.11e-16
global net current: -1.11e-16
flux=0 contrast: 1.000000000000
flux=pi/2 contrast: -5.551e-16
flux=pi contrast: -1.000000000000
```

Interpretation:

- The observable layer is now explicit and non-invasive.
- This is not a decision layer. It only exposes physical quantities needed to
  test substrate composition contracts.

### 6. Unitary Interferometer Cell Contract

This test asks whether the substrate can grow one usable physical primitive
without adding a measurement layer or a classifier.

Cell geometry:

```text
input port -> two arms -> out+ / out-
```

The output ports are ordinary state components, not absorbing detectors. The
read time is a fixed time-of-flight for the cell geometry.

Validated facts:

- The cell is closed and unitary.
- Wilson holonomy routes amplitude between `out+` and `out-`.
- `flux = 0` routes to `out+`.
- `flux = pi` routes to `out-`.
- `flux = pi/2` splits equally.
- Gauge-equivalent phase placement gives the same routing probabilities.
- Global phase does not route.

Latest local result:

```text
graph_wave_interferometer_cell_contract_test: 4 / 4 PASS
max norm drift: 1.33e-14
flux=0    P(out+)=0.999998201919 P(out-)=7.468e-31
flux=pi/2 P(out+)=0.499999100959 P(out-)=0.499999100959
flux=pi   P(out+)=7.319e-31 P(out-)=0.999998201919
gauge-covariant state error: 5.58e-15
routing probability diff after gauge transform: 9.21e-15
routing probability diff after global phase: 4.55e-15
```

Interpretation:

- This is a real substrate-grown operator, not a V26 integration point.
- It is still not a decision system. It is one unitary physical primitive that
  can later be composed if composition is tested separately.

### 6b. Port Transfer Composition Contract

This test defines the first graph-wave port tensor:

```text
two input rail amplitudes -> two output rail amplitudes
```

The cell Hamiltonian is Hermitian and block off-diagonal:

```text
H = [ 0      K(phi)^dagger ]
    [ K(phi)      0        ]
```

where `K(phi)` is a 2x2 flux-controlled unitary routing kernel. At the
time-of-flight boundary the port transfer is:

```text
T(phi) ~= -i K(phi)
```

Validated facts:

- a single cell has a closed port transfer matrix
- flux routes a two-rail input: identity at `0`, split at `pi/2`, swap at `pi`
- port transfer is gauge-covariant across input and output frames
- the transfer is linear and global phase is not a routing signal
- two unitary cells compose by matrix multiplication:

```text
T_B * T_A ~= scale^2 * K(phi_A + phi_B)
```
- noncommuting unitary kernels also compose correctly and preserve layer order

Latest local result:

```text
graph_wave_port_transfer_composition_contract_test: 6 / 6 PASS
effective angle: 1.570795094718
residual power: 1.518e-12
transfer power: 0.999999999998
max |T - (-i sin(theta)) K(phi)|: 3.68e-14
max |K_eff^dag K_eff - I|: 8.79e-14
gauge transfer frame error: 2.13e-15
linearity error: 1.52e-15
max |T_B*T_A - scale^2*K(phi_A+phi_B)|: 8.26e-14
max |K_B*K_A - K(phi_A+phi_B)|: 1.24e-16
noncommuting max |T_B*T_A - scale^2*(K_B*K_A)|: 2.61e-14
noncommuting order gap |K_B*K_A - K_A*K_B|: 0.837
routed output order gap: 0.675
```

Interpretation:

- This is the first stable composition grammar for graph-wave cells.
- It is still purely unitary, so it deliberately confirms the linear-collapse
  boundary: stacked unitary cells are still one larger linear operator.
- Any nonlinear boundary is out of scope for this contract:
  absorber/observable -> renormalized or conditional re-injection.

### 6c. Physical Port Wiring Contract

This test asks whether the algebraic composition contract can be realized as
one physical Hermitian graph-wave body rather than only as an external matrix
product.

The wired body has three 2-rail frames:

```text
input rails -> shared bus rails -> output rails
```

with block path Hamiltonian:

```text
H = [ 0        K_A^dagger     0       ]
    [ K_A          0        K_B^dagger ]
    [ 0          K_B          0       ]
```

At the calibrated time-of-flight boundary, the physical transfer is:

```text
T_physical ~= -K_B K_A ~= -K(phi_A + phi_B)
```

Validated facts:

- the full body is Hermitian
- physical wiring realizes the same transfer as algebraic composition
- flux pairs route through the wired body according to `phi_A + phi_B`
- whole-body gauge covariance holds across input, bus, and output frames
- the wiring is clocked: half-flight leaves power in input/bus/output, while
  full-flight transfers all power to output
- the body remains linear without a measurement/re-injection boundary
- noncommuting physical kernels preserve layer order through the wired body

Latest local result:

```text
graph_wave_physical_port_wiring_contract_test: 6 / 6 PASS
effective path angle: 3.141592653590
max |T_physical - (-K(phi_A+phi_B))|: 4.89e-15
max |T^dag T - I|: 1.13e-14
output power: 1.000000000000
residual input+bus: 1.326e-28
half-flight power input/bus/output: 0.25 / 0.50 / 0.25
full-flight power input/bus/output: 1.227e-28 / 1.444e-30 / 1.00
gauge transfer frame error: 3.70e-15
linearity error: 3.88e-15
noncommuting max |T_physical - (-(K_B*K_A))|: 2.89e-15
noncommuting order gap |K_B*K_A - K_A*K_B|: 0.837
routed output order gap: 0.675
```

Interpretation:

- A narrow, impedance-matched physical wiring exists for the port-transfer
  algebra.
- This supports building a graph-wave circuit/body without decision logic.
- It does not remove the linear-collapse boundary. Nonlinearity still has to
  stay outside this contract.

### 6d. Physical Wiring Robustness Contract

This test is deliberately a falsification check, not a success demo. It asks
whether the narrow physical wiring from 6c has a local tolerance basin under
small deterministic defects.

Perturbations tested:

- coupling magnitude perturbation: `0.001`, `0.01`, `0.05`
- phase perturbation in radians: `0.001`, `0.01`, `0.05`
- combined coupling + phase perturbation at the same levels
- uniform coupling scale `1.020`, with time-of-flight retiming scan
- deterministic pseudo-random coupling + phase disorder ensemble at `0.01`
  and `0.05` over 24 fixed seeds

Metrics:

- fixed-clock transfer error against the ideal noncommuting transfer
- best transfer error after a small time-of-flight scan
- residual power left in input/bus rails at the fixed clock
- routed order gap against the reversed noncommuting order

Latest local result:

```text
graph_wave_port_wiring_robustness_contract_test: 6 / 6 PASS
ideal fixed error: 2.89e-15
ideal residual: 1.26e-28
ideal routed order gap: 0.675

coupling eps=0.001 fixed_err=0.0002 best_err=0.0002 residual=0.0000 order_gap=0.6753
coupling eps=0.010 fixed_err=0.0021 best_err=0.0021 residual=0.0002 order_gap=0.6771
coupling eps=0.050 fixed_err=0.0107 best_err=0.0104 residual=0.0040 order_gap=0.6850

phase eps=0.001 rad fixed_err=0.0008 best_err=0.0008 residual=0.0000 order_gap=0.6743
phase eps=0.010 rad fixed_err=0.0082 best_err=0.0077 residual=0.0000 order_gap=0.6671
phase eps=0.050 rad fixed_err=0.0409 best_err=0.0376 residual=0.0000 order_gap=0.6351

combined eps=0.001 fixed_err=0.0008 best_err=0.0008 residual=0.0000 order_gap=0.6745
combined eps=0.010 fixed_err=0.0079 best_err=0.0075 residual=0.0002 order_gap=0.6692
combined eps=0.050 fixed_err=0.0386 best_err=0.0362 residual=0.0040 order_gap=0.6469

uniform scale=1.020 fixed_err=0.0009 best_err=0.0000 best_steps=1004 expected_steps~=1004

random eps=0.010 seeds=24 mean_err=0.0083 max_err=0.0127 max_residual=0.0003 min_order_gap=0.6631
random eps=0.050 seeds=24 mean_err=0.0413 max_err=0.0632 max_residual=0.0083 min_order_gap=0.6153
```

Interpretation:

- The ideal wiring is not an infinitesimal needle for these deterministic local
  defects.
- Up to the tested `5%` magnitude / `0.05 rad` phase scale, transfer error
  remains bounded and the noncommuting order signal remains visible.
- Uniform global coupling drift is mostly a clock detuning and is recovered by
  retiming.
- This is still not a global robustness proof. It does not test random disorder
  distributions broadly enough to claim production tolerance, larger graphs,
  many stacked cells, topology changes, measurement/re-injection, or decision
  usefulness.

### 6e. Physical Depth Chain Contract

This test asks whether the physical wiring idea survives one more depth step:

```text
input rails -> bus1 rails -> bus2 rails -> output rails
```

The first result is intentionally negative. A uniform three-link chain is not a
valid full-transfer depth primitive:

```text
uniform gains 1,1,1:
fixed_err: 0.550
residual in non-output frames: 0.805
output power: 0.195
order gap: 0.819
```

This matters because it prevents the wrong conclusion that arbitrary stacked
unitary wiring automatically gives usable depth.

The test then uses mirror-coupled path gains:

```text
sqrt(3), 2, sqrt(3)
```

Validated facts:

- mirror-coupled depth transfers the ordered noncommuting kernel product
- clocked power flow reaches the output frame at full flight time
- noncommuting layer order remains visible through three cells
- deterministic random disorder at `0.01` and `0.05` remains bounded in this
  narrow model

Latest local result:

```text
graph_wave_physical_depth_chain_contract_test: 5 / 5 PASS

uniform depth guardrail:
fixed_err=0.550
residual=0.805
output_power=0.195
expected_fail_detected=YES

mirror-coupled gains=sqrt(3),2,sqrt(3):
fixed_err=1.72e-12
residual=3.43e-12
transfer_unitary_err=3.50e-12
order_gap=1.128

one-third frame powers: 0.4219 / 0.4219 / 0.1406 / 0.0156
two-third frame powers: 0.0156 / 0.1406 / 0.4219 / 0.4219
full frame powers: 6.747e-13 / 8.995e-24 / 2.755e-12 / 0.999999999997

forward-vs-reversed transfer gap: 1.210
routed output order gap: 1.128

random eps=0.010 seeds=24 mean_err=0.0075 mean_best=0.0073 max_err=0.0132 max_residual=0.0004 min_order_gap=1.1217
random eps=0.050 seeds=24 mean_err=0.0375 mean_best=0.0366 max_err=0.0651 max_residual=0.0102 min_order_gap=1.0966
```

Interpretation:

- Physical depth is possible, but not with naive uniform wiring.
- Depth requires engineered inter-frame coupling geometry, at least in this
  minimal path model.
- The mirror-coupled chain gives a cleaner depth primitive than the uniform
  chain and keeps the ordered signal under the tested local disorder.
- This still does not introduce nonlinearity. It is a deeper linear/unitary
  body. Measurement/observable/re-injection is out of scope for this contract.

### 6f. Fan-In Aggregation Contract

This test adds the first graph-neighborhood primitive. It asks whether several
incoming rails can feed one aggregate mode while keeping the whole transfer
unitary.

The key discipline is that unitary fan-in is not free compression:

```text
3 input rails -> aggregate mode + 2 orthogonal residual modes
```

The aggregate mode is the GNN-analog neighborhood sum, physically realized as a
coherent symmetric port. The residual modes are required to preserve
information. Any state-discarding boundary must be opened as a separate
owner-approved contract, not as a V26 legacy trainer, GA, readout, or decision
shortcut.

Validated facts:

- the full fan-in transfer frame is Hermitian/unitary
- phases control constructive vs destructive aggregation
- a basis input sends only `1/3` of its power to the aggregate and `2/3` to
  residual modes, proving the no-compression boundary
- permuting neighbors keeps the aggregate invariant when physical couplings move
  with the input labels; the residual basis may rotate
- local gauge covariance holds across input and output frames
- deterministic random disorder at `0.01` and `0.05` leaves the aggregate signal
  bounded in this narrow model

Latest local result:

```text
graph_wave_fanin_aggregation_contract_test: 6 / 6 PASS
max |T - (-i W)|: 8.77e-15
max |T^dag T - I|: 3.06e-14
residual input power: 1.55e-28

aligned aggregate: 1.000000000000
aligned residual modes: 2.315e-32
destructive aggregate: 1.326e-32
destructive residual modes: 1.000000000000

basis input aggregate: 0.333333333333
basis input residual modes: 0.666666666667

neighbor permutation aggregate amplitude error: 2.22e-16
gauge transfer frame error: 2.07e-15

random eps=0.010 seeds=24 mean_err=0.0048 max_err=0.0069 min_aligned_agg=0.9998 max_destructive_agg=0.0000 max_input_residual=0.0002
random eps=0.050 seeds=24 mean_err=0.0239 max_err=0.0348 min_aligned_agg=0.9940 max_destructive_agg=0.0007 max_input_residual=0.0061
```

Interpretation:

- The substrate now has a minimal graph-neighborhood aggregation primitive, not
  only serial layer composition.
- The aggregate output is physically meaningful only together with its residual
  modes unless a separate measurement/re-injection contract is explicitly
  opened.
- This is still linear/unitary and does not perform learning or decision.

### 6g. GNN-Analog Node Update Contract

This test adds the first explicit node-update primitive:

```text
center self-state + 3 neighbor messages -> updated center port + 3 residual modes
```

This is the closest substrate contract so far to a GNN layer update:

```text
h_v, {h_u}, edge phases/couplings -> h'_v
```

The key discipline is unchanged: a unitary substrate cannot compress four input
amplitudes into one output amplitude without residual modes.

Validated facts:

- the full node-update transfer frame is Hermitian/unitary
- self-state and edge couplings set visible update contributions
- neighbor edge phases control constructive/destructive message interference
- relabeling neighbors keeps the updated-center port invariant when couplings
  move with the input labels
- residual modes prevent false node-state compression
- local gauge covariance holds across input and output frames
- deterministic random disorder at `0.01` and `0.05` leaves the update signal
  bounded in this narrow model

Latest local result:

```text
graph_wave_node_update_contract_test: 7 / 7 PASS
max |T - (-i W)|: 3.41e-14
max |T^dag T - I|: 6.27e-14
residual input power: 5.98e-28

update power self=0.481605351171
update power neighbor0=0.270903010033
update power neighbor1=0.163879598662
update power neighbor2=0.083612040134

aligned update: 1.000000000000
destructive update: 2.038e-31

neighbor relabel updated-center amplitude error: 6.50e-16
neighbor relabel updated-center power error: 5.55e-17

basis neighbor update: 0.163879598662
basis neighbor residual modes: 0.836120401338

gauge transfer frame error: 1.17e-15
gauge-covariant output state error: 7.95e-16

random eps=0.010 seeds=24 mean_update_err=0.0022 max_update_err=0.0043 min_update_power=0.4788 max_input_residual=0.0002
random eps=0.050 seeds=24 mean_update_err=0.0112 max_update_err=0.0214 min_update_power=0.4652 max_input_residual=0.0045
```

Interpretation:

- The graph-wave substrate now has the minimal GNN-analog node update grammar:
  self-state, neighbor messages, edge parameters, updated-center port, and
  residual modes.
- This still does not define a predictor, loss, trainer, legacy GA/readout path,
  or V26 runtime integration.

### 6h. Shared Edge Consistency Contract

This test asks whether two GNN-analog node-update primitives can share one
physical edge consistently:

```text
A self, B self, A external, B external
-> A update, B update, residual modes
```

The reciprocal shared-edge rule is intentionally strict. The valid rows use an
antisymmetric reciprocal orientation; a same-sign shared-edge rule is rejected
because it breaks update-row orthogonality.

Validated facts:

- reciprocal shared-edge update rows are orthogonal
- a bad same-sign edge convention is detected as invalid
- the full two-node transfer frame is Hermitian/unitary
- shared-edge contribution power is equal in both directions
- shared-edge phase controls pair interference
- residual modes prevent false compression
- local gauge covariance holds across the shared-edge body
- deterministic random disorder at `0.01` and `0.05` leaves the pair update
  bounded in this narrow model

Latest local result:

```text
graph_wave_shared_edge_consistency_contract_test: 7 / 7 PASS
good row overlap: 0.00e+00
bad same-sign overlap: 0.800
reciprocal shared-edge coefficient error: 0.00e+00

max |T - (-i W)|: 1.20e-14
max |T^dag T - I|: 3.18e-14
residual input power: 1.41e-28

expected shared-edge update power: 0.320000000000
A->B update power: 0.320000000000
B->A update power: 0.320000000000
transfer reciprocal error: 5.55e-17

phase 0 A_update/B_update: 0.810000000000 / 0.010000000000
phase pi A_update/B_update: 0.010000000000 / 0.810000000000

external-A update_A: 0.180000000000
external-A update_B: 1.528e-33
external-A residual modes: 0.820000000000

gauge transfer frame error: 3.22e-15
gauge-covariant output state error: 1.68e-15

random eps=0.010 seeds=24 mean_update_err=0.0037 max_update_err=0.0066 max_power_err=0.0028 max_input_residual=0.0002
random eps=0.050 seeds=24 mean_update_err=0.0186 max_update_err=0.0323 max_power_err=0.0145 max_input_residual=0.0039
```

Interpretation:

- The substrate now has a minimal two-node GNN-analog consistency rule for a
  shared edge.
- This is still not a general graph layer. It proves only one reciprocal
  two-node edge contract.
- It still does not define a predictor, loss, trainer, legacy GA/readout path,
  or V26 runtime integration.

### 6i. Degree-2 Path Guardrail Contract

This test asks the next smallest graph question:

```text
A - B - C
```

The dangerous shortcut is classical GNN-style broadcast: one `B` rail feeding
both `A` and `C` updates. In a unitary substrate that is not a free operation.
The test therefore first proves that naive one-rail broadcast is invalid, then
validates an edge-lifted path with explicit oriented message rails:

```text
A_self, B_self, C_self, A->B, B->A, B->C, C->B
-> A_update, B_update, C_update, residual modes
```

Validated facts:

- naive degree-2 one-rail broadcast is rejected by row-overlap accounting
- the edge-lifted degree-2 path transfer frame is Hermitian/unitary
- oriented edge message rails set the intended node update contributions
- edge phase controls local leaf interference
- path reversal preserves update ports when edge labels move with the path
- local gauge covariance holds across the lifted path body
- deterministic random disorder at `0.01` and `0.05` leaves the path update
  bounded in this narrow model

Latest local result:

```text
graph_wave_degree2_path_guardrail_contract_test: 7 / 7 PASS
naive leaf rows sharing one B rail overlap: 0.500
expected fail detected: YES

max |T - (-i W)|: 2.88e-14
max |T^dag T - I|: 8.00e-14
residual input power: 6.67e-28

B->A update power: 0.390243902439
A->B update power: 0.223744292237
C->B update power: 0.223744292237
B->C update power: 0.390243902439

phase 0 A_update: 0.987804878049
phase pi A_update: 0.012195121951

path reversal update amplitude errors A/B/C: 0.00e+00 / 0.00e+00 / 0.00e+00

gauge transfer frame error: 2.40e-15
gauge-covariant output state error: 1.23e-15

random eps=0.010 seeds=24 mean_update_err=0.0030 max_update_err=0.0049 max_power_err=0.0031 max_input_residual=0.0002
random eps=0.050 seeds=24 mean_update_err=0.0151 max_update_err=0.0244 max_power_err=0.0151 max_input_residual=0.0051
```

Interpretation:

- The substrate cannot import classical GNN broadcast semantics directly.
- Degree greater than one requires explicit edge-message rails or another
  separately tested unitary message-materialization contract.
- This is still not a general graph layer, predictor, loss, trainer, legacy
  GA/readout path, or V26 runtime integration.

### 7. Pure Spectral Substrate Comparison

This test asks only whether a golden/gauge graph-wave substrate has a distinct
spectral gap hierarchy compared with two controls. It does not prove a decision
layer and it does not claim a thermodynamic Cantor dimension from one finite
graph.

Compared substrates:

- plain 2D lattice
- deterministic random-phase lattice
- direct graph-wave golden-flux lattice with `alpha = 8/13`

Latest local result:

```text
substrate_spectral_comparison_test: PASS
plain         dim=0.585 gaps>2%/1%/0.5%/0.25% = 6/46/68/78
random-phase  dim=0.833 gaps>2%/1%/0.5%/0.25% = 0/14/92/146
golden-flux   dim=0.627 gaps>2%/1%/0.5%/0.25% = 18/22/46/80
L1(golden, plain): 1.172
L1(golden, random): 0.970
Cantor/fractal dimension claim: DEFERRED
```

### 8. Flux-Family Substrate Scan

This test asks only whether the golden-flux signature survives basic flux-family
controls. It does not use measurement or decision logic.

Compared direct graph-wave Hamiltonians:

- `alpha = 0`
- `alpha = 1/2`
- `alpha = 1/3`
- `alpha = 8/13`
- `alpha = 5/13`, the complement of `8/13`
- deterministic random-phase lattice

Latest local result:

```text
substrate_flux_scan_test: PASS
alpha=0    dim=0.585 gaps>2%/1%/0.5% = 6/46/68
alpha=1/2  dim=0.433 gaps>2%/1%/0.5% = 16/34/42
alpha=1/3  dim=0.642 gaps>2%/1%/0.5% = 14/26/54
alpha=8/13 dim=0.627 gaps>2%/1%/0.5% = 18/22/46
alpha=5/13 dim=0.627 gaps>2%/1%/0.5% = 18/22/46
random     dim=0.838 gaps>2%/1%/0.5% = 0/10/110
max eigen diff alpha=8/13 vs 5/13: 3.91e-14
L1(8/13, 0): 1.160
L1(8/13, 1/2): 0.923
L1(8/13, 1/3): 0.497
L1(8/13, random): 1.089
```

Interpretation:

- `alpha` and `1-alpha` give the same spectrum, as expected.
- `8/13` differs from zero, simple rational, and random-phase controls.
- This supports the golden/gauge substrate as a real flux-family effect.
- This still does not prove a decision layer or real information-flow
  usefulness.

### 9. Measurement Layer (Separate, Not Substrate)

The substrate remains Hermitian. Measurement is one explicit non-Hermitian layer
at detector sites:

```text
H_eff = H_substrate - iW
W >= 0, diagonal
```

Absorption is detection. The conserved accounting identity becomes:

```text
detected(t) + remaining_in_field(t) = 1
```

The strict accounting gate uses telescoped norm loss. The continuum current

```text
J_i(t) = 2 * Gamma_i * |z_i(t)|^2
```

is kept as a diagnostic because the Cayley step is discrete; telescoped norm
loss is the exact discrete accounting path.

Validated invariants:

- `W=0` remains unitary
- `W>0` is monotone contraction
- detected probability plus remaining field probability closes to machine
  precision
- absorbing screen produces a real double-slit pattern with nonzero captured
  beam probability

Latest local result:

```text
absorbing_screen_test: 3 / 3 PASS
W=0 operator unitarity error: 1.11e-15
W>0 norm after 300 steps: 0.0228
detected + remaining closure: 2.09e-14
discrete detector allocation error: 8.88e-16
continuum current diagnostic error: 3.10e-04
single slit caught: 0.100
double slit caught: 0.045
interior deep nulls: single=0, double=2
double maxima: 3
double centre-of-mass: 22.1
L1(single,double): 0.445
```

## What This Means

This preserves a clean physical foundation:

```text
complex state -> Hermitian graph-wave propagation -> gauge/Wilson loop physics
```

Measurement has also been tested as a separate layer:

```text
unitary propagation -> explicit absorber W -> detector accounting
```

The result is not a production predictor. It proves isolated physical facts
only; it does not open a decision, trainer, GA, readout, or integration track.

## What This Does Not Prove

- It does not prove improvement over the existing V26 control path.
- It does not prove, define, or require a learning rule.
- It does not prove, define, or require a production decision engine.
- It does not replace Living Silicon.
- It does not authorize editing, deleting, reusing, or reviving legacy GA/readout
  code.
- It does not prove that the golden/gauge substrate is sufficient for real
  information flow.
- It does not prove a thermodynamic fractal/Cantor spectrum from the finite
  spectral diagnostic.

## Safety Rule

The old V26 system remains authoritative. This checkpoint is not a replacement
loop and it does not reject V26. It prevents the graph-wave substrate from being
absorbed into uncontrolled legacy coupling. Graph-wave work may only advance
through isolated `EXCLUDE_FROM_ALL` lab targets, docs, and explicit substrate
experiments.

No graph-wave code should be called from production paths. This checkpoint does
not define criteria for production use, migration, decision loops, legacy
readout, legacy GA, or any trainer.

Any proposal for measurement, nonlinear boundary, decision, or V26 migration
work must be opened as a separate owner-approved checkpoint with an explicit
boundary/adapter contract. It must not be smuggled into substrate tests.

## Current Research Discipline

Do not mix goals. Each test must answer one question only:

```text
substrate physics != measurement != decision != learning != V26 boundary/integration
```

The current allowed work is substrate-only unless explicitly changed:

- refine direct graph-wave substrate candidates
- maintain GNN-analog port, depth, fan-in, node-update, shared-edge, and
  degree-2 path guardrail substrate grammar contracts
- compare spectra against controls
- keep measurement tests separate
- do not add V26 legacy readout, decision, learning JSON, GA, trainers, or
  optimizers to substrate tests

Choosing a substrate does not automatically authorize measurement, decision,
trainer, GA, readout, or integration work. Those are separate tracks and are out
of scope here.
