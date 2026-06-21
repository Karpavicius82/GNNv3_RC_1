# Graph-Wave Legacy Contact Audit - 2026-06-17

This document quarantines tests that touch the old V26 runtime model.

They are not part of the graph-wave substrate foundation. They do not define the
new engine. They only answer one migration question:

```text
If a future replacement ever needs to consume old V26 stream state, can that
boundary be made deterministic, normalized, gauge-compatible, and outcome-clean?
```

## Hard Boundary

Foundation tests must not link these legacy types:

- `RunState`
- `StateEncoder`
- `Reducer`
- Living Silicon
- GA/readout code
- production service/runtime paths

The tests in this file deliberately do link some of those types, so they are
legacy-contact audits. They are allowed to fail without invalidating the
substrate. They are allowed to pass without authorizing integration.

## Why Keep These Tests

The old V26 system may be replaced substantially. Still, deleting all contact
checks would be blind. These tests preserve a narrow fact: the old stream can be
adapted into a wave-shaped input without immediate outcome leakage or gauge
breakage.

That is useful only as migration evidence. It is not architectural permission.

## Contact Test 1: Stream -> Wave Injection

Target:

```text
RunState -> real StateEncoder -> normalized wave injection
```

Not included:

- propagation
- measurement
- readout
- decision
- learning

Latest local result:

```text
stream_to_wave_injection_contract_test: 4 / 4 PASS
repeat max diff: 0.00e+00
energy: 1.000000000000
completed-vs-failed injection max diff: 0.00e+00
read-vs-destructive L2 distance: 1.005
read-vs-destructive mean phase distance: 0.317 rad
run/session id only max diff: 0.00e+00
```

Interpretation:

- Decision-time V26 state can be converted to a normalized wave-shaped input.
- Outcome-bearing fields can be stripped before encoding.
- This does not prove that `StateEncoder` is the right input method for the new
  engine.

## Contact Test 2: Stream -> Wave -> Gauge Substrate

Target:

```text
RunState -> real StateEncoder -> normalized wave injection -> gauge substrate
```

Not included:

- measurement
- readout
- decision
- learning

Latest local result:

```text
stream_wave_gauge_contact_test: 4 / 4 PASS
max gauge-covariant state error: 7.18e-16
max physical probability diff: 1.60e-16
global phase propagated probability diff: 3.19e-16
probability L1 initial: 0.675
probability L1 propagated: 0.902
completed-vs-failed propagated probability diff: 0.00e+00
```

Interpretation:

- A V26-derived wave injection can enter the gauge substrate without violating
  local gauge covariance.
- Global phase remains unobservable.
- Outcome fields still do not leak after propagation.
- This does not prove usefulness, decision quality, or correctness of the old
  encoder as the future input layer.

## Migration Rule

These tests must stay behind this rule:

```text
legacy contact audit != new substrate foundation
```

Do not use these passing tests to integrate graph-wave code into V26 production.
Do not use them to preserve V26 architecture if the replacement needs a cleaner
input model.

They are only a controlled boundary check.
