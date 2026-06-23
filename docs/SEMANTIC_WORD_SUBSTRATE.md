# Word → substrate conversion

Two contracts on this topic, kept side by side.

## 1. The order primitive (original probe)

`tools/graph_wave_semantic_superposition_phase_contract_test.cpp`

Words are injected as real impulses at different times; the unitary evolution
between injections manufactures a relative phase; reading the gauge-invariant loop
**circulation** gives a signed scalar. For `A before B` vs `B before A` the
circulation is exactly antisymmetric (−0.3366 / +0.3366), and gauge-invariant.

What it proves is **real but narrow**: order can become an emergent, gauge-invariant
phase. The framing ("semantic transduction") overclaims:

- the ± antisymmetry is just the `A↔B` reflection symmetry (`psi_ba = P·psi_ab`,
  circulation is odd under the swap) — a 1-bit *reverse-or-not* detector;
- the relation token in the middle is inert (the reflection fixes it);
- one scalar per loop → endless collisions, no injective encoding;
- each word is still a fixed **node index** — the arbitrariness moved from phase to
  node, it was not removed;
- the "phase erasure kills the signal" check is a tautology (circulation *is* an
  imaginary part; zeroing phase zeroes it by definition).

## 2. The word substrate (corrected)

`tools/graph_wave_semantic_word_substrate_contract_test.cpp`

Same good idea — words as real impulses, phase born from evolution, gauge-invariant
readout — applied to the actual task, with the flaws fixed:

| flaw | fix |
|---|---|
| word = arbitrary node index | word = **co-occurrence pattern** (meaning enters the input) |
| readout = one scalar | **high-dim fingerprint**: `\|psi_i\|^2` channel (semantics) + edge-current channel (order) |
| only reverse-detection | **field-space retrieval** + **injectivity** over several phrases / 3 topics |
| no semantics test | nearest field is same-topic, 6/6 |
| tautological / tuned checks | replaced by a **bag-of-words control**: a bag is exactly order-blind (cos 1.0000), the substrate is not |

Verified (5/5):

```text
[1] embedding semantic       intra 0.667 > inter 0.000
[2] field retrieval          6/6 nearest-field same topic
[3] bag vs substrate         bag cos 1.0000 (order-blind); substrate mag 0.978, current +0.520
[4] injective                max pairwise field cos 0.761
[5] gauge-invariant          1.94e-16
```

Honest scope: the corpus is a 12-word toy with clean clusters — a proof of
principle, not a trained embedding; and the order signal concentrates in the
current channel (margin 0.46) rather than flipping perfectly. The point stands: a
single gauge-invariant field carries both *which words* (semantics) and *their
order* (emergent phase) — something a text bag fundamentally cannot.
