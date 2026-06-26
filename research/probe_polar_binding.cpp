// probe_polar_binding
// ----------------------------------------------------------------------------
// Direction (a) foundation, done right: does a REAL GNNv2 field's entanglement
// carry input identity? POLAR + physics only, with a fidelity ground-truth gate
// FIRST so the phase-blindness trap (mistaking blind collapse for "symmetry")
// cannot recur. See docs/PHYSICS_ONLY_DISCIPLINE.md.
//
// Real substrate: golden-flux 4x4 lattice (the selected critical candidate),
// 16 nodes = 4 qubits. Each "input" = a localized injection; propagate by
// gw::Stepper. Signature = per-qubit entropy (polar |r|) + <Z> + per-pair <ZZ>
// + POLAR coherence (M, phi) from the interference fringe (NO Cartesian XX/XY).
// ----------------------------------------------------------------------------
#include "qubit_physics.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>
using gw::Vec; using gw::cd;

static Vec stateOf(int inj) {
 gw::Graph g = gw::golden(4, 8.0 / 13.0); gw::Stepper U; U.build(g.h, 0.15);
 Vec f(16, cd(0, 0)); f[inj] = cd(1, 0); for (int t = 0; t < 32; ++t) f = U.step(f); return f;
}
static double fidelity(const Vec& a, const Vec& b) { cd o(0, 0); for (size_t k = 0; k < a.size(); ++k) o += std::conj(a[k]) * b[k]; return std::norm(o); }

struct Sig { double ent[4], z[4]; double zz[6], M[6], phi[6]; };
static Sig sigOf(const Vec& f) {
 Sig s; for (int q = 0; q < 4; ++q) { s.ent[q] = qp::entropyOf(f, 4, q); s.z[q] = qp::expectZ(f, 4, q); }
 int k = 0; for (int a = 0; a < 4; ++a) for (int b = a + 1; b < 4; ++b) {
  s.zz[k] = qp::expectZZ(f, 4, a, b) - qp::expectZ(f, 4, a) * qp::expectZ(f, 4, b);
  s.M[k] = qp::pairMag(f, 4, a, b); s.phi[k] = qp::pairPhase(f, 4, a, b); ++k; }
 return s;
}
static double sigDist(const Sig& a, const Sig& b) {
 double d = 0;
 for (int q = 0; q < 4; ++q) { d += (a.ent[q] - b.ent[q]) * (a.ent[q] - b.ent[q]); d += (a.z[q] - b.z[q]) * (a.z[q] - b.z[q]); }
 for (int k = 0; k < 6; ++k) { d += (a.zz[k] - b.zz[k]) * (a.zz[k] - b.zz[k]); const double pc = qp::polarCoherenceDist(a.M[k], a.phi[k], b.M[k], b.phi[k]); d += pc * pc; }
 return std::sqrt(d);
}

int main() {
 std::vector<int> in = {0, 5, 10, 15};
 std::vector<Vec> st; for (int i : in) st.push_back(stateOf(i));
 std::vector<Sig> sg; for (auto& s : st) sg.push_back(sigOf(s));

 // GATE 1 (ground truth): are the states actually distinct?
 double maxFid = 0; for (size_t i = 0; i < in.size(); ++i) for (size_t j = i + 1; j < in.size(); ++j) maxFid = std::max(maxFid, fidelity(st[i], st[j]));
 std::printf("[1] ground-truth fidelity: max off-diagonal = %.2e (must be ~0 = distinct states)\n", maxFid);

 // GATE 2 (polar signature): does it separate the distinct states?
 double minOff = 1e9; std::printf("[2] POLAR signature distances (entropy|r| + Z + ZZ + polar (M,phi); NO Cartesian):\n");
 for (size_t i = 0; i < in.size(); ++i) for (size_t j = i + 1; j < in.size(); ++j) { const double d = sigDist(sg[i], sg[j]); std::printf("    %2d-%2d: %.4f\n", in[i], in[j], d); minOff = std::min(minOff, d); }
 const double rep = sigDist(sigOf(stateOf(5)), sigOf(stateOf(5)));
 std::printf("    reproducibility = %.2e ; min off-diagonal = %.4f\n", rep, minOff);

 const bool ok = maxFid < 1e-9 && rep < 1e-9 && minOff > 0.1;
 std::printf("\n RESULT : %s\n", ok
  ? "POLAR signature separates all (distinct) inputs -> complete, no Cartesian, no info lost"
  : "FAIL -- blind or non-reproducible signature");
 return ok ? 0 : 1;
}
