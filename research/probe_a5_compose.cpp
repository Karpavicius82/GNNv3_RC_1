// A5: COMPOSITION -- does exact flux routing survive being wired into a real
// message-passing layer? This is the GNN kernel, weights-free:
// input --(fan-out)--> branch A, branch B (one node speaks to many neighbours)
// each branch deepens through a SECOND flux gate (a 2nd layer)
// Node state = complex amplitude (superposition). Edges = unitary hops (phases).
// Routing/"attention" = gauge flux, NOT a learned weight. flux pi = exact OFF.
// Production claim under test: outputs gate INDEPENDENTLY and EXACTLY, the layer
// CONSERVES (unitary) at all depth, and an upstream block exactly kills its
// whole subtree. If any of these is only approximate, it is a dead-end -- say so.
#include "../tools/graph_wave_substrate.hpp"
#include <cstdio>
#include <complex>
using namespace gw;

// hub indices: 0=I 1=hubA 2=outA 3=hubB 4=outB ; intermediates 5..12
static CMat net(double fA, double fA2, double fB, double fB2) {
 int n = 13; CMat H(n, Vec(n, cd(0, 0)));
 auto add = [&](int a, int b, double ph) { cd v = std::exp(cd(0, ph)); H[a][b] += v; H[b][a] += std::conj(v); };
 auto rh = [&](int a, int b, int U, int D, double flux) { add(a, U, 0); add(U, b, 0); add(a, D, 0); add(D, b, flux); };
 rh(0, 1, 5, 6, fA); // I -> hubA (gate fA)
 rh(1, 2, 7, 8, fA2); // hubA -> outA(gate fA2)
 rh(0, 3, 9, 10, fB); // I -> hubB (gate fB)
 rh(3, 4, 11, 12, fB2); // hubB -> outB(gate fB2)
 return H;
}

struct Out { double reachA, reachB, normErr; };
static Out evolve(const CMat& H, double dt = 0.3, double Tmax = 40) {
 int n = (int)H.size(); Stepper st; st.build(H, dt);
 Vec psi(n, cd(0, 0)); psi[0] = cd(1, 0);
 int steps = (int)(Tmax / dt); double rA = 0, rB = 0, ne = 0;
 for (int k = 0; k < steps; k++) { psi = st.step(psi); rA = std::max(rA, std::abs(psi[2])); rB = std::max(rB, std::abs(psi[4])); ne = std::max(ne, std::abs(power(psi) - 1.0)); }
 return { rA, rB, ne };
}

int main() {
 const double P = kPi, O = 0.0;
 std::printf("=== A5: composed flux-routed message-passing layer (weights-free GNN kernel) ===\n\n");
 std::printf("fan-out OPEN (fA=fB=0); 2nd-layer gates fA2,fB2 vary. reach=max|psi_out| over time.\n");
 std::printf(" fA2 fB2 | reach(outA) reach(outB) | maxNormErr\n");
 struct Row { double fA2, fB2; const char* tag; };
 Row rows[] = {{O, O, "A on , B on "}, {O, P, "A on , B OFF"}, {P, O, "A OFF, B on "}, {P, P, "A OFF, B OFF"}};
 for (auto& r : rows) {
 Out o = evolve(net(O, r.fA2, O, r.fB2));
 std::printf(" %-3s %-3s | %.4f %.2e | %.1e\n",
 r.fA2 == 0 ? "0" : "pi", r.fB2 == 0 ? "0" : "pi", o.reachA, o.reachB, o.normErr);
 }
 std::printf("\nupstream dominance: block branch A at the ENTRY (fA=pi), leave fA2=0:\n");
 Out u = evolve(net(P, O, O, O));
 std::printf(" fA=pi,fA2=0 -> reach(outA)=%.2e (whole A subtree exactly dead), reach(outB)=%.4f\n", u.reachA, u.reachB);
 std::printf("\nDead-end check: if reach(OFF) is ~1e-14 or below and independent of the other\nbranch, and normErr ~ machine zero, the composition is EXACT -> not a dead-end.\n");
 return 0;
}
