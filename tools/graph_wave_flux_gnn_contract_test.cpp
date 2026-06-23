// graph_wave_flux_gnn_contract_test
// ----------------------------------------------------------------------------
// The phases ARE the GNN. Message passing runs ON the substrate: a plaquette with
// magnetic FLUX (Hofstadter/golden-flux physics). A message leaving node 0 reaches
// node 2 by two paths (via 1 and via 3) and AGGREGATES by interference; the
// aggregate is set by the gauge-invariant flux through the loop, not by any
// absolute phase -- the meaning lives in the flux.
//
// Quantities are ROBUST (max reach over all time), not a single snapshot -- the
// plaquette is a closed oscillator whose snapshot swings with time (the plaquette
// diagnostic showed P2(flux=0) reading 0.98/0.0016/0.93 at different times).
// [1] the propagation is unitary (message passing conserves the state)
// [2] the gauge-invariant FLUX gates the aggregation at node 2: fully reachable
// at flux 0, EXACTLY decoupled at flux pi for all time (the phases compute)
// [3] absolute/gauge phase is NOT a signal: a node-gauge transform that keeps
// the loop flux fixed leaves the node-2 reachability invariant
// [4] honest: on ONE plaquette golden flux is NOT special (fully reachable like
// a generic flux); golden CRITICALITY is a lattice property, not a cell one
//
// Built on the shared substrate core (graph_wave_substrate.hpp): the unitary
// Cayley stepper comes from the single tested source. No decision, no learning,
// no trainer.
// ----------------------------------------------------------------------------
#include <cstdio>
#include "graph_wave_substrate.hpp"
namespace {
using namespace gw;
const double kDt = 0.1; const int kSteps = 200; // t up to 20: covers many recurrences (robust max)
// plaquette 0-1-2-3-0; flux carried on edge 3->0; optional node gauge g[]
Graph plaquette(double flux, const std::vector<double>& g) {
 Graph G(4); G.addEdge(0, 1, 1, 0); G.addEdge(1, 2, 1, 0); G.addEdge(2, 3, 1, 0); G.addEdge(3, 0, 1, flux); // loop encloses 'flux'
 if (!g.empty()) for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) G.h[i][j] *= std::exp(cd(0, g[i] - g[j]));
 return G;
}
// node-2 REACHABILITY: max population over the whole evolution (a snapshot is a
// fragile reading of a closed oscillator -- see the plaquette diagnostic).
double evolveP2(double flux, const std::vector<double>& g, int startNode) {
 Graph G = plaquette(flux, g); Stepper S; S.build(G.h, kDt);
 Vec psi(4, cd(0, 0)); psi[startNode] = (g.empty() ? cd(1, 0) : std::exp(cd(0, g[startNode]))); // gauge-transform the source too
 double mx = 0; for (int s = 0; s < kSteps; s++) { psi = S.step(psi); mx = std::max(mx, std::norm(psi[2])); } return mx;
}
double evolveNorm(double flux) {
 Graph G = plaquette(flux, {}); Stepper S; S.build(G.h, kDt); Vec psi(4, cd(0, 0)); psi[0] = cd(1, 0);
 double mn = 1, mx = 1; for (int s = 0; s < kSteps; s++) { psi = S.step(psi); double nm = power(psi); mn = std::min(mn, nm); mx = std::max(mx, nm); }
 return std::max(std::abs(mx - 1), std::abs(mn - 1));
}
bool report(const char* n, bool ok) { std::printf(" => %s\n", ok ? "PASS" : "FAIL"); if (!ok) std::printf(" !! %s\n", n); return ok; }
}
int main() {
 std::printf("=====================================================================\n");
 std::printf(" GRAPH-WAVE FLUX-GNN CONTRACT TEST (message passing where the FLUX computes)\n");
 std::printf(" Plaquette 0-1-2-3: two paths to node 2 aggregate by interference; flux is the signal.\n");
 std::printf("=====================================================================\n");
 int pass = 0, total = 0;

 std::printf("\n[1] THE PROPAGATION IS UNITARY (message passing conserves the state)\n");
 { double d = evolveNorm(0.7); std::printf(" max norm drift over %d steps = %.2e\n", kSteps, d);
 ++total; pass += report("propagation not unitary", d < 1e-12); }

 std::printf("\n[2] THE GAUGE-INVARIANT FLUX GATES THE AGGREGATION (robust: max reach over all time)\n");
 { double c = evolveP2(0.0, {}, 0), d = evolveP2(kPi, {}, 0);
 std::printf(" node-2 reachability: flux=0 (constructive)=%.4f flux=pi (destructive)=%.2e\n", c, d);
 std::printf(" (flux=pi decouples node 2 EXACTLY for all time -- not a snapshot; see plaquette diagnostic)\n");
 ++total; pass += report("flux did not gate the interference", c > 0.9 && d < 1e-9); }

 std::printf("\n[3] ABSOLUTE/GAUGE PHASE IS NOT A SIGNAL (only the loop flux matters)\n");
 { double base = evolveP2(0.9, {}, 0);
 std::vector<double> g = { 0.6, -1.3, 2.1, 0.4 }; // arbitrary node gauge; keeps the loop flux fixed
 double gauged = evolveP2(0.9, g, 0);
 std::printf(" node-2 reach: no gauge=%.6f with node gauge=%.6f diff=%.2e\n", base, gauged, std::abs(base - gauged));
 ++total; pass += report("gauge phase leaked into the observable", std::abs(base - gauged) < 1e-9); }

 std::printf("\n[4] HONEST: ON ONE PLAQUETTE GOLDEN FLUX IS NOT SPECIAL (criticality needs the lattice)\n");
 { double gold = evolveP2(2 * kPi * 8.0 / 13.0, {}, 0), d = evolveP2(kPi, {}, 0);
 std::printf(" node-2 reach: golden=%.4f (fully reachable, like a generic flux) flux=pi=%.2e (the only special point)\n", gold, d);
 std::printf(" => the single cell shows the gauge/flux PRINCIPLE; golden CRITICALITY is a lattice property, not a plaquette one\n");
 ++total; pass += report("golden should be reachable on a single plaquette", gold > 0.5); }

 std::printf("\n RESULT : %d / %d verified (flux is the signal; robust invariants only)\n", pass, total);
 return pass == total ? 0 : 1;
}
