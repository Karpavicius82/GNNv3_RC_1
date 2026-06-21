// graph_wave_deep_round_contract_test
// ----------------------------------------------------------------------------
// DEPTH for the working GNN. HONEST FINDING first: chaining the explicit edge-rail
// message round does NOT cleanly deepen the receptive field, because a single
// unitary round leaves amplitude on edge/residual modes that does not re-enter the
// node subspace on the next round. The clean, chainable deep mechanism is direct
// FIELD PROPAGATION on the node graph: U = one Cayley step of the Hermitian path
// adjacency is unitary on the node states, and U^k deepens the reach. Path 0-1-2-3.
//
//   [1] U, U^2, U^3 are unitary and conserve the state
//   [2] the receptive field DEEPENS: |U^k[0][3]| grows strongly with depth k
//   [3] nearer nodes are reached earlier / more strongly than farther ones
//
// Built on the shared substrate core (graph_wave_substrate.hpp): the Cayley
// propagator and the matrix helpers come from the single tested source. No
// decision, no learning, no trainer, no V26.
// ----------------------------------------------------------------------------
#include <cstdio>
#include "graph_wave_substrate.hpp"
namespace {
using namespace gw;
const double kDt = 0.6;
bool report(const char* n, bool ok) { std::printf("    => %s\n", ok ? "PASS" : "FAIL"); if (!ok) std::printf("    !! %s\n", n); return ok; }
}
int main() {
  std::printf("=====================================================================\n");
  std::printf(" GRAPH-WAVE DEEP-ROUND CONTRACT TEST  (depth = field propagation on the graph)\n");
  std::printf(" Honest: edge-rail rounds do not chain cleanly; field propagation does. Path 0-1-2-3.\n");
  std::printf("=====================================================================\n");
  Graph G(4); G.addEdge(0, 1, 1, 0); G.addEdge(1, 2, 1, 0); G.addEdge(2, 3, 1, 0);   // path adjacency (real, phaseless)
  Stepper S; S.build(G.h, kDt);
  CMat U = S.asMatrix(); CMat U2 = matmul(U, U), U3 = matmul(U2, U);
  int pass = 0, total = 0;

  std::printf("\n[1] U, U^2, U^3 ARE UNITARY (state conserved at every depth)\n");
  { double a = unitErr(U), b = unitErr(U2), c = unitErr(U3); std::printf("    |U^dag U - I|: U=%.2e  U^2=%.2e  U^3=%.2e\n", a, b, c);
    ++total; pass += report("a depth is not unitary", a < 1e-12 && b < 1e-12 && c < 1e-12); }

  std::printf("\n[2] THE RECEPTIVE FIELD DEEPENS: reach to the far node (0->3) grows with depth\n");
  { double d1 = std::abs(U[0][3]), d2 = std::abs(U2[0][3]), d3 = std::abs(U3[0][3]);
    std::printf("    |U^k[0][3]| (node0 reaching node3):  k=1: %.4f   k=2: %.4f   k=3: %.4f\n", d1, d2, d3);
    ++total; pass += report("receptive field did not deepen with depth", d3 > d1 && d3 > 5.0 * d1); }

  std::printf("\n[3] NEARER NODES ARE REACHED MORE STRONGLY THAN FARTHER ONES (per depth)\n");
  { double r1 = std::abs(U[0][1]), r2 = std::abs(U[0][2]), r3 = std::abs(U[0][3]);
    std::printf("    one step |U[0][d]|:  dist1=%.4f > dist2=%.4f > dist3=%.4f\n", r1, r2, r3);
    ++total; pass += report("distance ordering wrong", r1 > r2 && r2 > r3); }

  std::printf("\n RESULT : %d / %d verified  (deep working GNN via chainable field propagation)\n", pass, total);
  return pass == total ? 0 : 1;
}
