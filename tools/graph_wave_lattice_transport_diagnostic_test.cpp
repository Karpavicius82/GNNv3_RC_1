// graph_wave_lattice_transport_diagnostic_test
// ----------------------------------------------------------------------------
// THE CONNECTION (data-only diagnostic, returns 0, cannot break the suite): does
// the substrate criticality established statically by the eigenstate triad (plain
// extended / golden-flux critical / disorder localized) also appear in the GNN's
// own currency -- message-passing DYNAMICS? A wavepacket starts at the lattice
// centre and propagates unitarily under each lattice Hamiltonian; spread (RMS
// distance) and participation are TIME-AVERAGED over the late window (the
// plaquette diagnostic taught us a single snapshot of a closed system is fragile).
//
// Now built on the shared substrate core (graph_wave_substrate.hpp): the lattice
// families and the unitary Cayley stepper come from the single tested source, not
// a private copy. No decision, no learning, no trainer. Data only.
// ----------------------------------------------------------------------------
#include <cstdio>
#include "graph_wave_substrate.hpp"
namespace {
using namespace gw;
struct Stat { double spread; double partic; double normDrift; };
// propagate from centre, time-average spread (RMS distance) and participation over the late window
Stat transport(const Graph& g, int w, double dt, int steps) {
 int n = w * w; int cx = w / 2, cy = w / 2; int c = cy * w + cx;
 Stepper S; S.build(g.h, dt); Vec psi(n, cd(0, 0)); psi[c] = cd(1, 0);
 double accS = 0, accP = 0, maxDrift = 0; int cnt = 0; int half = steps / 2;
 for (int s = 0; s < steps; s++) { psi = S.step(psi);
 double nm = 0; for (auto& v : psi) nm += std::norm(v); maxDrift = std::max(maxDrift, std::abs(nm - 1));
 if (s >= half) { double sig = 0, p4 = 0;
 for (int y = 0; y < w; y++) for (int x = 0; x < w; x++) { double pr = std::norm(psi[y * w + x]); double d2 = double((x - cx) * (x - cx) + (y - cy) * (y - cy)); sig += pr * d2; p4 += pr * pr; }
 accS += std::sqrt(sig); accP += (p4 > 0 ? 1.0 / p4 : 0); cnt++; } }
 return { cnt ? accS / cnt : 0, cnt ? accP / cnt : 0, maxDrift };
}
}
int main() {
 std::printf("=====================================================================\n");
 std::printf(" GRAPH-WAVE LATTICE-TRANSPORT DIAGNOSTIC (criticality in message-passing DYNAMICS)\n");
 std::printf(" Wavepacket from centre; spread/participation TIME-AVERAGED over the late window.\n");
 std::printf("=====================================================================\n");
 const double a = 8.0 / 13.0, Wdis = 20.0; const int w = 12; const double dt = 0.1; const int steps = 200;
 std::printf(" lattice %dx%d (%d nodes), dt=%.2f, %d steps (t=%.0f), late-window time average\n", w, w, w * w, dt, steps, steps * dt);

 Graph gp = plain(w), gg = golden(w, a), gd = disorder(w, Wdis);
 Stat sp = transport(gp, w, dt, steps), sg = transport(gg, w, dt, steps), sd = transport(gd, w, dt, steps);

 std::printf("\n substrate | RMS spread | participation | norm drift\n");
 std::printf(" ---------------------+------------+---------------+-----------\n");
 std::printf(" plain (extended) | %7.3f | %7.2f | %.1e\n", sp.spread, sp.partic, sp.normDrift);
 std::printf(" golden-flux (cand) | %7.3f | %7.2f | %.1e\n", sg.spread, sg.partic, sg.normDrift);
 std::printf(" disorder W=20 (loc) | %7.3f | %7.2f | %.1e\n", sd.spread, sd.partic, sd.normDrift);

 std::printf("\n DIAGNOSTIC READINGS:\n");
 std::printf(" - unitary message passing (norm drift < 1e-12 on every lattice): %s\n",
 (sp.normDrift < 1e-12 && sg.normDrift < 1e-12 && sd.normDrift < 1e-12) ? "YES" : "NO");
 bool ordered = (sp.spread > sg.spread && sg.spread > sd.spread);
 std::printf(" - transport triad plain > golden > disorder (criticality in dynamics): %s\n",
 ordered ? "YES -> the static triad carries into message-passing dynamics" : "NOT cleanly ordered (report as-is)");
 std::printf(" - golden spread fraction between the controls: %.2f (0=disorder, 1=plain)\n",
 (sp.spread > sd.spread) ? (sg.spread - sd.spread) / (sp.spread - sd.spread) : 0.0);

 std::printf("\n RESULT : DIAGNOSTIC (data gathered, not a pass/fail gate)\n");
 return 0;
}
