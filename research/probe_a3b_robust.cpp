// A3b: is the diffusion exponent REAL or a fit artifact?
// A real exponent must be invariant to box size w, time step dt, and fit window.
// If beta(golden) drifts with these, A3 is also speculative and I say so.
// Honest guard: fit ONLY in the pre-saturation window t < w/4 (before the
// ballistic front reaches the wall), and REPORT the saturation fraction so a
// contaminated fit is visible, not hidden.
#include "../tools/graph_wave_substrate.hpp"
#include <cstdio>
#include <complex>
using namespace gw;

struct Res { double beta; double satFrac; int npts; };

static Res run(const Graph& g, int w, double dt) {
 int N = w * w;
 auto id = [w](int x, int y) { return y * w + x; };
 double cx = (w - 1) / 2.0, cy = (w - 1) / 2.0;
 Stepper st; st.build(g.h, dt);
 Vec psi(N, cd(0, 0)); psi[id((int)cx, (int)cy)] = cd(1, 0);
 double tHi = w / 4.0; // pre-wall (group velocity ~2 in 2D TB)
 if (tHi > 12.0) tHi = 12.0;
 double tLo = 1.5;
 int steps = (int)(tHi / dt) + 2;
 std::vector<double> lt, ls; double sigMax = 0;
 for (int k = 1; k <= steps; k++) {
 psi = st.step(psi);
 double mx = 0, my = 0, m2 = 0, nrm = 0;
 for (int y = 0; y < w; y++)
 for (int x = 0; x < w; x++) {
 double p = std::norm(psi[id(x, y)]);
 nrm += p; mx += p * x; my += p * y; m2 += p * (x * x + y * y);
 }
 mx /= nrm; my /= nrm; m2 /= nrm;
 double sig = std::sqrt(std::max(0.0, m2 - (mx * mx + my * my)));
 sigMax = std::max(sigMax, sig);
 double t = k * dt;
 if (t >= tLo && t <= tHi) { lt.push_back(std::log(t)); ls.push_back(std::log(sig)); }
 }
 int n = (int)lt.size();
 double mt = 0, msg = 0;
 for (int i = 0; i < n; i++) { mt += lt[i]; msg += ls[i]; }
 mt /= n; msg /= n;
 double num = 0, den = 0;
 for (int i = 0; i < n; i++) { num += (lt[i] - mt) * (ls[i] - msg); den += (lt[i] - mt) * (lt[i] - mt); }
 return { num / den, sigMax / (0.5 * w), n }; // satFrac >~0.8 => contaminated by wall
}

int main() {
 std::printf("=== A3b robustness: beta vs (w, dt). Real exponent => invariant. ===\n");
 std::printf("satFrac = sigmaMax/(w/2); if >~0.8 the fit window touched the wall (suspect).\n\n");
 struct T { const char* nm; };
 for (const char* terp : {"plain", "golden 8/13", "disorder W=4"}) {
 std::printf("-- %s --\n w dt beta satFrac npts\n", terp);
 for (int w : {16, 24, 32, 40}) {
 for (double dt : {0.5, 0.25}) {
 Graph g = (std::string(terp) == "plain") ? plain(w)
 : (std::string(terp) == "golden 8/13") ? golden(w, 8.0 / 13)
 : disorder(w, 4.0);
 Res r = run(g, w, dt);
 std::printf(" %-4d %.2f %+.3f %.2f %d\n", w, dt, r.beta, r.satFrac, r.npts);
 }
 }
 std::printf("\n");
 }
 return 0;
}
