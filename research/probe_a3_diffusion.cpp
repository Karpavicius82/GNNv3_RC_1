// A3 SIM: anomalous diffusion exponent.
// Wavepacket from centre, unitary Cayley evolution, spread sigma^2(t) = <r^2>-<r>^2.
// Fit sigma ~ t^beta:  ballistic beta=1 (extended), diffusive 0.5, localized 0.
// Critical/multifractal => anomalous (sub-ballistic) and bounded away from 0.
// Pure substrate dynamics; sharper DYNAMICAL criticality parash than the triad.
#include "../tools/graph_wave_substrate.hpp"
#include <cstdio>
#include <complex>
using namespace gw;

int main() {
  int w = 40, N = w * w;  // bigger box so the ballistic (plain) front does not hit the wall
  auto id = [w](int x, int y) { return y * w + x; };
  double cx = (w - 1) / 2.0, cy = (w - 1) / 2.0;
  double dt = 0.5;
  int steps = 60;

  struct Cfg { const char* nm; Graph g; };
  std::vector<Cfg> cfgs = {
      {"plain (extended)", plain(w)},
      {"golden 8/13", golden(w, 8.0 / 13)},
      {"disorder W=4", disorder(w, 4.0)},
  };

  for (auto& c : cfgs) {
    Stepper st; st.build(c.g.h, dt);
    Vec psi(N, cd(0, 0)); psi[id((int)cx, (int)cy)] = cd(1, 0);  // delta at centre
    // log-log least squares over a mid window (skip initial transient & boundary hit)
    std::vector<double> lt, ls;
    std::printf("\n== %-16s ==\n   t      sigma\n", c.nm);
    for (int k = 1; k <= steps; k++) {
      psi = st.step(psi);
      double mx = 0, my = 0, m2 = 0, nrm = 0;
      for (int y = 0; y < w; y++)
        for (int x = 0; x < w; x++) {
          double p = std::norm(psi[id(x, y)]);
          nrm += p; mx += p * x; my += p * y; m2 += p * (x * x + y * y);
        }
      mx /= nrm; my /= nrm; m2 /= nrm;
      double var = m2 - (mx * mx + my * my);
      double sig = std::sqrt(std::max(0.0, var));
      double t = k * dt;
      if (k % 6 == 0) std::printf("  %5.1f   %.3f\n", t, sig);
      if (t >= 2.0 && t <= 10.0) { lt.push_back(std::log(t)); ls.push_back(std::log(sig)); }
    }
    // slope = beta
    double mt = 0, msg = 0; int n = (int)lt.size();
    for (int i = 0; i < n; i++) { mt += lt[i]; msg += ls[i]; }
    mt /= n; msg /= n;
    double num = 0, den = 0;
    for (int i = 0; i < n; i++) { num += (lt[i] - mt) * (ls[i] - msg); den += (lt[i] - mt) * (lt[i] - mt); }
    double beta = num / den;
    std::printf("  -> diffusion exponent beta = %.3f  (ballistic 1.0, diffusive 0.5, localized 0)\n", beta);
  }
  return 0;
}
