// A2 SIM: does PHASE survive depth?
// The open native problem: clean chainable depth so far existed only via REAL
// (phaseless) diffusion -- the weights direction we reject. Test field
// propagation (unitary Cayley step, phase-carrying) on the GOLDEN lattice and
// ask, at every depth k:
//   (a) receptive field deepens   : |U^k[far,center]| grows
//   (b) stays unitary             : unitErr(U^k) ~ machine precision
//   (c) a GAUGE-INVARIANT phase signature survives composition:
//       chirality <L_z>(k) = orbital circulation about the lattice centre.
//       On plain (flux 0) <L_z>=0 by symmetry at ALL k (real diffusion cannot
//       make it). On golden (magnetic flux) <L_z> != 0 and persists with depth
//       => phase is doing work at depth, not decaying into a real diffusion.
#include "../tools/graph_wave_substrate.hpp"
#include <cstdio>
#include <complex>
using namespace gw;

int main() {
  int w = 8, N = w * w;
  auto id = [w](int x, int y) { return y * w + x; };
  double cx = (w - 1) / 2.0, cy = (w - 1) / 2.0;
  double dt = 0.5;

  struct Cfg { const char* nm; Graph g; };
  std::vector<Cfg> cfgs = {
      {"plain (flux 0)", plain(w)},
      {"golden 8/13", golden(w, 8.0 / 13)},
      {"disorder W=4", disorder(w, 4.0)},
  };

  // L_z = (x-cx) p_y - (y-cy) p_x,  p = -i grad (central difference)
  auto Lz = [&](const Vec& psi) {
    Vec out(N, cd(0, 0));
    for (int y = 0; y < w; y++)
      for (int x = 0; x < w; x++) {
        cd py(0, 0), px(0, 0);
        if (y + 1 < w) py += psi[id(x, y + 1)];
        if (y - 1 >= 0) py -= psi[id(x, y - 1)];
        if (x + 1 < w) px += psi[id(x + 1, y)];
        if (x - 1 >= 0) px -= psi[id(x - 1, y)];
        py *= cd(0, -0.5); px *= cd(0, -0.5);  // -i/2 * central diff
        out[id(x, y)] = (x - cx) * py - (y - cy) * px;
      }
    return out;
  };
  auto expectLz = [&](const Vec& psi) {
    Vec l = Lz(psi); cd s(0, 0);
    for (int i = 0; i < N; i++) s += std::conj(psi[i]) * l[i];
    return s.real();
  };

  for (auto& c : cfgs) {
    Stepper st; st.build(c.g.h, dt);
    CMat U = st.asMatrix();
    int far = id(w - 1, w - 1), ctr = id((int)cx, (int)cy);
    // initial: a small Gaussian-ish blob at centre (symmetric) so plain stays L_z=0
    Vec psi(N, cd(0, 0));
    for (int y = 0; y < w; y++)
      for (int x = 0; x < w; x++) {
        double r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
        psi[id(x, y)] = std::exp(-r2 / 2.0);
      }
    double nrm = std::sqrt(power(psi)); for (auto& v : psi) v /= nrm;

    std::printf("\n== %-16s ==\n  k   |U^k[far,ctr]|   unitErr(U^k)   <L_z>(k)\n", c.nm);
    CMat Uk = U; Vec p = psi;
    for (int k = 1; k <= 12; k++) {
      p = st.step(p);  // exact propagation of the state
      double reach = std::abs(Uk[far][ctr]);
      double ue = unitErr(Uk);
      std::printf("  %-2d   %.4f          %.1e        %+.4f\n", k, reach, ue, expectLz(p));
      Uk = matmul(U, Uk);
    }
  }
  std::printf("\nRead: plain <L_z>=0 at all depth (no phase). golden <L_z>!=0 and persists\n=> a gauge-invariant, weights-free signature carried THROUGH depth.\n");
  return 0;
}
