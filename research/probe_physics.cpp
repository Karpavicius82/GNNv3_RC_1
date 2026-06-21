// CONCRETE-PHYSICS verification of the production sparse engine. No ML, no tasks.
// Three hypotheses I asserted, now tested rigorously against exact ground truth:
//  H1: the t=5 degradation was ONLY my Bessel series (cancellation), not physics.
//      Fix = stable Miller downward recurrence -> propagator EXACT at ANY t.
//  H2: the sparse engine is GAUGE-INVARIANT (a local gauge leaves |psi_i|^2 alone)
//      -- the physical meaning lives in the flux, not absolute phase, AT SCALE.
//  H3: it reproduces EXACT destructive interference (flux=pi -> exact decoupling),
//      the actual computational primitive, through the production engine.
#include "../tools/graph_wave_substrate.hpp"
#include <cstdio>
#include <complex>
#include <vector>
#include <cmath>
#include <algorithm>
using namespace gw;

struct Sparse { int N; std::vector<std::vector<std::pair<int, cd>>> adj; };
static std::vector<cd> matvec(const Sparse& g, const std::vector<cd>& x) {
  std::vector<cd> y(g.N, cd(0, 0));
  for (int i = 0; i < g.N; i++) { cd s(0, 0); for (auto& e : g.adj[i]) s += e.second * x[e.first]; y[i] = s; }
  return y;
}
// STABLE Bessel J_0..J_kmax(x) via Miller downward recurrence + normalization.
static std::vector<double> besselAll(double x, int kmax) {
  int M = std::max(kmax, (int)x) + 60;
  std::vector<double> J(M + 2, 0.0);
  J[M + 1] = 0.0; J[M] = 1e-300;
  for (int k = M; k >= 1; k--) J[k - 1] = (2.0 * k / x) * J[k] - J[k + 1];
  double norm = J[0]; for (int k = 2; k <= M; k += 2) norm += 2.0 * J[k];
  for (int k = 0; k <= M; k++) J[k] /= norm;
  J.resize(kmax + 1); return J;
}
static std::vector<cd> chebProp(const Sparse& g, std::vector<cd> psi, double t, double a) {
  int K = (int)(a * t) + 40;
  std::vector<double> J = besselAll(a * t, K);
  std::vector<cd> Tkm1 = psi, Tk = matvec(g, psi); for (auto& v : Tk) v /= a;
  std::vector<cd> out(g.N, cd(0, 0)); cd i0(0, -1);
  for (int k = 0; k < K; k++) {
    cd coef = (k == 0 ? 1.0 : 2.0) * std::pow(i0, k) * J[k];
    if (k <= 1) { const std::vector<cd>& T = (k == 0) ? psi : Tk; for (int n = 0; n < g.N; n++) out[n] += coef * T[n]; }
    else { std::vector<cd> Tp = matvec(g, Tk); for (auto& v : Tp) v /= a; for (int n = 0; n < g.N; n++) Tp[n] = 2.0 * Tp[n] - Tkm1[n]; Tkm1 = Tk; Tk = Tp; for (int n = 0; n < g.N; n++) out[n] += coef * Tk[n]; }
  }
  return out;
}
static double pw(const std::vector<cd>& x) { double s = 0; for (auto& v : x) s += std::norm(v); return s; }

static Sparse fromDense(const CMat& H, double& a) {
  int N = (int)H.size(); Sparse s; s.N = N; s.adj.assign(N, {}); a = 0;
  for (int i = 0; i < N; i++) { double row = 0; for (int j = 0; j < N; j++) if (std::abs(H[i][j]) > 1e-15) { s.adj[i].push_back({j, H[i][j]}); row += std::abs(H[i][j]); } a = std::max(a, row); }
  a *= 1.05; return s;
}

int main() {
  // ---- H1: propagator exact at ANY t (stable Bessel) vs exact eigendecomposition ----
  int w = 7, N = w * w; Graph G = golden(w, 8.0 / 13.0);
  std::mt19937 rng(99u); std::uniform_real_distribution<double> u(-0.5, 0.5);
  for (int i = 0; i < N; i++) G.onsite(i, u(rng));
  const CMat& H = G.h; double a; Sparse sp = fromDense(H, a);
  Eig e = jacobiV(realRep(H));
  std::vector<int> idx(2 * N); for (int i = 0; i < 2 * N; i++) idx[i] = i;
  std::sort(idx.begin(), idx.end(), [&](int p, int q) { return e.ev[p] < e.ev[q]; });
  std::vector<double> lam(N); std::vector<Vec> U(N);
  for (int c = 0; c < N; c++) { int col = idx[2 * c]; lam[c] = e.ev[col]; Vec v(N);
    for (int n = 0; n < N; n++) v[n] = cd(e.V[n][col], e.V[n + N][col]);
    double nn = std::sqrt(rinner(v, v).real()); for (auto& z : v) z /= nn; U[c] = v; }

  std::printf("=== H1: is the propagator EXACT at ANY t? (stable Miller-Bessel vs exact eig) ===\n");
  std::printf("  t      max|cheb - exact|   normDrift\n");
  for (double t : {1.0, 5.0, 10.0, 20.0, 50.0}) {
    Vec psi(N, cd(0, 0)); psi[0] = cd(1, 0);
    Vec ch = chebProp(sp, psi, t, a);
    Vec ex(N, cd(0, 0));
    for (int c = 0; c < N; c++) { cd cf = std::exp(cd(0, -lam[c] * t)) * rinner(psi, U[c]); for (int n = 0; n < N; n++) ex[n] += cf * U[c][n]; }
    double diff = 0; for (int n = 0; n < N; n++) diff = std::max(diff, std::abs(ch[n] - ex[n]));
    std::printf("  %-5.0f  %.2e          %.2e\n", t, diff, std::abs(pw(ch) - 1.0));
  }

  // ---- H2: gauge invariance on the sparse engine (physical observable unchanged) ----
  std::printf("\n=== H2: GAUGE INVARIANCE of the sparse engine (|psi_i|^2 must not change) ===\n");
  std::vector<double> gphi(N); { std::uniform_real_distribution<double> pp(-kPi, kPi); for (int i = 0; i < N; i++) gphi[i] = pp(rng); }
  Sparse spg = sp; for (int i = 0; i < N; i++) for (auto& e2 : spg.adj[i]) e2.second *= std::exp(cd(0, gphi[i] - gphi[e2.first]));
  Vec p0(N, cd(0, 0)); p0[3] = cd(1, 0);
  Vec pg = p0; for (int i = 0; i < N; i++) pg[i] *= std::exp(cd(0, gphi[i]));
  Vec a1 = chebProp(sp, p0, 7.0, a), a2 = chebProp(spg, pg, 7.0, a);
  double obsDiff = 0; for (int i = 0; i < N; i++) obsDiff = std::max(obsDiff, std::abs(std::norm(a1[i]) - std::norm(a2[i])));
  std::printf("  max | |psi_i|^2 - |gauge(psi)_i|^2 |  after t=7 = %.2e   (must be ~machine)\n", obsDiff);

  // ---- H3: exact destructive interference (flux=pi gate) through the sparse engine ----
  std::printf("\n=== H3: EXACT flux=pi decoupling through the production engine ===\n");
  // diamond: 0 ->(up)-> 2 and 0 ->(down, phase phi)-> 2 ; at phi=pi node 2 stays dark
  for (double f : {0.0, kPi}) {
    Sparse d; d.N = 4; d.adj.assign(4, {});
    auto add = [&](int i, int j, double ph) { cd v = std::exp(cd(0, ph)); d.adj[i].push_back({j, v}); d.adj[j].push_back({i, std::conj(v)}); };
    add(0, 1, 0); add(1, 2, 0); add(0, 3, 0); add(3, 2, f);
    double mx = 0; for (double t = 0.1; t <= 12.0; t += 0.1) { Vec p(4, cd(0, 0)); p[0] = cd(1, 0); Vec o = chebProp(d, p, t, 2.2); mx = std::max(mx, std::abs(o[2])); }
    std::printf("  flux=%-4s node-2 max reach over time = %.2e\n", f == 0 ? "0" : "pi", mx);
  }
  std::printf("\nIf H1 ~1e-12 at all t, H2 ~machine, H3(pi) ~machine-zero: the production engine\nIS the concrete physics -- exact propagator, gauge-invariant, exact interference.\n");
  return 0;
}
