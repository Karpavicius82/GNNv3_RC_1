// THE CLOSED PIPELINE: FLOW -> DECORRELATION -> READOUT, on a real graph with
// CORRELATED classes (the regime that broke the un-glued system).
//   FLOW         : coherent neighbour sum over the sparse graph (denoise neighbourhood)
//   DECORRELATION: CG-whiten overlaps against the class Gram (G^-1, native recurrent)
//   READOUT      : argmax of the decorrelated class coefficients
// Compared to the NAIVE readout (argmax raw overlap) which collapses when classes
// overlap. Weights-free throughout. Real numbers, correlation sweep.
#include <complex>
#include <vector>
#include <cmath>
#include <cstdio>
#include <random>
using cd = std::complex<double>;
using Vec = std::vector<cd>;
static const double kPi = 3.14159265358979323846;

int main() {
  int N = 4000, C = 5, D = 256, deg = 12;
  double homophily = 0.75, noise = 1.0;
  std::mt19937 rng(2026u);
  std::uniform_int_distribution<int> randNode(0, N - 1), randClass(0, C - 1);
  std::uniform_real_distribution<double> uni(0, 1), ph(-kPi, kPi);
  std::normal_distribution<double> gn(0, noise);
  auto rnd = [&]() { Vec v(D); for (int d = 0; d < D; d++) v[d] = std::exp(cd(0, ph(rng))) / std::sqrt((double)D); return v; };
  auto inr = [&](const Vec& a, const Vec& b) { cd s(0, 0); for (int d = 0; d < D; d++) s += std::conj(a[d]) * b[d]; return s; };
  auto nrm = [&](Vec v) { double n = 0; for (auto& z : v) n += std::norm(z); n = std::sqrt(n); for (auto& z : v) z /= n; return v; };

  std::vector<int> y(N); for (int i = 0; i < N; i++) y[i] = randClass(rng);
  std::vector<std::vector<int>> byClass(C); for (int i = 0; i < N; i++) byClass[y[i]].push_back(i);
  // sparse homophily graph
  std::vector<std::vector<int>> adj(N);
  for (int i = 0; i < N; i++) for (int e = 0; e < deg; e++) {
    int j; if (uni(rng) < homophily) { auto& v = byClass[y[i]]; j = v[std::uniform_int_distribution<int>(0, (int)v.size() - 1)(rng)]; } else j = randNode(rng);
    if (j != i) { adj[i].push_back(j); adj[j].push_back(i); }
  }
  // CG solve G c = s  (C x C, native: matvec + inner products)
  auto cg = [&](const std::vector<std::vector<cd>>& G, const std::vector<cd>& s) {
    auto mv = [&](const std::vector<cd>& x) { std::vector<cd> yv(C, cd(0, 0)); for (int i = 0; i < C; i++) for (int j = 0; j < C; j++) yv[i] += G[i][j] * x[j]; return yv; };
    auto dot = [&](const std::vector<cd>& a, const std::vector<cd>& b) { cd r(0, 0); for (int i = 0; i < C; i++) r += std::conj(a[i]) * b[i]; return r; };
    std::vector<cd> c(C, cd(0, 0)), r = s, p = s; cd rs = dot(r, r);
    for (int t = 0; t < C + 2; t++) { auto Gp = mv(p); cd a = rs / dot(p, Gp); for (int i = 0; i < C; i++) { c[i] += a * p[i]; r[i] -= a * Gp[i]; } cd rs2 = dot(r, r); if (rs2.real() < 1e-26) break; cd b = rs2 / rs; for (int i = 0; i < C; i++) p[i] = r[i] + b * p[i]; rs = rs2; }
    return c;
  };

  std::printf("=== CLOSED PIPELINE: flow -> decorrelation -> readout ===\n");
  std::printf("RELATIONAL task: each node = superposition of TWO correlated classes; recover BOTH (top-2).\n");
  std::printf("N=%d, C=%d, D=%d, homophily=%.2f, noise=%.1f\n\n", N, C, D, homophily, noise);
  std::printf("  class-corr | raw(no flow) | FLOW+naive | FLOW+DECORRELATION (glued)\n");
  for (double beta : {0.0, 0.5, 0.7, 0.85}) {
    // correlated class signatures: sig_c = beta*base + sqrt(1-beta^2)*unique
    Vec base = rnd(); std::vector<Vec> sig(C);
    for (int c = 0; c < C; c++) { Vec u = rnd(), v(D); for (int d = 0; d < D; d++) v[d] = beta * base[d] + std::sqrt(1 - beta * beta) * u[d]; sig[c] = nrm(v); }
    std::vector<std::vector<cd>> G(C, std::vector<cd>(C)); for (int a = 0; a < C; a++) for (int b = 0; b < C; b++) G[a][b] = inr(sig[a], sig[b]);
    // RELATIONAL task: each node carries a SUPERPOSITION of two memberships
    // (its class + a second class). Recover BOTH (top-2). Correlated classes blur
    // the naive top-2; decorrelation must separate them.
    std::vector<int> y2(N); for (int i = 0; i < N; i++) { do { y2[i] = randClass(rng); } while (y2[i] == y[i]); }
    std::vector<Vec> feat(N); for (int i = 0; i < N; i++) { Vec f(D); for (int d = 0; d < D; d++) f[d] = (sig[y[i]][d] + sig[y2[i]][d]) * std::exp(cd(0, gn(rng))); feat[i] = nrm(f); }
    auto hop = [&](const std::vector<Vec>& X) { std::vector<Vec> Y(N, Vec(D, cd(0, 0))); for (int i = 0; i < N; i++) for (int j : adj[i]) for (int d = 0; d < D; d++) Y[i][d] += X[j][d]; return Y; };
    std::vector<Vec> h = hop(feat);   // 1-hop: neighbours share class y (homophily) -> reinforces y, y2 stays
    auto top2 = [&](const std::vector<cd>& coef) { int a = 0, b = 1; if (std::abs(coef[1]) > std::abs(coef[0])) { a = 1; b = 0; }
      for (int c = 2; c < C; c++) { double m = std::abs(coef[c]); if (m > std::abs(coef[a])) { b = a; a = c; } else if (m > std::abs(coef[b])) b = c; } return std::pair<int, int>(a, b); };
    auto acc = [&](const std::vector<Vec>& X, bool decor) {
      int ok = 0; for (int i = 0; i < N; i++) {
        std::vector<cd> s(C); for (int c = 0; c < C; c++) s[c] = inr(sig[c], X[i]);
        std::vector<cd> coef = decor ? cg(G, s) : s;
        auto [a, b] = top2(coef);
        bool got = ((a == y[i] || b == y[i]) && (a == y2[i] || b == y2[i]));
        ok += got;
      } return 100.0 * ok / N;
    };
    std::printf("  %.2f       |    %.1f%%      |   %.1f%%    |        %.1f%%\n", beta, acc(feat, false), acc(h, false), acc(h, true));
  }
  std::printf("\nIf at high class-correlation FLOW+DECORRELATION >> FLOW+naive, the glue substrate\ncloses the system on correlated data. Weights-free, native operations throughout.\n");
  return 0;
}
