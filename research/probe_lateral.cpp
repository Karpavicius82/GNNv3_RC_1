// Native realization of the DECORRELATION substrate: not a matrix inverse, but
// recurrent LATERAL INHIBITION. Each pattern's coefficient is its raw overlap
// MINUS the overlap-weighted activity of the others:  c <- s - E c  (E = off-
// diagonal Gram). Its fixed point is (I+E)c = s = G^-1 s -- the decorrelated
// (dual) coordinates, computed by local competition, no explicit inverse.
// Cross-check it against the exact G^-1 (am I truly not mistaken?) and route.
#include <complex>
#include <vector>
#include <cmath>
#include <cstdio>
#include <random>
using cd = std::complex<double>;
using Vec = std::vector<cd>;
static const double kPi = 3.14159265358979323846;

int main() {
  int D = 512, K = 8;
  std::mt19937 rng(7u); std::uniform_real_distribution<double> ph(-kPi, kPi);
  auto rnd = [&]() { Vec v(D); for (int d = 0; d < D; d++) v[d] = std::exp(cd(0, ph(rng))) / std::sqrt((double)D); return v; };
  auto inr = [&](const Vec& a, const Vec& b) { cd s(0, 0); for (int d = 0; d < D; d++) s += std::conj(a[d]) * b[d]; return s; };
  auto nrm = [&](Vec v) { double n = 0; for (auto& z : v) n += std::norm(z); n = std::sqrt(n); for (auto& z : v) z /= n; return v; };

  Vec kA = rnd(), kB = rnd(), vC = rnd();
  std::vector<Vec> key(K), val(K);
  auto build = [&](double corr) {
    auto mix = [&](const Vec& base) { Vec n = rnd(), r(D); for (int d = 0; d < D; d++) r[d] = corr * base[d] + (1 - corr) * n[d]; return nrm(r); };
    key[0] = kA; val[0] = kB; key[1] = kB; val[1] = vC;
    for (int i = 2; i < K; i++) { key[i] = mix(i % 2 ? kA : kB); val[i] = rnd(); }
  };
  auto gram = [&]() { std::vector<std::vector<cd>> G(K, std::vector<cd>(K)); for (int i = 0; i < K; i++) for (int j = 0; j < K; j++) G[i][j] = inr(key[i], key[j]); return G; };
  auto overlaps = [&](const Vec& q) { std::vector<cd> s(K); for (int i = 0; i < K; i++) s[i] = inr(key[i], q); return s; };
  // exact c = G^-1 s
  auto exact = [&](const std::vector<cd>& s) { auto G = gram(); std::vector<std::vector<cd>> I(K, std::vector<cd>(K, cd(0, 0))); for (int i = 0; i < K; i++) I[i][i] = cd(1, 0);
    for (int c = 0; c < K; c++) { int p = c; for (int r = c + 1; r < K; r++) if (std::abs(G[r][c]) > std::abs(G[p][c])) p = r; std::swap(G[p], G[c]); std::swap(I[p], I[c]); cd d = G[c][c];
      for (int j = 0; j < K; j++) { G[c][j] /= d; I[c][j] /= d; } for (int r = 0; r < K; r++) if (r != c) { cd f = G[r][c]; for (int j = 0; j < K; j++) { G[r][j] -= f * G[c][j]; I[r][j] -= f * I[c][j]; } } }
    std::vector<cd> c(K, cd(0, 0)); for (int i = 0; i < K; i++) for (int j = 0; j < K; j++) c[i] += I[i][j] * s[j]; return c; };
  // native: damped lateral inhibition  c <- (1-a) c + a (s - E c) ; E = G with zero diagonal
  auto lateral = [&](const std::vector<cd>& s, int T, double a) { auto G = gram(); std::vector<cd> c = s;
    for (int t = 0; t < T; t++) { std::vector<cd> nc(K); for (int i = 0; i < K; i++) { cd inh(0, 0); for (int j = 0; j < K; j++) if (j != i) inh += G[i][j] * c[j]; nc[i] = (1 - a) * c[i] + a * (s[i] - inh); } c = nc; } return c; };
  // better-conditioned native recurrent solver: conjugate gradient on G c = s
  // (G Hermitian pos-def -> exact in <=K steps for ANY correlation; only matvec + inner products)
  auto cg = [&](const std::vector<cd>& s, int T) { auto G = gram();
    auto mv = [&](const std::vector<cd>& x) { std::vector<cd> y(K, cd(0, 0)); for (int i = 0; i < K; i++) for (int j = 0; j < K; j++) y[i] += G[i][j] * x[j]; return y; };
    auto dot = [&](const std::vector<cd>& a, const std::vector<cd>& b) { cd s2(0, 0); for (int i = 0; i < K; i++) s2 += std::conj(a[i]) * b[i]; return s2; };
    std::vector<cd> c(K, cd(0, 0)), r = s, p = s; cd rs = dot(r, r);
    for (int t = 0; t < T; t++) { std::vector<cd> Gp = mv(p); cd a = rs / dot(p, Gp);
      for (int i = 0; i < K; i++) { c[i] += a * p[i]; r[i] -= a * Gp[i]; } cd rs2 = dot(r, r);
      if (rs2.real() < 1e-26) break; cd b = rs2 / rs; for (int i = 0; i < K; i++) p[i] = r[i] + b * p[i]; rs = rs2; }
    return c; };
  auto routeFrom = [&](const std::vector<cd>& c) { Vec r(D, cd(0, 0)); for (int i = 0; i < K; i++) for (int d = 0; d < D; d++) r[d] += c[i] * val[i][d]; return nrm(r); };

  std::printf("=== DECORRELATION as native recurrent LATERAL INHIBITION (no matrix inverse) ===\n");
  std::printf("cross-check vs exact G^-1, and route. T=400 iters, damping a=0.25.\n\n");
  std::printf("  corr | |c_cg - c_exact| | route Jacobi | route CG (<=K steps) | route exact\n");
  for (double corr : {0.0, 0.6, 0.8, 0.9, 0.95, 0.98}) {
    build(corr);
    std::vector<cd> sA = overlaps(kA);
    std::vector<cd> cE = exact(sA), cC = cg(sA, K + 2);
    double diff = 0; for (int i = 0; i < K; i++) diff = std::max(diff, std::abs(cE[i] - cC[i]));
    Vec r2e = routeFrom(exact(overlaps(routeFrom(exact(overlaps(kA))))));
    Vec r1j = routeFrom(lateral(overlaps(kA), 400, 0.25)); Vec r2j = routeFrom(lateral(overlaps(r1j), 400, 0.25));
    Vec r1c = routeFrom(cg(overlaps(kA), K + 2)); Vec r2c = routeFrom(cg(overlaps(r1c), K + 2));
    std::printf("  %.2f |     %.2e     |    %.3f     |       %.3f          |    %.3f\n",
                corr, diff, std::abs(inr(vC, r2j)), std::abs(inr(vC, r2c)), std::abs(inr(vC, r2e)));
  }
  std::printf("\nCG = native recurrent (matvec + inner products), exact in <=K steps for any\ncorrelation. If CG routes ~1.0 throughout, the decorrelation substrate has a\nstable native recurrent realization -- I am not mistaken about the glue.\n");
  return 0;
}
