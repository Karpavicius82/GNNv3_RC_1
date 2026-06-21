// DERIVED missing substrate: DECORRELATION / WHITENING (dual basis G^-1).
// Born and settling failed on correlated keys because raw overlaps <k_i,q> are
// contaminated by off-diagonal Gram terms. The exact fix: c = G^-1 <k,q> (the
// biorthogonal/dual coordinates). This is the glue between the unitary FLOW (which
// emits correlated overlaps) and MEMORY/SETTLING (which needs clean coordinates).
// Weights-free: G is the patterns' own overlap matrix, not a tuned parameter.
// Test: same correlated 2-hop routing where Born/settle gave 0.46-0.65.
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
  // Gram inverse times s  (Gauss-Jordan on K x K complex)
  auto gramSolve = [&](const Vec& q) {
    std::vector<std::vector<cd>> G(K, std::vector<cd>(K)), I(K, std::vector<cd>(K, cd(0, 0)));
    std::vector<cd> s(K);
    for (int i = 0; i < K; i++) { I[i][i] = cd(1, 0); s[i] = inr(key[i], q); for (int j = 0; j < K; j++) G[i][j] = inr(key[i], key[j]); }
    for (int c = 0; c < K; c++) { int p = c; for (int r = c + 1; r < K; r++) if (std::abs(G[r][c]) > std::abs(G[p][c])) p = r;
      std::swap(G[p], G[c]); std::swap(I[p], I[c]); cd d = G[c][c];
      for (int j = 0; j < K; j++) { G[c][j] /= d; I[c][j] /= d; }
      for (int r = 0; r < K; r++) if (r != c) { cd f = G[r][c]; for (int j = 0; j < K; j++) { G[r][j] -= f * G[c][j]; I[r][j] -= f * I[c][j]; } } }
    std::vector<cd> c(K, cd(0, 0)); for (int i = 0; i < K; i++) for (int j = 0; j < K; j++) c[i] += I[i][j] * s[j];  // c = G^-1 s
    return c;
  };
  auto raw = [&](const Vec& q) { Vec r(D, cd(0, 0)); for (int i = 0; i < K; i++) { double w = std::norm(inr(key[i], q)); for (int d = 0; d < D; d++) r[d] += w * val[i][d]; } return nrm(r); };
  auto whiten = [&](const Vec& q) { std::vector<cd> c = gramSolve(q); Vec r(D, cd(0, 0)); for (int i = 0; i < K; i++) for (int d = 0; d < D; d++) r[d] += c[i] * val[i][d]; return nrm(r); };

  std::printf("=== DECORRELATION (whitening, G^-1) -- the missing glue substrate ===\n");
  std::printf("2-hop kA->kB->vC, correlated distractors. raw Born vs whitened (G^-1).\n\n");
  std::printf("  corr | raw/Born final~vC | WHITENED final~vC\n");
  for (double corr : {0.0, 0.6, 0.8, 0.9, 0.95}) {
    build(corr);
    Vec b2 = raw(raw(kA));
    Vec w2 = whiten(whiten(kA));
    std::printf("  %.2f |       %.3f        |       %.3f\n", corr, std::abs(inr(vC, b2)), std::abs(inr(vC, w2)));
  }
  std::printf("\nIf whitened ~1.0 at all correlations where Born collapsed, the missing\nintermediate substrate is DECORRELATION (dual basis / lateral inhibition).\n");
  return 0;
}
