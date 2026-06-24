// graph_wave_nonlinear_compute_contract_test
// ----------------------------------------------------------------------------
// Is the Kerr nonlinearity a COMPUTE ENGINE, or decoration? A pure unitary /
// linear substrate e^{-iHt} can only realize LINEAR maps of its input; a LINEAR
// readout on it can therefore only compute linear (affine) functions. The Kerr
// term -g|psi|^2 psi breaks that: it mixes modes (cross-phase / four-wave mixing),
// generating the high-order products a linear system cannot, so a LINEAR readout
// on the Kerr-evolved field can compute NONLINEAR functions and INTERPOLATE them.
//
// We use the substrate as a reservoir: encode the input as a field, evolve it,
// and fit a LINEAR ridge readout on [Re psi_i, Im psi_i]. The encoding is affine
// in the input and we do NOT renormalize, so for g=0 the features are exactly
// affine in the input -> a linear readout can ONLY fit affine targets. Any gap on
// a nonlinear target is therefore the Kerr engine, not the readout.
//
// Gates:
//   [1] CLASSIFY  : 4-bit parity (not linearly separable). g>0 solves it; g=0 ~chance.
//   [2] INTERPOLATE: continuous target y = prod(2x_j-1) (degree 3). g>0 fits & inter-
//       polates to held-out inputs (high R^2); g=0 cannot (low R^2).
//   [3] CONTROL   : a LINEAR target y = x1-x2. BOTH g=0 and g>0 fit it well -- proving
//       the linear reservoir is competent and the gap above is specifically nonlinearity.
// Deterministic (fixed seeds). Substrate-only; ridge is the readout, not a substrate weight.
// ----------------------------------------------------------------------------
#define NOMINMAX
#include "graph_wave_substrate.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {
using gw::cd; using gw::Vec;

constexpr int N = 16, T = 40;
constexpr double DT = 0.3, BASE = 0.2;
constexpr double GK = 4.0;   // edge-of-chaos: strong enough to mix, not so strong it scrambles
const int IN[4] = {1, 5, 9, 13};

// real ridge: solve (X^T X + lam I) w = X^T y by Gaussian elimination
std::vector<double> ridge(const std::vector<std::vector<double>>& X, const std::vector<double>& y, double lam) {
  int M = (int)X.size(), D = (int)X[0].size();
  std::vector<std::vector<double>> A(D, std::vector<double>(D, 0.0)); std::vector<double> b(D, 0.0);
  for (int m = 0; m < M; ++m) for (int i = 0; i < D; ++i) { b[i] += X[m][i] * y[m]; for (int j = 0; j < D; ++j) A[i][j] += X[m][i] * X[m][j]; }
  for (int i = 0; i < D; ++i) A[i][i] += lam;
  for (int k = 0; k < D; ++k) {
    int p = k; double bv = std::abs(A[k][k]); for (int i = k + 1; i < D; ++i) { double v = std::abs(A[i][k]); if (v > bv) { bv = v; p = i; } }
    std::swap(A[p], A[k]); std::swap(b[p], b[k]); double piv = A[k][k];
    for (int i = k + 1; i < D; ++i) { double f = A[i][k] / piv; for (int j = k; j < D; ++j) A[i][j] -= f * A[k][j]; b[i] -= f * b[k]; }
  }
  std::vector<double> w(D, 0.0); for (int i = D - 1; i >= 0; --i) { double s = b[i]; for (int j = i + 1; j < D; ++j) s -= A[i][j] * w[j]; w[i] = s / A[i][i]; }
  return w;
}

std::vector<double> features(const gw::KerrStepper& KS, const std::vector<double>& x) {
  Vec psi(N, cd(BASE, 0));
  for (int j = 0; j < (int)x.size(); ++j) psi[IN[j]] = cd(BASE + x[j], 0);  // affine encoding, NO renormalize
  psi = KS.evolve(psi, T);
  std::vector<double> f; f.reserve(2 * N + 1); f.push_back(1.0);
  for (int i = 0; i < N; ++i) { f.push_back(std::real(psi[i])); f.push_back(std::imag(psi[i])); }
  return f;
}

double predict(const std::vector<double>& w, const std::vector<double>& f) { double d = 0; for (size_t i = 0; i < w.size(); ++i) d += w[i] * f[i]; return d; }

// classification accuracy on k-bit parity for a given g
double classifyAcc(double g, int k) {
  gw::Graph G(N); for (int i = 0; i < N; ++i) G.addEdge(i, (i + 1) % N, 1.0);
  gw::KerrStepper KS; KS.build(G.h, DT, g, true);
  std::mt19937 rng(7); std::normal_distribution<double> jit(0, 0.05);
  std::vector<std::vector<double>> Xtr, Xte; std::vector<double> ytr, yte;
  int P = 1 << k;
  for (int s = 0; s < 180; ++s) for (int p = 0; p < P; ++p) {
    std::vector<double> x(k); int par = 0; for (int j = 0; j < k; ++j) { int b = (p >> j) & 1; par ^= b; x[j] = b + jit(rng); }
    auto f = features(KS, x); double lbl = par ? 1.0 : -1.0;
    if (s < 130) { Xtr.push_back(f); ytr.push_back(lbl); } else { Xte.push_back(f); yte.push_back(lbl); }
  }
  auto w = ridge(Xtr, ytr, 1e-2);
  int ok = 0; for (size_t m = 0; m < Xte.size(); ++m) if ((predict(w, Xte[m]) >= 0 ? 1.0 : -1.0) == yte[m]) ok++;
  return 100.0 * ok / Xte.size();
}

// regression R^2 on a continuous target for a given g; target(x) supplied by caller
template <class F>
double regressR2(double g, int k, F target, unsigned seed) {
  gw::Graph G(N); for (int i = 0; i < N; ++i) G.addEdge(i, (i + 1) % N, 1.0);
  gw::KerrStepper KS; KS.build(G.h, DT, g, true);
  std::mt19937 rng(seed); std::uniform_real_distribution<double> u(0.0, 1.0);
  std::vector<std::vector<double>> Xtr, Xte; std::vector<double> ytr, yte;
  for (int s = 0; s < 1600; ++s) {
    std::vector<double> x(k); for (int j = 0; j < k; ++j) x[j] = u(rng);
    auto f = features(KS, x); double y = target(x);
    if (s < 1200) { Xtr.push_back(f); ytr.push_back(y); } else { Xte.push_back(f); yte.push_back(y); }
  }
  auto w = ridge(Xtr, ytr, 1e-3);
  double mean = 0; for (double v : yte) mean += v; mean /= yte.size();
  double ssr = 0, sst = 0; for (size_t m = 0; m < Xte.size(); ++m) { double e = yte[m] - predict(w, Xte[m]); ssr += e * e; double t = yte[m] - mean; sst += t * t; }
  return 1.0 - ssr / (sst + 1e-300);
}

bool report(const char* name, bool ok) { std::printf("   => %s\n", ok ? "PASS" : "FAIL"); if (!ok) std::printf("   !! %s\n", name); return ok; }
} // namespace

int main() {
  std::printf("=====================================================================\n");
  std::printf("  NONLINEAR COMPUTE ENGINE CONTRACT  (linear readout on a wave reservoir)\n");
  std::printf("=====================================================================\n");
  int pass = 0, total = 0;

  // optimal nonlinearity is task-dependent (edge of chaos): a 3-bit parity decision
  // wants more mixing (g=4); a degree-2 continuous map wants less (g=1). g=0 does neither.
  auto lin = [](const std::vector<double>& x) { return x[0] - x[1]; };
  auto cprod = [](const std::vector<double>& x) { return (2 * x[0] - 1) * (2 * x[1] - 1); };  // pure bilinear: ZERO linear part

  std::printf("\n[1] LINEAR reservoir is COMPETENT: fits a linear target y = x1 - x2\n");
  double L0 = regressR2(0.0, 3, lin, 13);
  std::printf("    g=0 R2=%.2f\n", L0);
  ++total; pass += report("linear reservoir cannot even fit a linear target -> reservoir is just weak", L0 > 0.95);

  std::printf("\n[2] ...but the LINEAR reservoir is LIMITED: cannot do 3-bit parity\n");
  double cl0 = classifyAcc(0.0, 3);
  std::printf("    g=0 parity acc=%.1f%%  (chance = 50%%)\n", cl0);
  ++total; pass += report("linear reservoir unexpectedly solved parity", cl0 < 62.0);

  std::printf("\n[3] KERR ENGINE computes the nonlinear function the linear one cannot (parity)\n");
  double cl4 = classifyAcc(GK, 3);
  std::printf("    g=%.0f parity acc=%.1f%%  (vs linear %.1f%%)\n", GK, cl4, cl0);
  ++total; pass += report("Kerr did not enable the nonlinear classification", cl4 > 85.0 && cl4 - cl0 > 25.0);

  std::printf("\n[4] KERR ENGINE INTERPOLATES a continuous nonlinear map (held-out inputs)\n");
  std::printf("    target y = (2x1-1)(2x2-1), pure degree-2 (no linear component)\n");
  double r0 = regressR2(0.0, 2, cprod, 11), r1 = regressR2(1.0, 2, cprod, 11);
  std::printf("    g=0 R2=%.2f   g=1 R2=%.2f\n", r0, r1);
  ++total; pass += report("Kerr did not interpolate the nonlinear map the linear one cannot", r1 > 0.85 && r0 < 0.20);

  std::printf("\n  RESULT : %d / %d verified\n", pass, total);
  return pass == total ? 0 : 1;
}
