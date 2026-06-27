// graph_wave_phased_bond_kerr_flow_contract_test
// -----------------------------------------------------------------------------
// Minimal contract for phased matrix-free Kerr bonds.
//
// This guards the smallest safe integration step:
//   SparseBond phase=0 must preserve the old matrix-free flow,
//   phase!=0 may steer phase current, and Kerr densification must remain intact.
// -----------------------------------------------------------------------------
#include "graph_wave_substrate.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>

namespace {
using gw::cd;
using gw::Vec;

double maxDiff(const Vec& a, const Vec& b) {
  double m = 0.0;
  for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
  return m;
}

void rotateOld(cd& a, cd& b, double theta) {
  const double c = std::cos(theta), s = std::sin(theta);
  const cd oldA = a, oldB = b;
  a = c * oldA + cd(0.0, -s) * oldB;
  b = c * oldB + cd(0.0, -s) * oldA;
}

Vec oldFlow(Vec psi, const std::vector<gw::SparseBond>& bonds, double dt, double g, int steps) {
  double maxW = 0.0;
  for (const auto& e : bonds) maxW = std::max(maxW, std::abs(e.w));
  const double inv = maxW > 1e-300 ? 1.0 / maxW : 0.0;
  for (int s = 0; s < steps; ++s) {
    for (const auto& e : bonds) rotateOld(psi[e.a], psi[e.b], 0.5 * dt * e.w * inv);
    if (g != 0.0) {
      for (auto& v : psi) v *= std::exp(cd(0.0, -g * std::norm(v) * dt));
    }
    for (auto it = bonds.rbegin(); it != bonds.rend(); ++it) {
      rotateOld(psi[it->a], psi[it->b], 0.5 * dt * it->w * inv);
    }
  }
  return psi;
}

std::vector<gw::SparseBond> ringBonds(int n, double phase) {
  std::vector<gw::SparseBond> bonds;
  bonds.reserve(n);
  for (int i = 0; i < n; ++i) bonds.push_back({i, (i + 1) % n, 1.0, phase});
  return bonds;
}

Vec broadPacket(int n) {
  Vec psi(n, cd(0, 0));
  const double c = n / 2.0, sig = 14.0;
  for (int x = 0; x < n; ++x) {
    const double d = x - c;
    psi[x] = std::exp(cd(-d * d / (2.0 * sig * sig), 0.0));
  }
  gw::normalizeInPlace(psi);
  return psi;
}

bool report(const char* name, bool ok) {
  std::printf("   => %s\n", ok ? "PASS" : "FAIL");
  if (!ok) std::printf("   !! %s\n", name);
  return ok;
}

}  // namespace

int main() {
  std::printf("=====================================================================\n");
  std::printf("  PHASED BOND KERR FLOW CONTRACT\n");
  std::printf("=====================================================================\n");
  int pass = 0, total = 0;

  std::printf("\n[1] phase=0 preserves the old matrix-free flow\n");
  {
    constexpr int N = 80;
    std::mt19937 rng(5);
    std::normal_distribution<double> gauss(0.0, 1.0);
    Vec seed(N);
    for (auto& z : seed) z = cd(gauss(rng), gauss(rng));
    gw::normalizeInPlace(seed);
    auto bonds = ringBonds(N, 0.0);
    Vec old = oldFlow(seed, bonds, 0.07, 9.0, 300);
    Vec now = gw::edgeLocalKerrFlow(seed, bonds, 0.07, 9.0, 300);
    double err = maxDiff(old, now);
    double drift = std::max(std::abs(gw::power(old) - 1.0), std::abs(gw::power(now) - 1.0));
    std::printf("    max |old-new|=%.3e  norm drift=%.3e\n", err, drift);
    ++total; pass += report("phase=0 changed existing flow", err < 1e-12 && drift < 1e-12);
  }

  std::printf("\n[2] Kerr still densifies under the phased-bond path\n");
  {
    constexpr int N = 160;
    auto bonds = ringBonds(N, 0.0);
    Vec seed = broadPacket(N);
    Vec lin = gw::edgeLocalKerrFlow(seed, bonds, 0.10, 0.0, 500);
    Vec ker = gw::edgeLocalKerrFlow(seed, bonds, 0.10, 60.0, 500);
    double prLin = gw::participationRatio(lin);
    double prKer = gw::participationRatio(ker);
    double drift = std::max(std::abs(gw::power(lin) - 1.0), std::abs(gw::power(ker) - 1.0));
    std::printf("    PR linear=%.2f  nonlinear=%.2f  compression=%.2fx  norm drift=%.3e\n",
                prLin, prKer, prLin / prKer, drift);
    ++total; pass += report("Kerr densification broke after adding bond phase", prKer < 0.65 * prLin && drift < 1e-11);
  }

  std::printf("\n[3] bond phase is visible to the physical current detector\n");
  {
    const cd a = cd(1.0 / std::sqrt(2.0), 0.0);
    const cd b = cd(1.0 / std::sqrt(2.0), 0.0);
    double j0 = gw::bondCurrent(a, b, 1.0, 0.0);
    double jp = gw::bondCurrent(a, b, 1.0, 0.40);
    double jm = gw::bondCurrent(a, b, 1.0, -0.40);
    std::printf("    j(0)=%.3e  j(+phase)=%.6f  j(-phase)=%.6f\n", j0, jp, jm);
    ++total; pass += report("bond phase does not steer phase current", std::abs(j0) < 1e-14 && jp > 0.3 && jm < -0.3);
  }

  std::printf("\n[4] opposite bond phases produce different unitary evolution, without norm loss\n");
  {
    constexpr int N = 64;
    Vec seed(N, cd(0, 0));
    seed[N / 2] = cd(1, 0);
    Vec plus = gw::edgeLocalKerrFlow(seed, ringBonds(N, 0.05), 0.11, 20.0, 240);
    Vec minus = gw::edgeLocalKerrFlow(seed, ringBonds(N, -0.05), 0.11, 20.0, 240);
    double diff = maxDiff(plus, minus);
    double drift = std::max(std::abs(gw::power(plus) - 1.0), std::abs(gw::power(minus) - 1.0));
    std::printf("    max |plus-minus|=%.3e  norm drift=%.3e\n", diff, drift);
    ++total; pass += report("phased evolution is inert or non-unitary", diff > 1e-3 && drift < 1e-11);
  }

  std::printf("\n  RESULT : %d / %d verified\n", pass, total);
  return pass == total ? 0 : 1;
}
