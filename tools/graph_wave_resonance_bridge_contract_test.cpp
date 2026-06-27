// ----------------------------------------------------------------------------
// RESONANCE BRIDGE CONTRACT.
//
// This verifies a substrate-level geometry change:
//   H -> H + B[psi]
//
// A long-range bridge is not a readout shortcut and not an arithmetic edge. It
// materializes only from observed phase coherence between two distant ports. The
// bridge phase is chosen from the observed relative phase so the locked witness
// state is not artificially pumped. A later impulse then moves through the
// changed substrate using the same unitary local-flow physics.
// ----------------------------------------------------------------------------
#define NOMINMAX
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

struct PhaseLock {
  bool stable = false;
  double score = 0.0;
  double phase = 0.0;
  double aligned_current = 0.0;
};

PhaseLock observePhaseLock(bool coherent) {
  constexpr int kFrames = 96;
  std::mt19937 rng(71);
  std::uniform_real_distribution<double> rnd(-gw::kPi, gw::kPi);
  cd acc(0.0, 0.0);
  cd lastA(0.0, 0.0), lastB(0.0, 0.0);
  for (int t = 0; t < kFrames; ++t) {
    double common = 0.071 * t + 0.003 * std::sin(0.13 * t);
    double rel = coherent ? 0.62 + 0.006 * std::sin(0.19 * t) : rnd(rng);
    lastA = std::polar(0.64 + 0.02 * std::sin(0.11 * t), common);
    lastB = std::polar(0.59 + 0.02 * std::cos(0.07 * t), common + rel);
    cd z = std::conj(lastA) * lastB;
    acc += z / (std::abs(z) + 1e-300);
  }
  cd mean = acc / (double)kFrames;
  PhaseLock lock;
  lock.score = std::abs(mean);
  lock.stable = lock.score > 0.985;
  lock.phase = -std::arg(mean);
  lock.aligned_current = gw::bondCurrent(lastA, lastB, 1.0, lock.phase);
  return lock;
}

std::vector<gw::SparseBond> baseLine(int n) {
  std::vector<gw::SparseBond> bonds;
  bonds.reserve(n - 1);
  for (int i = 0; i + 1 < n; ++i) bonds.push_back({i, i + 1, 1.0, 0.0});
  return bonds;
}

Vec delta(int n, int at) {
  Vec psi(n, cd(0.0, 0.0));
  psi[at] = cd(1.0, 0.0);
  return psi;
}

double regionPower(const Vec& psi, int center) {
  double s = 0.0;
  for (int i = std::max(0, center - 1); i <= std::min((int)psi.size() - 1, center + 1); ++i) {
    s += std::norm(psi[i]);
  }
  return s;
}

bool report(const char* name, bool ok) {
  std::printf("   => %s\n", ok ? "PASS" : "FAIL");
  if (!ok) std::printf("   !! %s\n", name);
  return ok;
}
}  // namespace

int main() {
  constexpr int kN = 96;
  constexpr int kPortA = 5;
  constexpr int kPortB = 90;
  constexpr double kDt = 0.28;
  constexpr int kSteps = 7;
  constexpr double kKappa = 1.45;

  PhaseLock coherent = observePhaseLock(true);
  PhaseLock random = observePhaseLock(false);

  std::vector<gw::SparseBond> base = baseLine(kN);
  std::vector<gw::SparseBond> bridged = base;
  if (coherent.stable) bridged.push_back({kPortA, kPortB, kKappa, coherent.phase});

  Vec psi0 = delta(kN, kPortA);
  Vec noBridge = gw::edgeLocalKerrFlow(psi0, base, kDt, 0.0, kSteps);
  Vec withBridge = gw::edgeLocalKerrFlow(psi0, bridged, kDt, 0.0, kSteps);
  Vec nonlinearBridge = gw::edgeLocalKerrFlow(psi0, bridged, kDt, 3.5, kSteps);

  double pNo = regionPower(noBridge, kPortB);
  double pBridge = regionPower(withBridge, kPortB);
  double pKerBridge = regionPower(nonlinearBridge, kPortB);
  double normNo = std::abs(gw::power(noBridge) - 1.0);
  double normBridge = std::abs(gw::power(withBridge) - 1.0);
  double normKerBridge = std::abs(gw::power(nonlinearBridge) - 1.0);
  double bridgeCurrent = std::abs(gw::bondCurrent(withBridge[kPortA], withBridge[kPortB],
                                                 1.0, coherent.phase));

  std::printf("=====================================================================\n");
  std::printf("  RESONANCE BRIDGE CONTRACT\n");
  std::printf("=====================================================================\n");
  std::printf("  phase-lock coherent score=%.6f phase=%.6f aligned_current=%.3e\n",
              coherent.score, coherent.phase, coherent.aligned_current);
  std::printf("  phase-lock random   score=%.6f stable=%d\n", random.score, random.stable ? 1 : 0);
  std::printf("  target power: no_bridge=%.3e bridge=%.3e nonlinear_bridge=%.3e\n",
              pNo, pBridge, pKerBridge);
  std::printf("  norm drift: no=%.3e bridge=%.3e nonlinear_bridge=%.3e bridge_current=%.3e\n",
              normNo, normBridge, normKerBridge, bridgeCurrent);

  int pass = 0, total = 0;
  total++; pass += report("coherent phase observation materializes bridge", coherent.stable);
  total++; pass += report("random phase observation does not materialize bridge", !random.stable);
  total++; pass += report("bridge is not an artificial phase pump", std::abs(coherent.aligned_current) < 0.02);
  total++; pass += report("changed substrate increases distant reach",
                          pBridge > 25.0 * std::max(pNo, 1e-300) && pBridge > 1e-4);
  total++; pass += report("bridge remains unitary",
                          normNo < 1e-12 && normBridge < 1e-12 && normKerBridge < 1e-12);
  total++; pass += report("nonlinear field still reaches through bridge",
                          pKerBridge > 10.0 * std::max(pNo, 1e-300) && pKerBridge > 1e-4);
  total++; pass += report("bridge carries physical current after impulse", bridgeCurrent > 1e-4);
  std::printf("\n  RESULT : %d / %d verified\n", pass, total);
  return pass == total ? 0 : 1;
}
