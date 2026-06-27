// ----------------------------------------------------------------------------
// DUAL LOCAL FLOW CONTRACT.
//
// The streaming system carries a linear control field and a Kerr field through
// the same physical substrate. This contract verifies that the optimized
// compiled carrier / dual-flow path is exactly the old local bond physics:
// same phase transport, same Kerr pressure, same norm, same detector visits.
// ----------------------------------------------------------------------------
#define NOMINMAX
#include "graph_wave_substrate.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {
using gw::cd;
using gw::Vec;

Vec seed(int n, double p) {
  Vec psi(n, cd(0, 0));
  for (int i = 0; i < n; ++i) psi[i] = cd(std::cos(p * i), std::sin((p + 0.17) * i));
  gw::normalizeInPlace(psi);
  return psi;
}

std::vector<gw::SparseBond> makeBonds(int n) {
  std::vector<gw::SparseBond> out;
  for (int i = 0; i < n; ++i) {
    for (int d = 1; d <= 4; ++d) {
      int j = (i + d) % n;
      if (i < j) out.push_back({i, j, 0.7 + 0.13 * d, 0.19 * ((i + 3 * d) % 11 - 5)});
    }
  }
  return out;
}

Vec legacyFlow(Vec psi, const std::vector<gw::SparseBond>& bs,
               double dt, double g, int steps, gw::LocalFlowStats* stats) {
  double max_w = 0.0;
  for (const auto& e : bs) max_w = std::max(max_w, std::abs(e.w));
  const double inv = max_w > 1e-300 ? 1.0 / max_w : 0.0;
  for (int s = 0; s < steps; ++s) {
    for (const auto& e : bs) {
      const double w = e.w * inv;
      if (stats) {
        const double j = gw::bondCurrentU(psi[e.a], psi[e.b], w, e.phase_u);
        const double rho = std::norm(psi[e.a]) + std::norm(psi[e.b]) + 1e-300;
        stats->current_abs += std::abs(j);
        stats->max_bond_speed = std::max(stats->max_bond_speed, std::abs(j) / rho);
        stats->bond_visits++;
      }
      gw::rotateBondU(psi[e.a], psi[e.b], 0.5 * dt * w, e.phase_u);
    }
    if (g != 0.0) for (auto& v : psi) v *= std::exp(cd(0, -g * std::norm(v) * dt));
    for (auto it = bs.rbegin(); it != bs.rend(); ++it) {
      const auto& e = *it;
      const double w = e.w * inv;
      if (stats) {
        const double j = gw::bondCurrentU(psi[e.a], psi[e.b], w, e.phase_u);
        const double rho = std::norm(psi[e.a]) + std::norm(psi[e.b]) + 1e-300;
        stats->current_abs += std::abs(j);
        stats->max_bond_speed = std::max(stats->max_bond_speed, std::abs(j) / rho);
        stats->bond_visits++;
      }
      gw::rotateBondU(psi[e.a], psi[e.b], 0.5 * dt * w, e.phase_u);
    }
  }
  return psi;
}

double maxDiff(const Vec& a, const Vec& b) {
  double d = 0.0;
  for (int i = 0; i < (int)a.size(); ++i) d = std::max(d, std::abs(a[i] - b[i]));
  return d;
}

bool report(const char* name, bool ok) {
  std::printf("   => %s\n", ok ? "PASS" : "FAIL");
  if (!ok) std::printf("   !! %s\n", name);
  return ok;
}
}  // namespace

int main() {
  constexpr int n = 80;
  constexpr double dt = 0.3;
  constexpr double g = 7.0;
  constexpr int steps = 3;
  auto bs = makeBonds(n);
  Vec lin0 = seed(n, 0.31);
  Vec ker0 = seed(n, 0.43);

  gw::LocalFlowStats legacyStats, compiledStats, pairStats;
  Vec legacyLin = legacyFlow(lin0, bs, dt, 0.0, steps, nullptr);
  Vec legacyKer = legacyFlow(ker0, bs, dt, g, steps, &legacyStats);
  Vec compiledKer = gw::edgeLocalKerrFlow(ker0, bs, dt, g, steps, &compiledStats);
  Vec pairLin = lin0;
  Vec pairKer = ker0;
  gw::edgeLocalKerrFlowPair(pairLin, pairKer, bs, dt, g, steps, &pairStats);

  double dCompiled = maxDiff(legacyKer, compiledKer);
  double dLin = maxDiff(legacyLin, pairLin);
  double dKer = maxDiff(legacyKer, pairKer);
  double normLin = std::abs(gw::power(pairLin) - 1.0);
  double normKer = std::abs(gw::power(pairKer) - 1.0);

  std::printf("=====================================================================\n");
  std::printf("  DUAL LOCAL FLOW CONTRACT\n");
  std::printf("=====================================================================\n");
  std::printf("  nodes=%d bonds=%d steps=%d\n", n, (int)bs.size(), steps);
  std::printf("  compiled diff=%.3e  pair lin diff=%.3e  pair ker diff=%.3e\n",
              dCompiled, dLin, dKer);
  std::printf("  norm drift lin=%.3e ker=%.3e  visits legacy=%lld pair=%lld\n",
              normLin, normKer, legacyStats.bond_visits, pairStats.bond_visits);

  int pass = 0, total = 0;
  total++; pass += report("compiled carrier matches legacy Kerr flow", dCompiled < 1e-12);
  total++; pass += report("dual-flow preserves linear control field", dLin < 1e-12);
  total++; pass += report("dual-flow preserves nonlinear Kerr field", dKer < 1e-12);
  total++; pass += report("dual-flow preserves norm", normLin < 1e-12 && normKer < 1e-12);
  total++; pass += report("dual-flow detector sees same Kerr visits",
                          legacyStats.bond_visits == pairStats.bond_visits);
  std::printf("\n  RESULT : %d / %d verified\n", pass, total);
  return pass == total ? 0 : 1;
}
