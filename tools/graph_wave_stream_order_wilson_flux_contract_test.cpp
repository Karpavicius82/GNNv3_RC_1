// graph_wave_stream_order_wilson_flux_contract_test
// -----------------------------------------------------------------------------
// Temporal order becomes physical only as gauge-invariant loop flux.
//
// A bare edge phase can be gauge. A closed triangle has Wilson flux:
//   Phi_ABC = phase(A->B) + phase(B->C) + phase(C->A)
//
// This contract guards the next nonlinear/streaming step: do not treat pair order
// as a physical signal unless it survives as loop flux.
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

struct Edge {
  double w = 0.0;
  double phase = 0.0;  // stored low -> high
};

double wrapPhase(double x) {
  while (x > gw::kPi) x -= 2.0 * gw::kPi;
  while (x < -gw::kPi) x += 2.0 * gw::kPi;
  return x;
}

struct Triangle {
  Edge ab, ac, bc;

  void touchPair(int older, int newer) {
    const int lo = std::min(older, newer);
    const int hi = std::max(older, newer);
    const double sign = (older == lo && newer == hi) ? 1.0 : -1.0;
    Edge* e = nullptr;
    if (lo == 0 && hi == 1) e = &ab;
    if (lo == 0 && hi == 2) e = &ac;
    if (lo == 1 && hi == 2) e = &bc;
    if (!e) return;
    e->w += 1.0;
    e->phase = std::clamp(wrapPhase(e->phase + sign * 0.035), -1.2, 1.2);
  }

  void burst(const std::vector<int>& order) {
    for (int i = 0; i < (int)order.size(); ++i) {
      for (int j = 0; j < i; ++j) touchPair(order[j], order[i]);
    }
  }

  double fluxABC() const {
    return wrapPhase(ab.phase + bc.phase - ac.phase);  // A->B + B->C + C->A
  }

  std::vector<gw::SparseBond> bonds() const {
    return {
      {0, 1, std::max(1.0, ab.w), ab.phase},
      {0, 2, std::max(1.0, ac.w), ac.phase},
      {1, 2, std::max(1.0, bc.w), bc.phase}
    };
  }
};

double loopCurrent(const Vec& psi, const std::vector<gw::SparseBond>& bonds) {
  double j01 = 0.0, j12 = 0.0, j20 = 0.0;
  for (const auto& e : bonds) {
    const double j = gw::bondCurrent(psi[e.a], psi[e.b], e.w, e.phase);
    if (e.a == 0 && e.b == 1) j01 = j;
    if (e.a == 1 && e.b == 2) j12 = j;
    if (e.a == 0 && e.b == 2) j20 = -j;  // stored 0->2, loop wants 2->0
  }
  return (j01 + j12 + j20) / 3.0;
}

struct Result {
  double flux = 0.0;
  double current = 0.0;
  double drift = 0.0;
};

Result runCase(int mode) {
  Triangle tri;
  std::mt19937 rng(17);
  for (int n = 0; n < 400; ++n) {
    if (mode > 0) tri.burst({0, 1, 2});
    else if (mode < 0) tri.burst({2, 1, 0});
    else {
      std::vector<int> o = {0, 1, 2};
      std::shuffle(o.begin(), o.end(), rng);
      tri.burst(o);
    }
  }

  Vec psi(3, cd(1.0, 0.0));
  gw::normalizeInPlace(psi);
  const double n0 = gw::power(psi);
  const auto bonds = tri.bonds();
  double drift = 0.0;
  for (int s = 0; s < 300; ++s) {
    psi = gw::edgeLocalKerrFlow(std::move(psi), bonds, 0.05, 2.0, 1);
    drift = std::max(drift, std::abs(gw::power(psi) - n0));
  }

  return {tri.fluxABC(), loopCurrent(psi, bonds), drift};
}

bool report(const char* name, bool ok) {
  std::printf("   => %s\n", ok ? "PASS" : "FAIL");
  if (!ok) std::printf("   !! %s\n", name);
  return ok;
}

}  // namespace

int main() {
  std::printf("=====================================================================\n");
  std::printf("  STREAM ORDER -> WILSON FLUX CONTRACT\n");
  std::printf("=====================================================================\n");

  Result plus = runCase(+1);
  Result minus = runCase(-1);
  Result random = runCase(0);

  std::printf("  ORDER+ flux=% .6f current=% .6f drift=%.3e\n", plus.flux, plus.current, plus.drift);
  std::printf("  ORDER- flux=% .6f current=% .6f drift=%.3e\n", minus.flux, minus.current, minus.drift);
  std::printf("  RANDOM flux=% .6f current=% .6f drift=%.3e\n", random.flux, random.current, random.drift);

  int pass = 0, total = 0;
  ++total; pass += report("ordered and reverse streams did not create opposite Wilson flux",
                          plus.flux > 0.2 && minus.flux < -0.2);
  ++total; pass += report("random order produced ordered-strength loop flux",
                          std::abs(random.flux) < 0.6 * std::abs(plus.flux));
  ++total; pass += report("loop current did not change sign with flux orientation",
                          plus.current * minus.current < 0.0);
  ++total; pass += report("phased loop evolution lost norm",
                          std::max({plus.drift, minus.drift, random.drift}) < 1e-12);

  std::printf("\n  RESULT : %d / %d verified\n", pass, total);
  return pass == total ? 0 : 1;
}
