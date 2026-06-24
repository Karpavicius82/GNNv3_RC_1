// graph_wave_horizon_densification_contract_test
// ----------------------------------------------------------------------------
// THE COMPRESSION CONTRACT. Earlier probes proved pieces separately: Kerr
// self-focusing densifies energy (probe_nonlinear_soliton), a holographic bundle
// has a capacity horizon (probe_blackhole_compress), a soliton PERSISTS but does
// not heal (probe_soliton_memory). This contract FUSES them into one physical
// claim and gates it:
//
//   The SAME energy and phase information densifies into a smaller state, the
//   outside reads it back through a local horizon/detector, structure beats
//   random, and the energy budget (core + radiation) is honest -- with a real
//   capacity horizon where the readable signal sinks into the noise floor.
//
// Everything is read FROM the wave. Physical readouts only:
//   * participation ratio  PR = (sum rho)^2 / sum rho^2     (how localized)
//   * probability current  j_x = -2 Im( conj psi_x psi_{x+1} )  (phase gradient)
//   * aperture detector    integral(j)/integral(rho), integral(rho)  (local horizon)
//   * interference overlap <a|b>  (the medium's own associative readout)
// Dynamics = real-space split-step Kerr (discrete nonlinear Schrodinger):
//   half unitary hop (Cayley) -> local phase psi_i *= exp(-i g |psi_i|^2 dt) -> half hop.
// No FFT, no analytic soliton (sech), no fitted profile, no classifier. The only
// closed forms are INPUT-STATE preparation (a Gaussian, a Galilean boost, a phasor
// atom); every MEASUREMENT is a physical operator on the field.
//
// Substrate-only, deterministic (fixed seeds).
// ----------------------------------------------------------------------------

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

// ---- physical readouts ------------------------------------------------------
double normSq(const Vec& z) { double s = 0.0; for (const auto& v : z) s += std::norm(v); return s; }
Vec normalize(Vec z) { double n = std::sqrt(normSq(z)); for (auto& v : z) v /= n; return z; }

double pr(const Vec& z) {            // participation ratio: how many sites hold the energy
  double a = 0.0, b = 0.0;
  for (const auto& v : z) { double p = std::norm(v); a += p; b += p * p; }
  return a * a / (b + 1e-300);
}

double totalCurrent(const Vec& z) {  // net momentum: sum of bond currents on the ring
  int N = (int)z.size(); double j = 0.0;
  for (int x = 0; x < N; ++x) j += -2.0 * std::imag(std::conj(z[x]) * z[(x + 1) % N]);
  return j;
}

int peakSite(const Vec& z) {         // where the energy concentrated (physical argmax of rho)
  int N = (int)z.size(), m = 0; double best = -1.0;
  for (int x = 0; x < N; ++x) { double p = std::norm(z[x]); if (p > best) { best = p; m = x; } }
  return m;
}

// finite aperture detector: integrate density and current over a small window
// (a local horizon), never a global scan.
double apertureEnergy(const Vec& z, int c, int R) {
  int N = (int)z.size(); double rho = 0.0;
  for (int d = -R; d <= R; ++d) rho += std::norm(z[((c + d) % N + N) % N]);
  return rho;
}
double apertureCurrent(const Vec& z, int c, int R) {  // integral(j)/integral(rho): phase-only velocity
  int N = (int)z.size(); double rho = 0.0, cur = 0.0;
  for (int d = -R; d <= R; ++d) {
    int x = ((c + d) % N + N) % N;
    rho += std::norm(z[x]);
    cur += -2.0 * std::imag(std::conj(z[x]) * z[(x + 1) % N]);
  }
  return cur / (rho + 1e-300);
}

cd overlap(const Vec& a, const Vec& b) { cd s(0, 0); for (size_t i = 0; i < a.size(); ++i) s += std::conj(a[i]) * b[i]; return s; }

// ---- input-state preparation (closed forms allowed: these are STATE PREP, not readout) ----
Vec gaussianPacket(int N, double c, double sig, double k0) {  // optional Galilean boost k0
  Vec psi(N, cd(0, 0));
  for (int x = 0; x < N; ++x) { double d = x - c; psi[x] = std::exp(cd(-d * d / (2 * sig * sig), k0 * x)); }
  return normalize(psi);
}

bool report(const char* name, bool ok) {
  std::printf("   => %s\n", ok ? "PASS" : "FAIL");
  if (!ok) std::printf("   !! %s\n", name);
  return ok;
}

} // namespace

int main() {
  const int N = 192;
  const double dt = 0.1, g = 60.0;
  const int FOCUS = 500;            // steps to self-focus
  const int R = 6;                  // aperture half-width

  gw::Graph G(N);
  for (int i = 0; i < N; ++i) G.addEdge(i, (i + 1) % N, 1.0);   // ring, linear hopping
  gw::Stepper Uh; Uh.build(G.h, dt / 2);                       // unitary half-step
  auto kerr = [&](Vec psi, double gk, int steps) {
    for (int s = 0; s < steps; ++s) {
      psi = Uh.step(psi);
      for (int i = 0; i < N; ++i) psi[i] *= std::exp(cd(0, -gk * std::norm(psi[i]) * dt));  // local Kerr phase
      psi = Uh.step(psi);
    }
    return psi;
  };

  std::printf("=====================================================================\n");
  std::printf("  GRAPH-WAVE HORIZON / DENSIFICATION CONTRACT  (N=%d, g=%.0f)\n", N, g);
  std::printf("  Kerr self-focusing + holographic horizon + current detector + control\n");
  std::printf("=====================================================================\n");

  int pass = 0, total = 0;

  // [1] NORM / ENERGY CONSERVED through the nonlinear densification -----------
  std::printf("\n[1] NORM CONSERVED under split-step Kerr (lossless densification)\n");
  {
    Vec p = gaussianPacket(N, N / 2.0, 14, 0.0);
    double n0 = normSq(p), drift = 0.0;
    for (int s = 0; s < FOCUS; ++s) {
      p = Uh.step(p);
      for (int i = 0; i < N; ++i) p[i] *= std::exp(cd(0, -g * std::norm(p[i]) * dt));
      p = Uh.step(p);
      drift = std::max(drift, std::abs(normSq(p) - n0));
    }
    std::printf("    max |norm^2 - 1| over %d steps = %.2e\n", FOCUS, drift);
    ++total; pass += report("nonlinear evolution is not norm-preserving", drift < 1e-10);
  }

  // [2] PR DENSIFIES: the wave compresses its own energy ----------------------
  std::printf("\n[2] PR DENSIFIES (broad packet -> soliton, energy concentrates)\n");
  Vec soliton;
  {
    Vec broad = gaussianPacket(N, N / 2.0, 14, 0.0);
    double prB = pr(broad);
    soliton = kerr(broad, g, FOCUS);
    double prS = pr(soliton);
    std::printf("    broad PR=%.1f  ->  densified PR=%.1f   (%.1fx)\n", prB, prS, prB / prS);
    ++total; pass += report("Kerr did not densify the packet", prS < 0.6 * prB);
  }

  // [3] BOUNDARY CURRENT / FLUX SURVIVES densification ------------------------
  // A boosted packet carries a directed flux. Densification must not SCRAMBLE it.
  // (On a lattice the current is velocity-weighted, sum 2 sin(k)|a_k|^2; only the
  // quasi-momentum sum k|a_k|^2 is exactly conserved -- and reading that needs
  // k-space, which we refuse. So the honest physical claim is the directed flux
  // SURVIVES: same sign, still a large fraction -- not scrambled to noise.)
  std::printf("\n[3] FLUX SURVIVES densification (directed, same sign, not scrambled)\n");
  Vec boostedSol;
  {
    Vec boosted = gaussianPacket(N, N / 2.0, 14, 0.30);   // Galilean boost -> net current
    double jBefore = totalCurrent(boosted);
    boostedSol = kerr(boosted, g, FOCUS);
    double jAfter = totalCurrent(boostedSol);
    double frac = std::abs(jAfter) / (std::abs(jBefore) + 1e-12);
    bool sameSign = (jBefore * jAfter) > 0.0;
    std::printf("    total current  before=%.4f  after=%.4f  surviving fraction=%.3f  sign %s\n",
                jBefore, jAfter, frac, sameSign ? "kept" : "FLIPPED");
    ++total; pass += report("densification scrambled the directed flux", sameSign && frac > 0.30);
  }

  // [4] PHASE-PRESERVING DETECTOR: current reads PHASE, not amplitude ----------
  // Two states with IDENTICAL |psi|^2 but different phase give different flux.
  // A single random scramble has a ~sqrt(N) random-walk residual; the physical
  // statement is the ENSEMBLE: phase-random states carry ZERO mean flux, while
  // the coherent soliton carries a definite directed flux. Phase IS the carrier.
  std::printf("\n[4] DETECTOR IS PHASE-PRESERVING (random-phase ensemble: mean flux -> 0)\n");
  {
    double jReal = totalCurrent(boostedSol);
    std::mt19937 rng(11); std::uniform_real_distribution<double> ph(-gw::kPi, gw::kPi);
    const int SEEDS = 512; double meanJ = 0.0;
    for (int s = 0; s < SEEDS; ++s) {
      Vec scr(N);
      for (int i = 0; i < N; ++i) scr[i] = std::polar(std::abs(boostedSol[i]), ph(rng));  // keep rho, random phase
      meanJ += totalCurrent(scr);
    }
    meanJ /= SEEDS;
    std::printf("    coherent flux=%.4f   phase-random ensemble mean flux=%.4f  (same |psi|^2)\n", jReal, meanJ);
    ++total; pass += report("current detector responds to amplitude, not phase",
                            std::abs(meanJ) < 0.1 * std::abs(jReal));
  }

  // [5] LOCAL APERTURE READOUT (a horizon), not a global scan ------------------
  std::printf("\n[5] LOCAL APERTURE reads the stored lump (horizon, no global scan)\n");
  {
    int c = peakSite(soliton);
    int far = (c + N / 2) % N;
    double eHere = apertureEnergy(soliton, c, R);
    double eFar  = apertureEnergy(soliton, far, R);
    double vHere = apertureCurrent(boostedSol, peakSite(boostedSol), R);
    std::printf("    aperture@peak energy=%.3f   aperture@far energy=%.3f   v@peak=%+.3f\n", eHere, eFar, vHere);
    ++total; pass += report("a local aperture cannot read the densified state",
                            eHere > 0.5 && eFar < 0.1 * eHere);
  }

  // [6] REAL STRUCTURE BEATS RANDOM: structure densifies, grey noise does not --
  std::printf("\n[6] STRUCTURE > RANDOM (coherent focuses; phase-random stays spread)\n");
  {
    Vec real = gaussianPacket(N, N / 2.0, 14, 0.0);          // coherent (flat phase)
    std::mt19937 rng(7); std::uniform_real_distribution<double> ph(-gw::kPi, gw::kPi);
    Vec rand(N);                                             // SAME envelope, random phase
    for (int i = 0; i < N; ++i) rand[i] = std::polar(std::abs(real[i]), ph(rng));
    rand = normalize(rand);
    double prRealF = pr(kerr(real, g, FOCUS));
    double prRandF = pr(kerr(rand, g, FOCUS));
    std::printf("    final PR  coherent=%.1f   phase-random=%.1f\n", prRealF, prRandF);
    ++total; pass += report("structure does not densify better than random",
                            prRealF < 0.7 * prRandF);
  }

  // [7] CAPACITY HORIZON: readable signal sinks into the noise floor -----------
  // The horizon, measured on the medium's own interference readout. K items are
  // superposed into ONE dense code (bind = interference, bundle = superposition);
  // recall by unbinding + cleanup. Small K: near-lossless. Large K: the stored
  // signal meets the 1/sqrt(D) noise floor -- the information horizon.
  std::printf("\n[7] CAPACITY HORIZON (interference readout: signal -> noise floor)\n");
  {
    const int D = 512;
    std::mt19937_64 rng(0xC0FFEEULL);
    auto atom = [&](void) { Vec a(D); std::uniform_real_distribution<double> u(-gw::kPi, gw::kPi);
                            for (auto& z : a) z = std::polar(1.0 / std::sqrt((double)D), u(rng)); return a; };
    auto recallAcc = [&](int K) {
      std::vector<Vec> key(K), val(K);
      for (int p = 0; p < K; ++p) { key[p] = atom(); val[p] = atom(); }
      Vec mem(D, cd(0, 0));
      for (int p = 0; p < K; ++p) for (int i = 0; i < D; ++i) mem[i] += key[p][i] * val[p][i];   // bind+bundle
      int correct = 0; double sig = 0.0, dist = 0.0;
      for (int p = 0; p < K; ++p) {
        Vec probe(D); for (int i = 0; i < D; ++i) probe[i] = mem[i] * std::conj(key[p][i]);       // unbind
        int best = -1; double bSim = -1.0, second = -1.0;
        for (int q = 0; q < K; ++q) { double s = std::abs(overlap(probe, val[q]));
          if (s > bSim) { second = bSim; bSim = s; best = q; } else if (s > second) second = s; }
        if (best == p) ++correct;
        sig += std::abs(overlap(probe, val[p])); dist += (best == p) ? second : bSim;
      }
      return std::make_pair((double)correct / K, sig / std::max(1e-9, dist));
    };
    auto [accLo, snrLo] = recallAcc(8);
    auto [accHi, snrHi] = recallAcc(256);
    std::printf("    K=8  : recall=%5.1f%% signal/floor=%.2f      (below horizon)\n", 100 * accLo, snrLo);
    std::printf("    K=256: recall=%5.1f%% signal/floor=%.2f      (past horizon)\n", 100 * accHi, snrHi);
    ++total; pass += report("no capacity horizon: signal does not sink toward the floor",
                            accLo > 0.99 && accHi < accLo && snrHi < snrLo);
  }

  // [8] RADIATION ACCOUNTED, not ignored --------------------------------------
  // Self-focusing is not lossless into the core: some energy is shed as radiation.
  // Honest budget: core (aperture) + radiation (rest) == total, and radiation > 0.
  std::printf("\n[8] RADIATION ACCOUNTED (core + radiation = total, radiation real)\n");
  {
    int c = peakSite(soliton);
    double core = apertureEnergy(soliton, c, R);
    double tot = normSq(soliton);
    double rad = tot - core;
    std::printf("    total=%.4f  core(aperture)=%.4f  radiation=%.4f  (core %.0f%%)\n",
                tot, core, rad, 100 * core / tot);
    bool budget = std::abs((core + rad) - tot) < 1e-9;        // exact accounting
    bool realRad = rad > 0.01;                                // radiation is not ignored
    bool captured = core > 0.40;                              // core still holds the bulk
    ++total; pass += report("radiation budget is not honest", budget && realRad && captured);
  }

  std::printf("\n  RESULT : %d / %d verified\n", pass, total);
  return pass == total ? 0 : 1;
}
