// A4: a primitive that DOES NOT SINK -- exact, finite-N, weights-free.
// Diamond/rhombus chain (Aharonov-Bohm cage). Each rhombus carries a gauge flux.
// At flux=pi the two paths between consecutive hubs interfere EXACTLY
// destructively => a particle injected at hub 0 is confined to a finite cage and
// can NEVER reach the far hub. This is an algebraic identity true at ANY length L
// and ALL time -- not an asymptotic exponent, not a fit, not an extrapolation.
// Contrast A3: there beta(golden) drifted with box size and never converged.
// Here: reach(flux=pi) ~ machine zero, INVARIANT of L. A ship that does not sink.
#include "../tools/graph_wave_substrate.hpp"
#include <cstdio>
#include <complex>
using namespace gw;

// build the diamond chain of L rhombi. hubs 0..L ; per rhombus i two legs U,D.
static CMat diamond(int L, double flux) {
 int nHub = L + 1, n = nHub + 2 * L;
 CMat H(n, Vec(n, cd(0, 0)));
 auto add = [&](int a, int b, double ph) {
 cd v = std::exp(cd(0, ph)); H[a][b] += v; H[b][a] += std::conj(v);
 };
 for (int i = 0; i < L; i++) {
 int hi = i, hj = i + 1, U = nHub + 2 * i, D = nHub + 2 * i + 1;
 add(hi, U, 0.0); add(U, hj, 0.0); // upper path: phase 0
 add(hi, D, 0.0); add(D, hj, flux); // lower path: carries the loop flux
 }
 return H;
}

// max over time of |psi_far| when injected at hub 0 (probability that ever leaks out)
static double maxReach(int L, double flux, double dt = 0.3, double Tmax = -1) {
 CMat H = diamond(L, flux);
 int n = (int)H.size(), src = 0, far = L; // hub 0 -> hub L
 Stepper st; st.build(H, dt);
 Vec psi(n, cd(0, 0)); psi[src] = cd(1, 0);
 if (Tmax < 0) Tmax = 6.0 * L; // long enough for ballistic crossing at flux 0
 int steps = (int)(Tmax / dt);
 double mx = 0;
 for (int k = 0; k < steps; k++) { psi = st.step(psi); mx = std::max(mx, std::abs(psi[far])); }
 return mx;
}

int main() {
 std::printf("=== A4: Aharonov-Bohm cage -- exact flux-gated confinement ===\n");
 std::printf("max-over-time amplitude reaching the FAR hub from hub 0.\n");
 std::printf("flux=0 transmits (O(1)); flux=pi confines EXACTLY, invariant of length L.\n\n");
 std::printf(" L reach(flux=0) reach(flux=pi)\n");
 for (int L : {3, 6, 12, 24, 48}) {
 double r0 = maxReach(L, 0.0);
 double rp = maxReach(L, kPi);
 std::printf(" %-4d %.4f %.2e\n", L, r0, rp);
 }
 std::printf("\nflux scan at L=12 (is pi special, or did I tune it?):\n flux/pi reach\n");
 for (double f = 0.0; f <= 1.0001; f += 0.125)
 std::printf(" %.3f %.4f\n", f, maxReach(12, f * kPi));
 std::printf("\nNon-sinking test: reach(pi) stays at machine zero for EVERY L,\nwith NO fit, NO N->infinity, NO window choice. An exact gauge-invariant gate.\n");
 return 0;
}
