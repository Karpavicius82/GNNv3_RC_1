// A1 SIM: sharpen the golden signal.
// New, sharper criticality probe = level-spacing gap-ratio <r> (pure spectral),
// alongside the existing participation exponent tau. alpha-scan over Fibonacci
// ratios + two sizes, vs plain (extended) and disorder (localized) controls.
// <r>: Poisson(localized) ~0.386, GOE(extended/chaotic) ~0.530, critical between.
// tau = <log(PR)/log(N)> over central band: localized->0, extended->1.
#include "../tools/graph_wave_substrate.hpp"
#include <cstdio>
#include <complex>
using namespace gw;

// gap-ratio <r> from eigenvalues (degeneracy-robust: r=0 at exact degeneracies)
static double gapRatio(std::vector<double> ev) {
 std::sort(ev.begin(), ev.end());
 int n = (int)ev.size();
 // central 80% to avoid spectral edges
 int lo = n / 10, hi = n - n / 10;
 double acc = 0; int cnt = 0;
 for (int i = lo + 1; i < hi - 1; i++) {
 double s1 = ev[i] - ev[i - 1], s2 = ev[i + 1] - ev[i];
 double mn = std::min(s1, s2), mx = std::max(s1, s2);
 if (mx < 1e-12) { acc += 0.0; cnt++; continue; }
 acc += mn / mx; cnt++;
 }
 return cnt ? acc / cnt : 0.0;
}

// participation exponent tau over the central third, from Hermitian eigenvectors
// (via the 2N real embedding; complex psi_k = v[k] + i v[k+N]).
static double tauExp(const CMat& h) {
 int n = (int)h.size();
 Eig e = jacobiV(realRep(h));
 // pair up: sort indices by eigenvalue, take every other (doubled spectrum)
 std::vector<int> idx(2 * n);
 for (int i = 0; i < 2 * n; i++) idx[i] = i;
 std::sort(idx.begin(), idx.end(), [&](int a, int b) { return e.ev[a] < e.ev[b]; });
 int lo = (2 * n) / 3, hi = 2 * (2 * n) / 3;
 double acc = 0; int cnt = 0;
 for (int k = lo; k < hi; k += 2) { // every other => one per complex state
 int c = idx[k];
 double s2 = 0, s4 = 0;
 for (int r = 0; r < n; r++) {
 double p = e.V[r][c] * e.V[r][c] + e.V[r + n][c] * e.V[r + n][c];
 s2 += p; s4 += p * p;
 }
 if (s2 < 1e-30) continue;
 double pr = (s2 * s2) / s4; // participation ratio
 acc += std::log(pr) / std::log((double)n); // PR ~ N^tau
 cnt++;
 }
 return cnt ? acc / cnt : 0.0;
}

static void row(const char* name, const CMat& h) {
 int n = (int)h.size();
 double r = gapRatio(eigvals(h));
 double t = tauExp(h);
 std::printf(" %-22s N=%-4d <r>=%.3f tau=%.3f\n", name, n, r, t);
}

int main() {
 std::printf("=== A1: alpha-scan (Fibonacci ratios -> golden 0.618) + controls ===\n");
 struct AB { const char* nm; double a; };
 AB alphas[] = {{"golden 3/5", 3.0 / 5}, {"golden 5/8", 5.0 / 8},
 {"golden 8/13", 8.0 / 13}, {"golden 13/21", 13.0 / 21},
 {"golden 21/34", 21.0 / 34}};
 for (int w : {10, 14}) {
 std::printf("\n-- lattice width %d --\n", w);
 row("plain (extended)", plain(w).h);
 for (auto& a : alphas) row(a.nm, golden(w, a.a).h);
 row("disorder W=4", disorder(w, 4.0).h);
 row("randomPhase", randomPhase(w).h);
 }
 std::printf("\nReference: <r> Poisson=0.386, GOE=0.530. tau extended->1, localized->0.\n");
 return 0;
}
