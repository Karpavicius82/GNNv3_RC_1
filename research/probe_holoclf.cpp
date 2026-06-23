// HOLO-CLF v2: weights-free diffractive matched-filter classifier, DONE RIGHT.
// v1 failed because a matched aperture is a PLANE WAVE -> it does not focus ->
// low peak (the prediction was exactly inverted). Fix: the matched filter must
// include a FOCUSING element. Time-reversal gives the exact lens for free:
// point at focal node T_c -> propagate K -> conjugate the aperture field = L_c
// (illuminating the aperture with L_c refocuses to a bright spot at T_c).
// Matched filter for class c: mask_c(x) = conj(t_c(x)) * L_c(x).
// input s=t_a, c=a : t_a*conj(t_a)*L_a = L_a -> focuses at T_a (bright).
// input s=t_a, c!=a: extra random phase t_a*conj(t_c) wrecks the focus (diffuse).
// Class = argmax_c intensity at T_c. No weights, no training; masks ARE the data.
#include "../tools/graph_wave_substrate.hpp"
#include <cstdio>
#include <complex>
#include <random>
using namespace gw;

int main() {
 int w = 32, N = w * w, M = 4, K = 14;
 auto id = [w](int x, int y) { return y * w + x; };
 int ax0 = 6, ax1 = 26, yA = 1;
 Stepper st; st.build(plain(w).h, 0.5);
 auto evolve = [&](Vec p) { for (int k = 0; k < K; k++) p = st.step(p); return p; };

 // class templates: fixed random phase patterns (the "images")
 std::vector<std::vector<double>> phi(M, std::vector<double>(w, 0.0));
 std::mt19937 rng(12345u); std::uniform_int_distribution<int> q(0, 3);
 for (int c = 0; c < M; c++) for (int x = ax0; x < ax1; x++) phi[c][x] = q(rng) * (kPi / 2);

 // focal node per class + its time-reversal focusing mask L_c on the aperture
 std::vector<int> Tc(M);
 std::vector<std::vector<cd>> L(M, std::vector<cd>(w, cd(0, 0)));
 for (int c = 0; c < M; c++) {
 int tx = (c + 1) * w / (M + 1), ty = w - 4; Tc[c] = id(tx, ty);
 Vec pt(N, cd(0, 0)); pt[Tc[c]] = cd(1, 0);
 Vec f = evolve(pt);
 for (int x = ax0; x < ax1; x++) L[c][x] = std::conj(f[id(x, yA)]);
 }

 auto score = [&](const std::vector<cd>& s, int c) {
 Vec p(N, cd(0, 0));
 for (int x = ax0; x < ax1; x++) p[id(x, yA)] = s[x] * std::exp(cd(0, -phi[c][x])) * L[c][x];
 double nrm = std::sqrt(power(p)); if (nrm < 1e-30) return 0.0; for (auto& v : p) v /= nrm;
 Vec o = evolve(p);
 return std::norm(o[Tc[c]]); // intensity at the class focal node
 };
 auto templAper = [&](int a) { std::vector<cd> s(w, cd(0, 0)); for (int x = ax0; x < ax1; x++) s[x] = std::exp(cd(0, phi[a][x])); return s; };
 // IDEAL readout = on-axis far-field of a perfect lens = coherent aperture sum
 auto idealScore = [&](const std::vector<cd>& s, int c) { cd acc(0, 0); for (int x = ax0; x < ax1; x++) acc += s[x] * std::exp(cd(0, -phi[c][x])); return std::norm(acc); };

 std::printf("=== weights-free matched-filter classifier: PRINCIPLE vs REALIZATION (M=%d) ===\n\n", M);
 std::printf("IDEAL readout = |coherent aperture sum| (perfect-lens focal value). ROW=true, COL=filter.\n ");
 for (int c = 0; c < M; c++) std::printf(" c%d ", c);
 std::printf(" pred\n");
 int correctI = 0, correctP = 0;
 for (int a = 0; a < M; a++) {
 auto s = templAper(a);
 int best = 0; double bv = -1; std::vector<double> row(M);
 for (int c = 0; c < M; c++) { row[c] = idealScore(s, c); if (row[c] > bv) { bv = row[c]; best = c; } }
 std::printf(" t%d ", a);
 for (int c = 0; c < M; c++) std::printf(" %5.1f", row[c]);
 std::printf(" %d %s\n", best, best == a ? "OK" : "X");
 correctI += (best == a);
 }
 std::printf("IDEAL accuracy: %d/%d (does the matched-filter PRINCIPLE hold?)\n\n", correctI, M);

 std::printf("PHYSICAL readout = intensity at lattice focal node T_c (x1000) -- focus-limited.\n ");
 for (int c = 0; c < M; c++) std::printf(" c%d ", c);
 std::printf(" pred\n");
 for (int a = 0; a < M; a++) {
 auto s = templAper(a);
 int best = 0; double bv = -1; std::vector<double> row(M);
 for (int c = 0; c < M; c++) { row[c] = score(s, c); if (row[c] > bv) { bv = row[c]; best = c; } }
 std::printf(" t%d ", a);
 for (int c = 0; c < M; c++) std::printf(" %5.1f", 1000 * row[c]);
 std::printf(" %d %s\n", best, best == a ? "OK" : "X");
 correctP += (best == a);
 }
 std::printf("PHYSICAL accuracy: %d/%d (does the lattice actually FOCUS it?)\n", correctP, M);
 int correct = correctP;

 std::printf("\nnoise robustness (per-site input phase noise sigma):\n sigma accuracy\n");
 for (double sg : {0.0, 0.4, 0.8, 1.2, 1.6}) {
 std::mt19937 ng(777u); std::normal_distribution<double> nd(0, sg);
 int ok = 0, tot = 0;
 for (int rep = 0; rep < 8; rep++) for (int a = 0; a < M; a++) {
 auto s = templAper(a);
 for (int x = ax0; x < ax1; x++) s[x] *= std::exp(cd(0, nd(ng)));
 int best = 0; double bv = -1;
 for (int c = 0; c < M; c++) { double v = score(s, c); if (v > bv) { bv = v; best = c; } }
 ok += (best == a); tot++;
 }
 std::printf(" %.1f %d/%d\n", sg, ok, tot);
 }
 std::printf("\nThe lens (time-reversal) is what v1 was missing. Matched -> focuses at T_c.\n");
 return 0;
}
