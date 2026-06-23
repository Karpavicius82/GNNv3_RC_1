// Does iterated dissipative SETTLING (resonator cleanup) close the depth slip?
// Same 2-hop routing kA->kB->vC with CORRELATED distractor keys, where single-shot
// Born failed (0.65 @corr0.6, 0.48 @corr0.8). Now each layer SETTLES the query
// onto the nearest key-attractor by ITERATED competition (the parked resonator
// cleanup), then forwards its value. Settling = live, weights-free, no softmax beta.
// If cleanup-routing ~1.0 where Born collapsed, the missing piece was SETTLING.
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

 // single-shot Born aggregation (the failing baseline)
 auto born = [&](const Vec& q) { Vec r(D, cd(0, 0)); for (int i = 0; i < K; i++) { double w = std::norm(inr(key[i], q)); for (int d = 0; d < D; d++) r[d] += w * val[i][d]; } return nrm(r); };
 // iterated dissipative SETTLING onto nearest key-attractor, then forward value
 auto settle = [&](Vec p, int T) { for (int t = 0; t < T; t++) { Vec np(D, cd(0, 0));
 for (int i = 0; i < K; i++) { cd s = inr(key[i], p); double w = std::norm(s) * std::norm(s); for (int d = 0; d < D; d++) np[d] += w * key[i][d]; } p = nrm(np); } return p; };
 auto route_settle = [&](const Vec& q, int T) { Vec p = settle(q, T); Vec r(D, cd(0, 0)); for (int i = 0; i < K; i++) { cd w = inr(key[i], p); for (int d = 0; d < D; d++) r[d] += w * val[i][d]; } return nrm(r); };
 auto maxKeySim = [&](const Vec& p) { double m = 0; for (int i = 0; i < K; i++) m = std::max(m, std::abs(inr(key[i], p))); return m; };

 std::printf("=== does iterated SETTLING close the depth slip Born could not? ===\n");
 std::printf("2-hop kA->kB->vC, correlated distractors. settle = %d cleanup iterations.\n\n", 12);
 std::printf(" corr | Born final~vC | SETTLE final~vC | settled sharpness (max key-sim)\n");
 for (double corr : {0.0, 0.6, 0.8, 0.9}) {
 build(corr);
 Vec b1 = born(kA), b2 = born(b1);
 Vec s1 = route_settle(kA, 12), s2 = route_settle(s1, 12);
 double sharp = maxKeySim(settle(kA, 12));
 std::printf(" %.1f | %.3f | %.3f | %.3f\n",
 corr, std::abs(inr(vC, b2)), std::abs(inr(vC, s2)), sharp);
 }
 std::printf("\nIf SETTLE ~1.0 where Born collapsed (0.48-0.65), the missing piece was the\nparked LIVE dissipative settling -- weights-free, sharp, native to the theory.\n");
 return 0;
}
