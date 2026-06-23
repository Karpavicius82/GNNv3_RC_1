// Is a PHYSICAL measurement nonlinearity (Born rule |amp|^2 + renorm) the missing
// ingredient for depth? The slip: pure linear/unitary composition collapses ->
// 2-layer routing fails (deep_attention [4]: linear overlap 0.510 ~ chance 1/sqrtK).
// Test 2-hop content routing kA -> kB -> vC with different aggregation weights:
// uniform / linear : linear in the query -> 2 layers collapse -> should be chance
// Born |<q,k>|^2 : the SYSTEM's own measurement nonlinearity (probability),
// NO weights, NO imported softmax -> does it route?
// softmax : reference nonlinearity (deep_attention used this)
// If Born routes (final overlap with vC ~ 1) the missing boundary is measurement.
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
 auto rand = [&]() { Vec v(D); for (int d = 0; d < D; d++) v[d] = std::exp(cd(0, ph(rng))) / std::sqrt((double)D); return v; };
 auto inner = [&](const Vec& a, const Vec& b) { cd s(0, 0); for (int d = 0; d < D; d++) s += std::conj(a[d]) * b[d]; return s; };
 auto normv = [&](Vec v) { double n = 0; for (auto& z : v) n += std::norm(z); n = std::sqrt(n); for (auto& z : v) z /= n; return v; };

 Vec kA = rand(), kB = rand(), vC = rand();
 std::vector<Vec> key(K), val(K);
 enum Mode { UNIFORM, LINEAR, BORN, SOFTMAX };
 // build slots with distractor keys CORRELATED to the queries by `corr`
 // (corr=0 -> orthogonal/easy; corr>0 -> linear weighting blurs, isolates nonlinearity)
 auto build = [&](double corr) {
 auto mix = [&](const Vec& base) { Vec r(D); Vec n = rand(); for (int d = 0; d < D; d++) r[d] = corr * base[d] + (1 - corr) * n[d];
 double nn = 0; for (auto& z : r) nn += std::norm(z); nn = std::sqrt(nn); for (auto& z : r) z /= nn; return r; };
 key[0] = kA; val[0] = kB; key[1] = kB; val[1] = vC;
 for (int i = 2; i < K; i++) { key[i] = mix(i % 2 ? kA : kB); val[i] = rand(); } // distractors look like the queries
 };

 auto attend = [&](const Vec& q, Mode m) {
 std::vector<cd> s(K); for (int i = 0; i < K; i++) s[i] = inner(key[i], q);
 std::vector<cd> w(K);
 if (m == SOFTMAX) { double mx = -1e9; for (int i = 0; i < K; i++) mx = std::max(mx, s[i].real());
 double Z = 0; for (int i = 0; i < K; i++) { double e = std::exp(8.0 * (s[i].real() - mx)); w[i] = e; Z += e; } for (int i = 0; i < K; i++) w[i] /= Z; }
 else for (int i = 0; i < K; i++) {
 if (m == UNIFORM) w[i] = 1.0 / K;
 else if (m == LINEAR) w[i] = s[i]; // linear in q
 else /*BORN*/ w[i] = std::norm(s[i]); // |<q,k>|^2 (measurement / probability)
 }
 Vec r(D, cd(0, 0)); for (int i = 0; i < K; i++) for (int d = 0; d < D; d++) r[d] += w[i] * val[i][d];
 return normv(r);
 };

 std::printf("=== does a measurement (Born) nonlinearity close the depth slip? ===\n");
 std::printf("2-hop routing kA->kB->vC. D=%d, K=%d. Distractor keys correlated to queries.\n", D, K);
 const char* nm[] = {"uniform", "linear", "Born", "softmax"};
 for (double corr : {0.0, 0.6, 0.8}) {
 build(corr);
 std::printf("\n-- distractor correlation = %.1f --\n mode | layer1 ~kB | final ~vC\n", corr);
 for (int m = 0; m < 4; m++) {
 Vec r1 = attend(kA, (Mode)m), r2 = attend(r1, (Mode)m);
 std::printf(" %-9s | %.3f | %.3f\n", nm[m], std::abs(inner(kB, r1)), std::abs(inner(vC, r2)));
 }
 }
 std::printf("\nchance ~ 1/sqrt(D)=%.3f. If at high correlation LINEAR collapses but BORN still\nroutes, the system's own measurement nonlinearity is the missing depth ingredient.\n", 1.0 / std::sqrt((double)D));
 return 0;
}
