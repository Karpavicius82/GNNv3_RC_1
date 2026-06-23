// CLOSED PIPELINE on a WELL-POSED task that needs all parts:
// communities each carry a SET of 2 CORRELATED topics, SHARED by the neighbourhood.
// node feature = noisy sample of ONE of its community's two topics.
// task = recover the community's TOPIC SET (both topics, top-2).
// FLOW (neighbour sum) recovers the shared set (neighbours collectively sample both
// + denoise); DECORRELATION (G^-1, native CG) separates the two CORRELATED topics;
// naive readout blurs them. If FLOW+DECOR >> FLOW+naive at high correlation, the
// glued three-substrate pipeline closes the system on correlated relational data.
#include <complex>
#include <vector>
#include <cmath>
#include <cstdio>
#include <random>
#include <array>
using cd = std::complex<double>;
using Vec = std::vector<cd>;
static const double kPi = 3.14159265358979323846;

int main() {
 int N = 4000, M = 200, C = 8, D = 256, deg = 12;
 double noise = 1.2;
 std::mt19937 rng(2026u);
 std::uniform_int_distribution<int> randTopic(0, C - 1), bit(0, 1);
 std::uniform_real_distribution<double> ph(-kPi, kPi);
 std::normal_distribution<double> gn(0, noise);
 auto rnd = [&]() { Vec v(D); for (int d = 0; d < D; d++) v[d] = std::exp(cd(0, ph(rng))) / std::sqrt((double)D); return v; };
 auto inr = [&](const Vec& a, const Vec& b) { cd s(0, 0); for (int d = 0; d < D; d++) s += std::conj(a[d]) * b[d]; return s; };
 auto nrm = [&](Vec v) { double n = 0; for (auto& z : v) n += std::norm(z); n = std::sqrt(n); for (auto& z : v) z /= n; return v; };

 std::vector<int> comm(N); for (int i = 0; i < N; i++) comm[i] = i % M;
 std::vector<std::vector<int>> byComm(M); for (int i = 0; i < N; i++) byComm[comm[i]].push_back(i);
 std::vector<std::array<int, 2>> topics(M);
 for (int m = 0; m < M; m++) { int a = randTopic(rng), b; do { b = randTopic(rng); } while (b == a); topics[m] = {a, b}; }
 std::vector<std::vector<int>> adj(N);
 std::uniform_int_distribution<int> rN(0, N - 1);
 for (int i = 0; i < N; i++) for (int e = 0; e < deg; e++) { auto& v = byComm[comm[i]]; int j = v[std::uniform_int_distribution<int>(0, (int)v.size() - 1)(rng)]; if (j != i) { adj[i].push_back(j); adj[j].push_back(i); } }

 auto cg = [&](const std::vector<std::vector<cd>>& G, const std::vector<cd>& s) {
 auto mv = [&](const std::vector<cd>& x) { std::vector<cd> y(C, cd(0, 0)); for (int i = 0; i < C; i++) for (int j = 0; j < C; j++) y[i] += G[i][j] * x[j]; return y; };
 auto dot = [&](const std::vector<cd>& a, const std::vector<cd>& b) { cd r(0, 0); for (int i = 0; i < C; i++) r += std::conj(a[i]) * b[i]; return r; };
 std::vector<cd> c(C, cd(0, 0)), r = s, p = s; cd rs = dot(r, r);
 for (int t = 0; t < C + 2; t++) { auto Gp = mv(p); cd a = rs / dot(p, Gp); for (int i = 0; i < C; i++) { c[i] += a * p[i]; r[i] -= a * Gp[i]; } cd rs2 = dot(r, r); if (rs2.real() < 1e-26) break; cd b = rs2 / rs; for (int i = 0; i < C; i++) p[i] = r[i] + b * p[i]; rs = rs2; }
 return c; };

 std::printf("=== CLOSED PIPELINE (well-posed): community topic-set recovery ===\n");
 std::printf("N=%d, %d communities, C=%d topics, D=%d, noise=%.1f. recover both shared topics (top-2).\n\n", N, M, C, D, noise);
 std::printf(" topic-corr | raw(no flow) | FLOW+naive | FLOW+DECORRELATION (glued)\n");
 for (double beta : {0.85, 0.92, 0.96, 0.98}) {
 Vec base = rnd(); std::vector<Vec> sig(C);
 for (int c = 0; c < C; c++) { Vec u = rnd(), v(D); for (int d = 0; d < D; d++) v[d] = beta * base[d] + std::sqrt(1 - beta * beta) * u[d]; sig[c] = nrm(v); }
 std::vector<std::vector<cd>> G(C, std::vector<cd>(C)); for (int a = 0; a < C; a++) for (int b = 0; b < C; b++) G[a][b] = inr(sig[a], sig[b]);
 std::vector<Vec> feat(N); for (int i = 0; i < N; i++) { int tp = topics[comm[i]][bit(rng)]; Vec f(D); for (int d = 0; d < D; d++) f[d] = sig[tp][d] * std::exp(cd(0, gn(rng))); feat[i] = f; }
 auto hop = [&](const std::vector<Vec>& X) { std::vector<Vec> Y(N, Vec(D, cd(0, 0))); for (int i = 0; i < N; i++) for (int j : adj[i]) for (int d = 0; d < D; d++) Y[i][d] += X[j][d]; return Y; };
 std::vector<Vec> h = hop(hop(feat));
 auto acc = [&](const std::vector<Vec>& X, bool decor) {
 int ok = 0; for (int i = 0; i < N; i++) {
 std::vector<cd> s(C); for (int c = 0; c < C; c++) s[c] = inr(sig[c], X[i]);
 std::vector<cd> coef = decor ? cg(G, s) : s;
 int a = 0, b = 1; if (std::abs(coef[1]) > std::abs(coef[0])) { a = 1; b = 0; }
 for (int c = 2; c < C; c++) { double m = std::abs(coef[c]); if (m > std::abs(coef[a])) { b = a; a = c; } else if (m > std::abs(coef[b])) b = c; }
 int t0 = topics[comm[i]][0], t1 = topics[comm[i]][1];
 ok += ((a == t0 || b == t0) && (a == t1 || b == t1));
 } return 100.0 * ok / N; };
 std::printf(" %.2f | %.1f%% | %.1f%% | %.1f%%\n", beta, acc(feat, false), acc(h, false), acc(h, true));
 }
 std::printf("\nFLOW recovers the shared set; DECORRELATION separates the correlated topics.\n");
 return 0;
}
