// PRODUCTION-SHAPE GNN, well-posed task: node has class y; its OWN feature is a
// heavily phase-noised copy of its class signature (one noisy copy -> weak). The
// graph is HOMOPHILIC (neighbours mostly share y), so the coherent sum of many
// neighbour features denoises toward sig[y] -> message passing should beat the
// own-feature readout. Classify directly by overlap with class signatures (no
// prototype training, so the baseline is not rescued). Real numbers, no promises.
#include "../tools/graph_wave_substrate.hpp"
#include <cstdio>
#include <complex>
#include <vector>
#include <cmath>
#include <random>
using namespace gw;

struct Sparse { int N; std::vector<std::vector<std::pair<int, cd>>> adj; };
static std::vector<cd> matvec(const Sparse& g, const std::vector<cd>& x) {
 std::vector<cd> y(g.N, cd(0, 0));
 for (int i = 0; i < g.N; i++) { cd s(0, 0); for (auto& e : g.adj[i]) s += e.second * x[e.first]; y[i] = s; }
 return y;
}

int main() {
 int N = 5000, C = 4, D = 256, deg = 12;
 double homophily = 0.85, noise = 1.6;
 std::mt19937 rng(2026u);
 std::uniform_int_distribution<int> randNode(0, N - 1), randClass(0, C - 1);
 std::uniform_real_distribution<double> uni(0, 1), ph(-kPi, kPi);
 std::normal_distribution<double> gn(0, noise);

 std::vector<int> y(N); for (int i = 0; i < N; i++) y[i] = randClass(rng); // class = label
 std::vector<std::vector<cd>> sig(C, std::vector<cd>(D));
 for (int c = 0; c < C; c++) for (int d = 0; d < D; d++) sig[c][d] = std::exp(cd(0, ph(rng)));
 // own feature = heavily phase-noised copy of own class signature (weak alone)
 std::vector<std::vector<cd>> feat(N, std::vector<cd>(D));
 for (int i = 0; i < N; i++) for (int d = 0; d < D; d++) feat[i][d] = sig[y[i]][d] * std::exp(cd(0, gn(rng)));
 // group nodes by class for homophilic wiring
 std::vector<std::vector<int>> byClass(C); for (int i = 0; i < N; i++) byClass[y[i]].push_back(i);

 Sparse g; g.N = N; g.adj.assign(N, {});
 auto add = [&](int i, int j) { if (i != j) { g.adj[i].push_back({j, cd(1, 0)}); g.adj[j].push_back({i, cd(1, 0)}); } };
 for (int i = 0; i < N; i++) for (int e = 0; e < deg; e++) {
 int j; if (uni(rng) < homophily) { auto& v = byClass[y[i]]; j = v[std::uniform_int_distribution<int>(0, (int)v.size() - 1)(rng)]; }
 else j = randNode(rng);
 add(i, j);
 }

 // message passing = coherent neighbour sum (1 and 2 hops)
 auto hop = [&](const std::vector<std::vector<cd>>& X) {
 std::vector<std::vector<cd>> Y(N, std::vector<cd>(D));
 for (int d = 0; d < D; d++) { std::vector<cd> col(N); for (int i = 0; i < N; i++) col[i] = X[i][d];
 std::vector<cd> o = matvec(g, col); for (int i = 0; i < N; i++) Y[i][d] = o[i]; }
 return Y;
 };
 auto acc = [&](const std::vector<std::vector<cd>>& X) {
 int ok = 0; for (int i = 0; i < N; i++) { int best = 0; double bv = -1;
 for (int c = 0; c < C; c++) { cd ov(0, 0); for (int d = 0; d < D; d++) ov += std::conj(sig[c][d]) * X[i][d]; double m = std::abs(ov); if (m > bv) { bv = m; best = c; } }
 ok += (best == y[i]); }
 return 100.0 * ok / N;
 };

 std::vector<std::vector<cd>> h1 = hop(feat), h2 = hop(h1);
 std::printf("=== integrated weights-free GNN at scale (well-posed homophily task) ===\n");
 std::printf("N=%d, C=%d, D=%d, deg=%d, homophily=%.2f, feature noise sigma=%.1f\n\n", N, C, D, deg, homophily, noise);
 std::printf(" own feature (NO message passing) : %.1f%% (one noisy copy)\n", acc(feat));
 std::printf(" GNN 1-hop neighbour aggregation : %.1f%%\n", acc(h1));
 std::printf(" GNN 2-hop neighbour aggregation : %.1f%%\n", acc(h2));
 std::printf("\n(chance=%.0f%%. message passing denoises across the homophilic neighbourhood.)\n", 100.0 / C);
 return 0;
}
