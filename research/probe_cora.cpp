// REAL-DATA production test: the weights-free GNN on the Cora citation graph.
// FLOW : graph message passing (neighbour feature aggregation)
// DECORRELATION: G^-1 on the class-prototype Gram (separates correlated classes)
// READOUT : holographic prototype (class = mean of training features) -> argmax
// No weights, no backprop. Compare own-features vs FLOW vs FLOW+decorrelation,
// against the known references (raw-features ~59%, GCN ~81.5%).
#include <cstdio>
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <cmath>
#include <random>
#include <algorithm>
using namespace std;

int main() {
 const int F = 1433, C = 7;
 unordered_map<string, int> labelId, nodeId;
 vector<vector<float>> feat; vector<int> lab; vector<string> labels;

 ifstream content("cora/cora.content");
 string line;
 while (getline(content, line)) {
 istringstream ss(line); string id; ss >> id;
 int idx = nodeId.size(); nodeId[id] = idx;
 vector<float> x(F); for (int f = 0; f < F; f++) ss >> x[f];
 string lb; ss >> lb; if (!labelId.count(lb)) { labelId[lb] = labelId.size(); labels.push_back(lb); }
 feat.push_back(move(x)); lab.push_back(labelId[lb]);
 }
 int N = feat.size();
 // L2-normalise features
 for (auto& x : feat) { double n = 0; for (float v : x) n += v * v; n = sqrt(n) + 1e-9; for (auto& v : x) v /= n; }
 // adjacency
 vector<vector<int>> adj(N);
 ifstream cites("cora/cora.cites");
 int edges = 0;
 while (getline(cites, line)) { istringstream ss(line); string a, b; ss >> a >> b;
 if (nodeId.count(a) && nodeId.count(b)) { int u = nodeId[a], v = nodeId[b]; if (u != v) { adj[u].push_back(v); adj[v].push_back(u); edges++; } } }
 printf("Cora loaded: N=%d nodes, %d edges, F=%d features, C=%d classes\n", N, edges, F, C);

 // train/test split: 20 nodes per class for training (standard low-label Cora), rest = test
 mt19937 rng(1);
 vector<int> order(N); for (int i = 0; i < N; i++) order[i] = i; shuffle(order.begin(), order.end(), rng);
 vector<int> isTrain(N, 0), perClass(C, 0);
 for (int i : order) if (perClass[lab[i]] < 20) { isTrain[i] = 1; perClass[lab[i]]++; }
 int ntr = 0; for (int i = 0; i < N; i++) ntr += isTrain[i];

 // FLOW: symmetric neighbour-sum aggregation, then L2-normalise (1 or 2 hops)
 auto hop = [&](const vector<vector<float>>& X) {
 vector<vector<float>> Y(N, vector<float>(F, 0));
 for (int i = 0; i < N; i++) { for (int f = 0; f < F; f++) Y[i][f] = X[i][f]; for (int j : adj[i]) for (int f = 0; f < F; f++) Y[i][f] += X[j][f]; }
 for (auto& y : Y) { double n = 0; for (float v : y) n += v * v; n = sqrt(n) + 1e-9; for (auto& v : y) v /= n; }
 return Y; };

 // build class prototypes from TRAIN nodes (holographic: mean of features), then classify all TEST
 auto evaluate = [&](const vector<vector<float>>& X, bool decor) {
 vector<vector<double>> proto(C, vector<double>(F, 0)); vector<int> cnt(C, 0);
 for (int i = 0; i < N; i++) if (isTrain[i]) { for (int f = 0; f < F; f++) proto[lab[i]][f] += X[i][f]; cnt[lab[i]]++; }
 for (int c = 0; c < C; c++) { double n = 0; for (int f = 0; f < F; f++) n += proto[c][f] * proto[c][f]; n = sqrt(n) + 1e-9; for (int f = 0; f < F; f++) proto[c][f] /= n; }
 // prototype Gram (cosine) for decorrelation
 vector<vector<double>> G(C, vector<double>(C, 0));
 for (int a = 0; a < C; a++) for (int b = 0; b < C; b++) { double s = 0; for (int f = 0; f < F; f++) s += proto[a][f] * proto[b][f]; G[a][b] = s; }
 auto cg = [&](const vector<double>& s) { // solve G c = s, C x C SPD
 auto mv = [&](const vector<double>& x) { vector<double> y(C, 0); for (int i = 0; i < C; i++) for (int j = 0; j < C; j++) y[i] += G[i][j] * x[j]; return y; };
 auto dot = [&](const vector<double>& a, const vector<double>& b) { double r = 0; for (int i = 0; i < C; i++) r += a[i] * b[i]; return r; };
 vector<double> c(C, 0), r = s, p = s; double rs = dot(r, r);
 for (int t = 0; t < C + 3; t++) { auto Gp = mv(p); double a = rs / (dot(p, Gp) + 1e-12); for (int i = 0; i < C; i++) { c[i] += a * p[i]; r[i] -= a * Gp[i]; } double rs2 = dot(r, r); if (rs2 < 1e-20) break; double b = rs2 / rs; for (int i = 0; i < C; i++) p[i] = r[i] + b * p[i]; rs = rs2; } return c; };
 int ok = 0, tot = 0;
 for (int i = 0; i < N; i++) { if (isTrain[i]) continue;
 vector<double> s(C); for (int c = 0; c < C; c++) { double v = 0; for (int f = 0; f < F; f++) v += proto[c][f] * X[i][f]; s[c] = v; }
 vector<double> sc = decor ? cg(s) : s;
 int best = 0; for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = c;
 ok += (best == lab[i]); tot++; }
 return 100.0 * ok / tot; };

 auto h1 = hop(feat); auto h2 = hop(h1);
 printf("train=%d (20/class), test=%d\n\n", ntr, N - ntr);
 printf(" method | naive | +DECORRELATION\n");
 printf(" own features (no message pass) | %.1f%% | %.1f%%\n", evaluate(feat, false), evaluate(feat, true));
 printf(" FLOW 1-hop | %.1f%% | %.1f%%\n", evaluate(h1, false), evaluate(h1, true));
 printf(" FLOW 2-hop | %.1f%% | %.1f%%\n", evaluate(h2, false), evaluate(h2, true));
 printf("\nreference: raw-features ~59%%, label-prop ~68%%, GCN (trained) ~81.5%%\n");
 printf("this pipeline is WEIGHTS-FREE: prototypes=means, G^-1=content Gram, FLOW=graph aggregation.\n");
 return 0;
}
