// probe_streaming_compression
// ----------------------------------------------------------------------------
// Streaming compression smoke probe using the GLOBAL substrate nonlinear physics.
// Compression is not a separate operation: every token event evolves
// the local field with matrix-free edge current + Kerr phase pressure.
// A horizon is only a detector condition over the already-evolved field.
// ----------------------------------------------------------------------------
#define NOMINMAX
#include "../tools/graph_wave_substrate.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <random>
#include <unordered_map>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "Psapi.lib")
#endif

using gw::cd;
using gw::Vec;

static double peakRamMb(){
#ifdef _WIN32
  PROCESS_MEMORY_COUNTERS_EX pmc;
  if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) return (double)pmc.PeakWorkingSetSize / (1024.0*1024.0);
#endif
  return -1.0;
}

constexpr int TOPICS = 3, PER = 24, WIN = 4, TOPK = 6;
constexpr double ETA = 1.0, DECAY = 0.99, EDGE_FLOOR = 1e-3, WMIN = 8.0, INJECT = 0.35;
constexpr int SMIN = 4, STEPS = 2;
constexpr double DT = 0.3, GK = 7.0;

struct Edge { double w = 0; int last = 0; };
struct Mem { std::unordered_map<int, cd> lin, ker; double prLin = 1, prKer = 1; };
struct Graph {
  std::vector<std::unordered_map<int, Edge>> adj; std::vector<int> seen; std::deque<int> win; int t = 0;
  int add(){ adj.push_back({}); seen.push_back(0); return (int)adj.size() - 1; }
  double incident(int a) const { double s = 0; for (auto& kv : adj[a]) s += kv.second.w; return s; }
  void decay(int a, int b){ auto it = adj[a].find(b); if (it == adj[a].end()) return; int dt = t - it->second.last; if (dt > 0) { it->second.w *= std::pow(DECAY, dt); it->second.last = t; } }
  void eraseEdge(int a, int b){ adj[a].erase(b); adj[b].erase(a); }
  void relaxNode(int a){ std::vector<int> dead; for (auto& kv : adj[a]) { int b = kv.first; int dt = t - kv.second.last; if (dt > 0) { kv.second.w *= std::pow(DECAY, dt); kv.second.last = t; auto back = adj[b].find(a); if (back != adj[b].end()) back->second = kv.second; } if (kv.second.w < EDGE_FLOOR) dead.push_back(b); } for (int b : dead) eraseEdge(a, b); }
  void prune(int a){ while ((int)adj[a].size() > TOPK) { int wk = -1; double mn = 1e100; for (auto& kv : adj[a]) if (kv.second.w < mn) { mn = kv.second.w; wk = kv.first; } eraseEdge(a, wk); } }
  void touch(int a, int b){ if (a == b) return; decay(a, b); decay(b, a); adj[a][b].w += ETA; adj[a][b].last = t; adj[b][a] = adj[a][b]; if ((int)adj[a].size() > TOPK) prune(a); if ((int)adj[b].size() > TOPK) prune(b); }
};

static unsigned hash3(int a, int b, int c){ unsigned x = 2166136261u; x = (x ^ (unsigned)a) * 16777619u; x = (x ^ (unsigned)(b + 0x9e3779b9u)) * 16777619u; x = (x ^ (unsigned)(c * 2654435761u)) * 16777619u; return x; }

static std::vector<int> hood(const Graph& g, int src){
  std::vector<int> nodes = {src}; std::unordered_map<int,int> seen; seen[src] = 0;
  for (int hop = 0; hop < 2; ++hop) { int sz = (int)nodes.size(); for (int h = 0; h < sz; ++h) for (auto& kv : g.adj[nodes[h]]) if (!seen.count(kv.first)) { seen[kv.first] = (int)nodes.size(); nodes.push_back(kv.first); } }
  return nodes;
}

static Vec project(const std::unordered_map<int, cd>& f, const std::unordered_map<int,int>& idx, int n, int src){
  Vec psi(n, cd(0,0)); for (auto& kv : f) { auto it = idx.find(kv.first); if (it != idx.end()) psi[it->second] += kv.second; }
  psi[idx.at(src)] += cd(INJECT, 0); gw::normalizeInPlace(psi); return psi;
}

static std::unordered_map<int, cd> unproject(const std::vector<int>& nodes, const Vec& psi){
  std::unordered_map<int, cd> out; for (int i = 0; i < (int)nodes.size(); ++i) if (std::norm(psi[i]) > 1e-14) out[nodes[i]] = psi[i]; return out;
}

struct Evo { bool horizon = false, stable = false; double prLin = 1, prKer = 1, phaseSpeed = 0; long long ops = 0; };

static std::vector<gw::SparseBond> bondsInLightCone(const Graph& g, const std::vector<int>& nodes, const std::unordered_map<int,int>& idx){
  std::vector<gw::SparseBond> bonds;
  for (int i = 0; i < (int)nodes.size(); ++i) for (auto& kv : g.adj[nodes[i]]) {
    auto it = idx.find(kv.first);
    if (it != idx.end() && i < it->second) bonds.push_back({i, it->second, kv.second.w});
  }
  return bonds;
}

static Evo evolve(Graph& g, int src, Mem& m){
  Evo e; auto nodes = hood(g, src); int n = (int)nodes.size();
  if (n < 3) { m.lin[src] = cd(1,0); m.ker[src] = cd(1,0); return e; }
  std::unordered_map<int,int> idx; for (int i = 0; i < n; ++i) idx[nodes[i]] = i;
  std::vector<gw::SparseBond> bonds = bondsInLightCone(g, nodes, idx);
  Vec lin = project(m.lin, idx, n, src), ker = project(m.ker, idx, n, src);
  gw::LocalFlowStats linStats, kerStats;
  lin = gw::edgeLocalKerrFlow(std::move(lin), bonds, DT, 0.0, STEPS, &linStats);
  ker = gw::edgeLocalKerrFlow(std::move(ker), bonds, DT, GK, STEPS, &kerStats);
  gw::normalizeInPlace(lin); gw::normalizeInPlace(ker);
  m.lin = unproject(nodes, lin); m.ker = unproject(nodes, ker);
  m.prLin = e.prLin = gw::participationRatio(lin); m.prKer = e.prKer = gw::participationRatio(ker);
  e.ops = linStats.bond_visits + kerStats.bond_visits;
  e.phaseSpeed = kerStats.max_bond_speed;
  e.stable = ((int)g.adj[src].size() >= TOPK) && g.incident(src) >= WMIN && g.seen[src] >= SMIN;
  // horizon = collapse beats dispersion: the nonlinear field is concentrated to less
  // than half the PR of its own linear (g=0) control. Scale-free, no magic cutoff.
  e.horizon = e.stable && e.prKer < 0.5 * e.prLin;
  return e;
}

static cd overlap(const std::unordered_map<int, cd>& a, const std::unordered_map<int, cd>& b){ cd s(0,0); for (auto& kv : a) { auto it = b.find(kv.first); if (it != b.end()) s += std::conj(kv.second) * it->second; } return s; }
static int argInt(char** argv, int argc, int i, int fallback){ return i < argc ? std::atoi(argv[i]) : fallback; }

struct System {
  Graph g; std::vector<Mem> mem; std::vector<bool> novel; bool randomize = false;
  long long updates = 0, stableHorizons = 0, novelHorizons = 0, horizonSamples = 0, maxOps = 0;
  double maxPhaseSpeed = 0;
  double sumHLin = 0, sumHKer = 0;
  explicit System(bool r) : randomize(r) { for (int i = 0; i < TOPICS * PER; ++i) add(false); }
  int add(bool n){ int id = g.add(); mem.push_back({}); novel.push_back(n); return id; }
  int randomEndpoint(int a, int b) const { int n = (int)g.adj.size(); int r = (int)(hash3(a,b,g.t) % (unsigned)n); return r == a ? (r + 1) % n : r; }
  void process(int node){
    g.t++; g.seen[node]++; g.relaxNode(node); for (int c : g.win) { g.relaxNode(c); g.touch(node, randomize ? randomEndpoint(node,c) : c); }
    g.win.push_back(node); if ((int)g.win.size() > WIN) g.win.pop_front();
    Evo e = evolve(g, node, mem[node]); updates++; maxOps = std::max(maxOps, e.ops);
    maxPhaseSpeed = std::max(maxPhaseSpeed, e.phaseSpeed);
    if (e.horizon) { horizonSamples++; sumHLin += e.prLin; sumHKer += e.prKer; if (novel[node]) novelHorizons++; else stableHorizons++; }
  }
};

static void run(System& s, int stream, int uniqueEvery){
  std::mt19937 r(11); std::uniform_int_distribution<int> tt(0, TOPICS-1), ww(0, PER-1);
  for (int i = 0; i < stream;) { int tp = tt(r); s.g.win.clear(); for (int b = 0; b < 6 && i < stream; ++b, ++i) { int node = (uniqueEvery > 0 && i % uniqueEvery == 0) ? s.add(true) : tp * PER + ww(r); s.process(node); } }
}

static double eval(System& s){
  std::vector<std::unordered_map<int,cd>> proto(TOPICS);
  for (int tp = 0; tp < TOPICS; ++tp) for (int w = 0; w < PER/2; ++w) for (auto& kv : s.mem[tp*PER+w].ker) proto[tp][kv.first] += kv.second;
  int ok = 0, total = 0; for (int tp = 0; tp < TOPICS; ++tp) for (int w = PER/2; w < PER; ++w) { int node = tp*PER+w, best = -1; double bs = -1; for (int c = 0; c < TOPICS; ++c) { double m = std::abs(overlap(s.mem[node].ker, proto[c])); if (m > bs) { bs = m; best = c; } } ok += (best == tp); total++; }
  return 100.0 * ok / total;
}

int main(int argc, char** argv){
  const int stream = argInt(argv, argc, 1, 60000), uniqueEvery = argInt(argv, argc, 2, 7);
  // real system trains on the full stream (timed); random control is capped (only a
  // value baseline) so 10M runs don't hold two giant systems in memory at once.
  auto t0 = std::chrono::steady_clock::now();
  System real(false); run(real, stream, uniqueEvery);
  auto tReal = std::chrono::steady_clock::now();
  System random(true); run(random, std::min(stream, 200000), uniqueEvery);
  double comp = real.sumHKer > 0 ? real.sumHLin / real.sumHKer : 0;
  double realAcc = eval(real), randomAcc = eval(random);
  double realSec = std::chrono::duration<double>(tReal - t0).count();
  std::printf("=== STREAMING COMPRESSION: nonlinear physics always on ===\n");
  std::printf("stream=%d uniqueEvery=%d field_updates=%lld nodes=%d horizons=%lld novel_horizons=%lld\n", stream, uniqueEvery, real.updates, (int)real.g.adj.size(), real.stableHorizons, real.novelHorizons);
  std::printf("horizon PR linear=%.2f nonlinear=%.2f compression=%.2fx\n", real.horizonSamples ? real.sumHLin/real.horizonSamples : 0, real.horizonSamples ? real.sumHKer/real.horizonSamples : 0, comp);
  std::printf("value REAL=%.1f%% RANDOM=%.1f%% max_bond_visits=%lld max_phase_speed=%.3f\n", realAcc, randomAcc, real.maxOps, real.maxPhaseSpeed);
  std::printf("train_sec=%.2f tokens_per_sec=%.0f peak_ram_mb=%.0f\n", realSec, stream/std::max(realSec,1e-9), peakRamMb());
  bool ok = real.updates == stream && real.stableHorizons > 0 && real.novelHorizons == 0 && comp > 1.35 && realAcc > randomAcc + 20.0 && real.maxOps <= 64LL * TOPK * 4 && real.maxPhaseSpeed < 8.0;
  return ok ? 0 : 1;
}
