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
constexpr int BASE_NODES = TOPICS * PER, BRIDGE_TOPK = 1;
constexpr double BRIDGE_MIN_COH = 0.20;

struct Edge { double w = 0; double phase = 0; int last = 0; };
struct BridgeBond { int a = 0, b = 0; double w = 0; double phase = 0; };
struct Mem { std::unordered_map<int, cd> lin, ker; std::unordered_map<int, double> sense; double prLin = 1, prKer = 1; };
struct BridgeStats {
  long long materialized = 0, trueBridge = 0, falseBridge = 0;
  double sameOverlap = 0, crossOverlap = 0, maxSame = 0, maxCross = 0;
  int samePairs = 0, crossPairs = 0;
};
struct Graph {
  std::vector<std::unordered_map<int, Edge>> adj; std::vector<BridgeBond> bridges; std::vector<int> seen; std::deque<int> win; int t = 0;
  int add(){ adj.push_back({}); seen.push_back(0); return (int)adj.size() - 1; }
  double incident(int a) const { double s = 0; for (auto& kv : adj[a]) s += kv.second.w; return s; }
  double meanBond(int a) const { return adj[a].empty() ? 0.0 : incident(a) / (double)adj[a].size(); }
  void decay(int a, int b){ auto it = adj[a].find(b); if (it == adj[a].end()) return; int dt = t - it->second.last; if (dt > 0) { it->second.w *= std::pow(DECAY, dt); it->second.last = t; } }
  void eraseEdge(int a, int b){ adj[a].erase(b); adj[b].erase(a); }
  static double wrapPhase(double x){ while (x > gw::kPi) x -= 2.0 * gw::kPi; while (x < -gw::kPi) x += 2.0 * gw::kPi; return x; }
  bool hasEdge(int a, int b) const { return adj[a].find(b) != adj[a].end(); }
  bool hasBridge(int a, int b) const { for (const auto& e : bridges) if ((e.a == a && e.b == b) || (e.a == b && e.b == a)) return true; return false; }
  int bridgeDegree(int a) const { int d = 0; for (const auto& e : bridges) if (e.a == a || e.b == a) d++; return d; }
  double orientedPhase(int u, int v) const { auto it = adj[u].find(v); if (it == adj[u].end()) return 0.0; return u < v ? it->second.phase : -it->second.phase; }
  double orientedBridgePhase(int u, int v) const { for (const auto& e : bridges) { if (e.a == u && e.b == v) return e.phase; if (e.a == v && e.b == u) return -e.phase; } return 0.0; }
  void addSignedPhase(int u, int v, double dphi){ auto it = adj[u].find(v); if (it == adj[u].end()) return; Edge e = it->second; double sign = u < v ? 1.0 : -1.0; e.phase = std::clamp(wrapPhase(e.phase + sign * dphi), -1.4, 1.4); adj[u][v] = e; adj[v][u] = e; }
  void addPlaquetteFlux(int a, int b, int c){ constexpr double dphi = 0.018; addSignedPhase(a, b, dphi / 3.0); addSignedPhase(b, c, dphi / 3.0); addSignedPhase(c, a, dphi / 3.0); }
  void closePlaquettes(int node){ for (int j = 1; j < (int)win.size(); ++j) for (int i = 0; i < j; ++i) { int a = win[i], b = win[j]; if (hasEdge(a, b) && hasEdge(a, node) && hasEdge(b, node)) addPlaquetteFlux(a, b, node); } }
  void relaxNode(int a){ std::vector<int> dead; for (auto& kv : adj[a]) { int b = kv.first; int dt = t - kv.second.last; if (dt > 0) { kv.second.w *= std::pow(DECAY, dt); kv.second.last = t; auto back = adj[b].find(a); if (back != adj[b].end()) back->second = kv.second; } if (kv.second.w < EDGE_FLOOR) dead.push_back(b); } for (int b : dead) eraseEdge(a, b); }
  void prune(int a){ while ((int)adj[a].size() > TOPK) { int wk = -1; double mn = 1e100; for (auto& kv : adj[a]) if (kv.second.w < mn) { mn = kv.second.w; wk = kv.first; } eraseEdge(a, wk); } }
  void touch(int a, int b){ if (a == b) return; decay(a, b); decay(b, a); adj[a][b].w += ETA; adj[a][b].last = t; adj[b][a] = adj[a][b]; if ((int)adj[a].size() > TOPK) prune(a); if ((int)adj[b].size() > TOPK) prune(b); }
  void setBridge(int a, int b, double w, double phaseAB){
    if (a == b || hasBridge(a, b)) return;
    if (a > b) { std::swap(a, b); phaseAB = -phaseAB; }
    bridges.push_back({a, b, w, wrapPhase(phaseAB)});
  }
};

static unsigned hash3(int a, int b, int c){ unsigned x = 2166136261u; x = (x ^ (unsigned)a) * 16777619u; x = (x ^ (unsigned)(b + 0x9e3779b9u)) * 16777619u; x = (x ^ (unsigned)(c * 2654435761u)) * 16777619u; return x; }

static std::vector<int> hood(const Graph& g, int src, bool useBridges = false){
  std::vector<int> nodes = {src}; std::unordered_map<int,int> seen; seen[src] = 0;
  for (int hop = 0; hop < 2; ++hop) {
    int sz = (int)nodes.size();
    for (int h = 0; h < sz; ++h) {
      int u = nodes[h];
      for (auto& kv : g.adj[u]) if (!seen.count(kv.first)) { seen[kv.first] = (int)nodes.size(); nodes.push_back(kv.first); }
    }
  }
  if (useBridges) {
    for (const auto& e : g.bridges) {
      int v = e.a == src ? e.b : (e.b == src ? e.a : -1);
      if (v >= 0 && !seen.count(v)) { seen[v] = (int)nodes.size(); nodes.push_back(v); }
    }
  }
  return nodes;
}

static Vec project(const std::unordered_map<int, cd>& f, const std::unordered_map<int,int>& idx, int n, int src){
  Vec psi(n, cd(0,0)); for (auto& kv : f) { auto it = idx.find(kv.first); if (it != idx.end()) psi[it->second] += kv.second; }
  psi[idx.at(src)] += cd(INJECT, 0); gw::normalizeInPlace(psi); return psi;
}

static std::unordered_map<int, cd> unproject(const std::vector<int>& nodes, const Vec& psi){
  std::unordered_map<int, cd> out; for (int i = 0; i < (int)nodes.size(); ++i) if (std::norm(psi[i]) > 1e-14) out[nodes[i]] = psi[i]; return out;
}

static std::unordered_map<int, double> feelMotion(const std::vector<int>& nodes, const Vec& before, const Vec& after){
  std::unordered_map<int, double> out;
  for (int i = 0; i < (int)nodes.size(); ++i) {
    double move = std::abs(after[i] - before[i]);
    double phase = std::abs(std::arg(std::conj(before[i]) * after[i])) * std::sqrt(std::norm(before[i]) * std::norm(after[i]));
    double sense = move + phase;
    if (sense > 1e-14) out[nodes[i]] = sense;
  }
  return out;
}

static cd feltOverlap(const Mem& a, const Mem& b){
  cd z(0,0);
  for (auto& kv : a.ker) {
    auto ib = b.ker.find(kv.first); if (ib == b.ker.end()) continue;
    auto sa = a.sense.find(kv.first), sb = b.sense.find(kv.first);
    if (sa == a.sense.end() || sb == b.sense.end()) continue;
    z += std::conj(kv.second) * ib->second * std::sqrt(sa->second * sb->second);
  }
  return z;
}

static double feltPower(const Mem& m){
  double s = 0;
  for (auto& kv : m.ker) { auto it = m.sense.find(kv.first); if (it != m.sense.end()) s += std::norm(kv.second) * it->second; }
  return s;
}

static double feltCoherence(const Mem& a, const Mem& b, cd* raw = nullptr){
  cd z = feltOverlap(a, b); if (raw) *raw = z;
  return std::abs(z) / (std::sqrt(feltPower(a) * feltPower(b)) + 1e-300);
}

struct Evo { bool horizon = false, stable = false; double prLin = 1, prKer = 1, phaseSpeed = 0; long long ops = 0; };

static std::vector<gw::SparseBond> bondsInLightCone(const Graph& g, const std::vector<int>& nodes, const std::unordered_map<int,int>& idx, bool useBridges = false){
  std::vector<gw::SparseBond> bonds;
  for (int i = 0; i < (int)nodes.size(); ++i) for (auto& kv : g.adj[nodes[i]]) {
    int u = nodes[i];
    auto it = idx.find(kv.first);
    if (it != idx.end() && i < it->second) bonds.push_back({i, it->second, kv.second.w, g.orientedPhase(u, kv.first)});
  }
  if (useBridges) {
    for (const auto& e : g.bridges) {
      auto ia = idx.find(e.a), ib = idx.find(e.b);
      if (ia != idx.end() && ib != idx.end()) {
        if (ia->second < ib->second) bonds.push_back({ia->second, ib->second, e.w, g.orientedBridgePhase(e.a, e.b)});
        else if (ib->second < ia->second) bonds.push_back({ib->second, ia->second, e.w, g.orientedBridgePhase(e.b, e.a)});
      }
    }
  }
  return bonds;
}

static Evo evolve(Graph& g, int src, Mem& m){
  Evo e; auto nodes = hood(g, src); int n = (int)nodes.size();
  if (n < 3) { m.lin[src] = cd(1,0); m.ker[src] = cd(1,0); return e; }
  std::unordered_map<int,int> idx; for (int i = 0; i < n; ++i) idx[nodes[i]] = i;
  std::vector<gw::SparseBond> bonds = bondsInLightCone(g, nodes, idx, false);
  Vec lin = project(m.lin, idx, n, src), ker = project(m.ker, idx, n, src);
  Vec kerBefore;
  if (src < BASE_NODES) kerBefore = ker;
  gw::LocalFlowStats kerStats;
  gw::edgeLocalKerrFlowPair(lin, ker, bonds, DT, GK, STEPS, &kerStats);
  gw::normalizeInPlace(lin); gw::normalizeInPlace(ker);
  m.lin = unproject(nodes, lin); m.ker = unproject(nodes, ker);
  if (src < BASE_NODES) m.sense = feelMotion(nodes, kerBefore, ker);
  else m.sense.clear();
  m.prLin = e.prLin = gw::participationRatio(lin); m.prKer = e.prKer = gw::participationRatio(ker);
  e.ops = (long long)bonds.size() * STEPS * 2 + kerStats.bond_visits;
  e.phaseSpeed = kerStats.max_bond_speed;
  e.stable = ((int)g.adj[src].size() >= TOPK) && g.incident(src) >= WMIN && g.seen[src] >= SMIN;
  // horizon = collapse beats dispersion: the nonlinear field is concentrated to less
  // than half the PR of its own linear (g=0) control. Scale-free, no magic cutoff.
  e.horizon = e.stable && e.prKer < 0.5 * e.prLin;
  return e;
}

static int argInt(char** argv, int argc, int i, int fallback){ return i < argc ? std::atoi(argv[i]) : fallback; }
static int topicOf(int node){ return node >= 0 && node < BASE_NODES ? node / PER : -1; }

struct System {
  Graph g; std::vector<Mem> mem; std::vector<bool> novel, horizonHub;
  std::vector<unsigned char> measuredPair, bridgePair;
  BridgeStats bridgeStats;
  bool randomize = false;
  long long updates = 0, stableHorizons = 0, novelHorizons = 0, horizonSamples = 0, maxOps = 0;
  double maxPhaseSpeed = 0;
  double sumHLin = 0, sumHKer = 0;
  explicit System(bool r) : randomize(r) { for (int i = 0; i < TOPICS * PER; ++i) add(false); measuredPair.assign(BASE_NODES * BASE_NODES, 0); bridgePair.assign(BASE_NODES * BASE_NODES, 0); }
  int add(bool n){ int id = g.add(); mem.push_back({}); novel.push_back(n); horizonHub.push_back(false); return id; }
  int randomEndpoint(int a, int b) const { int n = (int)g.adj.size(); int r = (int)(hash3(a,b,g.t) % (unsigned)n); return r == a ? (r + 1) % n : r; }
  int pairKey(int a, int b) const { if (a > b) std::swap(a, b); return a * BASE_NODES + b; }
  void observeCandidate(int a, int b, double coh){
    int key = pairKey(a, b); if (measuredPair[key]) return; measuredPair[key] = 1;
    if (topicOf(a) == topicOf(b)) { bridgeStats.sameOverlap += coh; bridgeStats.samePairs++; bridgeStats.maxSame = std::max(bridgeStats.maxSame, coh); }
    else { bridgeStats.crossOverlap += coh; bridgeStats.crossPairs++; bridgeStats.maxCross = std::max(bridgeStats.maxCross, coh); }
  }
  double bridgeImpedance(int a, int b, double coh) const { return std::sqrt(g.meanBond(a) * g.meanBond(b)) * coh * coh; }
  bool pressureCompatible(int a, int b, double w, double phaseAB) const {
    std::vector<int> nodes = hood(g, a, false); std::unordered_map<int,int> idx;
    for (int i = 0; i < (int)nodes.size(); ++i) idx[nodes[i]] = i;
    std::vector<int> hb = hood(g, b, false);
    for (int x : hb) if (!idx.count(x)) { idx[x] = (int)nodes.size(); nodes.push_back(x); }
    if (!idx.count(a) || !idx.count(b)) return false;
    std::vector<gw::SparseBond> bonds = bondsInLightCone(g, nodes, idx, false), bridged = bonds;
    int ia = idx[a], ib = idx[b];
    bridged.push_back({ia, ib, w, ia < ib ? phaseAB : -phaseAB});
    Vec psi(nodes.size(), cd(0,0));
    for (auto& kv : mem[a].ker) { auto it = idx.find(kv.first); if (it != idx.end()) psi[it->second] += kv.second; }
    for (auto& kv : mem[b].ker) { auto it = idx.find(kv.first); if (it != idx.end()) psi[it->second] += kv.second; }
    if (gw::power(psi) < 1e-300) psi[ia] = cd(1,0);
    gw::normalizeInPlace(psi);
    Vec noBridge = gw::edgeLocalKerrFlow(psi, bonds, DT, GK, STEPS);
    Vec withBridge = gw::edgeLocalKerrFlow(psi, bridged, DT, GK, STEPS);
    return gw::participationRatio(withBridge) <= gw::participationRatio(noBridge) + 1e-9;
  }
  int bestPartner(int a, cd* bestOverlap, double* bestCoh, bool observe){
    int best = -1; cd zBest(0,0); double cBest = BRIDGE_MIN_COH;
    for (int b = 0; b < BASE_NODES; ++b) {
      if (b == a || !horizonHub[b] || g.bridgeDegree(b) >= BRIDGE_TOPK) continue;
      int key = pairKey(a, b); if (bridgePair[key] || g.hasBridge(a, b)) continue;
      cd z(0,0); double coh = feltCoherence(mem[a], mem[b], &z);
      if (observe) observeCandidate(a, b, coh);
      if (coh > cBest) { best = b; cBest = coh; zBest = z; }
    }
    if (bestOverlap) *bestOverlap = zBest; if (bestCoh) *bestCoh = cBest; return best;
  }
  void tryMaterializeBridge(int a, bool horizon){
    if (!horizon || a >= BASE_NODES || novel[a]) return;
    horizonHub[a] = true; if (g.bridgeDegree(a) >= BRIDGE_TOPK) return;
    cd z(0,0); double coh = 0; int b = bestPartner(a, &z, &coh, true); if (b < 0) return;
    cd rz(0,0); double rcoh = 0; if (bestPartner(b, &rz, &rcoh, false) != a) return;
    double w = bridgeImpedance(a, b, coh); if (w <= 1e-12) return;
    double phaseAB = -std::arg(z); if (!pressureCompatible(a, b, w, phaseAB)) return;
    g.setBridge(a, b, w, phaseAB); bridgePair[pairKey(a, b)] = 1; bridgeStats.materialized++;
    if (topicOf(a) >= 0 && topicOf(a) == topicOf(b)) bridgeStats.trueBridge++; else bridgeStats.falseBridge++;
  }
  void process(int node){
    g.t++; g.seen[node]++; g.relaxNode(node); for (int c : g.win) { g.relaxNode(c); g.touch(node, randomize ? randomEndpoint(node,c) : c); }
    g.closePlaquettes(node);
    g.win.push_back(node); if ((int)g.win.size() > WIN) g.win.pop_front();
    Evo e = evolve(g, node, mem[node]); updates++; maxOps = std::max(maxOps, e.ops);
    maxPhaseSpeed = std::max(maxPhaseSpeed, e.phaseSpeed);
    if (e.horizon) { horizonSamples++; sumHLin += e.prLin; sumHKer += e.prKer; if (novel[node]) novelHorizons++; else stableHorizons++; }
    tryMaterializeBridge(node, e.horizon);
  }
};

static void run(System& s, int stream, int uniqueEvery){
  std::mt19937 r(11); std::uniform_int_distribution<int> tt(0, TOPICS-1), ww(0, PER-1);
  for (int i = 0; i < stream;) { int tp = tt(r); s.g.win.clear(); for (int b = 0; b < 6 && i < stream; ++b, ++i) { int node = (uniqueEvery > 0 && i % uniqueEvery == 0) ? s.add(true) : tp * PER + ww(r); s.process(node); } }
}

static double eval(System& s){
  std::vector<std::unordered_map<int,cd>> proto(TOPICS);
  for (int tp = 0; tp < TOPICS; ++tp) for (int w = 0; w < PER/2; ++w) for (auto& kv : s.mem[tp*PER+w].ker) proto[tp][kv.first] += kv.second;
  int ok = 0, total = 0; for (int tp = 0; tp < TOPICS; ++tp) for (int w = PER/2; w < PER; ++w) { int node = tp*PER+w, best = -1; double bs = -1; for (int c = 0; c < TOPICS; ++c) { double m = std::abs(gw::fieldOverlap(s.mem[node].ker, proto[c])); if (m > bs) { bs = m; best = c; } } ok += (best == tp); total++; }
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
  double sameBridgeOv = real.bridgeStats.samePairs ? real.bridgeStats.sameOverlap / real.bridgeStats.samePairs : 0.0;
  double crossBridgeOv = real.bridgeStats.crossPairs ? real.bridgeStats.crossOverlap / real.bridgeStats.crossPairs : 0.0;
  std::printf("=== STREAMING COMPRESSION: nonlinear physics always on ===\n");
  std::printf("stream=%d uniqueEvery=%d field_updates=%lld nodes=%d horizons=%lld novel_horizons=%lld\n", stream, uniqueEvery, real.updates, (int)real.g.adj.size(), real.stableHorizons, real.novelHorizons);
  std::printf("horizon PR linear=%.2f nonlinear=%.2f compression=%.2fx\n", real.horizonSamples ? real.sumHLin/real.horizonSamples : 0, real.horizonSamples ? real.sumHKer/real.horizonSamples : 0, comp);
  std::printf("bridges total=%lld true=%lld false=%lld overlap same=%.3f cross=%.3f max_same=%.3f max_cross=%.3f\n",
              real.bridgeStats.materialized, real.bridgeStats.trueBridge, real.bridgeStats.falseBridge,
              sameBridgeOv, crossBridgeOv, real.bridgeStats.maxSame, real.bridgeStats.maxCross);
  std::printf("value REAL=%.1f%% RANDOM=%.1f%% max_bond_visits=%lld max_phase_speed=%.3f\n", realAcc, randomAcc, real.maxOps, real.maxPhaseSpeed);
  std::printf("train_sec=%.2f tokens_per_sec=%.0f peak_ram_mb=%.0f\n", realSec, stream/std::max(realSec,1e-9), peakRamMb());
  bool ok = real.updates == stream && real.stableHorizons > 0 && real.novelHorizons == 0 && comp > 1.35 && realAcc > randomAcc + 20.0 && real.maxOps <= 64LL * TOPK * 4 && real.maxPhaseSpeed < 8.0;
  return ok ? 0 : 1;
}
