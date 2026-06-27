// graph_wave_overlap_bridge_contract_test
// ----------------------------------------------------------------------------
// In-situ bridge materialization by field interference.
//
// Drift is intentionally absent here. A bridge candidate is accepted only when
// the two hub memories show fringe visibility on the same support:
//   coherence(A,B) = |sum conj(A_i) B_i| / sqrt(power(A) power(B)).
//
// The topic labels below are never used to create a bridge; they are only the
// test oracle that counts true/false materializations after the physics has
// spoken.
// ----------------------------------------------------------------------------
#define NOMINMAX
#include "graph_wave_substrate.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <random>
#include <unordered_map>
#include <vector>

namespace {
using gw::cd;
using gw::Vec;

constexpr int kTopics = 3;
constexpr int kPerTopic = 24;
constexpr int kBaseNodes = kTopics * kPerTopic;
constexpr int kStream = 30000;
constexpr int kUniqueEvery = 7;
constexpr int kWindow = 4;
constexpr int kTopK = 6;
constexpr int kBridgeTopK = 1;
constexpr double kEta = 1.0;
constexpr double kDecay = 0.99;
constexpr double kEdgeFloor = 1e-3;
constexpr double kWMin = 8.0;
constexpr double kInject = 0.35;
constexpr double kBridgeMinCoherence = 0.20;
constexpr int kSMin = 4;
constexpr int kSteps = 2;
constexpr double kDt = 0.3;
constexpr double kG = 7.0;

struct Edge {
  double w = 0.0;
  double phase = 0.0;
  int last = 0;
};

struct Graph {
  std::vector<std::unordered_map<int, Edge>> adj;
  std::vector<std::unordered_map<int, Edge>> bridge;
  std::vector<int> seen;
  std::deque<int> win;
  int t = 0;

  int add() {
    adj.push_back({});
    bridge.push_back({});
    seen.push_back(0);
    return (int)adj.size() - 1;
  }

  double incident(int a) const {
    double s = 0.0;
    for (const auto& kv : adj[a]) s += kv.second.w;
    return s;
  }

  double meanBond(int a) const {
    return adj[a].empty() ? 0.0 : incident(a) / (double)adj[a].size();
  }

  static double wrapPhase(double x) {
    while (x > gw::kPi) x -= 2.0 * gw::kPi;
    while (x < -gw::kPi) x += 2.0 * gw::kPi;
    return x;
  }

  void decay(int a, int b) {
    auto it = adj[a].find(b);
    if (it == adj[a].end()) return;
    int dt = t - it->second.last;
    if (dt > 0) {
      it->second.w *= std::pow(kDecay, dt);
      it->second.last = t;
      auto back = adj[b].find(a);
      if (back != adj[b].end()) back->second = it->second;
    }
  }

  void eraseEdge(int a, int b) {
    adj[a].erase(b);
    adj[b].erase(a);
  }

  void eraseBridge(int a, int b) {
    bridge[a].erase(b);
    bridge[b].erase(a);
  }

  bool hasEdge(int a, int b) const {
    return adj[a].find(b) != adj[a].end();
  }

  bool hasBridge(int a, int b) const {
    return bridge[a].find(b) != bridge[a].end();
  }

  int bridgeDegree(int a) const {
    return (int)bridge[a].size();
  }

  double orientedPhaseFrom(const std::vector<std::unordered_map<int, Edge>>& plane,
                           int u, int v) const {
    auto it = plane[u].find(v);
    if (it == plane[u].end()) return 0.0;
    return u < v ? it->second.phase : -it->second.phase;
  }

  double orientedPhase(int u, int v) const {
    return orientedPhaseFrom(adj, u, v);
  }

  double orientedBridgePhase(int u, int v) const {
    return orientedPhaseFrom(bridge, u, v);
  }

  void addSignedPhase(int u, int v, double dphi) {
    auto it = adj[u].find(v);
    if (it == adj[u].end()) return;
    Edge e = it->second;
    const double sign = u < v ? 1.0 : -1.0;
    e.phase = std::clamp(wrapPhase(e.phase + sign * dphi), -1.4, 1.4);
    adj[u][v] = e;
    adj[v][u] = e;
  }

  void addPlaquetteFlux(int a, int b, int c) {
    constexpr double dphi = 0.018;
    addSignedPhase(a, b, dphi / 3.0);
    addSignedPhase(b, c, dphi / 3.0);
    addSignedPhase(c, a, dphi / 3.0);
  }

  void closePlaquettes(int node) {
    for (int j = 1; j < (int)win.size(); ++j) {
      for (int i = 0; i < j; ++i) {
        int a = win[i], b = win[j];
        if (hasEdge(a, b) && hasEdge(a, node) && hasEdge(b, node)) addPlaquetteFlux(a, b, node);
      }
    }
  }

  void relaxNode(int a) {
    std::vector<int> dead;
    for (auto& kv : adj[a]) {
      int b = kv.first;
      int dt = t - kv.second.last;
      if (dt > 0) {
        kv.second.w *= std::pow(kDecay, dt);
        kv.second.last = t;
        auto back = adj[b].find(a);
        if (back != adj[b].end()) back->second = kv.second;
      }
      if (kv.second.w < kEdgeFloor) dead.push_back(b);
    }
    for (int b : dead) eraseEdge(a, b);
  }

  void prune(int a) {
    while ((int)adj[a].size() > kTopK) {
      int weakest = -1;
      double ww = 1e100;
      for (const auto& kv : adj[a]) {
        if (kv.second.w < ww) {
          ww = kv.second.w;
          weakest = kv.first;
        }
      }
      eraseEdge(a, weakest);
    }
  }

  void pruneBridge(int a) {
    while ((int)bridge[a].size() > kBridgeTopK) {
      int weakest = -1;
      double ww = 1e100;
      for (const auto& kv : bridge[a]) {
        if (kv.second.w < ww) {
          ww = kv.second.w;
          weakest = kv.first;
        }
      }
      eraseBridge(a, weakest);
    }
  }

  void setBridge(int a, int b, double w, double phase_ab) {
    if (a == b) return;
    Edge e;
    e.w = w;
    e.phase = wrapPhase(a < b ? phase_ab : -phase_ab);
    e.last = t;
    bridge[a][b] = e;
    bridge[b][a] = e;
    pruneBridge(a);
    pruneBridge(b);
  }

  void touch(int a, int b) {
    if (a == b) return;
    decay(a, b);
    decay(b, a);
    adj[a][b].w += kEta;
    adj[a][b].last = t;
    adj[b][a] = adj[a][b];
    if ((int)adj[a].size() > kTopK) prune(a);
    if ((int)adj[b].size() > kTopK) prune(b);
  }
};

struct Memory {
  std::unordered_map<int, cd> lin;
  std::unordered_map<int, cd> ker;
  std::unordered_map<int, double> sense;
  double prLin = 1.0;
  double prKer = 1.0;
};

struct Update {
  bool horizon = false;
  bool stableDense = false;
  double prLin = 1.0;
  double prKer = 1.0;
  double maxPhaseSpeed = 0.0;
  long long ops = 0;
};

struct BridgeStats {
  long long materialized = 0;
  long long trueBridge = 0;
  long long falseBridge = 0;
  double sameOverlap = 0.0;
  double crossOverlap = 0.0;
  int samePairs = 0;
  int crossPairs = 0;
  double maxSame = 0.0;
  double maxCross = 0.0;
};

unsigned hash3(int a, int b, int c) {
  unsigned x = 2166136261u;
  x = (x ^ (unsigned)a) * 16777619u;
  x = (x ^ (unsigned)(b + 0x9e3779b9u)) * 16777619u;
  x = (x ^ (unsigned)(c * 2654435761u)) * 16777619u;
  return x;
}

int topicOf(int node) {
  return node >= 0 && node < kBaseNodes ? node / kPerTopic : -1;
}

std::vector<int> hood(const Graph& g, int src, bool useBridges) {
  std::vector<int> nodes = {src};
  std::unordered_map<int, int> seen;
  seen[src] = 0;
  for (int hop = 0; hop < 2; ++hop) {
    int sz = (int)nodes.size();
    for (int h = 0; h < sz; ++h) {
      int u = nodes[h];
      for (const auto& kv : g.adj[u]) {
        if (!seen.count(kv.first)) {
          seen[kv.first] = (int)nodes.size();
          nodes.push_back(kv.first);
        }
      }
    }
  }
  if (useBridges) {
    for (const auto& kv : g.bridge[src]) {
      if (!seen.count(kv.first)) {
        seen[kv.first] = (int)nodes.size();
        nodes.push_back(kv.first);
      }
    }
  }
  return nodes;
}

Vec project(const std::unordered_map<int, cd>& f, const std::vector<int>& nodes,
            const std::unordered_map<int, int>& idx, int src) {
  Vec psi(nodes.size(), cd(0, 0));
  for (const auto& kv : f) {
    auto it = idx.find(kv.first);
    if (it != idx.end()) psi[it->second] += kv.second;
  }
  psi[idx.at(src)] += cd(kInject, 0);
  gw::normalizeInPlace(psi);
  return psi;
}

std::unordered_map<int, cd> unproject(const std::vector<int>& nodes, const Vec& psi) {
  std::unordered_map<int, cd> out;
  for (int i = 0; i < (int)nodes.size(); ++i) {
    if (std::norm(psi[i]) > 1e-14) out[nodes[i]] = psi[i];
  }
  return out;
}

std::unordered_map<int, double> feelMotion(const std::vector<int>& nodes,
                                           const Vec& before,
                                           const Vec& after) {
  std::unordered_map<int, double> out;
  for (int i = 0; i < (int)nodes.size(); ++i) {
    const double move = std::abs(after[i] - before[i]);
    const double phase =
        std::abs(std::arg(std::conj(before[i]) * after[i])) *
        std::sqrt(std::norm(before[i]) * std::norm(after[i]));
    const double sense = move + phase;
    if (sense > 1e-14) out[nodes[i]] = sense;
  }
  return out;
}

cd feltOverlap(const Memory& a, const Memory& b) {
  cd z(0, 0);
  for (const auto& kv : a.ker) {
    auto ib = b.ker.find(kv.first);
    if (ib == b.ker.end()) continue;
    auto sa = a.sense.find(kv.first);
    auto sb = b.sense.find(kv.first);
    if (sa == a.sense.end() || sb == b.sense.end()) continue;
    const double live = std::sqrt(sa->second * sb->second);
    z += std::conj(kv.second) * ib->second * live;
  }
  return z;
}

double feltPower(const Memory& m) {
  double s = 0.0;
  for (const auto& kv : m.ker) {
    auto it = m.sense.find(kv.first);
    if (it != m.sense.end()) s += std::norm(kv.second) * it->second;
  }
  return s;
}

double feltCoherence(const Memory& a, const Memory& b, cd* raw_overlap = nullptr) {
  cd z = feltOverlap(a, b);
  if (raw_overlap) *raw_overlap = z;
  return std::abs(z) / (std::sqrt(feltPower(a) * feltPower(b)) + 1e-300);
}

std::vector<gw::SparseBond> bondsInLightCone(const Graph& g, const std::vector<int>& nodes,
                                             const std::unordered_map<int, int>& idx,
                                             bool useBridges) {
  std::vector<gw::SparseBond> bonds;
  for (int i = 0; i < (int)nodes.size(); ++i) {
    int u = nodes[i];
    for (const auto& kv : g.adj[u]) {
      auto it = idx.find(kv.first);
      if (it != idx.end() && i < it->second) {
        bonds.push_back({i, it->second, kv.second.w, g.orientedPhase(u, kv.first)});
      }
    }
    if (useBridges) {
      for (const auto& kv : g.bridge[u]) {
        auto it = idx.find(kv.first);
        if (it != idx.end() && i < it->second) {
          bonds.push_back({i, it->second, kv.second.w, g.orientedBridgePhase(u, kv.first)});
        }
      }
    }
  }
  return bonds;
}

Update evolve(Graph& g, int src, Memory& mem, bool useBridges) {
  Update u;
  std::vector<int> nodes = hood(g, src, useBridges);
  int n = (int)nodes.size();
  if (n < 3) {
    mem.lin[src] = cd(1, 0);
    mem.ker[src] = cd(1, 0);
    return u;
  }
  std::unordered_map<int, int> idx;
  for (int i = 0; i < n; ++i) idx[nodes[i]] = i;
  std::vector<gw::SparseBond> bonds = bondsInLightCone(g, nodes, idx, useBridges);
  Vec lin = project(mem.lin, nodes, idx, src);
  Vec ker = project(mem.ker, nodes, idx, src);
  Vec kerBefore;
  if (src < kBaseNodes) kerBefore = ker;
  gw::LocalFlowStats kerStats;
  gw::edgeLocalKerrFlowPair(lin, ker, bonds, kDt, kG, kSteps, &kerStats);
  gw::normalizeInPlace(lin);
  gw::normalizeInPlace(ker);
  mem.lin = unproject(nodes, lin);
  mem.ker = unproject(nodes, ker);
  if (src < kBaseNodes) mem.sense = feelMotion(nodes, kerBefore, ker);
  else mem.sense.clear();
  mem.prLin = u.prLin = gw::participationRatio(lin);
  mem.prKer = u.prKer = gw::participationRatio(ker);
  u.ops = (long long)bonds.size() * kSteps * 2 + kerStats.bond_visits;
  u.maxPhaseSpeed = kerStats.max_bond_speed;
  u.stableDense = ((int)g.adj[src].size() >= kTopK) && g.incident(src) >= kWMin &&
                  g.seen[src] >= kSMin;
  u.horizon = u.stableDense && u.prKer < 0.5 * u.prLin;
  return u;
}

struct System {
  Graph g;
  std::vector<Memory> mem;
  std::vector<bool> novel;
  std::vector<bool> horizonHub;
  std::vector<unsigned char> measuredPair;
  std::vector<unsigned char> bridgePair;
  BridgeStats bridgeStats;
  bool randomize = false;
  bool bridgeEnabled = false;
  long long updates = 0;
  long long horizons = 0;
  long long novelHorizons = 0;
  long long maxOps = 0;
  double maxPhaseSpeed = 0.0;
  double sumHLin = 0.0;
  double sumHKer = 0.0;

  System(bool randomEdges, bool enableBridge) : randomize(randomEdges), bridgeEnabled(enableBridge) {
    for (int i = 0; i < kBaseNodes; ++i) add(false);
    measuredPair.assign(kBaseNodes * kBaseNodes, 0);
    bridgePair.assign(kBaseNodes * kBaseNodes, 0);
  }

  int add(bool isNovel) {
    int id = g.add();
    mem.push_back({});
    novel.push_back(isNovel);
    horizonHub.push_back(false);
    return id;
  }

  int randomEndpoint(int a, int b) const {
    int n = (int)g.adj.size();
    int r = (int)(hash3(a, b, g.t) % (unsigned)n);
    return r == a ? (r + 1) % n : r;
  }

  int pairKey(int a, int b) const {
    if (a > b) std::swap(a, b);
    return a * kBaseNodes + b;
  }

  void observeCandidate(int a, int b, double coh) {
    int key = pairKey(a, b);
    if (measuredPair[key]) return;
    measuredPair[key] = 1;
    if (topicOf(a) == topicOf(b)) {
      bridgeStats.sameOverlap += coh;
      bridgeStats.samePairs++;
      bridgeStats.maxSame = std::max(bridgeStats.maxSame, coh);
    } else {
      bridgeStats.crossOverlap += coh;
      bridgeStats.crossPairs++;
      bridgeStats.maxCross = std::max(bridgeStats.maxCross, coh);
    }
  }

  double bridgeImpedance(int a, int b, double coh) const {
    return std::sqrt(g.meanBond(a) * g.meanBond(b)) * coh * coh;
  }

  bool pressureCompatible(int a, int b, double w, double phaseAB) const {
    std::vector<int> nodes = hood(g, a, true);
    std::unordered_map<int, int> idx;
    for (int i = 0; i < (int)nodes.size(); ++i) idx[nodes[i]] = i;
    std::vector<int> hb = hood(g, b, true);
    for (int x : hb) {
      if (!idx.count(x)) {
        idx[x] = (int)nodes.size();
        nodes.push_back(x);
      }
    }
    if (!idx.count(a) || !idx.count(b)) return false;
    std::vector<gw::SparseBond> bonds = bondsInLightCone(g, nodes, idx, true);
    std::vector<gw::SparseBond> bridged = bonds;
    int ia = idx[a], ib = idx[b];
    bridged.push_back({ia, ib, w, ia < ib ? phaseAB : -phaseAB});

    Vec psi(nodes.size(), cd(0, 0));
    for (const auto& kv : mem[a].ker) {
      auto it = idx.find(kv.first);
      if (it != idx.end()) psi[it->second] += kv.second;
    }
    for (const auto& kv : mem[b].ker) {
      auto it = idx.find(kv.first);
      if (it != idx.end()) psi[it->second] += kv.second;
    }
    if (gw::power(psi) < 1e-300) psi[ia] = cd(1, 0);
    gw::normalizeInPlace(psi);

    Vec noBridge = gw::edgeLocalKerrFlow(psi, bonds, kDt, kG, kSteps);
    Vec withBridge = gw::edgeLocalKerrFlow(psi, bridged, kDt, kG, kSteps);
    return gw::participationRatio(withBridge) <= gw::participationRatio(noBridge) + 1e-9;
  }

  int bestPartner(int a, cd* bestOverlap, double* bestCoh, bool observe) {
    int best = -1;
    cd zBest(0, 0);
    double cBest = kBridgeMinCoherence;
    for (int b = 0; b < kBaseNodes; ++b) {
      if (b == a || !horizonHub[b] || g.bridgeDegree(b) >= kBridgeTopK) continue;
      int key = pairKey(a, b);
      if (bridgePair[key] || g.hasBridge(a, b)) continue;
      cd z(0, 0);
      const double coh = feltCoherence(mem[a], mem[b], &z);
      if (observe) observeCandidate(a, b, coh);
      if (coh > cBest) {
        best = b;
        cBest = coh;
        zBest = z;
      }
    }
    if (bestOverlap) *bestOverlap = zBest;
    if (bestCoh) *bestCoh = cBest;
    return best;
  }

  void tryMaterializeBridge(int a, bool horizon) {
    if (!bridgeEnabled || !horizon || a >= kBaseNodes || novel[a]) return;
    horizonHub[a] = true;
    if (g.bridgeDegree(a) >= kBridgeTopK) return;
    cd bestOverlap(0, 0);
    double bestCoh = 0.0;
    int best = bestPartner(a, &bestOverlap, &bestCoh, true);
    if (best >= 0) {
      cd reciprocalOverlap(0, 0);
      double reciprocalCoh = 0.0;
      const int reciprocal = bestPartner(best, &reciprocalOverlap, &reciprocalCoh, false);
      if (reciprocal != a) return;
      const double w = bridgeImpedance(a, best, bestCoh);
      if (w <= 1e-12) return;
      const double phaseAB = -std::arg(bestOverlap);
      if (!pressureCompatible(a, best, w, phaseAB)) return;
      g.setBridge(a, best, w, phaseAB);
      bridgePair[pairKey(a, best)] = 1;
      bridgeStats.materialized++;
      if (topicOf(a) >= 0 && topicOf(a) == topicOf(best)) bridgeStats.trueBridge++;
      else bridgeStats.falseBridge++;
    }
  }

  void process(int node) {
    g.t++;
    g.seen[node]++;
    g.relaxNode(node);
    for (int c : g.win) {
      g.relaxNode(c);
      g.touch(node, randomize ? randomEndpoint(node, c) : c);
    }
    g.closePlaquettes(node);
    g.win.push_back(node);
    if ((int)g.win.size() > kWindow) g.win.pop_front();

    Update u = evolve(g, node, mem[node], false);
    updates++;
    maxOps = std::max(maxOps, u.ops);
    maxPhaseSpeed = std::max(maxPhaseSpeed, u.maxPhaseSpeed);
    if (u.horizon) {
      horizons++;
      sumHLin += u.prLin;
      sumHKer += u.prKer;
      if (novel[node]) novelHorizons++;
    }
    tryMaterializeBridge(node, u.horizon);
  }
};

void run(System& s) {
  std::mt19937 r(11);
  std::uniform_int_distribution<int> tt(0, kTopics - 1), ww(0, kPerTopic - 1);
  for (int i = 0; i < kStream;) {
    int tp = tt(r);
    s.g.win.clear();
    for (int b = 0; b < 6 && i < kStream; ++b, ++i) {
      int node = (kUniqueEvery > 0 && i % kUniqueEvery == 0) ? s.add(true) : tp * kPerTopic + ww(r);
      s.process(node);
    }
  }
}

double eval(const System& s) {
  std::vector<std::unordered_map<int, cd>> proto(kTopics);
  for (int tp = 0; tp < kTopics; ++tp) {
    for (int w = 0; w < kPerTopic / 2; ++w) {
      int node = tp * kPerTopic + w;
      for (const auto& kv : s.mem[node].ker) proto[tp][kv.first] += kv.second;
    }
  }
  int ok = 0, total = 0;
  for (int tp = 0; tp < kTopics; ++tp) {
    for (int w = kPerTopic / 2; w < kPerTopic; ++w) {
      int node = tp * kPerTopic + w;
      int best = -1;
      double bs = -1.0;
      for (int c = 0; c < kTopics; ++c) {
        double score = std::abs(gw::fieldOverlap(s.mem[node].ker, proto[c]));
        if (score > bs) {
          bs = score;
          best = c;
        }
      }
      ok += (best == tp);
      total++;
    }
  }
  return 100.0 * ok / total;
}

bool firstBridgePair(const System& s, int* a, int* b) {
  for (int i = 0; i < kBaseNodes; ++i) {
    for (const auto& kv : s.g.bridge[i]) {
      if (i < kv.first && topicOf(i) == topicOf(kv.first)) {
        *a = i;
        *b = kv.first;
        return true;
      }
    }
  }
  return false;
}

double targetPowerAfter(const Graph& g, int src, int dst, bool useBridge, int steps) {
  std::vector<int> nodes = hood(g, src, useBridge);
  std::unordered_map<int, int> idx;
  for (int i = 0; i < (int)nodes.size(); ++i) idx[nodes[i]] = i;
  auto itDst = idx.find(dst);
  if (itDst == idx.end()) return 0.0;
  std::vector<gw::SparseBond> bonds = bondsInLightCone(g, nodes, idx, useBridge);
  Vec psi(nodes.size(), cd(0, 0));
  psi[idx[src]] = cd(1, 0);
  psi = gw::edgeLocalKerrFlow(psi, bonds, kDt, kG, steps);
  return std::norm(psi[itDst->second]);
}

bool report(const char* name, bool ok) {
  std::printf("   => %s\n", ok ? "PASS" : "FAIL");
  if (!ok) std::printf("   !! %s\n", name);
  return ok;
}
}  // namespace

int main() {
  auto t0 = std::chrono::steady_clock::now();
  System base(false, false);
  System bridged(false, true);
  System randomBridged(true, true);
  run(base);
  run(bridged);
  run(randomBridged);
  auto t1 = std::chrono::steady_clock::now();

  BridgeStats ov = bridged.bridgeStats;
  const double sameMean = ov.samePairs ? ov.sameOverlap / ov.samePairs : 0.0;
  const double crossMean = ov.crossPairs ? ov.crossOverlap / ov.crossPairs : 0.0;
  const double baseComp = base.sumHKer > 0.0 ? base.sumHLin / base.sumHKer : 0.0;
  const double bridgeComp = bridged.sumHKer > 0.0 ? bridged.sumHLin / bridged.sumHKer : 0.0;
  const double baseAcc = eval(base);
  const double bridgeAcc = eval(bridged);
  const double randomAcc = eval(randomBridged);
  int a = -1, b = -1;
  const bool hasPair = firstBridgePair(bridged, &a, &b);
  const double pNo = hasPair ? targetPowerAfter(bridged.g, a, b, false, 8) : 0.0;
  const double pBr = hasPair ? targetPowerAfter(bridged.g, a, b, true, 8) : 0.0;
  const double sec = std::chrono::duration<double>(t1 - t0).count();

  std::printf("=====================================================================\n");
  std::printf("  OVERLAP BRIDGE CONTRACT  (drift-free in-situ bridge filter)\n");
  std::printf("=====================================================================\n");
  std::printf("  overlap mean: same=%.4f cross=%.4f  max_same=%.4f max_cross=%.4f\n",
              sameMean, crossMean, ov.maxSame, ov.maxCross);
  std::printf("  bridges: real total=%lld true=%lld false=%lld | random total=%lld false_by_label=%lld\n",
              bridged.bridgeStats.materialized, bridged.bridgeStats.trueBridge,
              bridged.bridgeStats.falseBridge, randomBridged.bridgeStats.materialized,
              randomBridged.bridgeStats.falseBridge);
  std::printf("  compression: base=%.2fx bridged=%.2fx | recognition base=%.1f%% bridged=%.1f%% random=%.1f%%\n",
              baseComp, bridgeComp, baseAcc, bridgeAcc, randomAcc);
  if (hasPair) {
    std::printf("  reach sample: %d(topic%d) -> %d(topic%d), power no_bridge=%.3e bridged=%.3e gain=%.1fx\n",
                a, topicOf(a), b, topicOf(b), pNo, pBr, pBr / (pNo + 1e-300));
  }
  std::printf("  work: max_ops base=%lld bridged=%lld max_phase_speed=%.3f elapsed=%.2fs\n",
              base.maxOps, bridged.maxOps, bridged.maxPhaseSpeed, sec);

  int pass = 0, total = 0;
  ++total;
  pass += report("same-topic overlap separates from cross-topic overlap",
                 sameMean > 0.05 && ov.maxSame > kBridgeMinCoherence &&
                     crossMean < 0.01 && ov.maxCross < 0.02);
  ++total;
  pass += report("bridge materialization has no cross-topic false positives",
                 bridged.bridgeStats.trueBridge > 0 && bridged.bridgeStats.falseBridge == 0);
  ++total;
  pass += report("bridge improves physical reach without readout shortcut",
                 hasPair && pBr > 10.0 * std::max(pNo, 1e-300) && pBr > 1e-6);
  ++total;
  pass += report("densification survives bridge materialization",
                 bridgeComp > 1.35 && bridgeComp > 0.9 * baseComp);
  ++total;
  pass += report("semantic value survives bridge materialization",
                 bridgeAcc >= baseAcc - 1e-9 && bridgeAcc > randomAcc + 20.0);
  ++total;
  pass += report("work remains bounded by the local bridged light cone",
                 bridged.maxOps <= 4 * std::max(base.maxOps, 1LL) && bridged.maxPhaseSpeed < 8.0);

  std::printf("\n  RESULT : %d / %d verified\n", pass, total);
  return pass == total ? 0 : 1;
}
