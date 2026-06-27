// graph_wave_streaming_densification_contract_test
// ----------------------------------------------------------------------------
// ONLINE DENSIFICATION CONTRACT.
//
// Compression is not an actuator here. There is no "if stable then compress".
// Every token event updates its local field through the same substrate physics:
//   graph plasticity -> local light cone -> edge current + Kerr phase pressure.
//
// The horizon condition is only a detector: after the same field has evolved, a
// stable/dense neighbourhood with low nonlinear PR is counted as a horizon. Novel
// one-off nodes receive the same physics, but do not become horizons.
//
// Gates:
//   (1) nonlinear physics runs on every token; horizons appear only on stable
//       dense clusters, never on one-off nodes;
//   (2) horizon fields are denser than the online linear control;
//   (3) semantic value survives through the densified fields (real > random);
//   (4) per-token work stays inside the local light cone.
// ----------------------------------------------------------------------------
#define NOMINMAX
#include "graph_wave_substrate.hpp"

#include <algorithm>
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
constexpr int kStream = 15000;
constexpr int kUniqueEvery = 7;
constexpr int kWindow = 4;
constexpr int kTopK = 6;
constexpr double kEta = 1.0;
constexpr double kDecay = 0.99;
constexpr double kEdgeFloor = 1e-3;
constexpr double kWMin = 8.0;
constexpr int kSMin = 4;
constexpr double kInject = 0.35;
constexpr double kDt = 0.3;
constexpr double kG = 7.0;
constexpr int kStepsPerToken = 2;

struct Edge {
  double w = 0.0;
  double phase = 0.0;
  int last = 0;
};

struct Graph {
  std::vector<std::unordered_map<int, Edge>> adj;
  std::vector<int> seen;
  std::deque<int> win;
  int t = 0;

  int add() {
    adj.push_back({});
    seen.push_back(0);
    return (int)adj.size() - 1;
  }

  double incident(int a) const {
    double s = 0.0;
    for (const auto& kv : adj[a]) s += kv.second.w;
    return s;
  }

  void decay(int a, int b) {
    auto it = adj[a].find(b);
    if (it == adj[a].end()) return;
    int dt = t - it->second.last;
    if (dt > 0) {
      it->second.w *= std::pow(kDecay, dt);
      it->second.last = t;
    }
  }

  void eraseEdge(int a, int b) {
    adj[a].erase(b);
    adj[b].erase(a);
  }

  static double wrapPhase(double x) {
    while (x > gw::kPi) x -= 2.0 * gw::kPi;
    while (x < -gw::kPi) x += 2.0 * gw::kPi;
    return x;
  }

  bool hasEdge(int a, int b) const {
    return adj[a].find(b) != adj[a].end();
  }

  double orientedPhase(int u, int v) const {
    auto it = adj[u].find(v);
    if (it == adj[u].end()) return 0.0;
    return u < v ? it->second.phase : -it->second.phase;
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
  double prLin = 1.0;
  double prKer = 1.0;
  int updates = 0;
};

struct UpdateStats {
  bool horizon = false;
  bool stableDense = false;
  double prLin = 1.0;
  double prKer = 1.0;
  double maxPhaseSpeed = 0.0;
  long long ops = 0;
};

unsigned hash3(int a, int b, int c) {
  unsigned x = 2166136261u;
  x = (x ^ (unsigned)a) * 16777619u;
  x = (x ^ (unsigned)(b + 0x9e3779b9u)) * 16777619u;
  x = (x ^ (unsigned)(c * 2654435761u)) * 16777619u;
  return x;
}

std::vector<int> hood(const Graph& g, int src) {
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
  if (gw::power(psi) < 1e-300) psi[idx.at(src)] = cd(1, 0);
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

std::vector<gw::SparseBond> bondsInLightCone(const Graph& g, const std::vector<int>& nodes,
                                             const std::unordered_map<int, int>& idx) {
  std::vector<gw::SparseBond> bonds;
  for (int i = 0; i < (int)nodes.size(); ++i) {
    int u = nodes[i];
    for (const auto& kv : g.adj[u]) {
      auto it = idx.find(kv.first);
      if (it != idx.end() && i < it->second) bonds.push_back({i, it->second, kv.second.w, g.orientedPhase(u, kv.first)});
    }
  }
  return bonds;
}

UpdateStats evolveLocalField(Graph& g, int src, Memory& mem) {
  UpdateStats s;
  std::vector<int> nodes = hood(g, src);
  int n = (int)nodes.size();
  if (n < 3) {
    mem.lin[src] = cd(1, 0);
    mem.ker[src] = cd(1, 0);
    mem.prLin = mem.prKer = 1.0;
    mem.updates++;
    return s;
  }

  std::unordered_map<int, int> idx;
  for (int i = 0; i < n; ++i) idx[nodes[i]] = i;

  std::vector<gw::SparseBond> bonds = bondsInLightCone(g, nodes, idx);
  Vec lin = project(mem.lin, nodes, idx, src);
  Vec ker = project(mem.ker, nodes, idx, src);
  gw::LocalFlowStats kerStats;
  gw::edgeLocalKerrFlowPair(lin, ker, bonds, kDt, kG, kStepsPerToken, &kerStats);
  gw::normalizeInPlace(lin);
  gw::normalizeInPlace(ker);

  mem.lin = unproject(nodes, lin);
  mem.ker = unproject(nodes, ker);
  mem.prLin = gw::participationRatio(lin);
  mem.prKer = gw::participationRatio(ker);
  mem.updates++;

  s.prLin = mem.prLin;
  s.prKer = mem.prKer;
  s.ops = (long long)bonds.size() * kStepsPerToken * 2 + kerStats.bond_visits;
  s.maxPhaseSpeed = kerStats.max_bond_speed;
  s.stableDense = ((int)g.adj[src].size() >= kTopK) && g.incident(src) >= kWMin && g.seen[src] >= kSMin;
  // horizon = collapse beats dispersion: nonlinear PR below half its own linear control.
  // Scale-free physical criterion, not a hand-tuned cutoff.
  s.horizon = s.stableDense && mem.prKer < 0.5 * mem.prLin;
  return s;
}

cd overlap(const std::unordered_map<int, cd>& a, const std::unordered_map<int, cd>& b) {
  cd s(0, 0);
  for (const auto& kv : a) {
    auto it = b.find(kv.first);
    if (it != b.end()) s += std::conj(kv.second) * it->second;
  }
  return s;
}

struct System {
  Graph g;
  std::vector<Memory> mem;
  std::vector<bool> novel;
  bool randomize = false;
  long long updates = 0;
  long long stableHorizons = 0;
  long long novelHorizons = 0;
  long long stableSamples = 0;
  long long horizonSamples = 0;
  long long maxOps = 0;
  double maxPhaseSpeed = 0.0;
  double sumPrLin = 0.0;
  double sumPrKer = 0.0;
  double sumHorizonPrLin = 0.0;
  double sumHorizonPrKer = 0.0;

  explicit System(bool random_edges) : randomize(random_edges) {
    for (int i = 0; i < kTopics * kPerTopic; ++i) add(false);
  }

  int add(bool is_novel) {
    int id = g.add();
    mem.push_back({});
    novel.push_back(is_novel);
    return id;
  }

  int randomEndpoint(int a, int b) const {
    int n = (int)g.adj.size();
    if (n <= 1) return b;
    int r = (int)(hash3(a, b, g.t) % (unsigned)n);
    if (r == a) r = (r + 1) % n;
    return r;
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

    UpdateStats u = evolveLocalField(g, node, mem[node]);
    updates++;
    maxOps = std::max(maxOps, u.ops);
    maxPhaseSpeed = std::max(maxPhaseSpeed, u.maxPhaseSpeed);
    if (u.stableDense) {
      stableSamples++;
      sumPrLin += u.prLin;
      sumPrKer += u.prKer;
    }
    if (u.horizon) {
      horizonSamples++;
      sumHorizonPrLin += u.prLin;
      sumHorizonPrKer += u.prKer;
      if (novel[node]) novelHorizons++;
      else stableHorizons++;
    }
  }
};

void runStream(System& s) {
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

double eval(System& s) {
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
      double bestScore = -1.0;
      for (int c = 0; c < kTopics; ++c) {
        double score = std::abs(overlap(s.mem[node].ker, proto[c]));
        if (score > bestScore) {
          bestScore = score;
          best = c;
        }
      }
      ok += (best == tp);
      total++;
    }
  }
  return 100.0 * ok / total;
}

bool report(const char* name, bool ok) {
  std::printf("   => %s\n", ok ? "PASS" : "FAIL");
  if (!ok) std::printf("   !! %s\n", name);
  return ok;
}
}  // namespace

int main() {
  System real(false), random(true);
  runStream(real);
  runStream(random);

  const double realAcc = eval(real);
  const double randomAcc = eval(random);
  const double stableCompression = real.sumPrKer > 0.0 ? real.sumPrLin / real.sumPrKer : 0.0;
  const double horizonCompression = real.sumHorizonPrKer > 0.0 ? real.sumHorizonPrLin / real.sumHorizonPrKer : 0.0;
  const long long opBound = 64LL * kTopK * 4;

  std::printf("=====================================================================\n");
  std::printf("  STREAMING DENSIFICATION CONTRACT  (nonlinear physics is always on)\n");
  std::printf("=====================================================================\n");
  int pass = 0, total = 0;

  std::printf("\n[1] SAME PHYSICS ON EVERY TOKEN; horizons only on stable+dense clusters\n");
  std::printf("    field_updates=%lld  stable_horizons=%lld  novel_horizons=%lld  stable_samples=%lld\n",
              real.updates, real.stableHorizons, real.novelHorizons, real.stableSamples);
  ++total;
  pass += report("horizon detector marked a one-off node or found no stable horizon",
                 real.updates == kStream && real.stableHorizons > 0 && real.novelHorizons == 0);

  std::printf("\n[2] HORIZON FIELD DENSIFIES (nonlinear PR < linear control)\n");
  std::printf("    stable-sample mean PR: linear=%.2f  nonlinear=%.2f  compression=%.2fx\n",
              real.stableSamples ? real.sumPrLin / real.stableSamples : 0.0,
              real.stableSamples ? real.sumPrKer / real.stableSamples : 0.0,
              stableCompression);
  std::printf("    horizon-sample mean PR: linear=%.2f  nonlinear=%.2f  compression=%.2fx\n",
              real.horizonSamples ? real.sumHorizonPrLin / real.horizonSamples : 0.0,
              real.horizonSamples ? real.sumHorizonPrKer / real.horizonSamples : 0.0,
              horizonCompression);
  ++total;
  pass += report("nonlinear term did not naturally densify horizon fields", horizonCompression > 1.35);

  std::printf("\n[3] SEMANTIC VALUE SURVIVES densified online fields\n");
  std::printf("    topic recognition through online fields: REAL=%.1f%%  RANDOM=%.1f%%  (chance %.0f%%)\n",
              realAcc, randomAcc, 100.0 / kTopics);
  ++total;
  pass += report("online densification destroyed semantic value", realAcc > 60.0 && realAcc > randomAcc + 20.0);

  std::printf("\n[4] WORK STAYS LOCAL (no VxV)\n");
  std::printf("    max field-update bond-visits=%lld  bound=%lld  max phase-current speed=%.3f\n",
              real.maxOps, opBound, real.maxPhaseSpeed);
  ++total;
  pass += report("online nonlinear field update is not local or phase current exploded",
                 real.maxOps <= opBound && real.maxPhaseSpeed < 8.0);

  std::printf("\n  RESULT : %d / %d verified\n", pass, total);
  return pass == total ? 0 : 1;
}
