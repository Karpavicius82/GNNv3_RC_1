// graph_wave_nonlinear_engine.hpp
// ----------------------------------------------------------------------------
// The CLOSED nonlinear GNNv2 engine: one pure-physics, polar (rho/phi),
// synchronous substrate carrying three coupled phase fields on a streaming,
// self-pruning plastic graph. Everything assembled in one place.
//
//   lin : linear reference (g=0)         -- the dispersion baseline.
//   psi : energy / content (Kerr g=7)    -- solitonic, localized memory.
//   chi : consolidation / feeling (g=30) -- driven by psi's motion, slow layer.
//   tau : per-node structure sensor      -- EMA of psi motion/tension.
//
// Synchronous tick: the whole local light cone is read from ONE snapshot, then all
// rho/phi changes are applied together. Edge pairs are canonicalized, so traversal
// order is not physics -- the step is exactly order-invariant (reversible).
//
// Pure physics, polar throughout: no complex Vec, no Re/Im, no projection, no
// labels. The engine processes integer token ids only; topic/label structure (if
// any) lives in the driver, never here.
//
// This is the substrate the perf strategy stands on: tau says WHERE structure is
// and the stable/horizon flags say WHEN it has crystallized -- so a later pass can
// skip the expensive nonlinear tick on quiet nodes ("stop only where needed").
// ----------------------------------------------------------------------------
#ifndef GRAPH_WAVE_NONLINEAR_ENGINE_HPP
#define GRAPH_WAVE_NONLINEAR_ENGINE_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace gwn {

// ---- engine configuration (physics + plasticity; NOT stream shape) ----------
struct Config {
  // graph plasticity
  int    topK      = 6;       // max neighbours kept per node
  int    window    = 4;       // co-occurrence window within one context
  double eta       = 1.0;     // edge reinforcement per touch
  double decay     = 0.99;    // edge weight decay per elapsed tick
  double edgeFloor = 1e-4;    // edges below this are pruned
  double wMin      = 8.0;     // incident-weight threshold for "stable"
  int    sMin      = 4;       // times-seen threshold for "stable"
  // local light-cone caps (bounded work per token -> scales)
  int    maxLocal  = 96;
  int    maxPairs  = 384;
  // field physics
  double inject    = 0.08;    // energy injected at the active node
  double dt        = 0.012;
  double psiG      = 7.0;     // psi Kerr nonlinearity
  double chiG      = 30.0;    // chi Kerr nonlinearity (strong self-focusing)
  double leak      = 0.9955;  // gentle dissipation
  double tauDecay  = 0.97;    // tau EMA retention
  double tauLearn  = 0.03;    // tau EMA learning
  double feelInject= 0.025;   // tau -> chi drive gain
  double horizon   = 0.75;    // psi PR < horizon*lin PR  => collapse beats dispersion
  bool   scrambleEdges = false; // diagnostic: wire each touch to a RANDOM node, destroying
                                // topological structure -- the baseline that makes tau collapse,
                                // i.e. the control proving tau senses real structure.
};

constexpr double kPi = 3.141592653589793238462643383279502884;

inline double wrapPhase(double x) {
  while (x >  kPi) x -= 2.0 * kPi;
  while (x < -kPi) x += 2.0 * kPi;
  return x;
}
inline double clampAbs(double x, double cap) { return std::max(-cap, std::min(cap, x)); }

inline unsigned hash3(int a, int b, int c) {
  unsigned x = 2166136261u;
  x = (x ^ (unsigned)a) * 16777619u;
  x = (x ^ (unsigned)(b + 0x9e3779b9u)) * 16777619u;
  x = (x ^ (unsigned)(c * 2654435761u)) * 16777619u;
  return x;
}

// a phase field sample on one node: polar (energy density + phase angle)
struct Field { double rho = 1e-9; double phi = 0.0; };

struct Edge { int to = -1; double w = 0.0; double phase = 0.0; int last = 0; };

enum FieldKind { kLin = 0, kPsi = 1, kChi = 2 };

struct Node {
  Edge   e[8];              // <= topK (topK<=8); fixed array is cache-friendly and bounded
  int    deg  = 0;
  int    seen = 0;
  Field  lin, psi, chi;
  double tauRho = 0.0;      // structure-presence magnitude (the WHERE signal)
  double tauPhi = 0.0;      // consolidated phase target
};

// running health/diagnostics the engine self-reports (no labels)
struct Stats {
  long long updates = 0, stable = 0, horizons = 0;
  double sumLinPr = 0, sumPsiPr = 0, sumChiPr = 0, sumTau = 0, maxRho = 0;
  double psiCompression() const { return sumPsiPr > 0 ? sumLinPr / sumPsiPr : 0.0; }
  double chiCompression() const { return sumChiPr > 0 ? sumLinPr / sumChiPr : 0.0; }
  double meanTau()        const { return updates  > 0 ? sumTau   / updates  : 0.0; }
};

// seeded gauge phase on an edge -- small, deterministic, gauge-invariant content
inline double seededPhase(int a, int b) {
  unsigned h = hash3(std::min(a, b), std::max(a, b), 0x51u);
  return 0.08 * (((h % 2001u) / 2000.0) - 0.5);
}

class Engine {
 public:
  explicit Engine(const Config& cfg = Config(), int capacity = 1 << 20)
      : cfg_(cfg), cap_(capacity), node_(new Node[capacity]) {}
  ~Engine() { delete[] node_; }
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  // grow a fresh node, return its id
  int addNode() {
    int id = count_++;
    double ph = 0.17 * (double)(hash3(id, 7, 11) % 37);
    node_[id].lin.phi = node_[id].psi.phi = node_[id].chi.phi = node_[id].tauPhi = ph;
    return id;
  }

  void beginContext() { winN_ = 0; }      // sentence / burst boundary

  // one token event: graph plasticity then a synchronous field tick
  void process(int id) {
    ++t_; ++stats_.updates; ++node_[id].seen;
    relaxNode(id);
    for (int i = 0; i < winN_; ++i) touch(id, win_[i]);
    if (winN_ < cfg_.window) win_[winN_++] = id;
    else { for (int i = 1; i < cfg_.window; ++i) win_[i - 1] = win_[i]; win_[cfg_.window - 1] = id; }
    physicalTick(id);
  }

  int          nodeCount() const { return count_; }
  const Node&  nodeAt(int i) const { return node_[i]; }
  const Stats& stats() const { return stats_; }
  const Config& config() const { return cfg_; }

  bool isStable(int a) const {
    return node_[a].deg >= cfg_.topK && incident(a) >= cfg_.wMin && node_[a].seen >= cfg_.sMin;
  }

 private:
  Field&       field(int id, FieldKind k) { return k == kLin ? node_[id].lin : (k == kPsi ? node_[id].psi : node_[id].chi); }
  const Field& field(int id, FieldKind k) const { return k == kLin ? node_[id].lin : (k == kPsi ? node_[id].psi : node_[id].chi); }

  double incident(int a) const { double s = 0; for (int i = 0; i < node_[a].deg; ++i) s += node_[a].e[i].w; return s; }
  int findEdge(int a, int b) const { for (int i = 0; i < node_[a].deg; ++i) if (node_[a].e[i].to == b) return i; return -1; }
  void removeDirected(int a, int b) { int s = findEdge(a, b); if (s < 0) return; node_[a].e[s] = node_[a].e[node_[a].deg - 1]; node_[a].deg--; }

  int slotFor(int a, int b) {
    int s = findEdge(a, b); if (s >= 0) return s;
    if (node_[a].deg < cfg_.topK) return node_[a].deg++;
    int weakest = 0;
    for (int i = 1; i < cfg_.topK; ++i) if (node_[a].e[i].w < node_[a].e[weakest].w) weakest = i;
    int old = node_[a].e[weakest].to; if (old >= 0) removeDirected(old, a);
    return weakest;
  }

  void decayEdge(int a, int s) {
    Edge& e = node_[a].e[s]; int dt = t_ - e.last;
    if (dt > 0) { e.w *= std::pow(cfg_.decay, dt); e.last = t_;
      int bk = findEdge(e.to, a); if (bk >= 0) { node_[e.to].e[bk].w = e.w; node_[e.to].e[bk].last = t_; } }
  }
  void relaxNode(int a) {
    for (int i = 0; i < node_[a].deg;) {
      decayEdge(a, i); int b = node_[a].e[i].to;
      if (node_[a].e[i].w < cfg_.edgeFloor) { removeDirected(b, a); removeDirected(a, b); } else ++i;
    }
  }
  void touchDirected(int a, int b, double ph) { int s = slotFor(a, b); node_[a].e[s].to = b; node_[a].e[s].w += cfg_.eta; node_[a].e[s].phase = ph; node_[a].e[s].last = t_; }
  void touch(int a, int b) {
    if (cfg_.scrambleEdges && count_ > 1) { int r = (int)(hash3(a, b, t_) % (unsigned)count_); b = (r == a) ? (r + 1) % count_ : r; }
    if (a == b) return; relaxNode(a); relaxNode(b); double ph = seededPhase(a, b); touchDirected(a, b, ph); touchDirected(b, a, ph);
  }
  double orientedPhase(int a, const Edge& e) const { return a < e.to ? e.phase : -e.phase; }

  // ---- one local light cone, gathered and canonicalized -------------------
  struct Pair { int ia, ib, a, b; double w, phaseAB; };

  int gatherLocal(int src, int* ids) const {
    int n = 0; ids[n++] = src;
    for (int i = 0; i < node_[src].deg && n < cfg_.maxLocal; ++i) { int v = node_[src].e[i].to; if (v >= 0 && !contains(ids, n, v)) ids[n++] = v; }
    for (int scan = 1; scan < n && n < cfg_.maxLocal; ++scan) { int u = ids[scan];
      for (int i = 0; i < node_[u].deg && n < cfg_.maxLocal; ++i) { int v = node_[u].e[i].to; if (v >= 0 && !contains(ids, n, v)) ids[n++] = v; } }
    return n;
  }
  static bool contains(const int* ids, int n, int id) { for (int i = 0; i < n; ++i) if (ids[i] == id) return true; return false; }
  static int  localIndex(const int* ids, int n, int id) { for (int i = 0; i < n; ++i) if (ids[i] == id) return i; return -1; }

  int gatherPairs(const int* ids, int n, Pair* pairs) const {
    int pc = 0;
    for (int i = 0; i < n; ++i) { int a = ids[i];
      for (int s = 0; s < node_[a].deg && pc < cfg_.maxPairs; ++s) {
        const Edge& e = node_[a].e[s]; int j = localIndex(ids, n, e.to);
        if (j > i) pairs[pc++] = {i, j, a, e.to, e.w, orientedPhase(a, e)};
      } }
    return pc;
  }
  // canonical order -> the synchronous step is independent of traversal order
  static void canonicalize(Pair* pairs, int pc) {
    std::sort(pairs, pairs + pc, [](const Pair& x, const Pair& y) {
      int xa = std::min(x.a, x.b), xb = std::max(x.a, x.b), ya = std::min(y.a, y.b), yb = std::max(y.a, y.b);
      return xa != ya ? xa < ya : xb < yb;
    });
  }

  // ---- the synchronous nonlinear flow on one field ------------------------
  void flowStep(const int* ids, int n, const Pair* pairs, int pc, FieldKind kind, double g) {
    double r0[96], p0[96], dr[96], dp[96];
    for (int i = 0; i < n; ++i) { const Field& f = field(ids[i], kind); r0[i] = f.rho; p0[i] = f.phi; dr[i] = 0.0; dp[i] = -0.25 * g * r0[i] * cfg_.dt; }
    for (int kk = 0; kk < pc; ++kk) {
      const Pair& e = pairs[kk]; int a = e.ia, b = e.ib;
      double root  = std::sqrt(std::max(0.0, r0[a] * r0[b]));
      double theta = wrapPhase(p0[b] - p0[a] + e.phaseAB);
      double cur   = 2.0 * e.w * root * std::sin(theta);
      if (g > 0.0) cur += 0.060 * g * e.w * (r0[a] - r0[b]);          // Kerr density pressure
      double flow  = cfg_.dt * cur;
      double cap   = (flow >= 0.0 ? 0.06 * r0[b] : 0.06 * r0[a]) + 1e-18;
      flow = clampAbs(flow, cap);
      dr[a] += flow; dr[b] -= flow;
      double ca = -0.03 * cfg_.dt * e.w * std::cos(theta) * std::sqrt((r0[b] + 1e-12) / (r0[a] + 1e-12));
      double cb = -0.03 * cfg_.dt * e.w * std::cos(theta) * std::sqrt((r0[a] + 1e-12) / (r0[b] + 1e-12));
      dp[a] += clampAbs(ca, 0.08); dp[b] += clampAbs(cb, 0.08);
    }
    for (int i = 0; i < n; ++i) { Field& f = field(ids[i], kind);
      f.rho = std::max(0.0, cfg_.leak * r0[i] + dr[i]); f.phi = wrapPhase(p0[i] + dp[i]);
      stats_.maxRho = std::max(stats_.maxRho, f.rho); }
  }

  // tau senses psi's motion; chi is driven purely by tau (consolidation)
  void updateTauAndChi(const int* ids, int n, const Pair* pairs, int pc, const double* bR, const double* bP) {
    double drive[96], sinA[96], cosA[96];
    for (int i = 0; i < n; ++i) {
      const Field& p = node_[ids[i]].psi;
      double motion = std::abs(p.rho - bR[i]);
      double twist  = std::abs(wrapPhase(p.phi - bP[i])) * std::sqrt(std::max(0.0, p.rho * bR[i]));
      drive[i] = motion + twist; sinA[i] = drive[i] * std::sin(p.phi); cosA[i] = drive[i] * std::cos(p.phi);
    }
    for (int kk = 0; kk < pc; ++kk) {
      const Pair& e = pairs[kk]; const Field& a = node_[e.a].psi; const Field& b = node_[e.b].psi;
      double root = std::sqrt(std::max(0.0, a.rho * b.rho));
      double theta = wrapPhase(b.phi - a.phi + e.phaseAB);
      double tension = e.w * root * std::abs(std::sin(theta));
      drive[e.ia] += 0.5 * tension; drive[e.ib] += 0.5 * tension;
      sinA[e.ia] += tension * std::sin(b.phi + e.phaseAB); cosA[e.ia] += tension * std::cos(b.phi + e.phaseAB);
      sinA[e.ib] += tension * std::sin(a.phi - e.phaseAB); cosA[e.ib] += tension * std::cos(a.phi - e.phaseAB);
    }
    for (int i = 0; i < n; ++i) {
      Node& nd = node_[ids[i]]; double target = std::atan2(sinA[i], cosA[i]);
      nd.tauRho = cfg_.tauDecay * nd.tauRho + cfg_.tauLearn * drive[i];
      nd.tauPhi = wrapPhase(nd.tauPhi + 0.22 * wrapPhase(target - nd.tauPhi));
      double pulse = cfg_.feelInject * nd.tauRho / (1.0 + nd.tauRho);
      nd.chi.rho += pulse;
      nd.chi.phi = wrapPhase(nd.chi.phi + 0.35 * wrapPhase(nd.tauPhi - nd.chi.phi));
      stats_.sumTau += nd.tauRho;
    }
  }

  static double localPr(const Engine& eng, const int* ids, int n, FieldKind k) {
    double s1 = 0, s2 = 0; for (int i = 0; i < n; ++i) { double r = eng.field(ids[i], k).rho; s1 += r; s2 += r * r; }
    return (s1 * s1) / (s2 + 1e-300);
  }

  void physicalTick(int src) {
    int ids[96]; Pair pairs[384];
    int n = gatherLocal(src, ids);
    int pc = gatherPairs(ids, n, pairs);
    canonicalize(pairs, pc);

    double bR[96], bP[96];
    for (int i = 0; i < n; ++i) { bR[i] = node_[ids[i]].psi.rho; bP[i] = node_[ids[i]].psi.phi; }

    // inject content into lin + psi; lin disperses (g=0), psi self-focuses
    node_[src].lin.rho += cfg_.inject / (1.0 + node_[src].lin.rho);
    node_[src].psi.rho += cfg_.inject / (1.0 + node_[src].psi.rho);

    flowStep(ids, n, pairs, pc, kLin, 0.0);
    flowStep(ids, n, pairs, pc, kPsi, cfg_.psiG);
    updateTauAndChi(ids, n, pairs, pc, bR, bP);
    flowStep(ids, n, pairs, pc, kChi, cfg_.chiG);

    if (isStable(src)) {
      double prL = localPr(*this, ids, n, kLin), prP = localPr(*this, ids, n, kPsi), prC = localPr(*this, ids, n, kChi);
      ++stats_.stable; stats_.sumLinPr += prL; stats_.sumPsiPr += prP; stats_.sumChiPr += prC;
      if (prP < cfg_.horizon * prL) ++stats_.horizons;
    }
  }

  Config cfg_;
  int    cap_;
  Node*  node_;
  int    count_ = 0, t_ = 0;
  int    win_[8] = {0}; int winN_ = 0;
  Stats  stats_;
};

}  // namespace gwn
#endif  // GRAPH_WAVE_NONLINEAR_ENGINE_HPP
