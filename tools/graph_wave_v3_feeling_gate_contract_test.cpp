// graph_wave_v3_feeling_gate_contract_test
// ----------------------------------------------------------------------------
// GNNv3 candidate gate without trigonometric readout.
//
// Phase is not stored as an angle. It is carried as a local two-component
// carrier. Edges carry a gauge carrier. The gate does not call angle functions
// and does not half-wave-rectify phase; it observes local current/stress activity
// already present between transported carriers.
// ----------------------------------------------------------------------------
#define NOMINMAX

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr int kTopics = 3;
constexpr int kPerTopic = 24;
constexpr int kBaseNodes = kTopics * kPerTopic;
constexpr int kDefaultStream = 60000;
constexpr int kUniqueEvery = 7;
constexpr int kMaxNodes = 200000;
constexpr int kTopK = 6;
constexpr int kWindow = 4;
constexpr int kMaxLocal = 96;
constexpr int kMaxPairs = 384;
constexpr int kSMin = 4;
constexpr double kEta = 1.0;
constexpr double kDecay = 0.99;
constexpr double kEdgeFloor = 1e-4;
constexpr double kWMin = 8.0;
constexpr double kInject = 0.08;
constexpr double kDt = 0.012;
constexpr double kPsiG = 7.0;
constexpr double kChiG = 30.0;
constexpr double kLeak = 0.9955;
constexpr double kTauDecay = 0.97;
constexpr double kTauLearn = 0.03;
constexpr double kFeelInject = 0.025;
constexpr double kApertureDecay = 0.985;
constexpr double kApertureLearn = 0.015;

enum FieldKind { kLin = 0, kPsi = 1, kChi = 2 };

struct Carrier {
  double x = 1.0;
  double y = 0.0;
};

struct Field {
  double rho = 1e-9;
  Carrier q;
};

struct Edge {
  int to = -1;
  double w = 0.0;
  Carrier gauge;
  int last = 0;
};

struct Node {
  Edge e[kTopK];
  int deg = 0;
  int seen = 0;
  int topic = -1;
  bool novel = false;
  Field lin;
  Field psi;
  Field chi;
  double tauRho = 0.0;
  Carrier tau;
  double aperture = 0.0;
};

struct Pair {
  int ia = 0;
  int ib = 0;
  int a = 0;
  int b = 0;
  double w = 0.0;
  Carrier gaugeAB;
};

double clampAbs(double x, double cap) {
  return std::max(-cap, std::min(cap, x));
}

Carrier normalize(Carrier c) {
  double n = std::sqrt(c.x * c.x + c.y * c.y);
  if (n < 1e-300) return {1.0, 0.0};
  return {c.x / n, c.y / n};
}

Carrier conjCarrier(Carrier a) {
  return {a.x, -a.y};
}

Carrier transport(Carrier gauge, Carrier q) {
  return {gauge.x * q.x - gauge.y * q.y, gauge.x * q.y + gauge.y * q.x};
}

double dotCarrier(Carrier a, Carrier b) {
  return a.x * b.x + a.y * b.y;
}

double crossCarrier(Carrier a, Carrier b) {
  return a.x * b.y - a.y * b.x;
}

double carrierDistance(Carrier a, Carrier b) {
  double dx = a.x - b.x, dy = a.y - b.y;
  return std::sqrt(dx * dx + dy * dy);
}

void pushCarrier(Carrier& q, double torque) {
  Carrier old = q;
  q.x += -torque * old.y;
  q.y += torque * old.x;
  q = normalize(q);
}

void blendCarrier(Carrier& q, Carrier target, double alpha) {
  q = normalize({(1.0 - alpha) * q.x + alpha * target.x,
                 (1.0 - alpha) * q.y + alpha * target.y});
}

unsigned hash3(int a, int b, int c) {
  unsigned x = 2166136261u;
  x = (x ^ (unsigned)a) * 16777619u;
  x = (x ^ (unsigned)(b + 0x9e3779b9u)) * 16777619u;
  x = (x ^ (unsigned)(c * 2654435761u)) * 16777619u;
  return x;
}

Carrier seededCarrier(int a, int b, int salt) {
  unsigned h1 = hash3(a, b, salt);
  unsigned h2 = hash3(b, a, salt + 0x6d2b79u);
  double x = ((int)(h1 % 2001u) - 1000) / 1000.0;
  double y = ((int)(h2 % 2001u) - 1000) / 1000.0;
  return normalize({x, y});
}

struct Metrics {
  long long updates = 0;
  long long stable = 0;
  long long horizons = 0;
  long long novelHorizons = 0;
  long long totalPairs = 0;
  long long activePairs = 0;
  double sumLinPr = 0.0;
  double sumPsiPr = 0.0;
  double sumChiPr = 0.0;
  double sumTau = 0.0;
  double sumAperture = 0.0;
  double maxRho = 0.0;
  double sameEdgeAct = 0.0;
  double crossEdgeAct = 0.0;
  double activeCurrentAbs = 0.0;
  double activeCurrentSigned = 0.0;
  double skippedCurrentAbs = 0.0;
  double skippedCurrentSigned = 0.0;
  double activeStressAbs = 0.0;
  double activeStressSigned = 0.0;
  double skippedStressAbs = 0.0;
  double skippedStressSigned = 0.0;
  long long skippedPairs = 0;
  int sameN = 0;
  int crossN = 0;
};

struct System {
  Node* node = nullptr;
  int count = 0;
  int t = 0;
  int win[kWindow];
  int winN = 0;
  bool reversePairs = false;
  bool randomEdges = false;
  bool feelingGate = false;
  Metrics m;

  System(bool reverse, bool randomize, bool gate)
      : node(new Node[kMaxNodes]), reversePairs(reverse), randomEdges(randomize),
        feelingGate(gate) {
    for (int tp = 0; tp < kTopics; ++tp) {
      for (int w = 0; w < kPerTopic; ++w) add(tp, false);
    }
  }

  ~System() { delete[] node; }

  int add(int topic, bool novel) {
    if (count >= kMaxNodes) {
      std::printf("  !! node capacity exceeded (%d)\n", kMaxNodes);
      std::exit(2);
    }
    int id = count++;
    node[id].topic = topic;
    node[id].novel = novel;
    Carrier q = seededCarrier(id, topic + 1, 7);
    node[id].lin.q = q;
    node[id].psi.q = q;
    node[id].chi.q = q;
    node[id].tau = q;
    return id;
  }

  Field& field(int id, FieldKind k) {
    if (k == kLin) return node[id].lin;
    if (k == kPsi) return node[id].psi;
    return node[id].chi;
  }

  const Field& field(int id, FieldKind k) const {
    if (k == kLin) return node[id].lin;
    if (k == kPsi) return node[id].psi;
    return node[id].chi;
  }

  int findEdge(int a, int b) const {
    for (int i = 0; i < node[a].deg; ++i) {
      if (node[a].e[i].to == b) return i;
    }
    return -1;
  }

  void removeDirected(int a, int b) {
    int s = findEdge(a, b);
    if (s < 0) return;
    node[a].e[s] = node[a].e[node[a].deg - 1];
    node[a].deg--;
  }

  int slotFor(int a, int b) {
    int s = findEdge(a, b);
    if (s >= 0) return s;
    if (node[a].deg < kTopK) return node[a].deg++;
    int weakest = 0;
    for (int i = 1; i < kTopK; ++i) {
      if (node[a].e[i].w < node[a].e[weakest].w) weakest = i;
    }
    int old = node[a].e[weakest].to;
    if (old >= 0) removeDirected(old, a);
    return weakest;
  }

  void decayEdge(int a, int slot) {
    Edge& e = node[a].e[slot];
    int dt = t - e.last;
    if (dt > 0) {
      e.w *= std::pow(kDecay, dt);
      e.last = t;
      int back = findEdge(e.to, a);
      if (back >= 0) {
        node[e.to].e[back].w = e.w;
        node[e.to].e[back].last = t;
      }
    }
  }

  void relaxNode(int a) {
    for (int i = 0; i < node[a].deg;) {
      decayEdge(a, i);
      int b = node[a].e[i].to;
      if (node[a].e[i].w < kEdgeFloor) {
        removeDirected(b, a);
        removeDirected(a, b);
      } else {
        ++i;
      }
    }
  }

  void touchDirected(int a, int b, Carrier gauge) {
    int s = slotFor(a, b);
    node[a].e[s].to = b;
    node[a].e[s].w += kEta;
    node[a].e[s].gauge = gauge;
    node[a].e[s].last = t;
  }

  void touch(int a, int b) {
    if (a == b) return;
    if (randomEdges) {
      int r = (int)(hash3(a, b, t) % (unsigned)count);
      b = r == a ? (r + 1) % count : r;
    }
    relaxNode(a);
    relaxNode(b);
    Carrier g = seededCarrier(std::min(a, b), std::max(a, b), 0x51u);
    touchDirected(a, b, a < b ? g : conjCarrier(g));
    touchDirected(b, a, b < a ? g : conjCarrier(g));
  }

  Carrier orientedGauge(int a, const Edge& e) const {
    return e.gauge;
  }

  bool contains(const int* ids, int n, int id) const {
    for (int i = 0; i < n; ++i) {
      if (ids[i] == id) return true;
    }
    return false;
  }

  int localIndex(const int* ids, int n, int id) const {
    for (int i = 0; i < n; ++i) {
      if (ids[i] == id) return i;
    }
    return -1;
  }

  int gatherLocal(int src, int* ids) const {
    int n = 0;
    ids[n++] = src;
    for (int i = 0; i < node[src].deg && n < kMaxLocal; ++i) {
      int v = node[src].e[i].to;
      if (v >= 0 && !contains(ids, n, v)) ids[n++] = v;
    }
    for (int scan = 1; scan < n && n < kMaxLocal; ++scan) {
      int u = ids[scan];
      for (int i = 0; i < node[u].deg && n < kMaxLocal; ++i) {
        int v = node[u].e[i].to;
        if (v >= 0 && !contains(ids, n, v)) ids[n++] = v;
      }
    }
    return n;
  }

  int gatherPairs(const int* ids, int n, Pair* pairs) const {
    int pc = 0;
    for (int i = 0; i < n; ++i) {
      int a = ids[i];
      for (int s = 0; s < node[a].deg && pc < kMaxPairs; ++s) {
        const Edge& e = node[a].e[s];
        int j = localIndex(ids, n, e.to);
        if (j > i) pairs[pc++] = {i, j, a, e.to, e.w, orientedGauge(a, e)};
      }
    }
    return pc;
  }

  void canonicalizePairs(Pair* pairs, int pc) const {
    if (reversePairs) {
      for (int i = 0; i < pc / 2; ++i) std::swap(pairs[i], pairs[pc - 1 - i]);
    }
    std::sort(pairs, pairs + pc, [](const Pair& x, const Pair& y) {
      int xa = std::min(x.a, x.b), xb = std::max(x.a, x.b);
      int ya = std::min(y.a, y.b), yb = std::max(y.a, y.b);
      if (xa != ya) return xa < ya;
      return xb < yb;
    });
  }

  double edgeActivity(Carrier qa, Carrier qb, Carrier gauge, double ra, double rb) const {
    Carrier tb = transport(gauge, qb);
    double root = std::sqrt(std::max(0.0, ra * rb));
    double current = std::abs(crossCarrier(qa, tb));
    double stress = std::abs(dotCarrier(qa, tb));
    return (root / (1.0 + root)) * (0.70 * current + 0.30 * stress);
  }

  double chiActivity(const Pair& e) const {
    const Field& a = node[e.a].chi;
    const Field& b = node[e.b].chi;
    return edgeActivity(a.q, b.q, e.gaugeAB, a.rho, b.rho);
  }

  double tauActivity(const Pair& e) const {
    return edgeActivity(node[e.a].tau, node[e.b].tau, e.gaugeAB,
                        node[e.a].tauRho, node[e.b].tauRho);
  }

  double apertureActivity(int id) const {
    double a = node[id].aperture;
    return a / (1.0 + a);
  }

  void observePhaseLoad(const Pair& e, bool active) {
    const Field& a = node[e.a].psi;
    const Field& b = node[e.b].psi;
    Carrier tb = transport(e.gaugeAB, b.q);
    double root = std::sqrt(std::max(0.0, a.rho * b.rho));
    double current = 2.0 * e.w * root * crossCarrier(a.q, tb);
    double stress = e.w * root * dotCarrier(a.q, tb);
    if (active) {
      m.activeCurrentAbs += std::abs(current);
      m.activeCurrentSigned += current;
      m.activeStressAbs += std::abs(stress);
      m.activeStressSigned += stress;
    } else {
      m.skippedCurrentAbs += std::abs(current);
      m.skippedCurrentSigned += current;
      m.skippedStressAbs += std::abs(stress);
      m.skippedStressSigned += stress;
      m.skippedPairs++;
    }
  }

  int selectActivePairs(const Pair* pairs, int pc, Pair* active, int src) {
    if (!feelingGate) {
      for (int i = 0; i < pc; ++i) {
        active[i] = pairs[i];
        observePhaseLoad(pairs[i], true);
      }
      m.totalPairs += pc;
      m.activePairs += pc;
      return pc;
    }
    int ac = 0;
    for (int i = 0; i < pc; ++i) {
      const Pair& e = pairs[i];
      bool sourceEdge = e.a == src || e.b == src;
      double edgeFeel = chiActivity(e) + tauActivity(e);
      double aperture = std::sqrt(std::max(0.0, apertureActivity(e.a) * apertureActivity(e.b)));
      bool felt = edgeFeel > aperture;
      bool activePair = sourceEdge || felt;
      observePhaseLoad(e, activePair);
      if (activePair) active[ac++] = e;
    }
    if (ac == 0 && pc > 0) {
      active[ac++] = pairs[0];
    }
    m.totalPairs += pc;
    m.activePairs += ac;
    return ac;
  }

  void inject(int src) {
    node[src].lin.rho += kInject / (1.0 + node[src].lin.rho);
    node[src].psi.rho += kInject / (1.0 + node[src].psi.rho);
  }

  void flowStep(const int* ids, int n, const Pair* pairs, int pc, FieldKind kind, double g) {
    double r0[kMaxLocal], dr[kMaxLocal], torque[kMaxLocal];
    Carrier q0[kMaxLocal];
    for (int i = 0; i < n; ++i) {
      const Field& f = field(ids[i], kind);
      r0[i] = f.rho;
      q0[i] = f.q;
      dr[i] = 0.0;
      torque[i] = -0.25 * g * r0[i] * kDt;
    }
    for (int kk = 0; kk < pc; ++kk) {
      const Pair& e = pairs[kk];
      int a = e.ia, b = e.ib;
      Carrier tb = transport(e.gaugeAB, q0[b]);
      double root = std::sqrt(std::max(0.0, r0[a] * r0[b]));
      double current = 2.0 * e.w * root * crossCarrier(q0[a], tb);
      if (g > 0.0) current += 0.060 * g * e.w * (r0[a] - r0[b]);
      double flow = kDt * current;
      double cap = flow >= 0.0 ? 0.06 * r0[b] : 0.06 * r0[a];
      flow = clampAbs(flow, cap + 1e-18);
      dr[a] += flow;
      dr[b] -= flow;

      double stress = dotCarrier(q0[a], tb);
      double ca = -0.03 * kDt * e.w * stress *
                  std::sqrt((r0[b] + 1e-12) / (r0[a] + 1e-12));
      double cb = -0.03 * kDt * e.w * stress *
                  std::sqrt((r0[a] + 1e-12) / (r0[b] + 1e-12));
      torque[a] += clampAbs(ca, 0.08);
      torque[b] += clampAbs(cb, 0.08);
    }
    for (int i = 0; i < n; ++i) {
      Field& f = field(ids[i], kind);
      f.rho = std::max(0.0, kLeak * r0[i] + dr[i]);
      f.q = q0[i];
      pushCarrier(f.q, torque[i]);
      m.maxRho = std::max(m.maxRho, f.rho);
    }
  }

  void updateTauAndChi(const int* ids, int n, const Pair* pairs, int pc,
                       const double* beforeR, const Carrier* beforeQ) {
    double drive[kMaxLocal];
    Carrier target[kMaxLocal];
    for (int i = 0; i < n; ++i) {
      const Field& p = node[ids[i]].psi;
      double motion = std::abs(p.rho - beforeR[i]);
      double phaseMotion = carrierDistance(p.q, beforeQ[i]) *
                           std::sqrt(std::max(0.0, p.rho * beforeR[i]));
      drive[i] = motion + phaseMotion;
      target[i] = {drive[i] * p.q.x, drive[i] * p.q.y};
    }
    for (int kk = 0; kk < pc; ++kk) {
      const Pair& e = pairs[kk];
      const Field& a = node[e.a].psi;
      const Field& b = node[e.b].psi;
      Carrier tb = transport(e.gaugeAB, b.q);
      Carrier ta = transport(conjCarrier(e.gaugeAB), a.q);
      double root = std::sqrt(std::max(0.0, a.rho * b.rho));
      double current = std::abs(crossCarrier(a.q, tb));
      double stress = std::abs(dotCarrier(a.q, tb));
      double tension = e.w * root * (0.70 * current + 0.30 * stress);
      drive[e.ia] += 0.5 * tension;
      drive[e.ib] += 0.5 * tension;
      target[e.ia].x += tension * tb.x;
      target[e.ia].y += tension * tb.y;
      target[e.ib].x += tension * ta.x;
      target[e.ib].y += tension * ta.y;
    }
    for (int i = 0; i < n; ++i) {
      Node& nd = node[ids[i]];
      nd.tauRho = kTauDecay * nd.tauRho + kTauLearn * drive[i];
      nd.aperture = kApertureDecay * nd.aperture + kApertureLearn * drive[i];
      blendCarrier(nd.tau, normalize(target[i]), 0.22);
      double pulse = kFeelInject * nd.tauRho / (1.0 + nd.tauRho);
      nd.chi.rho += pulse;
      blendCarrier(nd.chi.q, nd.tau, 0.35);
      m.sumTau += nd.tauRho;
      m.sumAperture += nd.aperture;
    }
  }

  double localPr(const int* ids, int n, FieldKind kind) const {
    double s1 = 0.0, s2 = 0.0;
    for (int i = 0; i < n; ++i) {
      double r = field(ids[i], kind).rho;
      s1 += r;
      s2 += r * r;
    }
    return (s1 * s1) / (s2 + 1e-300);
  }

  double incident(int a) const {
    double s = 0.0;
    for (int i = 0; i < node[a].deg; ++i) s += node[a].e[i].w;
    return s;
  }

  void physicalTick(int src) {
    int ids[kMaxLocal];
    Pair all[kMaxPairs], active[kMaxPairs];
    int n = gatherLocal(src, ids);
    int pc = gatherPairs(ids, n, all);
    canonicalizePairs(all, pc);
    int ac = selectActivePairs(all, pc, active, src);

    double beforeR[kMaxLocal];
    Carrier beforeQ[kMaxLocal];
    for (int i = 0; i < n; ++i) {
      beforeR[i] = node[ids[i]].psi.rho;
      beforeQ[i] = node[ids[i]].psi.q;
    }

    inject(src);
    flowStep(ids, n, active, ac, kLin, 0.0);
    flowStep(ids, n, active, ac, kPsi, kPsiG);
    updateTauAndChi(ids, n, all, pc, beforeR, beforeQ);
    flowStep(ids, n, active, ac, kChi, kChiG);

    bool stable = node[src].deg >= kTopK && incident(src) >= kWMin && node[src].seen >= kSMin;
    if (stable) {
      double prLin = localPr(ids, n, kLin);
      double prPsi = localPr(ids, n, kPsi);
      double prChi = localPr(ids, n, kChi);
      m.stable++;
      m.sumLinPr += prLin;
      m.sumPsiPr += prPsi;
      m.sumChiPr += prChi;
      if (prPsi < 0.75 * prLin) {
        m.horizons++;
        if (node[src].novel) m.novelHorizons++;
      }
    }
  }

  void process(int id) {
    t++;
    m.updates++;
    node[id].seen++;
    relaxNode(id);
    for (int i = 0; i < winN; ++i) touch(id, win[i]);
    if (winN < kWindow) {
      win[winN++] = id;
    } else {
      for (int i = 1; i < kWindow; ++i) win[i - 1] = win[i];
      win[kWindow - 1] = id;
    }
    physicalTick(id);
  }

  void clearWindow() { winN = 0; }

  void finalizeCoherence() {
    for (int a = 0; a < kBaseNodes; ++a) {
      for (int b = a + 1; b < kBaseNodes; ++b) {
        double act = 0.0;
        int es = findEdge(a, b);
        if (es >= 0) {
          const Edge& e = node[a].e[es];
          const Field& fa = node[a].chi;
          const Field& fb = node[b].chi;
          act = e.w * edgeActivity(fa.q, fb.q, orientedGauge(a, e), fa.rho, fb.rho);
        }
        if (node[a].topic == node[b].topic) {
          m.sameEdgeAct += act;
          m.sameN++;
        } else {
          m.crossEdgeAct += act;
          m.crossN++;
        }
      }
    }
    if (m.sameN) m.sameEdgeAct /= m.sameN;
    if (m.crossN) m.crossEdgeAct /= m.crossN;
  }
};

void run(System& s, int stream) {
  unsigned rng = 11u;
  auto next = [&]() {
    rng = 1664525u * rng + 1013904223u;
    return rng;
  };
  for (int i = 0; i < stream;) {
    int tp = (int)(next() % kTopics);
    s.clearWindow();
    for (int b = 0; b < 6 && i < stream; ++b, ++i) {
      int node = tp * kPerTopic + (int)(next() % kPerTopic);
      if (kUniqueEvery > 0 && i % kUniqueEvery == 0) node = s.add(tp, true);
      s.process(node);
    }
  }
  s.finalizeCoherence();
}

double syncError(const System& a, const System& b) {
  double e = 0.0;
  int n = std::min(a.count, b.count);
  for (int i = 0; i < n; ++i) {
    const Field& ap = a.node[i].psi;
    const Field& bp = b.node[i].psi;
    const Field& ac = a.node[i].chi;
    const Field& bc = b.node[i].chi;
    e += std::abs(ap.rho - bp.rho) + std::abs(ac.rho - bc.rho);
    e += carrierDistance(ap.q, bp.q) * std::sqrt(ap.rho + bp.rho + 1e-300);
    e += carrierDistance(ac.q, bc.q) * std::sqrt(ac.rho + bc.rho + 1e-300);
  }
  return e / (double)std::max(1, n);
}

struct Result {
  const char* name = "";
  double prLin = 0.0;
  double prPsi = 0.0;
  double prChi = 0.0;
  double psiComp = 0.0;
  double chiComp = 0.0;
  double tauMean = 0.0;
  double apertureMean = 0.0;
  double maxRho = 0.0;
  double activeFrac = 0.0;
  double sameAct = 0.0;
  double crossAct = 0.0;
  double activeCurrentAbs = 0.0;
  double activeCurrentBias = 0.0;
  double skippedCurrentAbs = 0.0;
  double skippedCurrentBias = 0.0;
  double activeStressAbs = 0.0;
  double activeStressBias = 0.0;
  double skippedStressAbs = 0.0;
  double skippedStressBias = 0.0;
  long long stable = 0;
  long long horizons = 0;
  long long novel = 0;
  long long skippedPairs = 0;
};

Result summarize(const char* name, const System& s) {
  Result r;
  r.name = name;
  r.stable = s.m.stable;
  r.horizons = s.m.horizons;
  r.novel = s.m.novelHorizons;
  r.prLin = s.m.stable ? s.m.sumLinPr / s.m.stable : 0.0;
  r.prPsi = s.m.stable ? s.m.sumPsiPr / s.m.stable : 0.0;
  r.prChi = s.m.stable ? s.m.sumChiPr / s.m.stable : 0.0;
  r.psiComp = r.prPsi > 0.0 ? r.prLin / r.prPsi : 0.0;
  r.chiComp = r.prChi > 0.0 ? r.prLin / r.prChi : 0.0;
  r.tauMean = s.m.updates ? s.m.sumTau / s.m.updates : 0.0;
  r.apertureMean = s.m.updates ? s.m.sumAperture / s.m.updates : 0.0;
  r.maxRho = s.m.maxRho;
  r.activeFrac = s.m.totalPairs ? (double)s.m.activePairs / (double)s.m.totalPairs : 0.0;
  r.sameAct = s.m.sameEdgeAct;
  r.crossAct = s.m.crossEdgeAct;
  r.activeCurrentAbs = s.m.activeCurrentAbs;
  r.activeCurrentBias = std::abs(s.m.activeCurrentSigned) / (s.m.activeCurrentAbs + 1e-300);
  r.skippedCurrentAbs = s.m.skippedCurrentAbs;
  r.skippedCurrentBias = std::abs(s.m.skippedCurrentSigned) / (s.m.skippedCurrentAbs + 1e-300);
  r.activeStressAbs = s.m.activeStressAbs;
  r.activeStressBias = std::abs(s.m.activeStressSigned) / (s.m.activeStressAbs + 1e-300);
  r.skippedStressAbs = s.m.skippedStressAbs;
  r.skippedStressBias = std::abs(s.m.skippedStressSigned) / (s.m.skippedStressAbs + 1e-300);
  r.skippedPairs = s.m.skippedPairs;
  return r;
}

void printResult(const Result& r) {
  std::printf("  [%s]\n", r.name);
  std::printf("    stable=%lld horizons=%lld novel=%lld active_pairs=%.3f\n",
              r.stable, r.horizons, r.novel, r.activeFrac);
  std::printf("    PR lin=%.3f psi=%.3f chi=%.3f compression psi=%.2fx chi=%.2fx\n",
              r.prLin, r.prPsi, r.prChi, r.psiComp, r.chiComp);
  std::printf("    tau_mean=%.6f aperture_mean=%.6f max_rho=%.3f chi_edge_activity same=%.6f cross=%.6f\n",
              r.tauMean, r.apertureMean, r.maxRho, r.sameAct, r.crossAct);
  std::printf("    phase_audit active |J|=%.3e bias=%.3f |S|=%.3e bias=%.3f\n",
              r.activeCurrentAbs, r.activeCurrentBias, r.activeStressAbs, r.activeStressBias);
  std::printf("    phase_audit skipped |J|=%.3e bias=%.3f |S|=%.3e bias=%.3f pairs=%lld\n",
              r.skippedCurrentAbs, r.skippedCurrentBias, r.skippedStressAbs,
              r.skippedStressBias, r.skippedPairs);
}

bool report(const char* name, bool ok) {
  std::printf("   => %s  %s\n", ok ? "PASS" : "FAIL", name);
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  int stream = argc > 1 ? std::atoi(argv[1]) : kDefaultStream;
  std::printf("=====================================================================\n");
  std::printf("  GNNv3 FEELING-GATED CARRIER-FIELD CONTRACT  (no trig gate)\n");
  std::printf("=====================================================================\n");
  std::printf("  stream=%d adaptive_aperture=yes\n", stream);

  System full(false, false, false);
  System gated(false, false, true);
  System gatedReverse(true, false, true);
  System random(false, true, true);
  run(full, stream);
  run(gated, stream);
  run(gatedReverse, stream);
  run(random, std::min(stream, 200000));

  Result rf = summarize("full physics control", full);
  Result rg = summarize("feeling-gated physics", gated);
  Result rr = summarize("matched-random gated control", random);
  double err = syncError(gated, gatedReverse);
  printResult(rf);
  printResult(rg);
  printResult(rr);
  std::printf("  gated_sync_error=%.6e\n", err);
  std::printf("  delta gated-full: psi_comp=%+.3f chi_comp=%+.3f active=%+.3f\n",
              rg.psiComp - rf.psiComp, rg.chiComp - rf.chiComp, rg.activeFrac - rf.activeFrac);

  int pass = 0, total = 0;
  ++total;
  pass += report("feeling gate keeps synchronous physics", err < 1e-9);
  ++total;
  pass += report("feeling gate reduces heavy edge-flow interactions", rg.activeFrac < 0.75);
  ++total;
  pass += report("gated psi still densifies", rg.psiComp > 1.5 && rg.psiComp > 0.55 * rf.psiComp);
  ++total;
  pass += report("gated chi remains denser than gated psi", rg.chiComp > rg.psiComp);
  ++total;
  pass += report("gated feeling is real, not matched-random", rg.tauMean > 2.0 * rr.tauMean);
  ++total;
  pass += report("gated field stays bounded", rg.maxRho < 50.0);
  ++total;
  pass += report("gated chi edge activity separates real support", rg.sameAct > rg.crossAct + 1e-6);
  ++total;
  pass += report("gate does not one-sidedly clip signed current",
                 rg.skippedCurrentBias < 0.10 && rg.skippedStressBias < 0.10);

  std::printf("  CONTRACT RESULT : %d / %d verified\n", pass, total);
  std::printf("\n  RESULT : %s\n", pass == total ? "PASS" : "FAIL");
  return pass == total ? 0 : 1;
}
