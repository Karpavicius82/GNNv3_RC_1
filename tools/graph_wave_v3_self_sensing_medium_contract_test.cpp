// graph_wave_v3_self_sensing_medium_contract_test
// ----------------------------------------------------------------------------
// GNNv3 lower-level self-sensing medium contract.
//
// This test keeps the carrier field in fixed SoA buffers. The active physics
// window is not selected from labels, token ids, or an external score. It is
// read from the field itself: chi/tau carrier activity against local aperture.
//
// The contract is intentionally C++ only and avoids trigonometric readout in
// the gate. Phase is carried by transported two-component carriers; the only
// observed quantities are current, stress, aperture, overlap and participation.
// ----------------------------------------------------------------------------
#define NOMINMAX

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr int kTopics = 3;
constexpr int kPerTopic = 32;
constexpr int kNodes = kTopics * kPerTopic;
constexpr int kMaxEdges = 384;
constexpr int kDefaultStream = 120000;
constexpr int kWarmup = 6000;
constexpr double kInject = 0.075;
constexpr double kDt = 0.010;
constexpr double kLeak = 0.996;
constexpr double kPsiPressure = 6.5;
constexpr double kChiPressure = 24.0;
constexpr double kTauDecay = 0.968;
constexpr double kTauLearn = 0.032;
constexpr double kApertureDecay = 0.986;
constexpr double kApertureLearn = 0.014;
constexpr double kChiInject = 0.021;

struct Carrier {
  double x = 1.0;
  double y = 0.0;
};

struct Stats {
  long long ticks = 0;
  long long activeEdges = 0;
  long long totalEdges = 0;
  long long skippedEdges = 0;
  long long samples = 0;
  double sumLinPr = 0.0;
  double sumPsiPr = 0.0;
  double sumChiPr = 0.0;
  double skippedCurrentAbs = 0.0;
  double skippedCurrentSigned = 0.0;
  double skippedStressAbs = 0.0;
  double skippedStressSigned = 0.0;
  double samePsi = 0.0;
  double crossPsi = 0.0;
  double sameChi = 0.0;
  double crossChi = 0.0;
  int sameN = 0;
  int crossN = 0;
  double maxRho = 0.0;
  double maxTau = 0.0;
};

double clampAbs(double x, double cap) {
  return std::max(-cap, std::min(cap, x));
}

Carrier normalize(Carrier c) {
  const double n = std::sqrt(c.x * c.x + c.y * c.y);
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
  const double dx = a.x - b.x;
  const double dy = a.y - b.y;
  return std::sqrt(dx * dx + dy * dy);
}

void pushCarrier(Carrier& q, double torque) {
  const Carrier old = q;
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
  x = (x ^ static_cast<unsigned>(a)) * 16777619u;
  x = (x ^ static_cast<unsigned>(b + 0x9e3779b9u)) * 16777619u;
  x = (x ^ static_cast<unsigned>(c * 2654435761u)) * 16777619u;
  return x;
}

Carrier seededCarrier(int a, int b, int salt) {
  const unsigned h1 = hash3(a, b, salt);
  const unsigned h2 = hash3(b, a, salt + 0x6d2b79u);
  const double x = (static_cast<int>(h1 % 2001u) - 1000) / 1000.0;
  const double y = (static_cast<int>(h2 % 2001u) - 1000) / 1000.0;
  return normalize({x, y});
}

int topicOf(int id) {
  return id / kPerTopic;
}

struct Medium {
  int edgeA[kMaxEdges];
  int edgeB[kMaxEdges];
  double edgeW[kMaxEdges];
  Carrier edgeGauge[kMaxEdges];
  int edgeN = 0;

  double linRho[kNodes];
  double psiRho[kNodes];
  double chiRho[kNodes];
  double tauRho[kNodes];
  double aperture[kNodes];
  Carrier linQ[kNodes];
  Carrier psiQ[kNodes];
  Carrier chiQ[kNodes];
  Carrier tauQ[kNodes];

  bool sensing = true;
  bool reverseEdges = false;
  Stats st;

  Medium(bool useSensing, bool reverse) : sensing(useSensing), reverseEdges(reverse) {
    for (int i = 0; i < kNodes; ++i) {
      const Carrier q = seededCarrier(i, topicOf(i) + 1, 7);
      linRho[i] = psiRho[i] = chiRho[i] = 1e-9;
      tauRho[i] = aperture[i] = 0.0;
      linQ[i] = psiQ[i] = chiQ[i] = tauQ[i] = q;
    }
    buildEdges();
  }

  bool hasEdge(int a, int b) const {
    const int lo = std::min(a, b);
    const int hi = std::max(a, b);
    for (int e = 0; e < edgeN; ++e) {
      if (edgeA[e] == lo && edgeB[e] == hi) return true;
    }
    return false;
  }

  void addEdge(int a, int b, double w, int salt) {
    if (a == b || edgeN >= kMaxEdges) return;
    const int lo = std::min(a, b);
    const int hi = std::max(a, b);
    if (hasEdge(lo, hi)) return;
    edgeA[edgeN] = lo;
    edgeB[edgeN] = hi;
    edgeW[edgeN] = w;
    edgeGauge[edgeN] = seededCarrier(lo, hi, salt);
    edgeN++;
  }

  void buildEdges() {
    for (int tp = 0; tp < kTopics; ++tp) {
      const int base = tp * kPerTopic;
      for (int i = 0; i < kPerTopic; ++i) {
        addEdge(base + i, base + ((i + 1) % kPerTopic), 1.00, 11);
        addEdge(base + i, base + ((i + 2) % kPerTopic), 0.72, 17);
        if ((i & 1) == 0) addEdge(base + i, base + ((i + 5) % kPerTopic), 0.46, 23);
      }
    }
    for (int tp = 0; tp < kTopics; ++tp) {
      const int aBase = tp * kPerTopic;
      const int bBase = ((tp + 1) % kTopics) * kPerTopic;
      for (int i = 0; i < kPerTopic; i += 8) {
        addEdge(aBase + i, bBase + ((i + 3) % kPerTopic), 0.13, 31);
      }
    }
  }

  int sourceAt(int t) const {
    int tp = (t / 53) % kTopics;
    if ((t % 211) == 0) tp = static_cast<int>(hash3(t, 13, 5) % kTopics);
    const int local = static_cast<int>(hash3(t, tp, 19) % kPerTopic);
    return tp * kPerTopic + local;
  }

  double apertureActivity(int id) const {
    return aperture[id] / (1.0 + aperture[id]);
  }

  double edgeActivity(int e, const double* rho, const Carrier* q) const {
    const int a = edgeA[e];
    const int b = edgeB[e];
    const Carrier tb = transport(edgeGauge[e], q[b]);
    const double root = std::sqrt(std::max(0.0, rho[a] * rho[b]));
    const double current = std::abs(crossCarrier(q[a], tb));
    const double stress = std::abs(dotCarrier(q[a], tb));
    return (root / (1.0 + root)) * (0.70 * current + 0.30 * stress);
  }

  bool sourceEdge(int e, int src) const {
    return edgeA[e] == src || edgeB[e] == src;
  }

  void observeSkippedLoad(int e) {
    const int a = edgeA[e];
    const int b = edgeB[e];
    const Carrier tb = transport(edgeGauge[e], psiQ[b]);
    const double root = std::sqrt(std::max(0.0, psiRho[a] * psiRho[b]));
    const double current = 2.0 * edgeW[e] * root * crossCarrier(psiQ[a], tb);
    const double stress = edgeW[e] * root * dotCarrier(psiQ[a], tb);
    st.skippedCurrentAbs += std::abs(current);
    st.skippedCurrentSigned += current;
    st.skippedStressAbs += std::abs(stress);
    st.skippedStressSigned += stress;
    st.skippedEdges++;
  }

  int selectActive(int src, bool* active) {
    int ac = 0;
    for (int pass = 0; pass < edgeN; ++pass) {
      const int e = reverseEdges ? edgeN - 1 - pass : pass;
      bool on = true;
      if (sensing) {
        const int a = edgeA[e];
        const int b = edgeB[e];
        const double feel = edgeActivity(e, chiRho, chiQ) + edgeActivity(e, tauRho, tauQ);
        const double ap = std::sqrt(std::max(0.0, apertureActivity(a) * apertureActivity(b)));
        on = sourceEdge(e, src) || feel > ap;
      }
      active[e] = on;
      if (on) {
        ac++;
      } else {
        observeSkippedLoad(e);
      }
    }
    if (sensing && ac == 0 && edgeN > 0) {
      active[0] = true;
      ac = 1;
    }
    st.totalEdges += edgeN;
    st.activeEdges += ac;
    return ac;
  }

  void inject(int src) {
    linRho[src] += kInject / (1.0 + linRho[src]);
    psiRho[src] += kInject / (1.0 + psiRho[src]);
  }

  void flowField(double* rho, Carrier* q, const bool* active, double pressure) {
    double r0[kNodes];
    double dr[kNodes];
    double torque[kNodes];
    Carrier q0[kNodes];
    for (int i = 0; i < kNodes; ++i) {
      r0[i] = rho[i];
      q0[i] = q[i];
      dr[i] = 0.0;
      torque[i] = -0.23 * pressure * r0[i] * kDt;
    }
    for (int e = 0; e < edgeN; ++e) {
      if (!active[e]) continue;
      const int a = edgeA[e];
      const int b = edgeB[e];
      const Carrier tb = transport(edgeGauge[e], q0[b]);
      const double root = std::sqrt(std::max(0.0, r0[a] * r0[b]));
      double current = 2.0 * edgeW[e] * root * crossCarrier(q0[a], tb);
      if (pressure > 0.0) current += 0.055 * pressure * edgeW[e] * (r0[a] - r0[b]);
      double flow = kDt * current;
      const double cap = flow >= 0.0 ? 0.055 * r0[b] : 0.055 * r0[a];
      flow = clampAbs(flow, cap + 1e-18);
      dr[a] += flow;
      dr[b] -= flow;

      const double stress = dotCarrier(q0[a], tb);
      const double ca = -0.026 * kDt * edgeW[e] * stress *
                        std::sqrt((r0[b] + 1e-12) / (r0[a] + 1e-12));
      const double cb = -0.026 * kDt * edgeW[e] * stress *
                        std::sqrt((r0[a] + 1e-12) / (r0[b] + 1e-12));
      torque[a] += clampAbs(ca, 0.075);
      torque[b] += clampAbs(cb, 0.075);
    }
    for (int i = 0; i < kNodes; ++i) {
      rho[i] = std::max(0.0, kLeak * r0[i] + dr[i]);
      q[i] = q0[i];
      pushCarrier(q[i], torque[i]);
      st.maxRho = std::max(st.maxRho, rho[i]);
    }
  }

  void updateFeeling(const double* beforeRho, const Carrier* beforeQ) {
    double drive[kNodes];
    Carrier target[kNodes];
    for (int i = 0; i < kNodes; ++i) {
      const double motion = std::abs(psiRho[i] - beforeRho[i]);
      const double phaseMotion = carrierDistance(psiQ[i], beforeQ[i]) *
                                 std::sqrt(std::max(0.0, psiRho[i] * beforeRho[i]));
      drive[i] = motion + phaseMotion;
      target[i] = {drive[i] * psiQ[i].x, drive[i] * psiQ[i].y};
    }
    for (int e = 0; e < edgeN; ++e) {
      const int a = edgeA[e];
      const int b = edgeB[e];
      const Carrier tb = transport(edgeGauge[e], psiQ[b]);
      const Carrier ta = transport(conjCarrier(edgeGauge[e]), psiQ[a]);
      const double root = std::sqrt(std::max(0.0, psiRho[a] * psiRho[b]));
      const double current = std::abs(crossCarrier(psiQ[a], tb));
      const double stress = std::abs(dotCarrier(psiQ[a], tb));
      const double tension = edgeW[e] * root * (0.70 * current + 0.30 * stress);
      drive[a] += 0.5 * tension;
      drive[b] += 0.5 * tension;
      target[a].x += tension * tb.x;
      target[a].y += tension * tb.y;
      target[b].x += tension * ta.x;
      target[b].y += tension * ta.y;
    }
    for (int i = 0; i < kNodes; ++i) {
      tauRho[i] = kTauDecay * tauRho[i] + kTauLearn * drive[i];
      aperture[i] = kApertureDecay * aperture[i] + kApertureLearn * drive[i];
      blendCarrier(tauQ[i], normalize(target[i]), 0.20);
      const double pulse = kChiInject * tauRho[i] / (1.0 + tauRho[i]);
      chiRho[i] += pulse;
      blendCarrier(chiQ[i], tauQ[i], 0.32);
      st.maxTau = std::max(st.maxTau, tauRho[i]);
    }
  }

  double participation(const double* rho) const {
    double s1 = 0.0;
    double s2 = 0.0;
    for (int i = 0; i < kNodes; ++i) {
      s1 += rho[i];
      s2 += rho[i] * rho[i];
    }
    return (s1 * s1) / (s2 + 1e-300);
  }

  void sample() {
    st.samples++;
    st.sumLinPr += participation(linRho);
    st.sumPsiPr += participation(psiRho);
    st.sumChiPr += participation(chiRho);
  }

  void tick(int t) {
    st.ticks++;
    const int src = sourceAt(t);
    bool active[kMaxEdges];
    selectActive(src, active);

    double beforeRho[kNodes];
    Carrier beforeQ[kNodes];
    for (int i = 0; i < kNodes; ++i) {
      beforeRho[i] = psiRho[i];
      beforeQ[i] = psiQ[i];
    }

    inject(src);
    flowField(linRho, linQ, active, 0.0);
    flowField(psiRho, psiQ, active, kPsiPressure);
    updateFeeling(beforeRho, beforeQ);
    flowField(chiRho, chiQ, active, kChiPressure);

    if (t > kWarmup && (t % 64) == 0) sample();
  }

  void run(int stream) {
    for (int t = 1; t <= stream; ++t) tick(t);
    finalizeChiSeparation();
  }

  void finalizeChiSeparation() {
    for (int e = 0; e < edgeN; ++e) {
      const double psiAct = edgeW[e] * edgeActivity(e, psiRho, psiQ);
      const double chiAct = edgeW[e] * edgeActivity(e, chiRho, chiQ);
      if (topicOf(edgeA[e]) == topicOf(edgeB[e])) {
        st.samePsi += psiAct;
        st.sameChi += chiAct;
        st.sameN++;
      } else {
        st.crossPsi += psiAct;
        st.crossChi += chiAct;
        st.crossN++;
      }
    }
    if (st.sameN) {
      st.samePsi /= st.sameN;
      st.sameChi /= st.sameN;
    }
    if (st.crossN) {
      st.crossPsi /= st.crossN;
      st.crossChi /= st.crossN;
    }
  }
};

double compression(const Stats& st, double sumPr) {
  return st.sumLinPr / (sumPr + 1e-300);
}

double activeFraction(const Stats& st) {
  return static_cast<double>(st.activeEdges) / static_cast<double>(st.totalEdges + 1);
}

double signedBias(double signedSum, double absSum) {
  return std::abs(signedSum) / (absSum + 1e-300);
}

double stateDistance(const Medium& a, const Medium& b) {
  double diff = 0.0;
  double scale = 0.0;
  for (int i = 0; i < kNodes; ++i) {
    diff += std::abs(a.psiRho[i] - b.psiRho[i]);
    diff += std::abs(a.chiRho[i] - b.chiRho[i]);
    diff += carrierDistance(a.psiQ[i], b.psiQ[i]);
    diff += carrierDistance(a.chiQ[i], b.chiQ[i]);
    scale += std::abs(a.psiRho[i]) + std::abs(a.chiRho[i]) + 2.0;
  }
  return diff / (scale + 1e-300);
}

void printStats(const char* name, const Medium& m) {
  const Stats& s = m.st;
  const double linPr = s.sumLinPr / (s.samples + 1e-300);
  const double psiPr = s.sumPsiPr / (s.samples + 1e-300);
  const double chiPr = s.sumChiPr / (s.samples + 1e-300);
  std::printf("  %-16s active=%.3f  PR lin=%.2f psi=%.2f chi=%.2f  "
              "comp psi=%.2fx chi=%.2fx  same/cross psi=%.2f chi=%.2f\n",
              name, activeFraction(s), linPr, psiPr, chiPr,
              compression(s, s.sumPsiPr), compression(s, s.sumChiPr),
              s.samePsi / (s.crossPsi + 1e-300),
              s.sameChi / (s.crossChi + 1e-300));
}

}  // namespace

int main(int argc, char** argv) {
  const int stream = argc > 1 ? std::atoi(argv[1]) : kDefaultStream;
  std::printf("=====================================================================\n");
  std::printf("  GNNv3 SELF-SENSING MEDIUM CONTRACT  (SoA carrier field, no trig gate)\n");
  std::printf("=====================================================================\n");
  std::printf("  stream=%d nodes=%d fixed_edges<=%d\n", stream, kNodes, kMaxEdges);

  Medium full(false, false);
  Medium sense(false, false);
  Medium senseRev(false, true);
  sense.sensing = true;
  senseRev.sensing = true;

  full.run(stream);
  sense.run(stream);
  senseRev.run(stream);

  printStats("full", full);
  printStats("self-sensing", sense);
  printStats("reverse", senseRev);

  const double psiComp = compression(sense.st, sense.st.sumPsiPr);
  const double chiComp = compression(sense.st, sense.st.sumChiPr);
  const double frac = activeFraction(sense.st);
  const double syncError = stateDistance(sense, senseRev);
  const double skippedCurrentBias = signedBias(sense.st.skippedCurrentSigned,
                                               sense.st.skippedCurrentAbs);
  const double skippedStressBias = signedBias(sense.st.skippedStressSigned,
                                              sense.st.skippedStressAbs);
  const double psiSep = sense.st.samePsi / (sense.st.crossPsi + 1e-300);
  const double chiSep = sense.st.sameChi / (sense.st.crossChi + 1e-300);

  std::printf("\n  internal support separation: psi=%.2f chi=%.2f\n", psiSep, chiSep);
  std::printf("  skipped phase audit: current_bias=%.3f stress_bias=%.3f skipped=%lld\n",
              skippedCurrentBias, skippedStressBias, sense.st.skippedEdges);
  std::printf("  reverse-order sync_error=%.4e\n", syncError);
  std::printf("  max_rho=%.3f max_tau=%.3f samples=%lld edges=%d\n",
              sense.st.maxRho, sense.st.maxTau, sense.st.samples, sense.edgeN);

  int pass = 0;
  int total = 8;
  auto check = [&](const char* label, bool ok) {
    std::printf("  %-44s : %s\n", label, ok ? "PASS" : "FAIL");
    if (ok) pass++;
  };

  check("field produced samples", sense.st.samples > 0);
  check("self-sensing reduces heavy flow", frac > 0.03 && frac < 0.85);
  check("psi still densifies", psiComp > 1.20);
  check("chi is denser than psi", chiComp > psiComp);
  check("psi support remains readable", psiSep > 1.05);
  check("chi separates internal support", chiSep > 1.20);
  check("skipped phase is not one-sided", skippedCurrentBias < 0.40 && skippedStressBias < 0.40);
  check("edge traversal stays synchronized", syncError < 5e-4);

  std::printf("\n  CONTRACT RESULT: %d / %d %s\n", pass, total,
              pass == total ? "PASS" : "FAIL");
  return pass == total ? 0 : 1;
}
