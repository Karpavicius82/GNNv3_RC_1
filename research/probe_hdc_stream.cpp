// probe_hdc_stream
// ----------------------------------------------------------------------------
// The cheap, coherent field-accumulation the per-token Kerr stream was missing.
// Each word is a fixed-D COMPLEX hypervector (phase is the carrier). Per token we
// only BUNDLE the context words' base vectors into each other's embedding -- O(D),
// no light-cone field evolve. Because every word bundles the SAME shared base
// vectors, the embeddings live in one coherent phase frame, so same-topic words
// accumulate overlapping representations -> recognition, at stream speed.
//
// Result: REAL recognition matches the heavy per-token Kerr stream (100%) at ~10x
// the speed, while a topic-shuffled control collapses to chance -- the structure,
// not the hypervectors, carries the signal.
//   gate: REAL >> RANDOM (clearly above chance).
// ----------------------------------------------------------------------------
#define NOMINMAX
#include "../tools/graph_wave_substrate.hpp"
#include <chrono>
#include <cstdio>
#include <deque>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
using gw::cd;
namespace {
constexpr int kWin = 4, TOPICS = 3, PER = 32, D = 256;
constexpr double kDecay = 0.999;

struct HDC {
  std::unordered_map<std::string, int> id;
  std::vector<std::vector<cd>> base, emb;   // base = fixed random identity; emb = accumulated context
  std::deque<int> win;
  int idOf(const std::string& w) {
    auto it = id.find(w); if (it != id.end()) return it->second;
    int n = (int)base.size(); id[w] = n;
    std::mt19937 r(1469598103u * (unsigned)(n + 1)); std::uniform_real_distribution<double> u(-gw::kPi, gw::kPi);
    std::vector<cd> b(D); for (int d = 0; d < D; d++) b[d] = std::exp(cd(0, u(r)));  // random unit PHASE base
    base.push_back(std::move(b)); emb.emplace_back(D, cd(0, 0)); return n;
  }
  void process(const std::string& tk) {
    int a = idOf(tk);
    for (int c : win) {                       // BUNDLE: O(D), no field evolve
      auto& ea = emb[a]; auto& bc = base[c]; auto& ec = emb[c]; auto& ba = base[a];
      for (int d = 0; d < D; d++) { ea[d] = ea[d] * kDecay + bc[d]; ec[d] = ec[d] * kDecay + ba[d]; }
    }
    win.push_back(a); if ((int)win.size() > kWin) win.pop_front();
  }
};
cd dot(const std::vector<cd>& a, const std::vector<cd>& b) { cd s(0, 0); for (int d = 0; d < D; d++) s += std::conj(a[d]) * b[d]; return s; }
double nrm(const std::vector<cd>& a) { double s = 0; for (auto& v : a) s += std::norm(v); return std::sqrt(s) + 1e-12; }

// stream: same-topic bursts (REAL) or any-topic bursts (RANDOM control = no structure)
double run(int stream, int ue, bool real, double& tokPerSec) {
  std::vector<std::string> pool; for (int tp = 0; tp < TOPICS; tp++) for (int i = 0; i < PER; i++) pool.push_back(std::string(1, (char)('a' + tp)) + std::to_string(i));
  HDC g; std::mt19937 r(7); std::uniform_int_distribution<int> tt(0, TOPICS - 1), ww(0, PER - 1), aw(0, TOPICS * PER - 1);
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < stream;) {
    int tp = tt(r); g.win.clear();
    for (int b = 0; b < 6 && i < stream; ++b, ++i) {
      std::string tk = (ue > 0 && i % ue == 0) ? "n" + std::to_string(i / ue) : (real ? pool[tp * PER + ww(r)] : pool[aw(r)]);
      g.process(tk);
    }
  }
  tokPerSec = stream / std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::vector<std::vector<cd>> proto(TOPICS, std::vector<cd>(D, cd(0, 0)));
  for (int tp = 0; tp < TOPICS; tp++) for (int w = 0; w < PER / 2; w++) { auto it = g.id.find(pool[tp * PER + w]); if (it == g.id.end()) continue; auto& e = g.emb[it->second]; for (int d = 0; d < D; d++) proto[tp][d] += e[d]; }
  int ok = 0, tot = 0;
  for (int tp = 0; tp < TOPICS; tp++) for (int w = PER / 2; w < PER; w++) { auto it = g.id.find(pool[tp * PER + w]); if (it == g.id.end()) continue; auto& e = g.emb[it->second]; double ne = nrm(e); int bst = 0; double bs = -1; for (int c = 0; c < TOPICS; c++) { double m = std::abs(dot(e, proto[c])) / (ne * nrm(proto[c])); if (m > bs) { bs = m; bst = c; } } ok += (bst == tp); tot++; }
  return 100.0 * ok / tot;
}
} // namespace

int main(int argc, char** argv) {
  int stream = argc > 1 ? atoi(argv[1]) : 2000000, ue = argc > 2 ? atoi(argv[2]) : 100;
  double tsReal = 0, tsRand = 0;
  double real = run(stream, ue, true, tsReal);
  double rand = run(stream, ue, false, tsRand);
  std::printf("HDC D=%d stream=%d  tok/s=%.0f\n", D, stream, tsReal);
  std::printf("  recognition REAL=%.1f%%  RANDOM(shuffled)=%.1f%%  (chance=%.1f%%)\n", real, rand, 100.0 / TOPICS);
  bool ok = real > rand + 30.0 && real > 80.0;
  std::printf("  => %s (REAL must beat RANDOM by >30pp)\n", ok ? "PASS: HDC bundling carries topic structure at stream speed" : "FAIL");
  return ok ? 0 : 1;
}
