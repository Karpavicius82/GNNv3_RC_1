#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {

using cd = std::complex<double>;
using CMat = std::vector<std::vector<cd>>;
using RMat = std::vector<std::vector<double>>;

constexpr double kPi = 3.141592653589793238462643383279502884;

struct Graph {
 CMat h;

 explicit Graph(const int n) : h(n, std::vector<cd>(n, cd(0, 0))) {}

 void addEdge(const int i, const int j, const double a, const double phase = 0.0) {
 const cd e = a * std::exp(cd(0, phase));
 h[i][j] += e;
 h[j][i] += std::conj(e);
 }

 double hermiticityError() const {
 const auto n = static_cast<int>(h.size());
 double err = 0.0;
 for (int i = 0; i < n; ++i) {
 for (int j = 0; j < n; ++j) {
 err = std::max(err, std::abs(h[i][j] - std::conj(h[j][i])));
 }
 }
 return err;
 }
};

RMat realRepresentation(const CMat& h) {
 const auto n = static_cast<int>(h.size());
 RMat r(2 * n, std::vector<double>(2 * n, 0.0));
 for (int i = 0; i < n; ++i) {
 for (int j = 0; j < n; ++j) {
 const double a = h[i][j].real();
 const double b = h[i][j].imag();
 r[i][j] = a;
 r[i][j + n] = -b;
 r[i + n][j] = b;
 r[i + n][j + n] = a;
 }
 }
 return r;
}

std::vector<double> jacobiEigenvalues(RMat a) {
 const int n = static_cast<int>(a.size());
 for (int sweep = 0; sweep < 260; ++sweep) {
 double off = 0.0;
 for (int p = 0; p < n; ++p) {
 for (int q = p + 1; q < n; ++q) off += a[p][q] * a[p][q];
 }
 if (off < 1e-22) break;

 for (int p = 0; p < n; ++p) {
 for (int q = p + 1; q < n; ++q) {
 const double apq = a[p][q];
 if (std::abs(apq) < 1e-300) continue;
 const double app = a[p][p];
 const double aqq = a[q][q];
 const double tau = (aqq - app) / (2.0 * apq);
 const double t = tau >= 0.0
 ? 1.0 / (tau + std::sqrt(1.0 + tau * tau))
 : -1.0 / (-tau + std::sqrt(1.0 + tau * tau));
 const double c = 1.0 / std::sqrt(1.0 + t * t);
 const double s = t * c;

 for (int k = 0; k < n; ++k) {
 if (k == p || k == q) continue;
 const double akp = a[k][p];
 const double akq = a[k][q];
 a[k][p] = a[p][k] = c * akp - s * akq;
 a[k][q] = a[q][k] = s * akp + c * akq;
 }
 a[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
 a[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
 a[p][q] = a[q][p] = 0.0;
 }
 }
 }

 std::vector<double> ev(n);
 for (int i = 0; i < n; ++i) ev[i] = a[i][i];
 std::sort(ev.begin(), ev.end());
 return ev;
}

std::vector<double> spectrum(const Graph& g) {
 return jacobiEigenvalues(realRepresentation(g.h));
}

Graph plainLattice(const int w, const int h) {
 Graph g(w * h);
 auto id = [w](const int x, const int y) { return y * w + x; };
 for (int y = 0; y < h; ++y) {
 for (int x = 0; x < w; ++x) {
 if (x + 1 < w) g.addEdge(id(x, y), id(x + 1, y), 1.0);
 if (y + 1 < h) g.addEdge(id(x, y), id(x, y + 1), 1.0);
 }
 }
 return g;
}

Graph randomPhaseLattice(const int w, const int h) {
 Graph g(w * h);
 auto id = [w](const int x, const int y) { return y * w + x; };
 std::mt19937 rng(0xC0FFEEu);
 std::uniform_real_distribution<double> phase(-kPi, kPi);
 for (int y = 0; y < h; ++y) {
 for (int x = 0; x < w; ++x) {
 if (x + 1 < w) g.addEdge(id(x, y), id(x + 1, y), 1.0, phase(rng));
 if (y + 1 < h) g.addEdge(id(x, y), id(x, y + 1), 1.0, phase(rng));
 }
 }
 return g;
}

Graph goldenFluxLattice(const int w, const int h, const double alpha) {
 Graph g(w * h);
 auto id = [w](const int x, const int y) { return y * w + x; };
 for (int y = 0; y < h; ++y) {
 for (int x = 0; x < w; ++x) {
 if (x + 1 < w) g.addEdge(id(x, y), id(x + 1, y), 1.0);
 if (y + 1 < h) g.addEdge(id(x, y), id(x, y + 1), 1.0, 2.0 * kPi * alpha * x);
 }
 }
 return g;
}

double boxDimension(std::vector<double> ev) {
 std::sort(ev.begin(), ev.end());
 const double lo = ev.front();
 const double hi = ev.back();
 const double width = hi - lo;
 if (width <= 0.0) return 0.0;

 const int bins[] = {16, 32, 64, 128, 256};
 constexpr int m = 5;
 double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
 for (const int bcount : bins) {
 std::vector<char> occupied(bcount, 0);
 for (const double e : ev) {
 int b = static_cast<int>(((e - lo) / width) * bcount);
 if (b < 0) b = 0;
 if (b >= bcount) b = bcount - 1;
 occupied[b] = 1;
 }
 int count = 0;
 for (const char b : occupied) count += b ? 1 : 0;
 const double lx = std::log(static_cast<double>(bcount));
 const double ly = std::log(static_cast<double>(count));
 sx += lx;
 sy += ly;
 sxx += lx * lx;
 sxy += lx * ly;
 }
 return (m * sxy - sx * sy) / (m * sxx - sx * sx);
}

int countGaps(std::vector<double> ev, const double min_gap) {
 std::sort(ev.begin(), ev.end());
 int count = 0;
 for (std::size_t i = 1; i < ev.size(); ++i) {
 if (ev[i] - ev[i - 1] > min_gap) ++count;
 }
 return count;
}

std::vector<double> histogram(const std::vector<double>& ev, const int bins) {
 const auto [mn, mx] = std::minmax_element(ev.begin(), ev.end());
 const double lo = *mn;
 const double hi = *mx;
 const double width = hi - lo;
 std::vector<double> hist(bins, 0.0);
 for (const double e : ev) {
 int b = static_cast<int>(((e - lo) / width) * bins);
 if (b < 0) b = 0;
 if (b >= bins) b = bins - 1;
 hist[b] += 1.0;
 }
 for (double& v : hist) v /= static_cast<double>(ev.size());
 return hist;
}

double l1(const std::vector<double>& a, const std::vector<double>& b) {
 double s = 0.0;
 for (std::size_t i = 0; i < a.size(); ++i) s += std::abs(a[i] - b[i]);
 return s;
}

struct Metrics {
 std::string name;
 double hermiticity = 0.0;
 double dim = 0.0;
 double width = 0.0;
 int gap2 = 0;
 int gap1 = 0;
 int gap05 = 0;
 int gap025 = 0;
 std::vector<double> hist;
};

Metrics measure(const std::string& name, const Graph& g) {
 const auto ev = spectrum(g);
 const double width = ev.back() - ev.front();
 Metrics m;
 m.name = name;
 m.hermiticity = g.hermiticityError();
 m.dim = boxDimension(ev);
 m.width = width;
 m.gap2 = countGaps(ev, 0.02 * width);
 m.gap1 = countGaps(ev, 0.01 * width);
 m.gap05 = countGaps(ev, 0.005 * width);
 m.gap025 = countGaps(ev, 0.0025 * width);
 m.hist = histogram(ev, 64);
 return m;
}

void printMetrics(const Metrics& m) {
 std::printf(" %-13s herm=%.1e dim=%.3f width=%.3f gaps>2%%/1%%/0.5%%/0.25%% = %d/%d/%d/%d\n",
 m.name.c_str(), m.hermiticity, m.dim, m.width,
 m.gap2, m.gap1, m.gap05, m.gap025);
}

} // namespace

int main() {
 std::printf("=====================================================================\n");
 std::printf(" PURE SPECTRAL SUBSTRATE COMPARISON\n");
 std::printf(" Question only: does golden/gauge graph-wave show a distinct gap hierarchy?\n");
 std::printf(" No measurement, no decision, no learning.\n");
 std::printf("=====================================================================\n");

 constexpr int w = 13;
 constexpr int h = 13;
 constexpr double alpha = 8.0 / 13.0;
 const Metrics plain = measure("plain", plainLattice(w, h));
 const Metrics random = measure("random-phase", randomPhaseLattice(w, h));
 const Metrics golden = measure("golden-flux", goldenFluxLattice(w, h, alpha));

 std::printf("\n[1] DIRECT GRAPH-WAVE HAMILTONIANS (N=%d, alpha=8/13)\n", w * h);
 printMetrics(plain);
 printMetrics(random);
 printMetrics(golden);

 const double l1_gp = l1(golden.hist, plain.hist);
 const double l1_gr = l1(golden.hist, random.hist);
 const double l1_pr = l1(plain.hist, random.hist);
 std::printf("\n[2] HISTOGRAM DISTANCES (64 bins over each spectrum support)\n");
 std::printf(" L1(golden, plain) = %.3f\n", l1_gp);
 std::printf(" L1(golden, random) = %.3f\n", l1_gr);
 std::printf(" L1(plain, random) = %.3f\n", l1_pr);

 const bool hermitian = plain.hermiticity < 1e-14 &&
 random.hermiticity < 1e-14 &&
 golden.hermiticity < 1e-14;
 const bool golden_has_large_gaps = golden.gap2 > plain.gap2 &&
 golden.gap2 > random.gap2 &&
 golden.gap05 >= 2 * golden.gap2;
 const bool golden_distinct = l1_gp > 0.20 && l1_gr > 0.12;
 const bool golden_sparser_than_random = golden.dim + 0.10 < random.dim;

 std::printf("\n[3] DECISION RULE FOR THIS TEST ONLY\n");
 std::printf(" hermitian=%s\n", hermitian ? "YES" : "NO");
 std::printf(" golden has stronger large-gap hierarchy than controls=%s\n",
 golden_has_large_gaps ? "YES" : "NO");
 std::printf(" golden distinct from both controls=%s\n", golden_distinct ? "YES" : "NO");
 std::printf(" golden spectrum sparser than random-phase control=%s (golden %.3f vs random %.3f)\n",
 golden_sparser_than_random ? "YES" : "NO", golden.dim, random.dim);
 std::printf(" Cantor/fractal dimension claim=DEFERRED (finite graph diagnostic only)\n");

 const bool pass = hermitian && golden_has_large_gaps && golden_distinct && golden_sparser_than_random;
 std::printf("\n=====================================================================\n");
 std::printf(" RESULT : %s\n", pass ? "PASS" : "FAIL");
 std::printf(" %s\n", pass
 ? "golden/gauge graph-wave is a distinct spectral substrate candidate, not a decision layer"
 : "do not use this substrate candidate yet; inspect spectral controls");
 std::printf("=====================================================================\n");
 return pass ? 0 : 1;
}
