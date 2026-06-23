#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using cd = std::complex<double>;
using Mat = std::vector<std::vector<cd>>;
using Vec = std::vector<cd>;

constexpr double kPi = 3.141592653589793238462643383279502884;

double norm2(const Vec& z) {
 double s = 0.0;
 for (const auto& v : z) s += std::norm(v);
 return s;
}

Vec matvec(const Mat& m, const Vec& z) {
 const auto n = static_cast<int>(m.size());
 Vec r(n, cd(0, 0));
 for (int i = 0; i < n; ++i) {
 cd s(0, 0);
 for (int j = 0; j < n; ++j) s += m[i][j] * z[j];
 r[i] = s;
 }
 return r;
}

struct Graph {
 int n;
 Mat h;
 explicit Graph(const int count) : n(count), h(count, Vec(count, cd(0, 0))) {}

 void addEdge(const int i, const int j, const double a, const double delta = 0.0) {
 const cd e = a * std::exp(cd(0, delta));
 h[i][j] += e;
 h[j][i] += std::conj(e);
 }

 void addPotential(const int i, const double v) { h[i][i] += cd(v, 0); }

 double hermiticityError() const {
 double m = 0.0;
 for (int i = 0; i < n; ++i) {
 for (int j = 0; j < n; ++j) {
 m = std::max(m, std::abs(h[i][j] - std::conj(h[j][i])));
 }
 }
 return m;
 }
};

struct LUC {
 int n = 0;
 Mat a;
 std::vector<int> piv;

 void factor(Mat m) {
 n = static_cast<int>(m.size());
 a = std::move(m);
 piv.resize(n);
 for (int i = 0; i < n; ++i) piv[i] = i;
 for (int k = 0; k < n; ++k) {
 double best = std::abs(a[k][k]);
 int br = k;
 for (int i = k + 1; i < n; ++i) {
 const double v = std::abs(a[i][k]);
 if (v > best) {
 best = v;
 br = i;
 }
 }
 if (br != k) {
 std::swap(a[br], a[k]);
 std::swap(piv[br], piv[k]);
 }
 const cd akk = a[k][k];
 for (int i = k + 1; i < n; ++i) {
 const cd f = a[i][k] / akk;
 a[i][k] = f;
 for (int j = k + 1; j < n; ++j) a[i][j] -= f * a[k][j];
 }
 }
 }

 Vec solve(const Vec& b) const {
 Vec x(n);
 for (int i = 0; i < n; ++i) x[i] = b[piv[i]];
 for (int i = 0; i < n; ++i) {
 cd s = x[i];
 for (int j = 0; j < i; ++j) s -= a[i][j] * x[j];
 x[i] = s;
 }
 for (int i = n - 1; i >= 0; --i) {
 cd s = x[i];
 for (int j = i + 1; j < n; ++j) s -= a[i][j] * x[j];
 x[i] = s / a[i][i];
 }
 return x;
 }
};

struct UnitaryStepper {
 int n = 0;
 Mat ap;
 Mat am;
 LUC lup;
 LUC lum;

 void build(const Mat& h, const double dt) {
 n = static_cast<int>(h.size());
 ap.assign(n, Vec(n));
 am.assign(n, Vec(n));
 const cd imag(0, 1);
 for (int i = 0; i < n; ++i) {
 for (int j = 0; j < n; ++j) {
 const cd k = imag * h[i][j] * (dt / 2.0);
 const cd id = i == j ? cd(1, 0) : cd(0, 0);
 ap[i][j] = id + k;
 am[i][j] = id - k;
 }
 }
 lup.factor(ap);
 lum.factor(am);
 }

 Vec step(const Vec& z) const { return lup.solve(matvec(am, z)); }
 Vec back(const Vec& z) const { return lum.solve(matvec(ap, z)); }

 Mat asMatrix() const {
 Mat u(n, Vec(n, cd(0, 0)));
 for (int j = 0; j < n; ++j) {
 Vec e(n, cd(0, 0));
 e[j] = cd(1, 0);
 const Vec c = step(e);
 for (int i = 0; i < n; ++i) u[i][j] = c[i];
 }
 return u;
 }
};

double operatorUnitarityError(const Mat& u) {
 const int n = static_cast<int>(u.size());
 double m = 0.0;
 for (int i = 0; i < n; ++i) {
 for (int j = 0; j < n; ++j) {
 cd s(0, 0);
 for (int k = 0; k < n; ++k) s += std::conj(u[k][i]) * u[k][j];
 const cd target = i == j ? cd(1, 0) : cd(0, 0);
 m = std::max(m, std::abs(s - target));
 }
 }
 return m;
}

std::vector<double> jacobiEigenvalues(std::vector<std::vector<double>> a) {
 const int n = static_cast<int>(a.size());
 for (int sweep = 0; sweep < 200; ++sweep) {
 double off = 0.0;
 for (int p = 0; p < n; ++p)
 for (int q = p + 1; q < n; ++q) off += a[p][q] * a[p][q];
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

bool testNormConservation() {
 std::printf("\n[1] NORM CONSERVATION\n");
 constexpr int w = 10;
 constexpr int n = w * w;
 Graph g(n);
 std::mt19937 rng(12345);
 std::uniform_real_distribution<double> phase(-kPi, kPi);
 std::uniform_real_distribution<double> pot(-0.3, 0.3);
 auto id = [](const int x, const int y) { return y * w + x; };
 for (int y = 0; y < w; ++y) {
 for (int x = 0; x < w; ++x) {
 if (x + 1 < w) g.addEdge(id(x, y), id(x + 1, y), 1.0, phase(rng));
 if (y + 1 < w) g.addEdge(id(x, y), id(x, y + 1), 1.0, phase(rng));
 g.addPotential(id(x, y), pot(rng));
 }
 }
 std::printf(" hermiticity error = %.2e\n", g.hermiticityError());
 UnitaryStepper u;
 u.build(g.h, 0.15);
 Vec z(n);
 std::normal_distribution<double> gauss(0, 1);
 for (auto& v : z) v = cd(gauss(rng), gauss(rng));
 const double base = std::sqrt(norm2(z));
 for (auto& v : z) v /= base;
 double maxdev = 0.0;
 for (int t = 0; t < 1000; ++t) {
 z = u.step(z);
 maxdev = std::max(maxdev, std::abs(norm2(z) - 1.0));
 }
 const bool pass = maxdev < 1e-9;
 std::printf(" max norm drift = %.3e %s\n", maxdev, pass ? "PASS" : "FAIL");
 return pass;
}

bool testReversibility() {
 std::printf("\n[2] TIME REVERSIBILITY\n");
 constexpr int w = 10;
 constexpr int n = w * w;
 Graph g(n);
 std::mt19937 rng(777);
 std::uniform_real_distribution<double> phase(-kPi, kPi);
 auto id = [](const int x, const int y) { return y * w + x; };
 for (int y = 0; y < w; ++y) {
 for (int x = 0; x < w; ++x) {
 if (x + 1 < w) g.addEdge(id(x, y), id(x + 1, y), 1.0, phase(rng));
 if (y + 1 < w) g.addEdge(id(x, y), id(x, y + 1), 1.0, phase(rng));
 }
 }
 UnitaryStepper u;
 u.build(g.h, 0.2);
 Vec z0(n);
 std::normal_distribution<double> gauss(0, 1);
 for (auto& v : z0) v = cd(gauss(rng), gauss(rng));
 const double base = std::sqrt(norm2(z0));
 for (auto& v : z0) v /= base;
 Vec z = z0;
 for (int t = 0; t < 500; ++t) z = u.step(z);
 for (int t = 0; t < 500; ++t) z = u.back(z);
 double err = 0.0;
 for (int i = 0; i < n; ++i) err = std::max(err, std::abs(z[i] - z0[i]));
 const double ue = operatorUnitarityError(u.asMatrix());
 const bool pass = err < 1e-9 && ue < 1e-9;
 std::printf(" return err = %.3e operator err = %.3e %s\n", err, ue,
 pass ? "PASS" : "FAIL");
 return pass;
}

bool testInterference() {
 std::printf("\n[3] INTERFERENCE\n");
 constexpr int s = 0;
 constexpr int a = 1;
 constexpr int b = 2;
 constexpr int tnode = 3;
 std::vector<double> transfer;
 for (int k = 0; k < 13; ++k) {
 const double phi = kPi * k / 12.0;
 Graph g(4);
 g.addEdge(s, a, 1.0);
 g.addEdge(s, b, 1.0);
 g.addEdge(a, tnode, 1.0);
 g.addEdge(b, tnode, 1.0, phi);
 UnitaryStepper u;
 u.build(g.h, 0.10);
 Vec z(4, cd(0, 0));
 z[s] = cd(1, 0);
 double acc = 0.0;
 for (int step = 0; step < 400; ++step) {
 z = u.step(z);
 acc += std::norm(z[tnode]);
 }
 transfer.push_back(acc / 400.0);
 }
 const double constructive = transfer.front();
 const double destructive = transfer.back();
 const bool pass = destructive < 0.1 * constructive && constructive > 0.05;
 std::printf(" constructive=%.3f destructive=%.3e ratio=%.1fx %s\n",
 constructive, destructive, constructive / std::max(destructive, 1e-12),
 pass ? "PASS" : "FAIL");
 return pass;
}

struct ScreenResult {
 std::vector<double> pattern;
 int peakStep;
};

ScreenResult runSlits(const int w, const int d, const int yBarrier,
 const std::vector<int>& slits, const double dt, const int tmax) {
 auto id = [w](const int x, const int y) { return y * w + x; };
 const int n = w * d;
 std::vector<char> blocked(n, 0);
 for (int x = 0; x < w; ++x) {
 const bool open = std::find(slits.begin(), slits.end(), x) != slits.end();
 if (!open) blocked[id(x, yBarrier)] = 1;
 }
 Graph g(n);
 for (int y = 0; y < d; ++y) {
 for (int x = 0; x < w; ++x) {
 const int here = id(x, y);
 if (blocked[here]) continue;
 if (x + 1 < w && !blocked[id(x + 1, y)]) g.addEdge(here, id(x + 1, y), 1.0);
 if (y + 1 < d && !blocked[id(x, y + 1)]) g.addEdge(here, id(x, y + 1), 1.0);
 }
 }
 UnitaryStepper u;
 u.build(g.h, dt);
 Vec z(n, cd(0, 0));
 const double xc = w / 2.0;
 const double y0 = 2.0;
 const double sx = 3.0;
 const double sy = 1.5;
 const double ky = kPi / 2.0;
 for (int y = 0; y < d; ++y) {
 for (int x = 0; x < w; ++x) {
 if (blocked[id(x, y)]) continue;
 const double env =
 std::exp(-((x - xc) * (x - xc) / (2 * sx * sx) +
 (y - y0) * (y - y0) / (2 * sy * sy)));
 z[id(x, y)] = env * std::exp(cd(0, ky * y));
 }
 }
 const double base = std::sqrt(norm2(z));
 for (auto& v : z) v /= base;
 std::vector<double> best(w, 0.0);
 double bestTot = -1.0;
 int bestStep = 0;
 for (int step = 0; step < tmax; ++step) {
 z = u.step(z);
 double total = 0.0;
 std::vector<double> pat(w, 0.0);
 for (int x = 0; x < w; ++x) {
 pat[x] = std::norm(z[id(x, d - 1)]);
 total += pat[x];
 }
 if (total > bestTot) {
 bestTot = total;
 best = pat;
 bestStep = step;
 }
 }
 double s = 0.0;
 for (double v : best) s += v;
 if (s > 0) for (double& v : best) v /= s;
 return {best, bestStep};
}

bool testDiffraction() {
 std::printf("\n[4] DIFFRACTION\n");
 constexpr int w = 41;
 constexpr int d = 18;
 constexpr int center = w / 2;
 const auto one = runSlits(w, d, 6, {center}, 0.25, 140);
 const auto two = runSlits(w, d, 6, {center - 3, center + 3}, 0.25, 140);
 double l1 = 0.0;
 for (int x = 0; x < w; ++x) l1 += std::abs(one.pattern[x] - two.pattern[x]);
 const double peak = *std::max_element(two.pattern.begin(), two.pattern.end());
 int maxima = 0;
 for (int x = 1; x < w - 1; ++x) {
 if (two.pattern[x] > 0.2 * peak && two.pattern[x] >= two.pattern[x - 1] &&
 two.pattern[x] > two.pattern[x + 1]) {
 ++maxima;
 }
 }
 const bool pass = l1 > 0.3 && maxima >= 2;
 std::printf(" L1(single,double)=%.3f double maxima=%d peak steps=%d/%d %s\n",
 l1, maxima, one.peakStep, two.peakStep, pass ? "PASS" : "FAIL");
 return pass;
}

std::string fibWord(const int iters) {
 std::string s = "A";
 for (int k = 0; k < iters; ++k) {
 std::string t;
 for (const char ch : s) t += ch == 'A' ? "AB" : "A";
 s.swap(t);
 }
 return s;
}

std::vector<double> chainSpectrum(const std::vector<double>& hop) {
 const int n = static_cast<int>(hop.size()) + 1;
 std::vector<std::vector<double>> h(n, std::vector<double>(n, 0.0));
 for (int b = 0; b < static_cast<int>(hop.size()); ++b) {
 h[b][b + 1] = hop[b];
 h[b + 1][b] = hop[b];
 }
 return jacobiEigenvalues(h);
}

int countGaps(const std::vector<double>& ev, const double minGap) {
 int c = 0;
 for (std::size_t i = 1; i < ev.size(); ++i) {
 if (ev[i] - ev[i - 1] > minGap) ++c;
 }
 return c;
}

bool testFractalResonance() {
 std::printf("\n[5] FRACTAL RESONANCE\n");
 const std::string word = fibWord(11);
 std::vector<double> fib;
 for (const char ch : word) fib.push_back(ch == 'A' ? 1.0 : 0.4);
 double mean = 0.0;
 for (const double h : fib) mean += h;
 mean /= static_cast<double>(fib.size());
 const std::vector<double> uniform(fib.size(), mean);
 const auto evFib = chainSpectrum(fib);
 const auto evUni = chainSpectrum(uniform);
 const double widthFib = evFib.back() - evFib.front();
 const double widthUni = evUni.back() - evUni.front();
 const int fibBands = countGaps(evFib, 0.05 * widthFib);
 const int uniBands = countGaps(evUni, 0.05 * widthUni);
 const bool pass = fibBands >= 2 && uniBands == 0;
 std::printf(" fib significant gaps=%d uniform significant gaps=%d %s\n",
 fibBands, uniBands, pass ? "PASS" : "FAIL");
 return pass;
}

} // namespace

int main() {
 std::printf("=====================================================================\n");
 std::printf(" GRAPH-WAVE SUBSTRATE : unitarity foundation test\n");
 std::printf("=====================================================================\n");
 int pass = 0;
 pass += testNormConservation() ? 1 : 0;
 pass += testReversibility() ? 1 : 0;
 pass += testInterference() ? 1 : 0;
 pass += testDiffraction() ? 1 : 0;
 pass += testFractalResonance() ? 1 : 0;
 std::printf("\nRESULT: %d / 5 physical invariants verified\n", pass);
 return pass == 5 ? 0 : 1;
}
