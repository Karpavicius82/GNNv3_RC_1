#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <string>
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
 std::vector<double> gamma;

 explicit Graph(const int count)
 : n(count), h(count, Vec(count, cd(0, 0))), gamma(count, 0.0) {}

 void addEdge(const int i, const int j, const double a, const double delta = 0.0) {
 const cd e = a * std::exp(cd(0, delta));
 h[i][j] += e;
 h[j][i] += std::conj(e);
 }

 void addPotential(const int i, const double v) {
 h[i][i] += cd(v, 0);
 }

 void addAbsorber(const int i, const double g) {
 gamma[i] += g;
 h[i][i] += cd(0, -g);
 }

 double hermiticityErrorOfSubstrate() const {
 double max_err = 0.0;
 for (int i = 0; i < n; ++i) {
 for (int j = 0; j < n; ++j) {
 cd hij = h[i][j];
 cd hji = h[j][i];
 if (i == j) {
 hij += cd(0, gamma[i]);
 hji += cd(0, gamma[i]);
 }
 max_err = std::max(max_err, std::abs(hij - std::conj(hji)));
 }
 }
 return max_err;
 }
};

struct LUC {
 int n = 0;
 Mat a;
 std::vector<int> piv;

 void factor(Mat input) {
 n = static_cast<int>(input.size());
 a = std::move(input);
 piv.resize(n);
 for (int i = 0; i < n; ++i) piv[i] = i;

 for (int k = 0; k < n; ++k) {
 double best = std::abs(a[k][k]);
 int best_row = k;
 for (int i = k + 1; i < n; ++i) {
 const double v = std::abs(a[i][k]);
 if (v > best) {
 best = v;
 best_row = i;
 }
 }
 if (best_row != k) {
 std::swap(a[best_row], a[k]);
 std::swap(piv[best_row], piv[k]);
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

struct Propagator {
 int n = 0;
 Mat ap;
 Mat am;
 LUC lu_p;

 void build(const Mat& h, const double dt) {
 n = static_cast<int>(h.size());
 ap.assign(n, Vec(n));
 am.assign(n, Vec(n));
 const cd i_unit(0, 1);

 for (int r = 0; r < n; ++r) {
 for (int c = 0; c < n; ++c) {
 const cd k = i_unit * h[r][c] * (dt / 2.0);
 const cd id = (r == c) ? cd(1, 0) : cd(0, 0);
 ap[r][c] = id + k;
 am[r][c] = id - k;
 }
 }
 lu_p.factor(ap);
 }

 Vec step(const Vec& z) const {
 return lu_p.solve(matvec(am, z));
 }

 Mat asMatrix() const {
 Mat u(n, Vec(n, cd(0, 0)));
 for (int j = 0; j < n; ++j) {
 Vec e(n, cd(0, 0));
 e[j] = cd(1, 0);
 const Vec col = step(e);
 for (int i = 0; i < n; ++i) u[i][j] = col[i];
 }
 return u;
 }
};

double operatorUnitarityError(const Mat& u) {
 const auto n = static_cast<int>(u.size());
 double max_err = 0.0;
 for (int i = 0; i < n; ++i) {
 for (int j = 0; j < n; ++j) {
 cd s(0, 0);
 for (int k = 0; k < n; ++k) s += std::conj(u[k][i]) * u[k][j];
 const cd target = (i == j) ? cd(1, 0) : cd(0, 0);
 max_err = std::max(max_err, std::abs(s - target));
 }
 }
 return max_err;
}

bool testLayerSeparation() {
 std::printf("\n[1] LAYER SEPARATION (substrate unitary; only W breaks it)\n");
 constexpr int width = 10;
 constexpr int n = width * width;
 auto id = [](const int x, const int y) { return y * width + x; };

 std::mt19937 rng(99);
 std::uniform_real_distribution<double> phase(-kPi, kPi);

 auto build_base = [&](Graph& g) {
 for (int y = 0; y < width; ++y) {
 for (int x = 0; x < width; ++x) {
 if (x + 1 < width) g.addEdge(id(x, y), id(x + 1, y), 1.0, phase(rng));
 if (y + 1 < width) g.addEdge(id(x, y), id(x, y + 1), 1.0, phase(rng));
 }
 }
 };

 rng.seed(99);
 Graph g0(n);
 build_base(g0);
 Propagator p0;
 p0.build(g0.h, 0.2);
 const double unitary_err = operatorUnitarityError(p0.asMatrix());
 std::printf(" W=0 : substrate hermiticity err=%.2e ||U^dag U - I||=%.2e\n",
 g0.hermiticityErrorOfSubstrate(), unitary_err);

 rng.seed(99);
 Graph g1(n);
 build_base(g1);
 for (int x = 0; x < width; ++x) {
 g1.addAbsorber(id(x, 0), 0.8);
 g1.addAbsorber(id(x, width - 1), 0.8);
 }
 Propagator p1;
 p1.build(g1.h, 0.2);

 Vec z(n);
 std::normal_distribution<double> gaussian(0, 1);
 for (auto& v : z) v = cd(gaussian(rng), gaussian(rng));
 const double inv_norm = 1.0 / std::sqrt(norm2(z));
 for (auto& v : z) v *= inv_norm;

 double prev = norm2(z);
 double max_increase = 0.0;
 for (int t = 0; t < 300; ++t) {
 z = p1.step(z);
 const double next = norm2(z);
 max_increase = std::max(max_increase, next - prev);
 prev = next;
 }

 std::printf(" W>0 : norm after 300 steps=%.4f max per-step INCREASE=%.2e\n",
 prev, max_increase);
 const bool pass = unitary_err < 1e-9 && prev < 0.5 && max_increase < 1e-12;
 std::printf(" => %s\n", pass ? "PASS" : "FAIL");
 return pass;
}

bool testProbabilityAccounting() {
 std::printf("\n[2] PROBABILITY ACCOUNTING (detected + remaining = 1)\n");
 constexpr int n = 200;
 Graph g(n);
 for (int i = 0; i + 1 < n; ++i) g.addEdge(i, i + 1, 1.0);

 constexpr int absorber_width = 20;
 constexpr double gamma_max = 1.2;
 for (int d = 0; d < absorber_width; ++d) {
 const double gg = gamma_max * std::pow((absorber_width - d) / double(absorber_width), 2);
 g.addAbsorber(d, gg);
 g.addAbsorber(n - 1 - d, gg);
 }

 constexpr double dt = 0.2;
 Propagator p;
 p.build(g.h, dt);

 Vec z(n, cd(0, 0));
 const double xc = n / 2.0;
 constexpr double sx = 8.0;
 constexpr double k = kPi / 2.0;
 for (int i = 0; i < n; ++i) {
 z[i] = std::exp(-((i - xc) * (i - xc)) / (2 * sx * sx)) * std::exp(cd(0, k * i));
 }
 const double inv_norm = 1.0 / std::sqrt(norm2(z));
 for (auto& v : z) v *= inv_norm;

 std::vector<double> detected_current(n, 0.0);
 std::vector<double> detected_discrete(n, 0.0);
 double detected_telescope = 0.0;
 double prev = norm2(z);
 for (int t = 0; t < 2000; ++t) {
 const Vec before = z;
 z = p.step(z);
 const double next = norm2(z);
 const double loss = std::max(0.0, prev - next);
 detected_telescope += loss;

 double weight_sum = 0.0;
 for (int i = 0; i < n; ++i) {
 weight_sum += g.gamma[i] * (std::norm(before[i]) + std::norm(z[i]));
 detected_current[i] += dt * 2.0 * g.gamma[i] * std::norm(z[i]);
 }
 if (weight_sum > 0.0) {
 for (int i = 0; i < n; ++i) {
 const double w = g.gamma[i] * (std::norm(before[i]) + std::norm(z[i]));
 detected_discrete[i] += loss * (w / weight_sum);
 }
 }
 prev = next;
 }

 const double remaining = norm2(z);
 double sum_current = 0.0;
 double sum_discrete = 0.0;
 for (int i = 0; i < n; ++i) {
 sum_current += detected_current[i];
 sum_discrete += detected_discrete[i];
 }

 const double closure = std::abs(detected_telescope + remaining - 1.0);
 const double discrete_match = std::abs(sum_discrete - detected_telescope);
 const double current_match = std::abs(sum_current - detected_telescope);
 std::printf(" detected(telescoped)=%.6f remaining=%.3e sum=%.6f\n",
 detected_telescope, remaining, detected_telescope + remaining);
 std::printf(" discrete detector allocation sum_i D_i = %.6f |D-loss|=%.2e\n",
 sum_discrete, discrete_match);
 std::printf(" continuum current integral sum_i D_i = %.6f |J-loss|=%.2e\n",
 sum_current, current_match);
 std::printf(" |detected+remaining - 1| = %.2e\n", closure);

 const bool pass = closure < 1e-12 && discrete_match < 1e-12 && current_match < 1e-2;
 std::printf(" => %s\n", pass ? "PASS" : "FAIL");
 return pass;
}

struct Pattern {
 std::vector<double> screen;
 double detected;
};

constexpr int kTopAbsorber = 4;
constexpr int kBottomAbsorber = 6;
constexpr int kSideAbsorber = 6;

Pattern runSlitsAbsorbing(const int width, const int depth, const int y_barrier,
 const std::vector<int>& slits, const double dt, const int max_t) {
 auto id = [width](const int x, const int y) { return y * width + x; };
 const int n = width * depth;
 std::vector<char> blocked(n, 0);
 for (int x = 0; x < width; ++x) {
 const bool open = std::find(slits.begin(), slits.end(), x) != slits.end();
 if (!open) blocked[id(x, y_barrier)] = 1;
 }

 Graph g(n);
 for (int y = 0; y < depth; ++y) {
 for (int x = 0; x < width; ++x) {
 const int a = id(x, y);
 if (blocked[a]) continue;
 if (x + 1 < width && !blocked[id(x + 1, y)]) g.addEdge(a, id(x + 1, y), 1.0);
 if (y + 1 < depth && !blocked[id(x, y + 1)]) g.addEdge(a, id(x, y + 1), 1.0);
 }
 }

 auto ramp = [](const int d, const int width, const double gamma_max) {
 return gamma_max * std::pow((width - d) / double(width), 2);
 };
 for (int y = 0; y < depth; ++y) {
 for (int x = 0; x < width; ++x) {
 const int a = id(x, y);
 if (blocked[a]) continue;
 double gg = 0.0;
 if (y < kTopAbsorber) gg = std::max(gg, ramp(y, kTopAbsorber, 1.0));
 if (y >= depth - kBottomAbsorber) {
 gg = std::max(gg, ramp(depth - 1 - y, kBottomAbsorber, 1.0));
 }
 if (x < kSideAbsorber) gg = std::max(gg, ramp(x, kSideAbsorber, 1.2));
 if (x >= width - kSideAbsorber) {
 gg = std::max(gg, ramp(width - 1 - x, kSideAbsorber, 1.2));
 }
 if (gg > 0.0) g.addAbsorber(a, gg);
 }
 }

 Propagator p;
 p.build(g.h, dt);

 Vec z(n, cd(0, 0));
 const double xc = width / 2.0;
 const double y0 = kTopAbsorber + 2.0;
 constexpr double sx = 3.0;
 constexpr double sy = 1.5;
 // With U ~= exp(-iHt), ky < 0 sends the packet toward increasing y
 // on this lattice, i.e. toward the absorbing screen below the slits.
 constexpr double ky = -kPi / 2.0;
 for (int y = 0; y < depth; ++y) {
 for (int x = 0; x < width; ++x) {
 if (blocked[id(x, y)]) continue;
 const double env = std::exp(-((x - xc) * (x - xc) / (2 * sx * sx) +
 (y - y0) * (y - y0) / (2 * sy * sy)));
 z[id(x, y)] = env * std::exp(cd(0, ky * y));
 }
 }
 const double inv_norm = 1.0 / std::sqrt(norm2(z));
 for (auto& v : z) v *= inv_norm;

 std::vector<double> screen(width, 0.0);
 for (int t = 0; t < max_t; ++t) {
 z = p.step(z);
 for (int y = depth - kBottomAbsorber; y < depth; ++y) {
 for (int x = kSideAbsorber; x < width - kSideAbsorber; ++x) {
 screen[x] += dt * 2.0 * g.gamma[id(x, y)] * std::norm(z[id(x, y)]);
 }
 }
 }

 double detected = 0.0;
 for (double v : screen) detected += v;
 if (detected > 0.0) {
 for (double& v : screen) v /= detected;
 }
 return {screen, detected};
}

bool testCleanDiffraction() {
 std::printf("\n[3] CLEAN FAR-FIELD DOUBLE SLIT (absorbing screen + walls)\n");
 constexpr int width = 45;
 constexpr int depth = 28;
 constexpr int y_barrier = 10;
 constexpr double dt = 0.25;
 constexpr int max_t = 220;
 constexpr int center = width / 2;

 const Pattern one = runSlitsAbsorbing(width, depth, y_barrier, {center}, dt, max_t);
 const Pattern two = runSlitsAbsorbing(width, depth, y_barrier, {center - 4, center + 4}, dt, max_t);

 const double peak_one = *std::max_element(one.screen.begin(), one.screen.end());
 const double peak_two = *std::max_element(two.screen.begin(), two.screen.end());
 auto bar = [](const double v, const double peak) {
 return std::string(static_cast<int>(std::round(38.0 * v / std::max(peak, 1e-12))), '#');
 };

 std::printf(" single slit (caught %.3f of the beam):\n", one.detected);
 for (int x = center - 14; x <= center + 14; ++x) {
 std::printf(" x=%2d |%s\n", x, bar(one.screen[x], peak_one).c_str());
 }
 std::printf(" double slit (caught %.3f of the beam):\n", two.detected);
 for (int x = center - 14; x <= center + 14; ++x) {
 std::printf(" x=%2d |%s\n", x, bar(two.screen[x], peak_two).c_str());
 }

 auto count_deep_nulls = [](const std::vector<double>& pattern, const double peak) {
 int count = 0;
 for (int x = kSideAbsorber + 3; x < static_cast<int>(pattern.size()) - kSideAbsorber - 3; ++x) {
 if (pattern[x] < 0.4 * peak &&
 pattern[x] <= pattern[x - 1] &&
 pattern[x] < pattern[x + 1]) {
 ++count;
 }
 }
 return count;
 };

 auto count_maxima = [](const std::vector<double>& pattern, const double peak) {
 int count = 0;
 for (int x = kSideAbsorber + 1; x < static_cast<int>(pattern.size()) - kSideAbsorber - 1; ++x) {
 if (pattern[x] > 0.3 * peak &&
 pattern[x] >= pattern[x - 1] &&
 pattern[x] > pattern[x + 1]) {
 ++count;
 }
 }
 return count;
 };

 const int nulls_one = count_deep_nulls(one.screen, peak_one);
 const int nulls_two = count_deep_nulls(two.screen, peak_two);
 const int maxima_two = count_maxima(two.screen, peak_two);

 double com = 0.0;
 double total = 0.0;
 double edge = 0.0;
 for (int x = 0; x < width; ++x) {
 com += x * two.screen[x];
 total += two.screen[x];
 if (x < 5 || x >= width - 5) edge += two.screen[x];
 }
 com /= total;

 double l1 = 0.0;
 for (int x = 0; x < width; ++x) l1 += std::abs(one.screen[x] - two.screen[x]);

 std::printf(" interior deep nulls: single=%d double=%d double maxima=%d\n",
 nulls_one, nulls_two, maxima_two);
 std::printf(" double centre-of-mass=%.1f (slits at %d) edge frac=%.3f L1(single,double)=%.3f\n",
 com, center, edge, l1);
 const bool pass = one.detected > 0.05 && two.detected > 0.02 &&
 nulls_two >= 2 && nulls_one == 0 &&
 maxima_two >= 3 && std::abs(com - center) < 3.0 &&
 edge < 0.15 && l1 > 0.3;
 std::printf(" => %s\n", pass ? "PASS" : "FAIL");
 return pass;
}

} // namespace

int main() {
 std::printf("=====================================================================\n");
 std::printf(" MEASUREMENT LAYER : absorbing screen over the unitary substrate\n");
 std::printf(" H_eff = H - i*W ; absorption = detection ; probability relocated\n");
 std::printf("=====================================================================\n");

 int pass = 0;
 constexpr int total = 3;
 pass += testLayerSeparation() ? 1 : 0;
 pass += testProbabilityAccounting() ? 1 : 0;
 pass += testCleanDiffraction() ? 1 : 0;

 std::printf("\n=====================================================================\n");
 std::printf(" RESULT : %d / %d measurement-layer invariants verified\n", pass, total);
 std::printf(" %s\n", pass == total
 ? "MEASUREMENT IS CLEAN -> non-unitarity lives only in the labelled layer"
 : "MEASUREMENT LAYER LEAKS -> fix before trusting detections");
 std::printf("=====================================================================\n");
 return pass == total ? 0 : 1;
}
