#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

using cd = std::complex<double>;
using Vec = std::vector<cd>;
using Mat = std::vector<std::vector<cd>>;

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr int kPorts = 2;
constexpr int kNodes = 6;
constexpr int kIn0 = 0;
constexpr int kIn1 = 1;
constexpr int kBus0 = 2;
constexpr int kBus1 = 3;
constexpr int kOut0 = 4;
constexpr int kOut1 = 5;
constexpr int kFlightSteps = 1024;

double dt() {
 return std::sqrt(2.0) * std::tan(kPi / (2.0 * static_cast<double>(kFlightSteps)));
}

double norm2(const Vec& z) {
 double s = 0.0;
 for (const auto& v : z) s += std::norm(v);
 return s;
}

Vec normalized(Vec z) {
 const double n = std::sqrt(norm2(z));
 for (auto& v : z) v /= n;
 return z;
}

double maxStateDiff(const Vec& a, const Vec& b) {
 double err = 0.0;
 for (std::size_t i = 0; i < a.size(); ++i) err = std::max(err, std::abs(a[i] - b[i]));
 return err;
}

double maxMatrixDiff(const Mat& a, const Mat& b) {
 double err = 0.0;
 for (std::size_t i = 0; i < a.size(); ++i) {
 for (std::size_t j = 0; j < a[i].size(); ++j) {
 err = std::max(err, std::abs(a[i][j] - b[i][j]));
 }
 }
 return err;
}

Vec matvec(const Mat& m, const Vec& z) {
 const auto n = static_cast<int>(m.size());
 const auto cols = static_cast<int>(z.size());
 Vec out(n, cd(0, 0));
 for (int i = 0; i < n; ++i) {
 cd s(0, 0);
 for (int j = 0; j < cols; ++j) s += m[i][j] * z[j];
 out[i] = s;
 }
 return out;
}

Mat matmul(const Mat& a, const Mat& b) {
 const auto n = static_cast<int>(a.size());
 const auto inner = static_cast<int>(b.size());
 const auto cols = static_cast<int>(b[0].size());
 Mat out(n, Vec(cols, cd(0, 0)));
 for (int i = 0; i < n; ++i) {
 for (int k = 0; k < inner; ++k) {
 for (int j = 0; j < cols; ++j) out[i][j] += a[i][k] * b[k][j];
 }
 }
 return out;
}

Mat dagger(const Mat& m) {
 const auto n = static_cast<int>(m.size());
 const auto cols = static_cast<int>(m[0].size());
 Mat out(cols, Vec(n, cd(0, 0)));
 for (int i = 0; i < n; ++i) {
 for (int j = 0; j < cols; ++j) out[j][i] = std::conj(m[i][j]);
 }
 return out;
}

Mat identity(const int n) {
 Mat out(n, Vec(n, cd(0, 0)));
 for (int i = 0; i < n; ++i) out[i][i] = cd(1, 0);
 return out;
}

double unitarityError(const Mat& m) {
 return maxMatrixDiff(matmul(dagger(m), m), identity(static_cast<int>(m[0].size())));
}

Mat scaled(const cd s, Mat m) {
 for (auto& row : m) {
 for (auto& v : row) v *= s;
 }
 return m;
}

double portPower(const Vec& z) {
 double s = 0.0;
 for (const auto& v : z) s += std::norm(v);
 return s;
}

struct Medium {
 Mat h;

 explicit Medium(const int n) : h(n, Vec(n, cd(0, 0))) {}

 void transferEdge(const int to, const int from, const cd value) {
 h[to][from] += value;
 h[from][to] += std::conj(value);
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
 int row = k;
 for (int i = k + 1; i < n; ++i) {
 const double v = std::abs(a[i][k]);
 if (v > best) {
 best = v;
 row = i;
 }
 }
 if (row != k) {
 std::swap(a[row], a[k]);
 std::swap(piv[row], piv[k]);
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

struct Stepper {
 Mat ap;
 Mat am;
 LUC lu;

 void build(const Mat& h, const double step_dt) {
 const auto n = static_cast<int>(h.size());
 ap.assign(n, Vec(n));
 am.assign(n, Vec(n));
 const cd imag(0, 1);
 for (int i = 0; i < n; ++i) {
 for (int j = 0; j < n; ++j) {
 const cd k = imag * h[i][j] * (step_dt / 2.0);
 const cd id = (i == j) ? cd(1, 0) : cd(0, 0);
 ap[i][j] = id + k;
 am[i][j] = id - k;
 }
 }
 lu.factor(ap);
 }

 Vec step(const Vec& z) const {
 return lu.solve(matvec(am, z));
 }

 Vec evolve(Vec z, const int steps) const {
 for (int t = 0; t < steps; ++t) z = step(z);
 return z;
 }
};

Mat su2X(const double theta) {
 const double c = std::cos(theta / 2.0);
 const double s = std::sin(theta / 2.0);
 return {
 {cd(c, 0), cd(0, -s)},
 {cd(0, -s), cd(c, 0)},
 };
}

Mat su2Z(const double phi) {
 return {
 {std::exp(cd(0, -phi / 2.0)), cd(0, 0)},
 {cd(0, 0), std::exp(cd(0, phi / 2.0))},
 };
}

Mat kernelA() {
 return matmul(su2Z(0.37 * kPi), su2X(0.29 * kPi));
}

Mat kernelB() {
 return matmul(su2X(0.41 * kPi), su2Z(-0.23 * kPi));
}

Medium physicalWiredComposite(const Mat& ka, const Mat& kb) {
 Medium m(kNodes);
 const int in[kPorts] = {kIn0, kIn1};
 const int bus[kPorts] = {kBus0, kBus1};
 const int out[kPorts] = {kOut0, kOut1};

 for (int row = 0; row < kPorts; ++row) {
 for (int col = 0; col < kPorts; ++col) {
 m.transferEdge(bus[row], in[col], ka[row][col]);
 m.transferEdge(out[row], bus[col], kb[row][col]);
 }
 }
 return m;
}

Vec injectInput(const Vec& input) {
 Vec z(kNodes, cd(0, 0));
 z[kIn0] = input[0];
 z[kIn1] = input[1];
 return z;
}

Vec extractOutput(const Vec& z) {
 return {z[kOut0], z[kOut1]};
}

Vec extractInput(const Vec& z) {
 return {z[kIn0], z[kIn1]};
}

Vec extractBus(const Vec& z) {
 return {z[kBus0], z[kBus1]};
}

Vec evolvePhysical(const Medium& m, const Vec& input, const int steps) {
 Stepper u;
 u.build(m.h, dt());
 return u.evolve(injectInput(input), steps);
}

Mat physicalTransferMatrix(const Medium& m, const int steps) {
 Mat t(kPorts, Vec(kPorts, cd(0, 0)));
 for (int col = 0; col < kPorts; ++col) {
 Vec input(kPorts, cd(0, 0));
 input[col] = cd(1, 0);
 const Vec out = extractOutput(evolvePhysical(m, input, steps));
 for (int row = 0; row < kPorts; ++row) t[row][col] = out[row];
 }
 return t;
}

double maxResidualPower(const Medium& m, const int steps) {
 double worst = 0.0;
 for (int col = 0; col < kPorts; ++col) {
 Vec input(kPorts, cd(0, 0));
 input[col] = cd(1, 0);
 const Vec z = evolvePhysical(m, input, steps);
 worst = std::max(worst, portPower(extractInput(z)) + portPower(extractBus(z)));
 }
 return worst;
}

double edgePattern(const int i, const int j, const int salt) {
 const int v = ((i + 1) * 37 + (j + 1) * 53 + salt * 97) % 11;
 return (static_cast<double>(v) - 5.0) / 5.0;
}

Medium perturbCouplings(Medium m, const double eps) {
 for (int i = 0; i < kNodes; ++i) {
 for (int j = i + 1; j < kNodes; ++j) {
 if (std::abs(m.h[i][j]) == 0.0) continue;
 const double scale = 1.0 + eps * edgePattern(i, j, 1);
 const cd v = m.h[i][j] * scale;
 m.h[i][j] = v;
 m.h[j][i] = std::conj(v);
 }
 }
 return m;
}

Medium perturbPhases(Medium m, const double eps_rad) {
 for (int i = 0; i < kNodes; ++i) {
 for (int j = i + 1; j < kNodes; ++j) {
 if (std::abs(m.h[i][j]) == 0.0) continue;
 const double phase = eps_rad * edgePattern(i, j, 2);
 const cd v = m.h[i][j] * std::exp(cd(0, phase));
 m.h[i][j] = v;
 m.h[j][i] = std::conj(v);
 }
 }
 return m;
}

Medium scaleAllCouplings(Medium m, const double scale) {
 for (int i = 0; i < kNodes; ++i) {
 for (int j = i + 1; j < kNodes; ++j) {
 if (std::abs(m.h[i][j]) == 0.0) continue;
 const cd v = m.h[i][j] * scale;
 m.h[i][j] = v;
 m.h[j][i] = std::conj(v);
 }
 }
 return m;
}

double randomSigned(std::uint32_t& state) {
 state = state * 1664525u + 1013904223u;
 const double unit = static_cast<double>((state >> 8) & 0x00ffffffu) / static_cast<double>(0x00ffffffu);
 return 2.0 * unit - 1.0;
}

Medium perturbRandomDisorder(Medium m, const double coupling_eps, const double phase_eps_rad,
 const std::uint32_t seed) {
 std::uint32_t state = seed;
 for (int i = 0; i < kNodes; ++i) {
 for (int j = i + 1; j < kNodes; ++j) {
 if (std::abs(m.h[i][j]) == 0.0) continue;
 const double scale = 1.0 + coupling_eps * randomSigned(state);
 const double phase = phase_eps_rad * randomSigned(state);
 const cd v = m.h[i][j] * scale * std::exp(cd(0, phase));
 m.h[i][j] = v;
 m.h[j][i] = std::conj(v);
 }
 }
 return m;
}

struct Metrics {
 double fixed_error = 0.0;
 double fixed_residual = 0.0;
 double best_error = 0.0;
 double best_residual = 0.0;
 int best_steps = kFlightSteps;
 double order_gap = 0.0;
};

struct FixedMetrics {
 double fixed_error = 0.0;
 double fixed_residual = 0.0;
 double order_gap = 0.0;
};

FixedMetrics evaluateFixed(const Medium& m, const Mat& expected, const Mat& reversed, const Vec& sample) {
 FixedMetrics r;
 const Mat fixed = physicalTransferMatrix(m, kFlightSteps);
 r.fixed_error = maxMatrixDiff(fixed, expected);
 r.fixed_residual = maxResidualPower(m, kFlightSteps);
 r.order_gap = maxStateDiff(matvec(fixed, sample), matvec(reversed, sample));
 return r;
}

Metrics evaluate(const Medium& m, const Mat& expected, const Mat& reversed, const Vec& sample) {
 Metrics r;
 const Mat fixed = physicalTransferMatrix(m, kFlightSteps);
 r.fixed_error = maxMatrixDiff(fixed, expected);
 r.fixed_residual = maxResidualPower(m, kFlightSteps);
 r.order_gap = maxStateDiff(matvec(fixed, sample), matvec(reversed, sample));

 r.best_error = std::numeric_limits<double>::infinity();
 for (int steps = kFlightSteps - 96; steps <= kFlightSteps + 96; steps += 4) {
 const Mat t = physicalTransferMatrix(m, steps);
 const double err = maxMatrixDiff(t, expected);
 if (err < r.best_error) {
 r.best_error = err;
 r.best_steps = steps;
 r.best_residual = maxResidualPower(m, steps);
 }
 }
 return r;
}

double transferErrorLimit(const double eps) {
 if (eps <= 0.0011) return 0.015;
 if (eps <= 0.011) return 0.09;
 return 0.35;
}

double residualLimit(const double eps) {
 if (eps <= 0.0011) return 0.03;
 if (eps <= 0.011) return 0.18;
 return 0.55;
}

bool printAndCheckLadder(const char* label, const std::vector<Medium>& media,
 const std::vector<double>& levels, const Mat& expected,
 const Mat& reversed, const Vec& sample) {
 bool pass = true;
 double prev_fixed_error = -1.0;
 for (std::size_t i = 0; i < media.size(); ++i) {
 const Metrics r = evaluate(media[i], expected, reversed, sample);
 const double err_limit = transferErrorLimit(levels[i]);
 const double leak_limit = residualLimit(levels[i]);
 const bool bounded = r.fixed_error < err_limit && r.best_error < err_limit &&
 r.fixed_residual < leak_limit && r.order_gap > 0.25;
 const bool roughly_ordered = prev_fixed_error < 0.0 || r.fixed_error + 0.02 >= prev_fixed_error;
 pass = pass && bounded && roughly_ordered;
 prev_fixed_error = r.fixed_error;

 std::printf(" %s eps=%.3f fixed_err=%.4f best_err=%.4f best_steps=%d fixed_residual=%.4f order_gap=%.4f %s\n",
 label, levels[i], r.fixed_error, r.best_error, r.best_steps,
 r.fixed_residual, r.order_gap, bounded && roughly_ordered ? "OK" : "BAD");
 }
 return pass;
}

bool testIdealBaseline() {
 std::printf("\n[1] IDEAL WIRING BASELINE\n");
 const Mat ka = kernelA();
 const Mat kb = kernelB();
 const Medium ideal = physicalWiredComposite(ka, kb);
 const Mat expected = scaled(cd(-1, 0), matmul(kb, ka));
 const Mat reversed = scaled(cd(-1, 0), matmul(ka, kb));
 const Vec sample = normalized({cd(0.47, -0.11), cd(-0.25, 0.83)});
 const Metrics r = evaluate(ideal, expected, reversed, sample);
 const double order_basis_gap = maxMatrixDiff(matmul(kb, ka), matmul(ka, kb));
 const bool pass = ideal.hermiticityError() < 1e-14 && r.fixed_error < 1e-12 &&
 r.fixed_residual < 1e-20 && r.order_gap > 0.5 &&
 order_basis_gap > 0.5 && unitarityError(physicalTransferMatrix(ideal, kFlightSteps)) < 1e-12;

 std::printf(" hermiticity error=%.2e\n", ideal.hermiticityError());
 std::printf(" fixed_err=%.2e residual=%.2e order_basis_gap=%.3f routed_order_gap=%.3f\n",
 r.fixed_error, r.fixed_residual, order_basis_gap, r.order_gap);
 std::printf(" => %s\n", pass ? "PASS" : "FAIL");
 return pass;
}

bool testCouplingPerturbationLadder() {
 std::printf("\n[2] COUPLING PERTURBATION LADDER\n");
 const Mat ka = kernelA();
 const Mat kb = kernelB();
 const Medium ideal = physicalWiredComposite(ka, kb);
 const Mat expected = scaled(cd(-1, 0), matmul(kb, ka));
 const Mat reversed = scaled(cd(-1, 0), matmul(ka, kb));
 const Vec sample = normalized({cd(0.47, -0.11), cd(-0.25, 0.83)});
 const std::vector<double> levels = {0.001, 0.01, 0.05};
 const std::vector<Medium> media = {
 perturbCouplings(ideal, levels[0]),
 perturbCouplings(ideal, levels[1]),
 perturbCouplings(ideal, levels[2]),
 };
 const bool pass = printAndCheckLadder("coupling", media, levels, expected, reversed, sample);
 std::printf(" => %s\n", pass ? "PASS" : "FAIL");
 return pass;
}

bool testPhasePerturbationLadder() {
 std::printf("\n[3] PHASE PERTURBATION LADDER\n");
 const Mat ka = kernelA();
 const Mat kb = kernelB();
 const Medium ideal = physicalWiredComposite(ka, kb);
 const Mat expected = scaled(cd(-1, 0), matmul(kb, ka));
 const Mat reversed = scaled(cd(-1, 0), matmul(ka, kb));
 const Vec sample = normalized({cd(0.47, -0.11), cd(-0.25, 0.83)});
 const std::vector<double> levels = {0.001, 0.01, 0.05};
 const std::vector<Medium> media = {
 perturbPhases(ideal, levels[0]),
 perturbPhases(ideal, levels[1]),
 perturbPhases(ideal, levels[2]),
 };
 const bool pass = printAndCheckLadder("phase(rad)", media, levels, expected, reversed, sample);
 std::printf(" => %s\n", pass ? "PASS" : "FAIL");
 return pass;
}

bool testCombinedPerturbationLadder() {
 std::printf("\n[4] COMBINED COUPLING + PHASE PERTURBATION LADDER\n");
 const Mat ka = kernelA();
 const Mat kb = kernelB();
 const Medium ideal = physicalWiredComposite(ka, kb);
 const Mat expected = scaled(cd(-1, 0), matmul(kb, ka));
 const Mat reversed = scaled(cd(-1, 0), matmul(ka, kb));
 const Vec sample = normalized({cd(0.47, -0.11), cd(-0.25, 0.83)});
 const std::vector<double> levels = {0.001, 0.01, 0.05};
 const std::vector<Medium> media = {
 perturbPhases(perturbCouplings(ideal, levels[0]), levels[0]),
 perturbPhases(perturbCouplings(ideal, levels[1]), levels[1]),
 perturbPhases(perturbCouplings(ideal, levels[2]), levels[2]),
 };
 const bool pass = printAndCheckLadder("combined", media, levels, expected, reversed, sample);
 std::printf(" => %s\n", pass ? "PASS" : "FAIL");
 return pass;
}

bool testUniformScaleRetiming() {
 std::printf("\n[5] UNIFORM COUPLING SCALE IS A RETIMABLE CLOCK DETUNING\n");
 const Mat ka = kernelA();
 const Mat kb = kernelB();
 const Medium ideal = physicalWiredComposite(ka, kb);
 const Mat expected = scaled(cd(-1, 0), matmul(kb, ka));
 const Mat reversed = scaled(cd(-1, 0), matmul(ka, kb));
 const Vec sample = normalized({cd(0.47, -0.11), cd(-0.25, 0.83)});
 const Medium scaled_up = scaleAllCouplings(ideal, 1.02);
 const Metrics r = evaluate(scaled_up, expected, reversed, sample);
 const int expected_steps = static_cast<int>(std::lround(static_cast<double>(kFlightSteps) / 1.02));
 const bool pass = r.fixed_error > 1e-4 && r.best_error < 1e-6 &&
 std::abs(r.best_steps - expected_steps) <= 8 && r.order_gap > 0.5;

 std::printf(" scale=1.020 fixed_err=%.4f best_err=%.4f best_steps=%d expected_steps~%d order_gap=%.4f\n",
 r.fixed_error, r.best_error, r.best_steps, expected_steps, r.order_gap);
 std::printf(" => %s\n", pass ? "PASS" : "FAIL");
 return pass;
}

bool testRandomDisorderEnsemble() {
 std::printf("\n[6] DETERMINISTIC RANDOM DISORDER ENSEMBLE\n");
 const Mat ka = kernelA();
 const Mat kb = kernelB();
 const Medium ideal = physicalWiredComposite(ka, kb);
 const Mat expected = scaled(cd(-1, 0), matmul(kb, ka));
 const Mat reversed = scaled(cd(-1, 0), matmul(ka, kb));
 const Vec sample = normalized({cd(0.47, -0.11), cd(-0.25, 0.83)});

 struct Row {
 double eps = 0.0;
 double mean_error = 0.0;
 double max_error = 0.0;
 double max_residual = 0.0;
 double min_order_gap = std::numeric_limits<double>::infinity();
 };

 std::vector<Row> rows;
 for (const double eps : {0.01, 0.05}) {
 Row row;
 row.eps = eps;
 constexpr int kSeeds = 24;
 for (int seed = 1; seed <= kSeeds; ++seed) {
 const Medium m = perturbRandomDisorder(ideal, eps, eps, static_cast<std::uint32_t>(seed * 7919));
 const FixedMetrics r = evaluateFixed(m, expected, reversed, sample);
 row.mean_error += r.fixed_error / static_cast<double>(kSeeds);
 row.max_error = std::max(row.max_error, r.fixed_error);
 row.max_residual = std::max(row.max_residual, r.fixed_residual);
 row.min_order_gap = std::min(row.min_order_gap, r.order_gap);
 }
 rows.push_back(row);
 }

 const bool mild_pass = rows[0].max_error < 0.06 && rows[0].max_residual < 0.03 && rows[0].min_order_gap > 0.55;
 const bool stress_pass = rows[1].max_error < 0.25 && rows[1].max_residual < 0.12 && rows[1].min_order_gap > 0.35;
 const bool pass = mild_pass && stress_pass;

 for (const Row& row : rows) {
 std::printf(" random eps=%.3f seeds=24 mean_err=%.4f max_err=%.4f max_residual=%.4f min_order_gap=%.4f %s\n",
 row.eps, row.mean_error, row.max_error, row.max_residual, row.min_order_gap,
 (row.eps < 0.02 ? mild_pass : stress_pass) ? "OK" : "BAD");
 }
 std::printf(" => %s\n", pass ? "PASS" : "FAIL");
 return pass;
}

} // namespace

int main() {
 std::printf("=====================================================================\n");
 std::printf(" GRAPH-WAVE PORT WIRING ROBUSTNESS CONTRACT TEST\n");
 std::printf(" Falsification check: small coupling/phase defects around physical wiring.\n");
 std::printf(" No absorber, no measurement layer, no decision, no learning.\n");
 std::printf("=====================================================================\n");

 int pass = 0;
 constexpr int total = 6;
 pass += testIdealBaseline() ? 1 : 0;
 pass += testCouplingPerturbationLadder() ? 1 : 0;
 pass += testPhasePerturbationLadder() ? 1 : 0;
 pass += testCombinedPerturbationLadder() ? 1 : 0;
 pass += testUniformScaleRetiming() ? 1 : 0;
 pass += testRandomDisorderEnsemble() ? 1 : 0;

 std::printf("\n=====================================================================\n");
 std::printf(" RESULT : %d / %d verified\n", pass, total);
 std::printf(" %s\n", pass == total
 ? "physical wiring has a bounded local tolerance basin; clocking still matters"
 : "physical wiring is too brittle under the tested perturbations");
 std::printf("=====================================================================\n");
 return pass == total ? 0 : 1;
}
