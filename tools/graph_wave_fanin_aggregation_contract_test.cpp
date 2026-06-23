#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <utility>
#include <vector>

namespace {

using cd = std::complex<double>;
using Vec = std::vector<cd>;
using Mat = std::vector<std::vector<cd>>;

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr int kInputs = 3;
constexpr int kModes = 3;
constexpr int kNodes = kInputs + kModes;
constexpr int kSteps = 512;
constexpr int kIn0 = 0;
constexpr int kOut0 = 3;

double dt() {
 return 2.0 * std::tan(kPi / (4.0 * static_cast<double>(kSteps)));
}

double effectiveAngle() {
 return static_cast<double>(kSteps) * 2.0 * std::atan(dt() / 2.0);
}

cd transferScale() {
 return cd(0, -std::sin(effectiveAngle()));
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

Mat faninKernel(const std::vector<double>& phases) {
 Mat w(kModes, Vec(kInputs, cd(0, 0)));
 const double scale = 1.0 / std::sqrt(3.0);
 for (int mode = 0; mode < kModes; ++mode) {
 for (int in = 0; in < kInputs; ++in) {
 const double fourier = 2.0 * kPi * static_cast<double>(mode * in) / 3.0;
 w[mode][in] = scale * std::exp(cd(0, phases[in] + fourier));
 }
 }
 return w;
}

Medium faninMedium(const Mat& w) {
 Medium m(kNodes);
 for (int mode = 0; mode < kModes; ++mode) {
 for (int in = 0; in < kInputs; ++in) {
 m.transferEdge(kOut0 + mode, kIn0 + in, w[mode][in]);
 }
 }
 return m;
}

Medium faninMedium(const std::vector<double>& phases) {
 return faninMedium(faninKernel(phases));
}

Vec injectInput(const Vec& input) {
 Vec z(kNodes, cd(0, 0));
 for (int i = 0; i < kInputs; ++i) z[kIn0 + i] = input[i];
 return z;
}

Vec extractOutput(const Vec& z) {
 Vec out(kModes, cd(0, 0));
 for (int i = 0; i < kModes; ++i) out[i] = z[kOut0 + i];
 return out;
}

Vec extractInput(const Vec& z) {
 Vec out(kInputs, cd(0, 0));
 for (int i = 0; i < kInputs; ++i) out[i] = z[kIn0 + i];
 return out;
}

Vec evolvePhysical(const Medium& m, const Vec& input) {
 Stepper u;
 u.build(m.h, dt());
 return u.evolve(injectInput(input), kSteps);
}

Vec routeInput(const Medium& m, const Vec& input) {
 return extractOutput(evolvePhysical(m, input));
}

Mat transferMatrix(const Medium& m) {
 Mat t(kModes, Vec(kInputs, cd(0, 0)));
 for (int col = 0; col < kInputs; ++col) {
 Vec input(kInputs, cd(0, 0));
 input[col] = cd(1, 0);
 const Vec out = routeInput(m, input);
 for (int row = 0; row < kModes; ++row) t[row][col] = out[row];
 }
 return t;
}

double maxInputResidualPower(const Medium& m) {
 double worst = 0.0;
 for (int col = 0; col < kInputs; ++col) {
 Vec input(kInputs, cd(0, 0));
 input[col] = cd(1, 0);
 worst = std::max(worst, portPower(extractInput(evolvePhysical(m, input))));
 }
 return worst;
}

std::vector<double> localGauge() {
 return {0.23, -0.71, 1.13, -0.37, 0.89, -1.21};
}

Mat gaugeTransform(const Mat& h, const std::vector<double>& chi) {
 const auto n = static_cast<int>(h.size());
 Mat out(n, Vec(n, cd(0, 0)));
 for (int i = 0; i < n; ++i) {
 for (int j = 0; j < n; ++j) {
 out[i][j] = std::exp(cd(0, chi[i])) * h[i][j] * std::exp(cd(0, -chi[j]));
 }
 }
 return out;
}

Vec gaugeInput(const Vec& x, const std::vector<double>& chi) {
 Vec out(kInputs, cd(0, 0));
 for (int i = 0; i < kInputs; ++i) out[i] = std::exp(cd(0, chi[kIn0 + i])) * x[i];
 return out;
}

Vec gaugeOutput(const Vec& y, const std::vector<double>& chi) {
 Vec out(kModes, cd(0, 0));
 for (int i = 0; i < kModes; ++i) out[i] = std::exp(cd(0, chi[kOut0 + i])) * y[i];
 return out;
}

Mat expectedGaugeTransfer(const Mat& t, const std::vector<double>& chi) {
 Mat out(kModes, Vec(kInputs, cd(0, 0)));
 for (int row = 0; row < kModes; ++row) {
 for (int col = 0; col < kInputs; ++col) {
 out[row][col] = std::exp(cd(0, chi[kOut0 + row])) * t[row][col] *
 std::exp(cd(0, -chi[kIn0 + col]));
 }
 }
 return out;
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

bool testUnitaryFanInTransfer() {
 std::printf("\n[1] UNITARY FAN-IN TRANSFER FRAME\n");
 const std::vector<double> phases = {0.11 * kPi, -0.27 * kPi, 0.43 * kPi};
 const Mat w = faninKernel(phases);
 const Medium m = faninMedium(w);
 const Mat t = transferMatrix(m);
 const Mat expected = scaled(transferScale(), w);
 const double herm_err = m.hermiticityError();
 const double transfer_err = maxMatrixDiff(t, expected);
 const double unitary_err = unitarityError(t);
 const double residual = maxInputResidualPower(m);
 const bool pass = herm_err < 1e-14 && transfer_err < 1e-12 && unitary_err < 1e-12 && residual < 1e-12;

 std::printf(" hermiticity error=%.2e effective_angle=%.12f\n", herm_err, effectiveAngle());
 std::printf(" max |T - (-i W)|=%.2e max |T^dag T - I|=%.2e residual_input=%.2e\n",
 transfer_err, unitary_err, residual);
 std::printf(" => %s\n", pass ? "PASS" : "FAIL");
 return pass;
}

bool testPhaseControlledAggregation() {
 std::printf("\n[2] PHASE-CONTROLLED AGGREGATION PORT\n");
 const Vec equal = normalized({cd(1, 0), cd(1, 0), cd(1, 0)});
 const Vec aligned = routeInput(faninMedium({0.0, 0.0, 0.0}), equal);
 const Vec destructive = routeInput(faninMedium({0.0, 2.0 * kPi / 3.0, 4.0 * kPi / 3.0}), equal);
 const double aligned_agg = std::norm(aligned[0]);
 const double aligned_residual = std::norm(aligned[1]) + std::norm(aligned[2]);
 const double destructive_agg = std::norm(destructive[0]);
 const double destructive_residual = std::norm(destructive[1]) + std::norm(destructive[2]);
 const bool pass = aligned_agg > 0.999999 && aligned_residual < 1e-12 &&
 destructive_agg < 1e-12 && destructive_residual > 0.999999;

 std::printf(" aligned aggregate=%.12f residual_modes=%.3e\n", aligned_agg, aligned_residual);
 std::printf(" destructive aggregate=%.3e residual_modes=%.12f\n", destructive_agg, destructive_residual);
 std::printf(" => %s\n", pass ? "PASS" : "FAIL");
 return pass;
}

bool testResidualModesPreventFalseCompression() {
 std::printf("\n[3] RESIDUAL MODES PREVENT FALSE UNITARY COMPRESSION\n");
 const Medium m = faninMedium({0.0, 0.0, 0.0});
 const Vec basis = {cd(1, 0), cd(0, 0), cd(0, 0)};
 const Vec out = routeInput(m, basis);
 const double aggregate = std::norm(out[0]);
 const double residual = std::norm(out[1]) + std::norm(out[2]);
 const double total = portPower(out);
 const bool pass = std::abs(aggregate - 1.0 / 3.0) < 1e-12 &&
 std::abs(residual - 2.0 / 3.0) < 1e-12 &&
 std::abs(total - 1.0) < 1e-12;

 std::printf(" basis input aggregate=%.12f residual_modes=%.12f total_output=%.12f\n",
 aggregate, residual, total);
 std::printf(" no_compression_boundary=CONFIRMED\n");
 std::printf(" => %s\n", pass ? "PASS" : "FAIL");
 return pass;
}

bool testPermutationOfNeighborsKeepsAggregate() {
 std::printf("\n[4] NEIGHBOR PERMUTATION KEEPS THE AGGREGATE WHEN COUPLINGS MOVE WITH INPUT LABELS\n");
 const std::vector<double> phases = {0.17 * kPi, -0.31 * kPi, 0.08 * kPi};
 const Vec input = normalized({cd(0.4, -0.2), cd(-0.1, 0.8), cd(0.5, 0.3)});
 const int perm[kInputs] = {2, 0, 1};
 std::vector<double> phases_p(kInputs);
 Vec input_p(kInputs, cd(0, 0));
 for (int i = 0; i < kInputs; ++i) {
 phases_p[i] = phases[perm[i]];
 input_p[i] = input[perm[i]];
 }
 const Vec out = routeInput(faninMedium(phases), input);
 const Vec out_p = routeInput(faninMedium(phases_p), input_p);
 const double aggregate_amp_err = std::abs(out[0] - out_p[0]);
 const double aggregate_power_err = std::abs(std::norm(out[0]) - std::norm(out_p[0]));
 const bool pass = aggregate_amp_err < 1e-12 && aggregate_power_err < 1e-12;

 std::printf(" aggregate amplitude error=%.2e power error=%.2e\n",
 aggregate_amp_err, aggregate_power_err);
 std::printf(" residual basis is allowed to rotate under permutation\n");
 std::printf(" => %s\n", pass ? "PASS" : "FAIL");
 return pass;
}

bool testGaugeCovariantFanIn() {
 std::printf("\n[5] FAN-IN TRANSFER IS GAUGE-COVARIANT\n");
 const Medium m = faninMedium({0.11 * kPi, -0.27 * kPi, 0.43 * kPi});
 const auto chi = localGauge();
 Medium mg(kNodes);
 mg.h = gaugeTransform(m.h, chi);

 const Mat t = transferMatrix(m);
 const Mat tg = transferMatrix(mg);
 const Mat expected_tg = expectedGaugeTransfer(t, chi);
 const double matrix_err = maxMatrixDiff(tg, expected_tg);

 const Vec input = normalized({cd(0.4, -0.2), cd(-0.1, 0.8), cd(0.5, 0.3)});
 const Vec out = matvec(t, input);
 const Vec out_g = matvec(tg, gaugeInput(input, chi));
 const Vec expected_out_g = gaugeOutput(out, chi);
 const double state_err = maxStateDiff(out_g, expected_out_g);
 const double power_err = std::abs(std::norm(out_g[0]) - std::norm(out[0])) +
 std::abs(std::norm(out_g[1]) - std::norm(out[1])) +
 std::abs(std::norm(out_g[2]) - std::norm(out[2]));
 const bool pass = matrix_err < 1e-12 && state_err < 1e-12 && power_err < 1e-12;

 std::printf(" max |T_gauge - G_out T G_in^-1|=%.2e\n", matrix_err);
 std::printf(" gauge-covariant output state error=%.2e output-power L1 diff=%.2e\n",
 state_err, power_err);
 std::printf(" => %s\n", pass ? "PASS" : "FAIL");
 return pass;
}

bool testRandomDisorderKeepsAggregateSignal() {
 std::printf("\n[6] RANDOM DISORDER KEEPS AGGREGATE SIGNAL BOUNDED\n");
 const Medium ideal = faninMedium({0.0, 0.0, 0.0});
 const Mat expected = scaled(transferScale(), faninKernel({0.0, 0.0, 0.0}));
 const Vec aligned = normalized({cd(1, 0), cd(1, 0), cd(1, 0)});
 const Vec destructive = normalized({cd(1, 0), cd(1, 0), cd(1, 0)});

 struct Row {
 double eps = 0.0;
 double mean_error = 0.0;
 double max_error = 0.0;
 double min_aligned_agg = std::numeric_limits<double>::infinity();
 double max_destructive_agg = 0.0;
 double max_input_residual = 0.0;
 };

 std::vector<Row> rows;
 for (const double eps : {0.01, 0.05}) {
 Row row;
 row.eps = eps;
 constexpr int kSeeds = 24;
 for (int seed = 1; seed <= kSeeds; ++seed) {
 const Medium m = perturbRandomDisorder(ideal, eps, eps, static_cast<std::uint32_t>(seed * 65537));
 const Mat t = transferMatrix(m);
 const Vec out_aligned = routeInput(m, aligned);
 const Vec out_destructive =
 routeInput(perturbRandomDisorder(faninMedium({0.0, 2.0 * kPi / 3.0, 4.0 * kPi / 3.0}),
 eps, eps, static_cast<std::uint32_t>(seed * 65537)),
 destructive);
 const double err = maxMatrixDiff(t, expected);
 row.mean_error += err / static_cast<double>(kSeeds);
 row.max_error = std::max(row.max_error, err);
 row.min_aligned_agg = std::min(row.min_aligned_agg, std::norm(out_aligned[0]));
 row.max_destructive_agg = std::max(row.max_destructive_agg, std::norm(out_destructive[0]));
 row.max_input_residual = std::max(row.max_input_residual, maxInputResidualPower(m));
 }
 rows.push_back(row);
 }

 const bool mild_pass = rows[0].max_error < 0.04 && rows[0].min_aligned_agg > 0.95 &&
 rows[0].max_destructive_agg < 0.002 && rows[0].max_input_residual < 0.01;
 const bool stress_pass = rows[1].max_error < 0.18 && rows[1].min_aligned_agg > 0.82 &&
 rows[1].max_destructive_agg < 0.04 && rows[1].max_input_residual < 0.05;
 const bool pass = mild_pass && stress_pass;

 for (const Row& row : rows) {
 const bool row_pass = row.eps < 0.02 ? mild_pass : stress_pass;
 std::printf(" random eps=%.3f seeds=24 mean_err=%.4f max_err=%.4f min_aligned_agg=%.4f max_destructive_agg=%.4f max_input_residual=%.4f %s\n",
 row.eps, row.mean_error, row.max_error, row.min_aligned_agg,
 row.max_destructive_agg, row.max_input_residual, row_pass ? "OK" : "BAD");
 }
 std::printf(" => %s\n", pass ? "PASS" : "FAIL");
 return pass;
}

} // namespace

int main() {
 std::printf("=====================================================================\n");
 std::printf(" GRAPH-WAVE FAN-IN AGGREGATION CONTRACT TEST\n");
 std::printf(" Three input rails -> aggregate mode + two orthogonal residual modes.\n");
 std::printf(" No absorber, no measurement layer, no decision, no learning.\n");
 std::printf("=====================================================================\n");

 int pass = 0;
 constexpr int total = 6;
 pass += testUnitaryFanInTransfer() ? 1 : 0;
 pass += testPhaseControlledAggregation() ? 1 : 0;
 pass += testResidualModesPreventFalseCompression() ? 1 : 0;
 pass += testPermutationOfNeighborsKeepsAggregate() ? 1 : 0;
 pass += testGaugeCovariantFanIn() ? 1 : 0;
 pass += testRandomDisorderKeepsAggregateSignal() ? 1 : 0;

 std::printf("\n=====================================================================\n");
 std::printf(" RESULT : %d / %d verified\n", pass, total);
 std::printf(" %s\n", pass == total
 ? "fan-in aggregation is a unitary graph primitive with explicit residual modes"
 : "fan-in aggregation contract is not stable enough yet");
 std::printf("=====================================================================\n");
 return pass == total ? 0 : 1;
}
