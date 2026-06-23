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
constexpr int kInputs = 7;
constexpr int kModes = 7;
constexpr int kNodes = kInputs + kModes;
constexpr int kSteps = 512;
constexpr int kIn0 = 0;
constexpr int kOut0 = 7;

constexpr int kASelf = 0;
constexpr int kBSelf = 1;
constexpr int kCSelf = 2;
constexpr int kAToB = 3;
constexpr int kBToA = 4;
constexpr int kBToC = 5;
constexpr int kCToB = 6;

constexpr int kAUpdate = 0;
constexpr int kBUpdate = 1;
constexpr int kCUpdate = 2;

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

cd rowInner(const Vec& a, const Vec& b) {
 cd s(0, 0);
 for (int i = 0; i < static_cast<int>(a.size()); ++i) s += a[i] * std::conj(b[i]);
 return s;
}

bool addOrthogonalRow(Mat& rows, Vec candidate) {
 for (const auto& basis : rows) {
 const cd coeff = rowInner(candidate, basis);
 for (int i = 0; i < static_cast<int>(candidate.size()); ++i) candidate[i] -= coeff * basis[i];
 }

 const double n = std::sqrt(std::max(0.0, rowInner(candidate, candidate).real()));
 if (n < 1e-10) return false;
 for (auto& v : candidate) v /= n;
 rows.push_back(candidate);
 return true;
}

struct PathParams {
 double leaf_self_gain = 1.0;
 double leaf_in_gain = 0.8;
 double center_self_gain = 1.1;
 double center_in_gain = 0.7;
 double phase_b_to_a = 0.17 * kPi;
 double phase_b_to_c = -0.11 * kPi;
 double phase_a_to_b = -0.23 * kPi;
 double phase_c_to_b = 0.29 * kPi;
};

Vec pathRowA(const PathParams& p) {
 const double n = std::sqrt(p.leaf_self_gain * p.leaf_self_gain + p.leaf_in_gain * p.leaf_in_gain);
 Vec row(kInputs, cd(0, 0));
 row[kASelf] = cd(p.leaf_self_gain / n, 0);
 row[kBToA] = (p.leaf_in_gain / n) * std::exp(cd(0, p.phase_b_to_a));
 return row;
}

Vec pathRowB(const PathParams& p) {
 const double n = std::sqrt(p.center_self_gain * p.center_self_gain +
 2.0 * p.center_in_gain * p.center_in_gain);
 Vec row(kInputs, cd(0, 0));
 row[kBSelf] = cd(p.center_self_gain / n, 0);
 row[kAToB] = (p.center_in_gain / n) * std::exp(cd(0, p.phase_a_to_b));
 row[kCToB] = (p.center_in_gain / n) * std::exp(cd(0, p.phase_c_to_b));
 return row;
}

Vec pathRowC(const PathParams& p) {
 const double n = std::sqrt(p.leaf_self_gain * p.leaf_self_gain + p.leaf_in_gain * p.leaf_in_gain);
 Vec row(kInputs, cd(0, 0));
 row[kCSelf] = cd(p.leaf_self_gain / n, 0);
 row[kBToC] = (p.leaf_in_gain / n) * std::exp(cd(0, p.phase_b_to_c));
 return row;
}

Mat completeKernelFromRows(const Vec& row_a, const Vec& row_b, const Vec& row_c) {
 Mat rows;
 rows.push_back(row_a);
 rows.push_back(row_b);
 rows.push_back(row_c);

 for (int i = 0; i < kInputs && static_cast<int>(rows.size()) < kModes; ++i) {
 Vec candidate(kInputs, cd(0, 0));
 candidate[i] = cd(1, 0);
 addOrthogonalRow(rows, candidate);
 }

 for (int mode = 1; static_cast<int>(rows.size()) < kModes && mode < 32; ++mode) {
 Vec candidate(kInputs, cd(0, 0));
 for (int j = 0; j < kInputs; ++j) {
 const double angle = 2.0 * kPi * static_cast<double>(mode * j) / static_cast<double>(kInputs);
 candidate[j] = std::exp(cd(0, angle));
 }
 addOrthogonalRow(rows, candidate);
 }

 return rows;
}

Mat pathKernel(const PathParams& p) {
 return completeKernelFromRows(pathRowA(p), pathRowB(p), pathRowC(p));
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

Medium pathMedium(const Mat& w) {
 Medium m(kNodes);
 for (int mode = 0; mode < kModes; ++mode) {
 for (int in = 0; in < kInputs; ++in) {
 m.transferEdge(kOut0 + mode, kIn0 + in, w[mode][in]);
 }
 }
 return m;
}

Medium pathMedium(const PathParams& p) {
 return pathMedium(pathKernel(p));
}

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
 return {0.23, -0.71, 1.13, -0.37, 0.89, -1.21, 0.52,
 -0.46, 0.34, -0.58, 0.77, -0.91, 0.19, -0.27};
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

bool testNaiveDegreeTwoBroadcastFails() {
 std::printf("\n[1] NAIVE DEGREE-2 BROADCAST IS REJECTED\n");
 const Vec row_a = normalized({cd(1, 0), cd(1, 0), cd(0, 0)});
 const Vec row_c = normalized({cd(0, 0), cd(1, 0), cd(1, 0)});
 const double overlap = std::abs(rowInner(row_a, row_c));
 const bool pass = overlap > 0.45;

 std::printf(" naive leaf rows sharing one B rail overlap=%.3f\n", overlap);
 std::printf(" expected_fail_detected=%s\n", pass ? "YES" : "NO");
 std::printf(" => %s\n", pass ? "PASS" : "FAIL");
 return pass;
}

bool testEdgeLiftedPathTransfer() {
 std::printf("\n[2] EDGE-LIFTED DEGREE-2 PATH TRANSFER FRAME\n");
 const PathParams p;
 const Mat w = pathKernel(p);
 const Medium m = pathMedium(w);
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

bool testIncomingEdgeChannelsSetUpdates() {
 std::printf("\n[3] ORIENTED EDGE MESSAGE RAILS SET NODE UPDATES\n");
 const PathParams p;
 const Mat w = pathKernel(p);
 const Medium m = pathMedium(w);
 bool pass = true;

 struct Probe {
 int input = 0;
 int update = 0;
 const char* label = "";
 };

 const Probe probes[] = {
 {kBToA, kAUpdate, "B->A"},
 {kAToB, kBUpdate, "A->B"},
 {kCToB, kBUpdate, "C->B"},
 {kBToC, kCUpdate, "B->C"},
 };

 for (const auto& probe : probes) {
 Vec input(kInputs, cd(0, 0));
 input[probe.input] = cd(1, 0);
 const Vec out = routeInput(m, input);
 const double observed = std::norm(out[probe.update]);
 const double expected = std::norm(w[probe.update][probe.input]);
 pass = pass && std::abs(observed - expected) < 1e-12;
 std::printf(" %s update_power=%.12f expected=%.12f\n", probe.label, observed, expected);
 }

 std::printf(" => %s\n", pass ? "PASS" : "FAIL");
 return pass;
}

bool testEdgePhaseControlsLeafInterference() {
 std::printf("\n[4] EDGE PHASE CONTROLS LOCAL LEAF INTERFERENCE\n");
 PathParams aligned;
 aligned.phase_b_to_a = 0.0;
 PathParams flipped = aligned;
 flipped.phase_b_to_a = kPi;

 Vec input(kInputs, cd(0, 0));
 input[kASelf] = cd(1, 0);
 input[kBToA] = cd(1, 0);
 input = normalized(input);

 const Vec aligned_out = routeInput(pathMedium(aligned), input);
 const Vec flipped_out = routeInput(pathMedium(flipped), input);
 const double aligned_update = std::norm(aligned_out[kAUpdate]);
 const double flipped_update = std::norm(flipped_out[kAUpdate]);
 const bool pass = aligned_update > 0.98 && flipped_update < 0.02;

 std::printf(" phase 0 A_update=%.12f\n", aligned_update);
 std::printf(" phase pi A_update=%.12f\n", flipped_update);
 std::printf(" => %s\n", pass ? "PASS" : "FAIL");
 return pass;
}

bool testPathReversalEquivariance() {
 std::printf("\n[5] PATH REVERSAL PRESERVES UPDATE PORTS WHEN EDGE LABELS MOVE\n");
 const PathParams p;
 PathParams rp = p;
 rp.phase_b_to_a = p.phase_b_to_c;
 rp.phase_b_to_c = p.phase_b_to_a;
 rp.phase_a_to_b = p.phase_c_to_b;
 rp.phase_c_to_b = p.phase_a_to_b;

 const Vec input = normalized({cd(0.6, -0.1), cd(-0.2, 0.7), cd(0.5, 0.3),
 cd(0.1, -0.4), cd(0.3, 0.2), cd(-0.5, 0.1),
 cd(0.2, 0.6)});
 Vec reversed(kInputs, cd(0, 0));
 reversed[kASelf] = input[kCSelf];
 reversed[kBSelf] = input[kBSelf];
 reversed[kCSelf] = input[kASelf];
 reversed[kAToB] = input[kCToB];
 reversed[kBToA] = input[kBToC];
 reversed[kBToC] = input[kBToA];
 reversed[kCToB] = input[kAToB];

 const Vec out = routeInput(pathMedium(p), input);
 const Vec out_r = routeInput(pathMedium(rp), reversed);
 const double a_err = std::abs(out[kAUpdate] - out_r[kCUpdate]);
 const double b_err = std::abs(out[kBUpdate] - out_r[kBUpdate]);
 const double c_err = std::abs(out[kCUpdate] - out_r[kAUpdate]);
 const bool pass = a_err < 1e-12 && b_err < 1e-12 && c_err < 1e-12;

 std::printf(" A<->C update amplitude errors: A=%.2e B=%.2e C=%.2e\n", a_err, b_err, c_err);
 std::printf(" residual basis may rotate under path reversal\n");
 std::printf(" => %s\n", pass ? "PASS" : "FAIL");
 return pass;
}

bool testGaugeCovariantDegreeTwoPath() {
 std::printf("\n[6] EDGE-LIFTED DEGREE-2 PATH IS GAUGE-COVARIANT\n");
 const Medium m = pathMedium(PathParams{});
 const auto chi = localGauge();
 Medium mg(kNodes);
 mg.h = gaugeTransform(m.h, chi);

 const Mat t = transferMatrix(m);
 const Mat tg = transferMatrix(mg);
 const Mat expected_tg = expectedGaugeTransfer(t, chi);
 const double matrix_err = maxMatrixDiff(tg, expected_tg);

 const Vec input = normalized({cd(0.6, -0.1), cd(-0.2, 0.7), cd(0.5, 0.3),
 cd(0.1, -0.4), cd(0.3, 0.2), cd(-0.5, 0.1),
 cd(0.2, 0.6)});
 const Vec out = matvec(t, input);
 const Vec out_g = matvec(tg, gaugeInput(input, chi));
 const Vec expected_out_g = gaugeOutput(out, chi);
 const double state_err = maxStateDiff(out_g, expected_out_g);
 double power_err = 0.0;
 for (int i = 0; i < kModes; ++i) power_err += std::abs(std::norm(out_g[i]) - std::norm(out[i]));
 const bool pass = matrix_err < 1e-12 && state_err < 1e-12 && power_err < 1e-12;

 std::printf(" max |T_gauge - G_out T G_in^-1|=%.2e\n", matrix_err);
 std::printf(" gauge-covariant output state error=%.2e output-power L1 diff=%.2e\n",
 state_err, power_err);
 std::printf(" => %s\n", pass ? "PASS" : "FAIL");
 return pass;
}

bool testRandomDisorderKeepsPathBounded() {
 std::printf("\n[7] RANDOM DISORDER KEEPS DEGREE-2 PATH UPDATE BOUNDED\n");
 const Medium ideal = pathMedium(PathParams{});
 const Vec input = normalized({cd(0.6, -0.1), cd(-0.2, 0.7), cd(0.5, 0.3),
 cd(0.1, -0.4), cd(0.3, 0.2), cd(-0.5, 0.1),
 cd(0.2, 0.6)});
 const Vec ideal_out = routeInput(ideal, input);

 struct Row {
 double eps = 0.0;
 double mean_update_err = 0.0;
 double max_update_err = 0.0;
 double max_power_err = 0.0;
 double max_input_residual = 0.0;
 };

 Row rows[2];
 rows[0].eps = 0.01;
 rows[1].eps = 0.05;
 for (auto& row : rows) {
 for (int seed = 0; seed < 24; ++seed) {
 const Medium perturbed = perturbRandomDisorder(ideal, row.eps, row.eps, 0xD362A1u + seed * 173u);
 const Vec full = evolvePhysical(perturbed, input);
 const Vec out = extractOutput(full);
 double update_err = 0.0;
 double power_err = 0.0;
 for (int i = 0; i < 3; ++i) {
 update_err = std::max(update_err, std::abs(out[i] - ideal_out[i]));
 power_err += std::abs(std::norm(out[i]) - std::norm(ideal_out[i]));
 }
 row.mean_update_err += update_err;
 row.max_update_err = std::max(row.max_update_err, update_err);
 row.max_power_err = std::max(row.max_power_err, power_err);
 row.max_input_residual = std::max(row.max_input_residual, portPower(extractInput(full)));
 }
 row.mean_update_err /= 24.0;
 }

 const bool mild_pass = rows[0].max_update_err < 0.05 && rows[0].max_input_residual < 0.03;
 const bool stress_pass = rows[1].max_update_err < 0.22 && rows[1].max_input_residual < 0.12;
 const bool pass = mild_pass && stress_pass;
 for (const auto& row : rows) {
 const bool row_pass = row.eps < 0.02 ? mild_pass : stress_pass;
 std::printf(" random eps=%.3f seeds=24 mean_update_err=%.4f max_update_err=%.4f max_power_err=%.4f max_input_residual=%.4f %s\n",
 row.eps, row.mean_update_err, row.max_update_err, row.max_power_err,
 row.max_input_residual, row_pass ? "OK" : "BAD");
 }
 std::printf(" => %s\n", pass ? "PASS" : "FAIL");
 return pass;
}

} // namespace

int main() {
 std::printf("=====================================================================\n");
 std::printf(" GRAPH-WAVE DEGREE-2 PATH GUARDRAIL CONTRACT TEST\n");
 std::printf(" Three-node GNN-analog path: reject one-rail broadcast, use edge-message rails.\n");
 std::printf(" No no trainer, no production decision.\n");
 std::printf("=====================================================================\n");

 int pass = 0;
 const int total = 7;
 pass += testNaiveDegreeTwoBroadcastFails() ? 1 : 0;
 pass += testEdgeLiftedPathTransfer() ? 1 : 0;
 pass += testIncomingEdgeChannelsSetUpdates() ? 1 : 0;
 pass += testEdgePhaseControlsLeafInterference() ? 1 : 0;
 pass += testPathReversalEquivariance() ? 1 : 0;
 pass += testGaugeCovariantDegreeTwoPath() ? 1 : 0;
 pass += testRandomDisorderKeepsPathBounded() ? 1 : 0;

 std::printf("\n=====================================================================\n");
 std::printf(" RESULT : %d / %d verified\n", pass, total);
 std::printf(" %s\n", pass == total
 ? "degree-2 path requires edge-lifted messages and preserves GNN-analog updates"
 : "degree-2 path guardrail is not stable enough yet");
 std::printf("=====================================================================\n");
 return pass == total ? 0 : 1;
}
