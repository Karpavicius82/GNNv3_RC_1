#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
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
 // Cayley eigenphase calibration for the three-block path eigenvalue sqrt(2).
 return std::sqrt(2.0) * std::tan(kPi / (2.0 * static_cast<double>(kFlightSteps)));
}

double effectivePathAngle() {
 return static_cast<double>(kFlightSteps) *
 2.0 * std::atan(std::sqrt(2.0) * dt() / 2.0);
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

double wrapPhase(double x) {
 while (x <= -kPi) x += 2.0 * kPi;
 while (x > kPi) x -= 2.0 * kPi;
 return x;
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

Mat idealKernel(const double flux) {
 const cd e = std::exp(cd(0, flux));
 return {
 {0.5 * (cd(1, 0) + e), 0.5 * (cd(1, 0) - e)},
 {0.5 * (cd(1, 0) - e), 0.5 * (cd(1, 0) + e)},
 };
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

Medium physicalWiredComposite(const double flux_a, const double flux_b) {
 return physicalWiredComposite(idealKernel(flux_a), idealKernel(flux_b));
}

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

Mat noncommutingKernelA() {
 return matmul(su2Z(0.37 * kPi), su2X(0.29 * kPi));
}

Mat noncommutingKernelB() {
 return matmul(su2X(0.41 * kPi), su2Z(-0.23 * kPi));
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

Vec evolvePhysical(const Medium& m, const Vec& input, const int steps = kFlightSteps) {
 Stepper u;
 u.build(m.h, dt());
 return u.evolve(injectInput(input), steps);
}

Mat physicalTransferMatrix(const Medium& m) {
 Mat t(kPorts, Vec(kPorts, cd(0, 0)));
 for (int col = 0; col < kPorts; ++col) {
 Vec input(kPorts, cd(0, 0));
 input[col] = cd(1, 0);
 const Vec out = extractOutput(evolvePhysical(m, input));
 for (int row = 0; row < kPorts; ++row) t[row][col] = out[row];
 }
 return t;
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
 return {std::exp(cd(0, chi[kIn0])) * x[0], std::exp(cd(0, chi[kIn1])) * x[1]};
}

Vec gaugeOutput(const Vec& y, const std::vector<double>& chi) {
 return {std::exp(cd(0, chi[kOut0])) * y[0], std::exp(cd(0, chi[kOut1])) * y[1]};
}

Mat expectedGaugeTransfer(const Mat& t, const std::vector<double>& chi) {
 Mat out(kPorts, Vec(kPorts, cd(0, 0)));
 const int ins[kPorts] = {kIn0, kIn1};
 const int outs[kPorts] = {kOut0, kOut1};
 for (int row = 0; row < kPorts; ++row) {
 for (int col = 0; col < kPorts; ++col) {
 out[row][col] = std::exp(cd(0, chi[outs[row]])) * t[row][col] *
 std::exp(cd(0, -chi[ins[col]]));
 }
 }
 return out;
}

bool testPhysicalWiringMatchesAlgebraicComposition() {
 std::printf("\n[1] PHYSICAL WIRING REALIZES ALGEBRAIC PORT COMPOSITION\n");
 const double flux_a = 0.31 * kPi;
 const double flux_b = 0.44 * kPi;
 const double flux_eff = wrapPhase(flux_a + flux_b);
 const Medium m = physicalWiredComposite(flux_a, flux_b);
 const Mat t = physicalTransferMatrix(m);
 const Mat expected = scaled(cd(-1, 0), idealKernel(flux_eff));
 const double herm_err = m.hermiticityError();
 const double transfer_err = maxMatrixDiff(t, expected);
 const double unitary_err = maxMatrixDiff(matmul(dagger(t), t), identity(kPorts));

 Vec input(kPorts, cd(0, 0));
 input[0] = cd(1, 0);
 const Vec z = evolvePhysical(m, input);
 const double residual = portPower(extractInput(z)) + portPower(extractBus(z));
 const double output_power = portPower(extractOutput(z));

 const bool pass = herm_err < 1e-14 && transfer_err < 1e-12 && unitary_err < 1e-12 &&
 residual < 1e-10 && std::abs(output_power - 1.0) < 1e-10;

 std::printf(" hermiticity error=%.2e\n", herm_err);
 std::printf(" effective path angle=%.12f target pi=%.12f\n", effectivePathAngle(), kPi);
 std::printf(" max |T_physical - (-K(phi_A+phi_B))|=%.2e\n", transfer_err);
 std::printf(" max |T^dag T - I|=%.2e\n", unitary_err);
 std::printf(" output power=%.12f residual input+bus=%.3e\n", output_power, residual);
 std::printf(" => %s\n", pass ? "PASS" : "FAIL");
 return pass;
}

bool testFluxPairRoutesThroughWiredBody() {
 std::printf("\n[2] FLUX PAIRS ROUTE THROUGH THE WIRED BODY\n");
 const Vec upper = {cd(1, 0), cd(0, 0)};
 const Vec z00 = extractOutput(evolvePhysical(physicalWiredComposite(0.0, 0.0), upper));
 const Vec z0p = extractOutput(evolvePhysical(physicalWiredComposite(0.0, kPi), upper));
 const Vec zhp = extractOutput(evolvePhysical(physicalWiredComposite(kPi / 2.0, kPi / 2.0), upper));
 const Vec zh0 = extractOutput(evolvePhysical(physicalWiredComposite(kPi / 2.0, 0.0), upper));

 const double p00 = portPower(z00);
 const double p0p = portPower(z0p);
 const double php = portPower(zhp);
 const double ph0 = portPower(zh0);
 const double p00_out0 = std::norm(z00[0]) / p00;
 const double p00_out1 = std::norm(z00[1]) / p00;
 const double p0p_out0 = std::norm(z0p[0]) / p0p;
 const double p0p_out1 = std::norm(z0p[1]) / p0p;
 const double php_out0 = std::norm(zhp[0]) / php;
 const double php_out1 = std::norm(zhp[1]) / php;
 const double ph0_out0 = std::norm(zh0[0]) / ph0;
 const double ph0_out1 = std::norm(zh0[1]) / ph0;

 const bool pass = p00_out0 > 0.999999 && p00_out1 < 1e-12 &&
 p0p_out1 > 0.999999 && p0p_out0 < 1e-12 &&
 php_out1 > 0.999999 && php_out0 < 1e-12 &&
 std::abs(ph0_out0 - 0.5) < 1e-12 && std::abs(ph0_out1 - 0.5) < 1e-12;

 std::printf(" phiA=0 phiB=0 P(out0|ports)=%.12f P(out1|ports)=%.3e\n", p00_out0, p00_out1);
 std::printf(" phiA=0 phiB=pi P(out0|ports)=%.3e P(out1|ports)=%.12f\n", p0p_out0, p0p_out1);
 std::printf(" phiA=pi/2 phiB=pi/2 P(out0|ports)=%.3e P(out1|ports)=%.12f\n", php_out0, php_out1);
 std::printf(" phiA=pi/2 phiB=0 P(out0|ports)=%.12f P(out1|ports)=%.12f\n", ph0_out0, ph0_out1);
 std::printf(" => %s\n", pass ? "PASS" : "FAIL");
 return pass;
}

bool testGaugeCovariantPhysicalWiring() {
 std::printf("\n[3] WHOLE-BODY WIRING IS GAUGE-COVARIANT\n");
 const Medium m = physicalWiredComposite(0.37 * kPi, 0.22 * kPi);
 const auto chi = localGauge();
 Medium mg(kNodes);
 mg.h = gaugeTransform(m.h, chi);

 const Mat t = physicalTransferMatrix(m);
 const Mat tg = physicalTransferMatrix(mg);
 const Mat expected_tg = expectedGaugeTransfer(t, chi);
 const double matrix_err = maxMatrixDiff(tg, expected_tg);

 const Vec x = normalized({cd(0.61, 0.23), cd(-0.17, 0.72)});
 const Vec y = matvec(t, x);
 const Vec yg = matvec(tg, gaugeInput(x, chi));
 const Vec expected_yg = gaugeOutput(y, chi);
 const double state_err = maxStateDiff(yg, expected_yg);
 const double power_err = std::abs(std::norm(yg[0]) - std::norm(y[0])) +
 std::abs(std::norm(yg[1]) - std::norm(y[1]));

 const bool pass = matrix_err < 1e-12 && state_err < 1e-12 && power_err < 1e-12;
 std::printf(" max |T_gauge - G_out T G_in^-1|=%.2e\n", matrix_err);
 std::printf(" gauge-covariant output state error=%.2e\n", state_err);
 std::printf(" output port-power L1 diff=%.2e\n", power_err);
 std::printf(" => %s\n", pass ? "PASS" : "FAIL");
 return pass;
}

bool testClockedBoundaryIsRequired() {
 std::printf("\n[4] PHYSICAL WIRING HAS A CLOCKED TIME-OF-FLIGHT BOUNDARY\n");
 const Medium m = physicalWiredComposite(0.29 * kPi, 0.43 * kPi);
 const Vec input = {cd(1, 0), cd(0, 0)};
 const Vec half = evolvePhysical(m, input, kFlightSteps / 2);
 const Vec full = evolvePhysical(m, input, kFlightSteps);
 const double half_input = portPower(extractInput(half));
 const double half_bus = portPower(extractBus(half));
 const double half_output = portPower(extractOutput(half));
 const double full_input = portPower(extractInput(full));
 const double full_bus = portPower(extractBus(full));
 const double full_output = portPower(extractOutput(full));

 const bool pass = half_bus > 0.49 && half_input > 0.24 && half_output > 0.24 &&
 full_output > 0.999999 && (full_input + full_bus) < 1e-10;

 std::printf(" half-flight power: input=%.12f bus=%.12f output=%.12f\n",
 half_input, half_bus, half_output);
 std::printf(" full-flight power: input=%.3e bus=%.3e output=%.12f\n",
 full_input, full_bus, full_output);
 std::printf(" => %s\n", pass ? "PASS" : "FAIL");
 return pass;
}

bool testLinearityBoundaryRemains() {
 std::printf("\n[5] PHYSICAL WIRING IS STILL LINEAR WITHOUT MEASUREMENT\n");
 const Medium m = physicalWiredComposite(0.13 * kPi, 0.51 * kPi);
 const Vec x = normalized({cd(0.8, 0.1), cd(-0.2, 0.55)});
 const Vec y = normalized({cd(-0.1, 0.7), cd(0.4, -0.35)});
 const cd a(0.7, -0.2);
 const cd b(-0.3, 0.5);
 const Vec combo = {a * x[0] + b * y[0], a * x[1] + b * y[1]};
 const Vec routed_combo = extractOutput(evolvePhysical(m, combo));
 const Vec routed_x = extractOutput(evolvePhysical(m, x));
 const Vec routed_y = extractOutput(evolvePhysical(m, y));
 const Vec expected = {a * routed_x[0] + b * routed_y[0], a * routed_x[1] + b * routed_y[1]};

 const Vec shifted = {std::exp(cd(0, 1.2345)) * x[0], std::exp(cd(0, 1.2345)) * x[1]};
 const Vec routed_shifted = extractOutput(evolvePhysical(m, shifted));
 const double linearity_err = maxStateDiff(routed_combo, expected);
 const double phase_power_err = std::abs(std::norm(routed_shifted[0]) - std::norm(routed_x[0])) +
 std::abs(std::norm(routed_shifted[1]) - std::norm(routed_x[1]));

 const bool pass = linearity_err < 1e-12 && phase_power_err < 1e-12;
 std::printf(" max |H_phys(a*x+b*y)-a*H_phys(x)-b*H_phys(y)|=%.2e\n", linearity_err);
 std::printf(" global-phase output port-power L1 diff=%.2e\n", phase_power_err);
 std::printf(" nonlinear boundary=ABSENT (measurement/re-injection still separate)\n");
 std::printf(" => %s\n", pass ? "PASS" : "FAIL");
 return pass;
}

bool testNonCommutingPhysicalKernelsPreserveOrder() {
 std::printf("\n[6] NONCOMMUTING PHYSICAL KERNELS PRESERVE LAYER ORDER\n");
 const Mat ka = noncommutingKernelA();
 const Mat kb = noncommutingKernelB();
 const Medium m = physicalWiredComposite(ka, kb);
 const Mat t = physicalTransferMatrix(m);
 const Mat expected = scaled(cd(-1, 0), matmul(kb, ka));
 const double ka_unitary_err = unitarityError(ka);
 const double kb_unitary_err = unitarityError(kb);
 const double transfer_err = maxMatrixDiff(t, expected);
 const double physical_unitary_err = maxMatrixDiff(matmul(dagger(t), t), identity(kPorts));
 const double order_gap = maxMatrixDiff(matmul(kb, ka), matmul(ka, kb));

 const Vec input = normalized({cd(0.47, -0.11), cd(-0.25, 0.83)});
 const Vec forward = matvec(t, input);
 const Vec reversed = matvec(scaled(cd(-1, 0), matmul(ka, kb)), input);
 const double routed_order_gap = maxStateDiff(forward, reversed);

 const bool pass = ka_unitary_err < 1e-12 && kb_unitary_err < 1e-12 &&
 transfer_err < 1e-12 && physical_unitary_err < 1e-12 &&
 order_gap > 0.05 && routed_order_gap > 0.02;

 std::printf(" K_A unitarity error=%.2e K_B unitarity error=%.2e\n", ka_unitary_err, kb_unitary_err);
 std::printf(" max |T_physical - (-(K_B*K_A))|=%.2e\n", transfer_err);
 std::printf(" max |T^dag T - I|=%.2e\n", physical_unitary_err);
 std::printf(" noncommuting order gap |K_B*K_A - K_A*K_B|=%.3f\n", order_gap);
 std::printf(" routed output order gap=%.3f\n", routed_order_gap);
 std::printf(" => %s\n", pass ? "PASS" : "FAIL");
 return pass;
}

} // namespace

int main() {
 std::printf("=====================================================================\n");
 std::printf(" GRAPH-WAVE PHYSICAL PORT WIRING CONTRACT TEST\n");
 std::printf(" One Hermitian body: input rails -> shared bus rails -> output rails.\n");
 std::printf(" No absorber, no measurement layer, no decision, no learning.\n");
 std::printf("=====================================================================\n");

 int pass = 0;
 constexpr int total = 6;
 pass += testPhysicalWiringMatchesAlgebraicComposition() ? 1 : 0;
 pass += testFluxPairRoutesThroughWiredBody() ? 1 : 0;
 pass += testGaugeCovariantPhysicalWiring() ? 1 : 0;
 pass += testClockedBoundaryIsRequired() ? 1 : 0;
 pass += testLinearityBoundaryRemains() ? 1 : 0;
 pass += testNonCommutingPhysicalKernelsPreserveOrder() ? 1 : 0;

 std::printf("\n=====================================================================\n");
 std::printf(" RESULT : %d / %d verified\n", pass, total);
 std::printf(" %s\n", pass == total
 ? "physical port wiring can realize composed graph-wave transfer; nonlinearity remains separate"
 : "physical wiring is not a stable layer boundary yet; inspect the failing contract");
 std::printf("=====================================================================\n");
 return pass == total ? 0 : 1;
}
