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
constexpr int kNodes = 4;
constexpr int kIn0 = 0;
constexpr int kIn1 = 1;
constexpr int kOut0 = 2;
constexpr int kOut1 = 3;
constexpr int kFlightSteps = 512;
constexpr double kFlightTime = kPi / 2.0;
constexpr double kDt = kFlightTime / static_cast<double>(kFlightSteps);

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

Mat divided(const Mat& m, const cd s) {
    Mat out = m;
    for (auto& row : out) {
        for (auto& v : row) v /= s;
    }
    return out;
}

double portPower(const Vec& z) {
    double s = 0.0;
    for (const auto& v : z) s += std::norm(v);
    return s;
}

struct Medium {
    Mat h;

    explicit Medium(const int n) : h(n, Vec(n, cd(0, 0))) {}

    void transferEdge(const int out, const int in, const cd value) {
        h[out][in] += value;
        h[in][out] += std::conj(value);
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

    void build(const Mat& h, const double dt) {
        const auto n = static_cast<int>(h.size());
        ap.assign(n, Vec(n));
        am.assign(n, Vec(n));
        const cd imag(0, 1);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                const cd k = imag * h[i][j] * (dt / 2.0);
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

Medium portTransferCell(const Mat& k) {
    Medium m(kNodes);
    m.transferEdge(kOut0, kIn0, k[0][0]);
    m.transferEdge(kOut0, kIn1, k[0][1]);
    m.transferEdge(kOut1, kIn0, k[1][0]);
    m.transferEdge(kOut1, kIn1, k[1][1]);
    return m;
}

Medium portTransferCell(const double flux) {
    return portTransferCell(idealKernel(flux));
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

double effectiveAngle() {
    return static_cast<double>(kFlightSteps) * 2.0 * std::atan(kDt / 2.0);
}

cd transferScale() {
    return cd(0, -std::sin(effectiveAngle()));
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

Vec extractInputResidual(const Vec& z) {
    return {z[kIn0], z[kIn1]};
}

Vec routeInput(const Medium& m, const Vec& input) {
    Stepper u;
    u.build(m.h, kDt);
    return extractOutput(u.evolve(injectInput(input), kFlightSteps));
}

Mat transferMatrix(const Medium& m) {
    Mat t(kPorts, Vec(kPorts, cd(0, 0)));
    for (int col = 0; col < kPorts; ++col) {
        Vec input(kPorts, cd(0, 0));
        input[col] = cd(1, 0);
        const Vec out = routeInput(m, input);
        for (int row = 0; row < kPorts; ++row) t[row][col] = out[row];
    }
    return t;
}

double maxInputResidualPower(const Medium& m) {
    Stepper u;
    u.build(m.h, kDt);
    double worst = 0.0;
    for (int col = 0; col < kPorts; ++col) {
        Vec input(kPorts, cd(0, 0));
        input[col] = cd(1, 0);
        const Vec z = u.evolve(injectInput(input), kFlightSteps);
        worst = std::max(worst, portPower(extractInputResidual(z)));
    }
    return worst;
}

std::vector<double> localGauge() {
    return {0.23, -0.71, 1.13, -0.37};
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

bool testSingleCellTransferMatrix() {
    std::printf("\n[1] SINGLE CELL HAS A PORT TRANSFER MATRIX\n");
    const double flux = 0.37 * kPi;
    const Medium m = portTransferCell(flux);
    const Mat t = transferMatrix(m);
    const Mat expected = scaled(transferScale(), idealKernel(flux));
    const Mat normalized_t = divided(t, transferScale());
    const double herm_err = m.hermiticityError();
    const double transfer_err = maxMatrixDiff(t, expected);
    const double unitary_err = maxMatrixDiff(matmul(dagger(normalized_t), normalized_t), identity(kPorts));
    const double residual_power = maxInputResidualPower(m);
    const double scale_power = std::norm(transferScale());

    const bool pass = herm_err < 1e-14 && transfer_err < 1e-12 && unitary_err < 1e-12 &&
                      residual_power < 1e-10 && std::abs(scale_power + residual_power - 1.0) < 1e-10;

    std::printf("    hermiticity error=%.2e\n", herm_err);
    std::printf("    effective angle=%.12f residual power=%.3e transfer power=%.12f\n",
                effectiveAngle(), residual_power, scale_power);
    std::printf("    max |T - (-i sin(theta)) K(phi)|=%.2e\n", transfer_err);
    std::printf("    max |K_eff^dag K_eff - I|=%.2e\n", unitary_err);
    std::printf("    => %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

bool testFluxControlsRouting() {
    std::printf("\n[2] FLUX CONTROLS TWO-RAIL PORT ROUTING\n");
    const Vec upper = {cd(1, 0), cd(0, 0)};
    const Vec z0 = routeInput(portTransferCell(0.0), upper);
    const Vec zh = routeInput(portTransferCell(kPi / 2.0), upper);
    const Vec zp = routeInput(portTransferCell(kPi), upper);
    const double p0 = portPower(z0);
    const double ph = portPower(zh);
    const double pp = portPower(zp);
    const double z0_upper = std::norm(z0[0]) / p0;
    const double z0_lower = std::norm(z0[1]) / p0;
    const double h_upper = std::norm(zh[0]) / ph;
    const double h_lower = std::norm(zh[1]) / ph;
    const double p_upper = std::norm(zp[0]) / pp;
    const double p_lower = std::norm(zp[1]) / pp;

    const bool pass = z0_upper > 0.999999 && z0_lower < 1e-12 &&
                      std::abs(h_upper - 0.5) < 1e-12 && std::abs(h_lower - 0.5) < 1e-12 &&
                      p_lower > 0.999999 && p_upper < 1e-12;

    std::printf("    flux=0    P(out0|ports)=%.12f P(out1|ports)=%.3e\n", z0_upper, z0_lower);
    std::printf("    flux=pi/2 P(out0|ports)=%.12f P(out1|ports)=%.12f\n", h_upper, h_lower);
    std::printf("    flux=pi   P(out0|ports)=%.3e P(out1|ports)=%.12f\n", p_upper, p_lower);
    std::printf("    => %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

bool testGaugeCovariantTransfer() {
    std::printf("\n[3] PORT TRANSFER IS GAUGE-COVARIANT\n");
    const Medium m = portTransferCell(0.41 * kPi);
    const auto chi = localGauge();
    Medium mg(kNodes);
    mg.h = gaugeTransform(m.h, chi);

    const Mat t = transferMatrix(m);
    const Mat tg = transferMatrix(mg);
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
    std::printf("    max |T_gauge - G_out T G_in^-1|=%.2e\n", matrix_err);
    std::printf("    gauge-covariant output state error=%.2e\n", state_err);
    std::printf("    output port-power L1 diff=%.2e\n", power_err);
    std::printf("    => %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

bool testLinearityAndGlobalPhaseBoundary() {
    std::printf("\n[4] TRANSFER IS LINEAR; GLOBAL PHASE IS NOT A ROUTING SIGNAL\n");
    const Medium m = portTransferCell(0.29 * kPi);
    const Vec x = normalized({cd(0.8, 0.1), cd(-0.2, 0.55)});
    const Vec y = normalized({cd(-0.1, 0.7), cd(0.4, -0.35)});
    const cd a(0.7, -0.2);
    const cd b(-0.3, 0.5);
    const Vec combo = {a * x[0] + b * y[0], a * x[1] + b * y[1]};
    const Vec routed_combo = routeInput(m, combo);
    const Vec routed_x = routeInput(m, x);
    const Vec routed_y = routeInput(m, y);
    const Vec expected = {a * routed_x[0] + b * routed_y[0], a * routed_x[1] + b * routed_y[1]};

    const Vec shifted = {std::exp(cd(0, 1.2345)) * x[0], std::exp(cd(0, 1.2345)) * x[1]};
    const Vec routed_shifted = routeInput(m, shifted);
    const double linearity_err = maxStateDiff(routed_combo, expected);
    const double phase_power_err = std::abs(std::norm(routed_shifted[0]) - std::norm(routed_x[0])) +
                                   std::abs(std::norm(routed_shifted[1]) - std::norm(routed_x[1]));

    const bool pass = linearity_err < 1e-12 && phase_power_err < 1e-12;
    std::printf("    max |T(a*x+b*y) - a*T(x)-b*T(y)|=%.2e\n", linearity_err);
    std::printf("    global-phase output port-power L1 diff=%.2e\n", phase_power_err);
    std::printf("    => %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

bool testCompositionAlgebraAndLinearCollapse() {
    std::printf("\n[5] COMPOSED CELLS OBEY TRANSFER ALGEBRA AND STILL COLLAPSE LINEARLY\n");
    const double flux_a = 0.31 * kPi;
    const double flux_b = 0.44 * kPi;
    const double flux_eff = wrapPhase(flux_a + flux_b);
    const Mat ta = transferMatrix(portTransferCell(flux_a));
    const Mat tb = transferMatrix(portTransferCell(flux_b));
    const Mat composite = matmul(tb, ta);
    const Mat expected_composite = scaled(transferScale() * transferScale(), idealKernel(flux_eff));
    const Mat kernel_composite = matmul(idealKernel(flux_b), idealKernel(flux_a));
    const Mat kernel_eff = idealKernel(flux_eff);

    const Vec input = normalized({cd(0.47, -0.11), cd(-0.25, 0.83)});
    const Vec y_seq = matvec(tb, matvec(ta, input));
    const Vec y_mat = matvec(composite, input);

    const double transfer_err = maxMatrixDiff(composite, expected_composite);
    const double kernel_err = maxMatrixDiff(kernel_composite, kernel_eff);
    const double seq_err = maxStateDiff(y_seq, y_mat);
    const double out_power = portPower(y_seq);
    const double expected_power = std::norm(transferScale()) * std::norm(transferScale()) * portPower(input);

    const bool pass = transfer_err < 1e-12 && kernel_err < 1e-12 && seq_err < 1e-12 &&
                      std::abs(out_power - expected_power) < 1e-12;

    std::printf("    flux_A/pi=%.3f flux_B/pi=%.3f effective_flux/pi=%.3f\n",
                flux_a / kPi, flux_b / kPi, flux_eff / kPi);
    std::printf("    max |T_B*T_A - scale^2*K(phi_A+phi_B)|=%.2e\n", transfer_err);
    std::printf("    max |K_B*K_A - K(phi_A+phi_B)|=%.2e\n", kernel_err);
    std::printf("    sequential-vs-matrix output error=%.2e\n", seq_err);
    std::printf("    composite output port power=%.12f expected=%.12f\n", out_power, expected_power);
    std::printf("    linear-collapse boundary=CONFIRMED (no measurement/nonlinearity in this test)\n");
    std::printf("    => %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

bool testNonCommutingKernelsPreserveLayerOrder() {
    std::printf("\n[6] NONCOMMUTING PORT KERNELS PRESERVE LAYER ORDER\n");
    const Mat ka = noncommutingKernelA();
    const Mat kb = noncommutingKernelB();
    const Mat ta = transferMatrix(portTransferCell(ka));
    const Mat tb = transferMatrix(portTransferCell(kb));
    const Mat composite = matmul(tb, ta);
    const Mat expected = scaled(transferScale() * transferScale(), matmul(kb, ka));
    const double ka_unitary_err = unitarityError(ka);
    const double kb_unitary_err = unitarityError(kb);
    const double transfer_err = maxMatrixDiff(composite, expected);
    const double order_gap = maxMatrixDiff(matmul(kb, ka), matmul(ka, kb));

    const Vec input = normalized({cd(0.47, -0.11), cd(-0.25, 0.83)});
    const Vec forward = matvec(composite, input);
    const Vec reversed = matvec(scaled(transferScale() * transferScale(), matmul(ka, kb)), input);
    const double routed_order_gap = maxStateDiff(forward, reversed);

    const bool pass = ka_unitary_err < 1e-12 && kb_unitary_err < 1e-12 &&
                      transfer_err < 1e-12 && order_gap > 0.05 && routed_order_gap > 0.02;

    std::printf("    K_A unitarity error=%.2e K_B unitarity error=%.2e\n", ka_unitary_err, kb_unitary_err);
    std::printf("    max |T_B*T_A - scale^2*(K_B*K_A)|=%.2e\n", transfer_err);
    std::printf("    noncommuting order gap |K_B*K_A - K_A*K_B|=%.3f\n", order_gap);
    std::printf("    routed output order gap=%.3f\n", routed_order_gap);
    std::printf("    => %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

}  // namespace

int main() {
    std::printf("=====================================================================\n");
    std::printf(" GRAPH-WAVE PORT TRANSFER COMPOSITION CONTRACT TEST\n");
    std::printf(" Two-rail port tensor: input amplitudes -> output amplitudes.\n");
    std::printf(" No absorber, no measurement layer, no decision, no learning, no V26 runtime coupling.\n");
    std::printf("=====================================================================\n");

    int pass = 0;
    constexpr int total = 6;
    pass += testSingleCellTransferMatrix() ? 1 : 0;
    pass += testFluxControlsRouting() ? 1 : 0;
    pass += testGaugeCovariantTransfer() ? 1 : 0;
    pass += testLinearityAndGlobalPhaseBoundary() ? 1 : 0;
    pass += testCompositionAlgebraAndLinearCollapse() ? 1 : 0;
    pass += testNonCommutingKernelsPreserveLayerOrder() ? 1 : 0;

    std::printf("\n=====================================================================\n");
    std::printf(" RESULT : %d / %d verified\n", pass, total);
    std::printf(" %s\n", pass == total
                    ? "port transfer is a composable graph-wave layer contract; nonlinearity remains separate"
                    : "do not build stacked cells yet; inspect the failing port contract");
    std::printf("=====================================================================\n");
    return pass == total ? 0 : 1;
}
