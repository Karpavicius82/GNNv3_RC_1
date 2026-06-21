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
constexpr int kInputs = 4;  // A self, B self, A external neighbor, B external neighbor
constexpr int kModes = 4;   // A update, B update, two residual modes
constexpr int kNodes = kInputs + kModes;
constexpr int kSteps = 512;
constexpr int kIn0 = 0;
constexpr int kOut0 = 4;

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

struct PairParams {
    double self_gain = 1.0;
    double shared_gain = 0.8;
    double external_gain = 0.6;
    double shared_phase = 0.31 * kPi;
};

Vec sharedRowA(const PairParams& p) {
    const double n = std::sqrt(p.self_gain * p.self_gain + p.shared_gain * p.shared_gain +
                               p.external_gain * p.external_gain);
    Vec row(kInputs, cd(0, 0));
    row[0] = cd(p.self_gain / n, 0);
    row[1] = (p.shared_gain / n) * std::exp(cd(0, p.shared_phase));
    row[2] = cd(p.external_gain / n, 0);
    return row;
}

Vec sharedRowB(const PairParams& p) {
    const double n = std::sqrt(p.self_gain * p.self_gain + p.shared_gain * p.shared_gain +
                               p.external_gain * p.external_gain);
    Vec row(kInputs, cd(0, 0));
    row[0] = -(p.shared_gain / n) * std::exp(cd(0, -p.shared_phase));
    row[1] = cd(p.self_gain / n, 0);
    row[3] = cd(p.external_gain / n, 0);
    return row;
}

Vec badSameSignRowB(const PairParams& p) {
    const double n = std::sqrt(p.self_gain * p.self_gain + p.shared_gain * p.shared_gain +
                               p.external_gain * p.external_gain);
    Vec row(kInputs, cd(0, 0));
    row[0] = (p.shared_gain / n) * std::exp(cd(0, -p.shared_phase));
    row[1] = cd(p.self_gain / n, 0);
    row[3] = cd(p.external_gain / n, 0);
    return row;
}

Mat completeKernelFromRows(const Vec& row_a, const Vec& row_b) {
    Mat rows;
    rows.push_back(row_a);
    rows.push_back(row_b);

    for (int i = 0; i < kInputs && static_cast<int>(rows.size()) < kModes; ++i) {
        Vec candidate(kInputs, cd(0, 0));
        candidate[i] = cd(1, 0);
        addOrthogonalRow(rows, candidate);
    }

    for (int mode = 1; static_cast<int>(rows.size()) < kModes && mode < 16; ++mode) {
        Vec candidate(kInputs, cd(0, 0));
        for (int j = 0; j < kInputs; ++j) {
            const double angle = 2.0 * kPi * static_cast<double>(mode * j) / static_cast<double>(kInputs);
            candidate[j] = std::exp(cd(0, angle));
        }
        addOrthogonalRow(rows, candidate);
    }

    return rows;
}

Mat sharedEdgeKernel(const PairParams& p) {
    return completeKernelFromRows(sharedRowA(p), sharedRowB(p));
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

Medium sharedEdgeMedium(const Mat& w) {
    Medium m(kNodes);
    for (int mode = 0; mode < kModes; ++mode) {
        for (int in = 0; in < kInputs; ++in) {
            m.transferEdge(kOut0 + mode, kIn0 + in, w[mode][in]);
        }
    }
    return m;
}

Medium sharedEdgeMedium(const PairParams& p) {
    return sharedEdgeMedium(sharedEdgeKernel(p));
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
    return {0.23, -0.71, 1.13, -0.37, 0.89, -1.21, 0.52, -0.46};
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

bool testReciprocalSharedEdgeRows() {
    std::printf("\n[1] RECIPROCAL SHARED EDGE ROWS ARE ORTHOGONAL\n");
    const PairParams p;
    const Vec row_a = sharedRowA(p);
    const Vec row_b = sharedRowB(p);
    const Vec bad_b = badSameSignRowB(p);
    const double good_overlap = std::abs(rowInner(row_a, row_b));
    const double bad_overlap = std::abs(rowInner(row_a, bad_b));
    const double reciprocal_err = std::abs(row_a[1] + std::conj(row_b[0]));
    const bool pass = good_overlap < 1e-14 && bad_overlap > 0.50 && reciprocal_err < 1e-14;

    std::printf("    good row overlap=%.2e\n", good_overlap);
    std::printf("    bad same-sign overlap=%.3f\n", bad_overlap);
    std::printf("    reciprocal shared-edge coefficient error=%.2e\n", reciprocal_err);
    std::printf("    => %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

bool testPhysicalSharedEdgeTransfer() {
    std::printf("\n[2] PHYSICAL BODY REALIZES SHARED-EDGE TWO-NODE UPDATE\n");
    const PairParams p;
    const Mat w = sharedEdgeKernel(p);
    const Medium m = sharedEdgeMedium(w);
    const Mat t = transferMatrix(m);
    const Mat expected = scaled(transferScale(), w);
    const double herm_err = m.hermiticityError();
    const double transfer_err = maxMatrixDiff(t, expected);
    const double unitary_err = unitarityError(t);
    const double residual = maxInputResidualPower(m);
    const bool pass = herm_err < 1e-14 && transfer_err < 1e-12 && unitary_err < 1e-12 && residual < 1e-12;

    std::printf("    hermiticity error=%.2e effective_angle=%.12f\n", herm_err, effectiveAngle());
    std::printf("    max |T - (-i W)|=%.2e max |T^dag T - I|=%.2e residual_input=%.2e\n",
                transfer_err, unitary_err, residual);
    std::printf("    => %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

bool testSharedEdgeContributionIsReciprocal() {
    std::printf("\n[3] SHARED EDGE CONTRIBUTION IS RECIPROCAL\n");
    const PairParams p;
    const Mat w = sharedEdgeKernel(p);
    const Medium m = sharedEdgeMedium(w);
    const Mat t = transferMatrix(m);
    const double expected_shared_power = std::norm(w[0][1]);

    Vec only_a(kInputs, cd(0, 0));
    only_a[0] = cd(1, 0);
    Vec only_b(kInputs, cd(0, 0));
    only_b[1] = cd(1, 0);
    const Vec out_a = routeInput(m, only_a);
    const Vec out_b = routeInput(m, only_b);
    const double a_to_b_power = std::norm(out_a[1]);
    const double b_to_a_power = std::norm(out_b[0]);
    const double transfer_reciprocal_err = std::abs(t[0][1] - std::conj(t[1][0]));
    const bool pass = std::abs(a_to_b_power - expected_shared_power) < 1e-12 &&
                      std::abs(b_to_a_power - expected_shared_power) < 1e-12 &&
                      std::abs(a_to_b_power - b_to_a_power) < 1e-12 &&
                      transfer_reciprocal_err < 1e-12;

    std::printf("    expected shared-edge update power=%.12f\n", expected_shared_power);
    std::printf("    A->B update power=%.12f B->A update power=%.12f\n", a_to_b_power, b_to_a_power);
    std::printf("    transfer reciprocal error=%.2e\n", transfer_reciprocal_err);
    std::printf("    => %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

bool testEdgePhaseControlsPairInterference() {
    std::printf("\n[4] SHARED EDGE PHASE CONTROLS PAIR INTERFERENCE\n");
    PairParams aligned;
    aligned.shared_phase = 0.0;
    PairParams flipped = aligned;
    flipped.shared_phase = kPi;

    const Vec pair_equal = normalized({cd(1, 0), cd(1, 0), cd(0, 0), cd(0, 0)});
    const Vec aligned_out = routeInput(sharedEdgeMedium(aligned), pair_equal);
    const Vec flipped_out = routeInput(sharedEdgeMedium(flipped), pair_equal);
    const double aligned_a = std::norm(aligned_out[0]);
    const double aligned_b = std::norm(aligned_out[1]);
    const double flipped_a = std::norm(flipped_out[0]);
    const double flipped_b = std::norm(flipped_out[1]);
    const bool pass = aligned_a > 0.80 && aligned_b < 0.02 && flipped_a < 0.02 && flipped_b > 0.80;

    std::printf("    phase 0:  A_update=%.12f B_update=%.12f\n", aligned_a, aligned_b);
    std::printf("    phase pi: A_update=%.12f B_update=%.12f\n", flipped_a, flipped_b);
    std::printf("    => %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

bool testResidualModesPreventPairCompression() {
    std::printf("\n[5] RESIDUAL MODES PREVENT FALSE SHARED-EDGE COMPRESSION\n");
    const PairParams p;
    const Mat w = sharedEdgeKernel(p);
    const Medium m = sharedEdgeMedium(w);
    Vec ext_a(kInputs, cd(0, 0));
    ext_a[2] = cd(1, 0);
    const Vec out = routeInput(m, ext_a);
    const double update_a = std::norm(out[0]);
    const double update_b = std::norm(out[1]);
    const double residual = std::norm(out[2]) + std::norm(out[3]);
    const double total = portPower(out);
    const double expected_update = std::norm(w[0][2]);
    const bool pass = std::abs(update_a - expected_update) < 1e-12 &&
                      update_b < 1e-12 &&
                      std::abs(residual - (1.0 - expected_update)) < 1e-12 &&
                      std::abs(total - 1.0) < 1e-12;

    std::printf("    external-A update_A=%.12f update_B=%.3e residual_modes=%.12f total=%.12f\n",
                update_a, update_b, residual, total);
    std::printf("    no_compression_boundary=CONFIRMED\n");
    std::printf("    => %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

bool testGaugeCovariantSharedEdgeUpdate() {
    std::printf("\n[6] SHARED-EDGE UPDATE IS GAUGE-COVARIANT\n");
    const Medium m = sharedEdgeMedium(PairParams{});
    const auto chi = localGauge();
    Medium mg(kNodes);
    mg.h = gaugeTransform(m.h, chi);

    const Mat t = transferMatrix(m);
    const Mat tg = transferMatrix(mg);
    const Mat expected_tg = expectedGaugeTransfer(t, chi);
    const double matrix_err = maxMatrixDiff(tg, expected_tg);

    const Vec input = normalized({cd(0.6, -0.1), cd(-0.2, 0.7), cd(0.4, -0.3), cd(0.1, 0.5)});
    const Vec out = matvec(t, input);
    const Vec out_g = matvec(tg, gaugeInput(input, chi));
    const Vec expected_out_g = gaugeOutput(out, chi);
    const double state_err = maxStateDiff(out_g, expected_out_g);
    double power_err = 0.0;
    for (int i = 0; i < kModes; ++i) power_err += std::abs(std::norm(out_g[i]) - std::norm(out[i]));
    const bool pass = matrix_err < 1e-12 && state_err < 1e-12 && power_err < 1e-12;

    std::printf("    max |T_gauge - G_out T G_in^-1|=%.2e\n", matrix_err);
    std::printf("    gauge-covariant output state error=%.2e output-power L1 diff=%.2e\n",
                state_err, power_err);
    std::printf("    => %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

bool testRandomDisorderKeepsSharedEdgeBounded() {
    std::printf("\n[7] RANDOM DISORDER KEEPS SHARED-EDGE UPDATE BOUNDED\n");
    const Medium ideal = sharedEdgeMedium(PairParams{});
    const Vec input = normalized({cd(0.6, -0.1), cd(-0.2, 0.7), cd(0.4, -0.3), cd(0.1, 0.5)});
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
            const Medium perturbed = perturbRandomDisorder(ideal, row.eps, row.eps, 0x51A7E9u + seed * 131u);
            const Vec full = evolvePhysical(perturbed, input);
            const Vec out = extractOutput(full);
            const double update_err = std::max(std::abs(out[0] - ideal_out[0]), std::abs(out[1] - ideal_out[1]));
            const double power_err = std::abs(std::norm(out[0]) - std::norm(ideal_out[0])) +
                                     std::abs(std::norm(out[1]) - std::norm(ideal_out[1]));
            row.mean_update_err += update_err;
            row.max_update_err = std::max(row.max_update_err, update_err);
            row.max_power_err = std::max(row.max_power_err, power_err);
            row.max_input_residual = std::max(row.max_input_residual, portPower(extractInput(full)));
        }
        row.mean_update_err /= 24.0;
    }

    const bool mild_pass = rows[0].max_update_err < 0.05 && rows[0].max_input_residual < 0.03;
    const bool stress_pass = rows[1].max_update_err < 0.20 && rows[1].max_input_residual < 0.12;
    const bool pass = mild_pass && stress_pass;
    for (const auto& row : rows) {
        const bool row_pass = row.eps < 0.02 ? mild_pass : stress_pass;
        std::printf("    random eps=%.3f seeds=24 mean_update_err=%.4f max_update_err=%.4f max_power_err=%.4f max_input_residual=%.4f %s\n",
                    row.eps, row.mean_update_err, row.max_update_err, row.max_power_err,
                    row.max_input_residual, row_pass ? "OK" : "BAD");
    }
    std::printf("    => %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

}  // namespace

int main() {
    std::printf("=====================================================================\n");
    std::printf(" GRAPH-WAVE SHARED EDGE CONSISTENCY CONTRACT TEST\n");
    std::printf(" Two GNN-analog node updates share one reciprocal physical edge.\n");
    std::printf(" No V26 runtime coupling, no legacy GA/readout, no trainer, no production decision.\n");
    std::printf("=====================================================================\n");

    int pass = 0;
    const int total = 7;
    pass += testReciprocalSharedEdgeRows() ? 1 : 0;
    pass += testPhysicalSharedEdgeTransfer() ? 1 : 0;
    pass += testSharedEdgeContributionIsReciprocal() ? 1 : 0;
    pass += testEdgePhaseControlsPairInterference() ? 1 : 0;
    pass += testResidualModesPreventPairCompression() ? 1 : 0;
    pass += testGaugeCovariantSharedEdgeUpdate() ? 1 : 0;
    pass += testRandomDisorderKeepsSharedEdgeBounded() ? 1 : 0;

    std::printf("\n=====================================================================\n");
    std::printf(" RESULT : %d / %d verified\n", pass, total);
    std::printf(" %s\n", pass == total
                    ? "shared edge is consistent across two GNN-analog node updates"
                    : "shared-edge contract is not stable enough yet");
    std::printf("=====================================================================\n");
    return pass == total ? 0 : 1;
}
