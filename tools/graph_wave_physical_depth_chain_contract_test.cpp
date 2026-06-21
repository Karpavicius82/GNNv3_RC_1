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
constexpr int kPorts = 2;
constexpr int kFrames = 4;
constexpr int kNodes = kPorts * kFrames;
constexpr int kFlightSteps = 1536;

int node(const int frame, const int port) {
    return frame * kPorts + port;
}

double dt() {
    return (kPi / 2.0) / static_cast<double>(kFlightSteps);
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

Mat kernelC() {
    return matmul(su2Z(-0.19 * kPi), su2X(0.33 * kPi));
}

std::vector<double> uniformGains() {
    return {1.0, 1.0, 1.0};
}

std::vector<double> mirrorGains() {
    return {std::sqrt(3.0), 2.0, std::sqrt(3.0)};
}

Medium physicalDepthChain(const Mat& ka, const Mat& kb, const Mat& kc, const std::vector<double>& gains) {
    Medium m(kNodes);
    const Mat kernels[3] = {ka, kb, kc};
    for (int layer = 0; layer < 3; ++layer) {
        for (int row = 0; row < kPorts; ++row) {
            for (int col = 0; col < kPorts; ++col) {
                m.transferEdge(node(layer + 1, row), node(layer, col), gains[layer] * kernels[layer][row][col]);
            }
        }
    }
    return m;
}

Vec injectInput(const Vec& input) {
    Vec z(kNodes, cd(0, 0));
    z[node(0, 0)] = input[0];
    z[node(0, 1)] = input[1];
    return z;
}

Vec extractFrame(const Vec& z, const int frame) {
    return {z[node(frame, 0)], z[node(frame, 1)]};
}

double framePower(const Vec& z, const int frame) {
    return portPower(extractFrame(z, frame));
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
        const Vec out = extractFrame(evolvePhysical(m, input, steps), 3);
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
        worst = std::max(worst, framePower(z, 0) + framePower(z, 1) + framePower(z, 2));
    }
    return worst;
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

Metrics evaluate(const Medium& m, const Mat& expected, const Mat& reversed, const Vec& sample) {
    Metrics r;
    const Mat fixed = physicalTransferMatrix(m, kFlightSteps);
    r.fixed_error = maxMatrixDiff(fixed, expected);
    r.fixed_residual = maxResidualPower(m, kFlightSteps);
    r.order_gap = maxStateDiff(matvec(fixed, sample), matvec(reversed, sample));

    r.best_error = std::numeric_limits<double>::infinity();
    for (int steps = kFlightSteps - 160; steps <= kFlightSteps + 160; steps += 4) {
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

bool testUniformDepthGuardrail() {
    std::printf("\n[1] UNIFORM THREE-CELL DEPTH FAILS AS A FULL-TRANSFER PRIMITIVE\n");
    const Mat ka = kernelA();
    const Mat kb = kernelB();
    const Mat kc = kernelC();
    const Medium m = physicalDepthChain(ka, kb, kc, uniformGains());
    const Mat expected = scaled(cd(0, 1), matmul(kc, matmul(kb, ka)));
    const Mat reversed = scaled(cd(0, 1), matmul(ka, matmul(kb, kc)));
    const Vec sample = normalized({cd(0.47, -0.11), cd(-0.25, 0.83)});
    const Metrics r = evaluate(m, expected, reversed, sample);
    const double transfer_unitary_err = unitarityError(physicalTransferMatrix(m, kFlightSteps));
    const Vec z = evolvePhysical(m, {cd(1, 0), cd(0, 0)}, kFlightSteps);
    const bool pass = m.hermiticityError() < 1e-14 && r.fixed_error > 0.10 &&
                      r.fixed_residual > 0.10 && transfer_unitary_err > 0.10 &&
                      framePower(z, 3) < 0.90 && r.order_gap > 0.2;

    std::printf("    hermiticity error=%.2e\n", m.hermiticityError());
    std::printf("    fixed_err=%.3f residual=%.3f transfer_unitary_err=%.3f output_power=%.3f order_gap=%.3f\n",
                r.fixed_error, r.fixed_residual, transfer_unitary_err, framePower(z, 3), r.order_gap);
    std::printf("    expected_fail_detected=%s\n", pass ? "YES" : "NO");
    std::printf("    => %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

bool testEngineeredDepthTransfer() {
    std::printf("\n[2] MIRROR-COUPLED THREE-CELL DEPTH TRANSFER\n");
    const Mat ka = kernelA();
    const Mat kb = kernelB();
    const Mat kc = kernelC();
    const Medium m = physicalDepthChain(ka, kb, kc, mirrorGains());
    const Mat expected = scaled(cd(0, 1), matmul(kc, matmul(kb, ka)));
    const Mat reversed = scaled(cd(0, 1), matmul(ka, matmul(kb, kc)));
    const Vec sample = normalized({cd(0.47, -0.11), cd(-0.25, 0.83)});
    const Metrics r = evaluate(m, expected, reversed, sample);
    const double transfer_unitary_err = unitarityError(physicalTransferMatrix(m, kFlightSteps));
    const bool pass = m.hermiticityError() < 1e-14 && r.fixed_error < 1e-5 &&
                      r.fixed_residual < 1e-10 && transfer_unitary_err < 1e-5 &&
                      r.order_gap > 0.2;

    std::printf("    gains=sqrt(3),2,sqrt(3)\n");
    std::printf("    fixed_err=%.2e residual=%.2e transfer_unitary_err=%.2e order_gap=%.3f\n",
                r.fixed_error, r.fixed_residual, transfer_unitary_err, r.order_gap);
    std::printf("    => %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

bool testClockedDepthPowerFlow() {
    std::printf("\n[3] MIRROR-COUPLED DEPTH CHAIN HAS A CLOCKED POWER FLOW\n");
    const Medium m = physicalDepthChain(kernelA(), kernelB(), kernelC(), mirrorGains());
    const Vec input = {cd(1, 0), cd(0, 0)};
    const Vec third = evolvePhysical(m, input, kFlightSteps / 3);
    const Vec two_thirds = evolvePhysical(m, input, 2 * kFlightSteps / 3);
    const Vec full = evolvePhysical(m, input, kFlightSteps);
    const bool pass = framePower(full, 3) > 0.99999 &&
                      framePower(full, 0) + framePower(full, 1) + framePower(full, 2) < 1e-8 &&
                      framePower(third, 1) > 0.15 && framePower(two_thirds, 2) > 0.15;

    std::printf("    one-third powers: f0=%.4f f1=%.4f f2=%.4f f3=%.4f\n",
                framePower(third, 0), framePower(third, 1), framePower(third, 2), framePower(third, 3));
    std::printf("    two-third powers: f0=%.4f f1=%.4f f2=%.4f f3=%.4f\n",
                framePower(two_thirds, 0), framePower(two_thirds, 1), framePower(two_thirds, 2),
                framePower(two_thirds, 3));
    std::printf("    full powers: f0=%.3e f1=%.3e f2=%.3e f3=%.12f\n",
                framePower(full, 0), framePower(full, 1), framePower(full, 2), framePower(full, 3));
    std::printf("    => %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

bool testDepthOrderMatters() {
    std::printf("\n[4] MIRROR-COUPLED DEPTH PRESERVES NONCOMMUTING ORDER\n");
    const Mat ka = kernelA();
    const Mat kb = kernelB();
    const Mat kc = kernelC();
    const Medium forward_m = physicalDepthChain(ka, kb, kc, mirrorGains());
    const Medium reversed_m = physicalDepthChain(kc, kb, ka, mirrorGains());
    const Mat forward_t = physicalTransferMatrix(forward_m, kFlightSteps);
    const Mat reversed_t = physicalTransferMatrix(reversed_m, kFlightSteps);
    const double matrix_gap = maxMatrixDiff(forward_t, reversed_t);
    const Vec sample = normalized({cd(0.47, -0.11), cd(-0.25, 0.83)});
    const double routed_gap = maxStateDiff(matvec(forward_t, sample), matvec(reversed_t, sample));
    const bool pass = matrix_gap > 0.2 && routed_gap > 0.1;

    std::printf("    forward-vs-reversed transfer gap=%.3f\n", matrix_gap);
    std::printf("    routed output order gap=%.3f\n", routed_gap);
    std::printf("    => %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

bool testDepthRandomDisorderEnsemble() {
    std::printf("\n[5] MIRROR-COUPLED DEPTH RANDOM DISORDER ENSEMBLE\n");
    const Mat ka = kernelA();
    const Mat kb = kernelB();
    const Mat kc = kernelC();
    const Medium ideal = physicalDepthChain(ka, kb, kc, mirrorGains());
    const Mat expected = scaled(cd(0, 1), matmul(kc, matmul(kb, ka)));
    const Mat reversed = scaled(cd(0, 1), matmul(ka, matmul(kb, kc)));
    const Vec sample = normalized({cd(0.47, -0.11), cd(-0.25, 0.83)});

    struct Row {
        double eps = 0.0;
        double mean_error = 0.0;
        double max_error = 0.0;
        double max_residual = 0.0;
        double min_order_gap = std::numeric_limits<double>::infinity();
        double mean_best_error = 0.0;
    };

    std::vector<Row> rows;
    for (const double eps : {0.01, 0.05}) {
        Row row;
        row.eps = eps;
        constexpr int kSeeds = 24;
        for (int seed = 1; seed <= kSeeds; ++seed) {
            const Medium m = perturbRandomDisorder(ideal, eps, eps, static_cast<std::uint32_t>(seed * 104729));
            const Metrics r = evaluate(m, expected, reversed, sample);
            row.mean_error += r.fixed_error / static_cast<double>(kSeeds);
            row.mean_best_error += r.best_error / static_cast<double>(kSeeds);
            row.max_error = std::max(row.max_error, r.fixed_error);
            row.max_residual = std::max(row.max_residual, r.fixed_residual);
            row.min_order_gap = std::min(row.min_order_gap, r.order_gap);
        }
        rows.push_back(row);
    }

    const bool mild_pass = rows[0].max_error < 0.12 && rows[0].max_residual < 0.08 && rows[0].min_order_gap > 0.25;
    const bool stress_pass = rows[1].max_error < 0.45 && rows[1].max_residual < 0.30 && rows[1].min_order_gap > 0.10;
    const bool pass = mild_pass && stress_pass;

    for (const Row& row : rows) {
        const bool row_pass = row.eps < 0.02 ? mild_pass : stress_pass;
        std::printf("    random eps=%.3f seeds=24 mean_err=%.4f mean_best=%.4f max_err=%.4f max_residual=%.4f min_order_gap=%.4f %s\n",
                    row.eps, row.mean_error, row.mean_best_error, row.max_error, row.max_residual,
                    row.min_order_gap, row_pass ? "OK" : "BAD");
    }
    std::printf("    => %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

}  // namespace

int main() {
    std::printf("=====================================================================\n");
    std::printf(" GRAPH-WAVE PHYSICAL DEPTH CHAIN CONTRACT TEST\n");
    std::printf(" Four 2-rail frames: input -> bus1 -> bus2 -> output.\n");
    std::printf(" No absorber, no measurement layer, no decision, no learning, no V26 runtime coupling.\n");
    std::printf("=====================================================================\n");

    int pass = 0;
    constexpr int total = 5;
    pass += testUniformDepthGuardrail() ? 1 : 0;
    pass += testEngineeredDepthTransfer() ? 1 : 0;
    pass += testClockedDepthPowerFlow() ? 1 : 0;
    pass += testDepthOrderMatters() ? 1 : 0;
    pass += testDepthRandomDisorderEnsemble() ? 1 : 0;

    std::printf("\n=====================================================================\n");
    std::printf(" RESULT : %d / %d verified\n", pass, total);
    std::printf(" %s\n", pass == total
                    ? "three-cell physical depth preserves ordered transfer with bounded local disorder"
                    : "three-cell physical depth is not stable enough under this contract");
    std::printf("=====================================================================\n");
    return pass == total ? 0 : 1;
}
