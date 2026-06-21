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
constexpr double kDt = 0.02;
constexpr int kFlightSteps = 707;

constexpr int kIn = 0;
constexpr int kArmA = 1;
constexpr int kArmB = 2;
constexpr int kOutPlus = 3;
constexpr int kOutMinus = 4;
constexpr int kNodes = 5;

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

double maxProbabilityDiff(const Vec& a, const Vec& b) {
    double err = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        err = std::max(err, std::abs(std::norm(a[i]) - std::norm(b[i])));
    }
    return err;
}

Vec matvec(const Mat& m, const Vec& z) {
    const auto n = static_cast<int>(m.size());
    Vec out(n, cd(0, 0));
    for (int i = 0; i < n; ++i) {
        cd s(0, 0);
        for (int j = 0; j < n; ++j) s += m[i][j] * z[j];
        out[i] = s;
    }
    return out;
}

struct Medium {
    Mat h;

    explicit Medium(const int n) : h(n, Vec(n, cd(0, 0))) {}

    void edge(const int i, const int j, const double a, const double phase = 0.0) {
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

Medium interferometerCell(const double flux) {
    Medium m(kNodes);

    // The cell is a closed unitary scattering primitive:
    // input -> two arms -> two output ports.  No absorber, no detector layer.
    m.edge(kIn, kArmA, 1.0);
    m.edge(kIn, kArmB, 1.0);

    // out+ reads the symmetric arm combination with Wilson phase = flux.
    m.edge(kArmA, kOutPlus, 1.0);
    m.edge(kArmB, kOutPlus, 1.0, flux);

    // out- reads the complementary antisymmetric combination.
    m.edge(kArmA, kOutMinus, 1.0);
    m.edge(kArmB, kOutMinus, 1.0, flux + kPi);
    return m;
}

Vec inputPacket() {
    Vec z(kNodes, cd(0, 0));
    z[kIn] = cd(1, 0);
    return z;
}

Vec route(const double flux) {
    const Medium m = interferometerCell(flux);
    Stepper u;
    u.build(m.h, kDt);
    return u.evolve(inputPacket(), kFlightSteps);
}

std::vector<double> localGauge() {
    return {0.23, -0.71, 1.13, -0.37, 0.89};
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

Vec gaugeTransform(const Vec& z, const std::vector<double>& chi) {
    Vec out(z.size());
    for (std::size_t i = 0; i < z.size(); ++i) out[i] = std::exp(cd(0, chi[i])) * z[i];
    return out;
}

bool testClosedUnitaryCell() {
    std::printf("\n[1] CLOSED UNITARY CELL\n");
    const Medium m = interferometerCell(0.37 * kPi);
    Stepper u;
    u.build(m.h, kDt);
    Vec z = normalized({cd(0.31, 0.2), cd(-0.4, 0.1), cd(0.2, -0.6), cd(0.5, 0.1), cd(-0.3, 0.7)});

    double max_drift = 0.0;
    for (int t = 0; t < 1000; ++t) {
        z = u.step(z);
        max_drift = std::max(max_drift, std::abs(norm2(z) - 1.0));
    }
    const bool pass = m.hermiticityError() < 1e-14 && max_drift < 1e-12;
    std::printf("    hermiticity error=%.2e   max norm drift=%.2e\n", m.hermiticityError(), max_drift);
    std::printf("    => %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

bool testHolonomyRoutesWithoutMeasurement() {
    std::printf("\n[2] HOLONOMY ROUTES AMPLITUDE WITHOUT MEASUREMENT\n");
    const Vec zero = route(0.0);
    const Vec half = route(kPi / 2.0);
    const Vec pi = route(kPi);

    const double zp = std::norm(zero[kOutPlus]);
    const double zm = std::norm(zero[kOutMinus]);
    const double hp = std::norm(half[kOutPlus]);
    const double hm = std::norm(half[kOutMinus]);
    const double pp = std::norm(pi[kOutPlus]);
    const double pm = std::norm(pi[kOutMinus]);

    const bool pass = zp > 0.999 && zm < 1e-8 && pm > 0.999 && pp < 1e-8 &&
                      std::abs(hp - hm) < 1e-10 && hp > 0.30 && hm > 0.30;

    std::printf("    flux=0    P(out+)=%.12f P(out-)=%.3e\n", zp, zm);
    std::printf("    flux=pi/2 P(out+)=%.12f P(out-)=%.12f\n", hp, hm);
    std::printf("    flux=pi   P(out+)=%.3e P(out-)=%.12f\n", pp, pm);
    std::printf("    => %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

bool testGaugeEquivalentCellSameRouting() {
    std::printf("\n[3] GAUGE-EQUIVALENT CELL HAS SAME ROUTING\n");
    const Medium m = interferometerCell(0.61 * kPi);
    const auto chi = localGauge();
    Medium mg(kNodes);
    mg.h = gaugeTransform(m.h, chi);

    Stepper u;
    Stepper ug;
    u.build(m.h, kDt);
    ug.build(mg.h, kDt);

    Vec z = inputPacket();
    Vec zg = gaugeTransform(z, chi);
    z = u.evolve(std::move(z), kFlightSteps);
    zg = ug.evolve(std::move(zg), kFlightSteps);

    const Vec expected = gaugeTransform(z, chi);
    const double state_err = maxStateDiff(zg, expected);
    const double prob_err = maxProbabilityDiff(zg, z);
    const bool pass = state_err < 1e-12 && prob_err < 1e-12;
    std::printf("    max gauge-covariant state error=%.2e\n", state_err);
    std::printf("    max routing probability diff=%.2e\n", prob_err);
    std::printf("    => %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

bool testGlobalPhaseDoesNotRoute() {
    std::printf("\n[4] GLOBAL PHASE DOES NOT ROUTE\n");
    const Medium m = interferometerCell(0.43 * kPi);
    Stepper u;
    u.build(m.h, kDt);

    Vec z = inputPacket();
    Vec shifted = z;
    for (auto& v : shifted) v *= std::exp(cd(0, 1.2345));
    z = u.evolve(std::move(z), kFlightSteps);
    shifted = u.evolve(std::move(shifted), kFlightSteps);

    const double prob_err = maxProbabilityDiff(z, shifted);
    const bool pass = prob_err < 1e-12;
    std::printf("    max routing probability diff=%.2e\n", prob_err);
    std::printf("    => %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

}  // namespace

int main() {
    std::printf("=====================================================================\n");
    std::printf(" GRAPH-WAVE INTERFEROMETER CELL CONTRACT TEST\n");
    std::printf(" Pure unitary cell: one input port, two arms, two output ports.\n");
    std::printf(" No absorber, no measurement layer, no decision, no learning, no V26.\n");
    std::printf("=====================================================================\n");

    int pass = 0;
    constexpr int total = 4;
    pass += testClosedUnitaryCell() ? 1 : 0;
    pass += testHolonomyRoutesWithoutMeasurement() ? 1 : 0;
    pass += testGaugeEquivalentCellSameRouting() ? 1 : 0;
    pass += testGlobalPhaseDoesNotRoute() ? 1 : 0;

    std::printf("\n=====================================================================\n");
    std::printf(" RESULT : %d / %d verified\n", pass, total);
    std::printf(" %s\n", pass == total
                    ? "unitary interferometer cell is a valid substrate-grown primitive"
                    : "do not compose cells yet; inspect the failing primitive");
    std::printf("=====================================================================\n");
    return pass == total ? 0 : 1;
}
