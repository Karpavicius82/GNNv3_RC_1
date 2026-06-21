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

double wrapPhase(double x) {
    while (x <= -kPi) x += 2.0 * kPi;
    while (x > kPi) x -= 2.0 * kPi;
    return x;
}

double phaseDistance(const double a, const double b) {
    return std::abs(wrapPhase(a - b));
}

double maxStateDiff(const Vec& a, const Vec& b) {
    double err = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) err = std::max(err, std::abs(a[i] - b[i]));
    return err;
}

double maxDoubleDiff(const std::vector<double>& a, const std::vector<double>& b) {
    double err = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) err = std::max(err, std::abs(a[i] - b[i]));
    return err;
}

double maxMatrixDiff(const Mat& a, const Mat& b) {
    double err = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        for (std::size_t j = 0; j < a[i].size(); ++j) err = std::max(err, std::abs(a[i][j] - b[i][j]));
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

struct ObservableFrame {
    double norm = 0.0;
    std::vector<double> p;
    double participation = 0.0;
    double entropy = 0.0;
};

ObservableFrame observe(const Vec& z) {
    ObservableFrame out;
    out.p.resize(z.size());
    double p2 = 0.0;
    for (std::size_t i = 0; i < z.size(); ++i) {
        out.p[i] = std::norm(z[i]);
        out.norm += out.p[i];
        p2 += out.p[i] * out.p[i];
    }
    out.participation = (p2 > 0.0) ? (out.norm * out.norm / p2) : 0.0;
    for (const double p : out.p) {
        if (p > 0.0 && out.norm > 0.0) {
            const double q = p / out.norm;
            out.entropy -= q * std::log(q);
        }
    }
    return out;
}

double portPower(const ObservableFrame& o, const std::vector<int>& nodes) {
    double s = 0.0;
    for (const int n : nodes) s += o.p[static_cast<std::size_t>(n)];
    return s;
}

double contrast(const double a, const double b) {
    const double d = a + b;
    return d > 0.0 ? (a - b) / d : 0.0;
}

double edgeCurrent(const Mat& h, const Vec& z, const int i, const int j) {
    return 2.0 * std::imag(std::conj(z[static_cast<std::size_t>(i)]) *
                           h[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] *
                           z[static_cast<std::size_t>(j)]);
}

double wilsonPhase(const Mat& h, const std::vector<int>& loop) {
    cd product(1, 0);
    for (std::size_t k = 0; k < loop.size(); ++k) {
        const int i = loop[k];
        const int j = loop[(k + 1) % loop.size()];
        const cd e = h[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
        product *= e / std::abs(e);
    }
    return std::arg(product);
}

Medium interferometerCell(const double flux) {
    Medium m(kNodes);
    m.edge(kIn, kArmA, 1.0);
    m.edge(kIn, kArmB, 1.0);
    m.edge(kArmA, kOutPlus, 1.0);
    m.edge(kArmB, kOutPlus, 1.0, flux);
    m.edge(kArmA, kOutMinus, 1.0);
    m.edge(kArmB, kOutMinus, 1.0, flux + kPi);
    return m;
}

Vec inputPacket() {
    Vec z(kNodes, cd(0, 0));
    z[kIn] = cd(1, 0);
    return z;
}

Vec route(const Medium& m) {
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

bool testObservableIsNonInvasive() {
    std::printf("\n[1] OBSERVABLE IS NON-INVASIVE\n");
    const Medium m = interferometerCell(kPi / 2.0);
    const Vec z = route(m);
    const Vec z_before = z;
    const Mat h_before = m.h;

    const auto o = observe(z);
    const double plus = portPower(o, {kOutPlus});
    const double minus = portPower(o, {kOutMinus});
    const double c = contrast(plus, minus);
    const bool pass = maxStateDiff(z, z_before) == 0.0 && maxMatrixDiff(m.h, h_before) == 0.0 &&
                      std::abs(o.norm - 1.0) < 1e-12 && std::abs(c) < 1e-10 &&
                      o.participation > 1.0 && o.entropy > 0.0;

    std::printf("    state mutation error=%.2e   medium mutation error=%.2e\n",
                maxStateDiff(z, z_before), maxMatrixDiff(m.h, h_before));
    std::printf("    norm=%.12f P(out+)=%.12f P(out-)=%.12f contrast=%.3e\n",
                o.norm, plus, minus, c);
    std::printf("    participation=%.6f entropy=%.6f\n", o.participation, o.entropy);
    std::printf("    => %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

bool testGaugeAndGlobalPhaseInvariantObservables() {
    std::printf("\n[2] OBSERVABLES ARE GAUGE AND GLOBAL-PHASE SAFE\n");
    const Medium m = interferometerCell(0.61 * kPi);
    const Vec z = route(m);
    const auto chi = localGauge();

    Medium mg(kNodes);
    mg.h = gaugeTransform(m.h, chi);
    Stepper ug;
    ug.build(mg.h, kDt);
    Vec zg = ug.evolve(gaugeTransform(inputPacket(), chi), kFlightSteps);

    Vec shifted = inputPacket();
    for (auto& v : shifted) v *= std::exp(cd(0, 1.2345));
    Stepper u;
    u.build(m.h, kDt);
    shifted = u.evolve(std::move(shifted), kFlightSteps);

    const auto o = observe(z);
    const auto og = observe(zg);
    const auto os = observe(shifted);
    const double p_err_g = maxDoubleDiff(o.p, og.p);
    const double p_err_s = maxDoubleDiff(o.p, os.p);
    const double contrast_err = std::abs(
        contrast(portPower(o, {kOutPlus}), portPower(o, {kOutMinus})) -
        contrast(portPower(og, {kOutPlus}), portPower(og, {kOutMinus})));
    const double wilson_err = phaseDistance(wilsonPhase(m.h, {kIn, kArmA, kOutPlus, kArmB}),
                                            wilsonPhase(mg.h, {kIn, kArmA, kOutPlus, kArmB}));
    const double current_err = std::abs(edgeCurrent(m.h, z, kArmA, kOutPlus) -
                                        edgeCurrent(mg.h, zg, kArmA, kOutPlus));
    const bool pass = p_err_g < 1e-12 && p_err_s < 1e-12 && contrast_err < 1e-12 &&
                      std::abs(o.participation - og.participation) < 1e-12 &&
                      std::abs(o.entropy - og.entropy) < 1e-12 &&
                      wilson_err < 1e-12 && current_err < 1e-12;

    std::printf("    probability diff gauge=%.2e global=%.2e\n", p_err_g, p_err_s);
    std::printf("    contrast diff gauge=%.2e\n", contrast_err);
    std::printf("    Wilson phase diff gauge=%.2e\n", wilson_err);
    std::printf("    edge current diff gauge=%.2e\n", current_err);
    std::printf("    => %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

bool testEdgeCurrentContinuity() {
    std::printf("\n[3] EDGE CURRENT SATISFIES CONTINUITY\n");
    const Medium m = interferometerCell(0.37 * kPi);
    Vec z = normalized({cd(0.31, 0.2), cd(-0.4, 0.1), cd(0.2, -0.6), cd(0.5, 0.1), cd(-0.3, 0.7)});
    const Vec hz = matvec(m.h, z);

    double antisym_err = 0.0;
    double continuity_err = 0.0;
    double global_net = 0.0;
    for (int i = 0; i < kNodes; ++i) {
        double net = 0.0;
        for (int j = 0; j < kNodes; ++j) {
            const double jij = edgeCurrent(m.h, z, i, j);
            const double jji = edgeCurrent(m.h, z, j, i);
            antisym_err = std::max(antisym_err, std::abs(jij + jji));
            net += jij;
        }
        const double schrodinger_dpdt = 2.0 * std::real(std::conj(z[static_cast<std::size_t>(i)]) *
                                                        (cd(0, -1) * hz[static_cast<std::size_t>(i)]));
        continuity_err = std::max(continuity_err, std::abs(net - schrodinger_dpdt));
        global_net += net;
    }

    const bool pass = antisym_err < 1e-12 && continuity_err < 1e-12 && std::abs(global_net) < 1e-12;
    std::printf("    antisymmetry max |J_ij + J_ji|=%.2e\n", antisym_err);
    std::printf("    continuity max |sum_j J_ij - dp_i/dt|=%.2e\n", continuity_err);
    std::printf("    global net current=%.2e\n", global_net);
    std::printf("    => %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

bool testRoutingReadByObservables() {
    std::printf("\n[4] ROUTING IS READ BY PORT POWER AND CONTRAST\n");
    const auto oz = observe(route(interferometerCell(0.0)));
    const auto oh = observe(route(interferometerCell(kPi / 2.0)));
    const auto op = observe(route(interferometerCell(kPi)));

    const double cz = contrast(portPower(oz, {kOutPlus}), portPower(oz, {kOutMinus}));
    const double ch = contrast(portPower(oh, {kOutPlus}), portPower(oh, {kOutMinus}));
    const double cp = contrast(portPower(op, {kOutPlus}), portPower(op, {kOutMinus}));

    const bool pass = cz > 0.999 && std::abs(ch) < 1e-10 && cp < -0.999 &&
                      portPower(oz, {kOutPlus, kOutMinus}) > 0.999 &&
                      portPower(op, {kOutPlus, kOutMinus}) > 0.999;

    std::printf("    flux=0    Pout=%.12f contrast=%.12f\n", portPower(oz, {kOutPlus, kOutMinus}), cz);
    std::printf("    flux=pi/2 Pout=%.12f contrast=%.3e\n", portPower(oh, {kOutPlus, kOutMinus}), ch);
    std::printf("    flux=pi   Pout=%.12f contrast=%.12f\n", portPower(op, {kOutPlus, kOutMinus}), cp);
    std::printf("    => %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

}  // namespace

int main() {
    std::printf("=====================================================================\n");
    std::printf(" GRAPH-WAVE OBSERVABLE CONTRACT TEST\n");
    std::printf(" Non-invasive physical reads: probabilities, ports, Wilson phase, current.\n");
    std::printf(" No absorber, no decision, no learning, no V26.\n");
    std::printf("=====================================================================\n");

    int pass = 0;
    constexpr int total = 4;
    pass += testObservableIsNonInvasive() ? 1 : 0;
    pass += testGaugeAndGlobalPhaseInvariantObservables() ? 1 : 0;
    pass += testEdgeCurrentContinuity() ? 1 : 0;
    pass += testRoutingReadByObservables() ? 1 : 0;

    std::printf("\n=====================================================================\n");
    std::printf(" RESULT : %d / %d verified\n", pass, total);
    std::printf(" %s\n", pass == total
                    ? "observable layer is usable as a non-invasive substrate read boundary"
                    : "observable layer is not clean enough; inspect failing read");
    std::printf("=====================================================================\n");
    return pass == total ? 0 : 1;
}
