#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <utility>
#include <vector>

namespace {

using cd = std::complex<double>;
using Mat = std::vector<std::vector<cd>>;
using Vec = std::vector<cd>;

constexpr double kPi = 3.141592653589793238462643383279502884;

double wrapPhase(double x) {
    while (x <= -kPi) x += 2.0 * kPi;
    while (x > kPi) x -= 2.0 * kPi;
    return x;
}

double phaseDiff(const double a, const double b) {
    return std::abs(wrapPhase(a - b));
}

double norm2(const Vec& z) {
    double s = 0.0;
    for (const auto& v : z) s += std::norm(v);
    return s;
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

struct Graph {
    Mat h;

    explicit Graph(const int n) : h(n, Vec(n, cd(0, 0))) {}

    void addEdge(const int i, const int j, const double a, const double phase = 0.0) {
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
};

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

double wilsonPhase(const Mat& h, const std::vector<int>& loop) {
    cd product(1, 0);
    for (std::size_t k = 0; k < loop.size(); ++k) {
        const int i = loop[k];
        const int j = loop[(k + 1) % loop.size()];
        product *= h[i][j] / std::abs(h[i][j]);
    }
    return std::arg(product);
}

Graph diamond(const double flux) {
    Graph g(4);
    constexpr int s = 0;
    constexpr int upper = 1;
    constexpr int lower = 2;
    constexpr int target = 3;
    g.addEdge(s, upper, 1.0);
    g.addEdge(upper, target, 1.0);
    g.addEdge(target, lower, 1.0);
    g.addEdge(lower, s, 1.0, flux);
    return g;
}

double transferToTarget(const Mat& h) {
    Stepper u;
    u.build(h, 0.10);
    Vec z(4, cd(0, 0));
    z[0] = cd(1, 0);
    double acc = 0.0;
    constexpr int steps = 400;
    for (int t = 0; t < steps; ++t) {
        z = u.step(z);
        acc += std::norm(z[3]);
    }
    return acc / steps;
}

bool testWilsonLoopGaugeInvariant() {
    std::printf("\n[1] WILSON LOOP IS GAUGE-INVARIANT\n");
    constexpr double flux = 0.73 * kPi;
    Graph g(4);
    g.addEdge(0, 1, 1.0, 0.17);
    g.addEdge(1, 2, 1.0, -0.31);
    g.addEdge(2, 3, 1.0, 0.29);
    g.addEdge(3, 0, 1.0, flux - 0.17 + 0.31 - 0.29);
    const std::vector<int> loop{0, 1, 2, 3};
    const std::vector<double> chi{0.13, -0.91, 1.72, 0.44};
    const Mat gt = gaugeTransform(g.h, chi);

    const double before = wilsonPhase(g.h, loop);
    const double after = wilsonPhase(gt, loop);
    const bool pass = g.hermiticityError() < 1e-14 && phaseDiff(before, after) < 1e-12;
    std::printf("    requested flux=%.12f\n", wrapPhase(flux));
    std::printf("    Wilson before=%.12f after=%.12f diff=%.2e\n",
                before, after, phaseDiff(before, after));
    std::printf("    => %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

bool testGaugeCovariantEvolution() {
    std::printf("\n[2] EVOLUTION IS GAUGE-COVARIANT\n");
    Graph g(5);
    g.addEdge(0, 1, 0.9, 0.3);
    g.addEdge(1, 2, 1.1, -0.2);
    g.addEdge(2, 3, 0.7, 1.1);
    g.addEdge(3, 0, 1.0, -0.4);
    g.addEdge(1, 4, 0.8, 0.6);
    g.addEdge(4, 3, 1.2, -1.0);
    const std::vector<double> chi{0.8, -0.4, 1.7, -1.1, 0.25};
    const Mat gt = gaugeTransform(g.h, chi);

    Stepper u;
    Stepper ug;
    u.build(g.h, 0.14);
    ug.build(gt, 0.14);

    Vec z = {cd(0.2, 0.4), cd(-0.3, 0.1), cd(0.5, -0.2), cd(-0.1, -0.3), cd(0.4, 0.2)};
    const double inv = 1.0 / std::sqrt(norm2(z));
    for (auto& v : z) v *= inv;
    Vec zg = gaugeTransform(z, chi);

    double max_state_err = 0.0;
    double max_prob_err = 0.0;
    for (int t = 0; t < 80; ++t) {
        z = u.step(z);
        zg = ug.step(zg);
        const Vec expected = gaugeTransform(z, chi);
        max_state_err = std::max(max_state_err, maxStateDiff(zg, expected));
        max_prob_err = std::max(max_prob_err, maxProbabilityDiff(zg, z));
    }
    const bool pass = max_state_err < 1e-12 && max_prob_err < 1e-12;
    std::printf("    max |z_gauge - G*z| = %.2e\n", max_state_err);
    std::printf("    max probability difference = %.2e\n", max_prob_err);
    std::printf("    => %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

bool testFluxNotPhasePlacementControlsInterference() {
    std::printf("\n[3] INTERFERENCE DEPENDS ON LOOP FLUX, NOT PHASE PLACEMENT\n");
    constexpr double flux = 0.61 * kPi;
    const Graph g = diamond(flux);
    const std::vector<double> chi{0.0, 0.7, -1.3, 0.2};
    const Mat redistributed = gaugeTransform(g.h, chi);

    const double base = transferToTarget(g.h);
    const double moved = transferToTarget(redistributed);
    const double zero_flux = transferToTarget(diamond(0.0).h);
    const double pi_flux = transferToTarget(diamond(kPi).h);
    const bool pass = std::abs(base - moved) < 1e-12 &&
                      std::abs(zero_flux - pi_flux) > 0.1;
    std::printf("    same flux, original placement    <P_T>=%.12f\n", base);
    std::printf("    same flux, gauge-redistributed   <P_T>=%.12f diff=%.2e\n",
                moved, std::abs(base - moved));
    std::printf("    controls: flux=0 <P_T>=%.6f, flux=pi <P_T>=%.6e\n", zero_flux, pi_flux);
    std::printf("    => %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

Graph landauGrid(const int w, const int h, const double alpha) {
    Graph g(w * h);
    auto id = [w](const int x, const int y) { return y * w + x; };
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (x + 1 < w) g.addEdge(id(x, y), id(x + 1, y), 1.0);
            if (y + 1 < h) g.addEdge(id(x, y), id(x, y + 1), 1.0, 2.0 * kPi * alpha * x);
        }
    }
    return g;
}

bool testLandauPlaquetteFlux() {
    std::printf("\n[4] LANDAU GAUGE PLAQUETTE WILSON FLUX\n");
    constexpr int w = 5;
    constexpr int h = 4;
    constexpr double alpha = 3.0 / 8.0;
    const Graph g = landauGrid(w, h, alpha);
    auto id = [](const int x, const int y) { return y * w + x; };
    const double expected = wrapPhase(2.0 * kPi * alpha);
    double max_err = 0.0;
    for (int y = 0; y + 1 < h; ++y) {
        for (int x = 0; x + 1 < w; ++x) {
            const std::vector<int> loop{id(x, y), id(x + 1, y), id(x + 1, y + 1), id(x, y + 1)};
            max_err = std::max(max_err, phaseDiff(wilsonPhase(g.h, loop), expected));
        }
    }
    const bool pass = g.hermiticityError() < 1e-14 && max_err < 1e-12;
    std::printf("    expected plaquette flux=%.12f   max error=%.2e\n", expected, max_err);
    std::printf("    => %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

}  // namespace

int main() {
    std::printf("=====================================================================\n");
    std::printf(" GAUGE / WILSON INVARIANCE TEST FOR COMPLEX GRAPH-WAVE SUBSTRATE\n");
    std::printf(" Checks only substrate geometry: local gauge, loop flux, covariance\n");
    std::printf("=====================================================================\n");

    int pass = 0;
    constexpr int total = 4;
    pass += testWilsonLoopGaugeInvariant() ? 1 : 0;
    pass += testGaugeCovariantEvolution() ? 1 : 0;
    pass += testFluxNotPhasePlacementControlsInterference() ? 1 : 0;
    pass += testLandauPlaquetteFlux() ? 1 : 0;

    std::printf("\n=====================================================================\n");
    std::printf(" RESULT : %d / %d verified\n", pass, total);
    std::printf(" %s\n", pass == total
                    ? "gauge phases are physical holonomies, not coordinate artifacts"
                    : "gauge/Wilson claim not verified; inspect failing test");
    std::printf("=====================================================================\n");
    return pass == total ? 0 : 1;
}
