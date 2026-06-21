#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {

using cd = std::complex<double>;
using CMat = std::vector<std::vector<cd>>;
using RMat = std::vector<std::vector<double>>;

constexpr double kPi = 3.141592653589793238462643383279502884;

struct Graph {
    CMat h;

    explicit Graph(const int n) : h(n, std::vector<cd>(n, cd(0, 0))) {}

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

RMat realRepresentation(const CMat& h) {
    const auto n = static_cast<int>(h.size());
    RMat r(2 * n, std::vector<double>(2 * n, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const double a = h[i][j].real();
            const double b = h[i][j].imag();
            r[i][j] = a;
            r[i][j + n] = -b;
            r[i + n][j] = b;
            r[i + n][j + n] = a;
        }
    }
    return r;
}

std::vector<double> jacobiEigenvalues(RMat a) {
    const int n = static_cast<int>(a.size());
    for (int sweep = 0; sweep < 260; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) off += a[p][q] * a[p][q];
        }
        if (off < 1e-22) break;

        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                const double apq = a[p][q];
                if (std::abs(apq) < 1e-300) continue;
                const double app = a[p][p];
                const double aqq = a[q][q];
                const double tau = (aqq - app) / (2.0 * apq);
                const double t = tau >= 0.0
                    ? 1.0 / (tau + std::sqrt(1.0 + tau * tau))
                    : -1.0 / (-tau + std::sqrt(1.0 + tau * tau));
                const double c = 1.0 / std::sqrt(1.0 + t * t);
                const double s = t * c;

                for (int k = 0; k < n; ++k) {
                    if (k == p || k == q) continue;
                    const double akp = a[k][p];
                    const double akq = a[k][q];
                    a[k][p] = a[p][k] = c * akp - s * akq;
                    a[k][q] = a[q][k] = s * akp + c * akq;
                }
                a[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
                a[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
                a[p][q] = a[q][p] = 0.0;
            }
        }
    }

    std::vector<double> ev(n);
    for (int i = 0; i < n; ++i) ev[i] = a[i][i];
    std::sort(ev.begin(), ev.end());
    return ev;
}

Graph fluxLattice(const int w, const int h, const double alpha) {
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

Graph randomPhaseLattice(const int w, const int h) {
    Graph g(w * h);
    auto id = [w](const int x, const int y) { return y * w + x; };
    std::mt19937 rng(0xBADC0DEu);
    std::uniform_real_distribution<double> phase(-kPi, kPi);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (x + 1 < w) g.addEdge(id(x, y), id(x + 1, y), 1.0, phase(rng));
            if (y + 1 < h) g.addEdge(id(x, y), id(x, y + 1), 1.0, phase(rng));
        }
    }
    return g;
}

std::vector<double> spectrum(const Graph& g) {
    return jacobiEigenvalues(realRepresentation(g.h));
}

double boxDimension(std::vector<double> ev) {
    std::sort(ev.begin(), ev.end());
    const double lo = ev.front();
    const double hi = ev.back();
    const double width = hi - lo;
    const int bins[] = {16, 32, 64, 128, 256};
    constexpr int m = 5;
    double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
    for (const int bcount : bins) {
        std::vector<char> occupied(bcount, 0);
        for (const double e : ev) {
            int b = static_cast<int>(((e - lo) / width) * bcount);
            if (b < 0) b = 0;
            if (b >= bcount) b = bcount - 1;
            occupied[b] = 1;
        }
        int count = 0;
        for (const char b : occupied) count += b ? 1 : 0;
        const double lx = std::log(static_cast<double>(bcount));
        const double ly = std::log(static_cast<double>(count));
        sx += lx;
        sy += ly;
        sxx += lx * lx;
        sxy += lx * ly;
    }
    return (m * sxy - sx * sy) / (m * sxx - sx * sx);
}

int countGaps(std::vector<double> ev, const double min_gap) {
    std::sort(ev.begin(), ev.end());
    int count = 0;
    for (std::size_t i = 1; i < ev.size(); ++i) {
        if (ev[i] - ev[i - 1] > min_gap) ++count;
    }
    return count;
}

std::vector<double> histogram(const std::vector<double>& ev, const double lo, const double hi, const int bins) {
    std::vector<double> hist(bins, 0.0);
    const double width = hi - lo;
    for (const double e : ev) {
        int b = static_cast<int>(((e - lo) / width) * bins);
        if (b < 0) b = 0;
        if (b >= bins) b = bins - 1;
        hist[b] += 1.0;
    }
    for (double& v : hist) v /= static_cast<double>(ev.size());
    return hist;
}

double l1(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) s += std::abs(a[i] - b[i]);
    return s;
}

double maxEigenDiff(const std::vector<double>& a, const std::vector<double>& b) {
    double err = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) err = std::max(err, std::abs(a[i] - b[i]));
    return err;
}

struct Case {
    std::string name;
    double herm = 0.0;
    double dim = 0.0;
    double width = 0.0;
    int gap2 = 0;
    int gap1 = 0;
    int gap05 = 0;
    std::vector<double> ev;
    std::vector<double> hist;
};

Case makeCase(const std::string& name, const Graph& g) {
    Case c;
    c.name = name;
    c.herm = g.hermiticityError();
    c.ev = spectrum(g);
    c.width = c.ev.back() - c.ev.front();
    c.dim = boxDimension(c.ev);
    c.gap2 = countGaps(c.ev, 0.02 * c.width);
    c.gap1 = countGaps(c.ev, 0.01 * c.width);
    c.gap05 = countGaps(c.ev, 0.005 * c.width);
    return c;
}

void printCase(const Case& c) {
    std::printf("    %-10s herm=%.1e dim=%.3f width=%.3f gaps>2%%/1%%/0.5%%=%d/%d/%d\n",
                c.name.c_str(), c.herm, c.dim, c.width, c.gap2, c.gap1, c.gap05);
}

}  // namespace

int main() {
    std::printf("=====================================================================\n");
    std::printf(" SUBSTRATE FLUX SCAN\n");
    std::printf(" Question only: is the golden-flux signature a real flux-family effect?\n");
    std::printf(" No measurement, no decision, no learning, no V26 runtime.\n");
    std::printf("=====================================================================\n");

    constexpr int w = 13;
    constexpr int h = 13;
    Case zero = makeCase("alpha=0", fluxLattice(w, h, 0.0));
    Case half = makeCase("alpha=1/2", fluxLattice(w, h, 0.5));
    Case third = makeCase("alpha=1/3", fluxLattice(w, h, 1.0 / 3.0));
    Case fib = makeCase("alpha=8/13", fluxLattice(w, h, 8.0 / 13.0));
    Case comp = makeCase("alpha=5/13", fluxLattice(w, h, 5.0 / 13.0));
    Case random = makeCase("random", randomPhaseLattice(w, h));

    std::vector<Case*> cases{&zero, &half, &third, &fib, &comp, &random};
    double lo = cases.front()->ev.front();
    double hi = cases.front()->ev.back();
    for (const auto* c : cases) {
        lo = std::min(lo, c->ev.front());
        hi = std::max(hi, c->ev.back());
    }
    for (auto* c : cases) c->hist = histogram(c->ev, lo, hi, 64);

    std::printf("\n[1] DIRECT GRAPH-WAVE FLUX FAMILY (N=%d)\n", w * h);
    for (const auto* c : cases) printCase(*c);

    const double complement_err = maxEigenDiff(fib.ev, comp.ev);
    const double l1_fib_zero = l1(fib.hist, zero.hist);
    const double l1_fib_half = l1(fib.hist, half.hist);
    const double l1_fib_third = l1(fib.hist, third.hist);
    const double l1_fib_random = l1(fib.hist, random.hist);

    std::printf("\n[2] FLUX SANITY CHECKS\n");
    std::printf("    max eigen diff alpha=8/13 vs 5/13 complement = %.2e\n", complement_err);
    std::printf("    L1(8/13, 0)      = %.3f\n", l1_fib_zero);
    std::printf("    L1(8/13, 1/2)    = %.3f\n", l1_fib_half);
    std::printf("    L1(8/13, 1/3)    = %.3f\n", l1_fib_third);
    std::printf("    L1(8/13, random) = %.3f\n", l1_fib_random);

    const bool hermitian = std::all_of(cases.begin(), cases.end(), [](const Case* c) {
        return c->herm < 1e-14;
    });
    const bool complement_symmetry = complement_err < 1e-10;
    const bool flux_moves_spectrum = l1_fib_zero > 0.5 && l1_fib_random > 0.5;
    const bool not_simple_rational = l1_fib_half > 0.25 && l1_fib_third > 0.25;
    const bool gap_family_not_random = fib.gap2 > random.gap2 && fib.gap1 >= random.gap1;

    std::printf("\n[3] DECISION RULE FOR THIS TEST ONLY\n");
    std::printf("    all Hermitian=%s\n", hermitian ? "YES" : "NO");
    std::printf("    alpha and 1-alpha spectral symmetry=%s\n", complement_symmetry ? "YES" : "NO");
    std::printf("    8/13 differs from zero/random controls=%s\n", flux_moves_spectrum ? "YES" : "NO");
    std::printf("    8/13 differs from simple rational controls=%s\n", not_simple_rational ? "YES" : "NO");
    std::printf("    8/13 gap family is not random-phase=%s\n", gap_family_not_random ? "YES" : "NO");

    const bool pass = hermitian && complement_symmetry && flux_moves_spectrum &&
                      not_simple_rational && gap_family_not_random;

    std::printf("\n=====================================================================\n");
    std::printf(" RESULT : %s\n", pass ? "PASS" : "FAIL");
    std::printf(" %s\n", pass
                    ? "golden-flux signature survives flux-family controls; still substrate-only"
                    : "golden-flux signature not established against flux-family controls");
    std::printf("=====================================================================\n");
    return pass ? 0 : 1;
}
