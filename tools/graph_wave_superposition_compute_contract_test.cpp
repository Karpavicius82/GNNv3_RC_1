// graph_wave_superposition_compute_contract_test
// ----------------------------------------------------------------------------
// One question: does SUPERPOSITION become COMPUTATION on the linear graph-wave
// substrate, through the Born-rule readout |psi|^2 -- without any nonlinearity
// in the Hamiltonian, without measurement back-action, without a trainer?
//
// The rest of the suite proved the substrate is unitary, gauge-covariant, and
// that pure transfer "COLLAPSES LINEARLY" (port_transfer_composition [5]). This
// test takes that same linear substrate and crosses the boundary the suite
// deliberately left open: it shows that
//   * amplitude evolution stays linear, BUT
//   * the probability readout |.|^2 is a NONLINEAR map of the input (the Born
//     cross/interference term), and
//   * a COHERENT superposition carries a relative-phase aggregate that an
//     INCOHERENT (classical magnitude-sum) aggregation structurally cannot.
//
// Substrate only: 2-node Hermitian edge, Cayley/Crank-Nicolson unitary step,
// same idiom as the other graph_wave_* contracts. No V26 types, no GA, no
// decision path.
// ----------------------------------------------------------------------------

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

double norm2(const Vec& z) {
    double s = 0.0;
    for (const auto& v : z) s += std::norm(v);
    return s;
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

struct Medium {
    Mat h;
    explicit Medium(const int n) : h(n, Vec(n, cd(0, 0))) {}

    // Hermitian edge: out<-in carries `value`, in<-out carries its conjugate.
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

// LU with partial pivoting -- solves the implicit half of the Cayley step.
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
                if (v > best) { best = v; row = i; }
            }
            if (row != k) { std::swap(a[row], a[k]); std::swap(piv[row], piv[k]); }
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

// Crank-Nicolson / Cayley unitary step: z_{n+1} = (I + iH dt/2)^{-1}(I - iH dt/2) z_n.
struct Stepper {
    Mat am;
    LUC lu;

    void build(const Mat& h, const double dt) {
        const auto n = static_cast<int>(h.size());
        Mat ap(n, Vec(n));
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

    Vec evolve(Vec z, const int steps) const {
        for (int t = 0; t < steps; ++t) z = lu.solve(matvec(am, z));
        return z;
    }
};

// Probability read at output node 0 -- the Born-rule readout under test.
double readP0(const Mat& u, const Vec& in) { return std::norm(matvec(u, in)[0]); }

bool report(const char* name, const bool ok) {
    std::printf("    => %s\n", ok ? "PASS" : "FAIL");
    if (!ok) std::printf("    !! %s\n", name);
    return ok;
}

}  // namespace

int main() {
    // --- substrate: a single Hermitian edge tuned to a ~50/50 mixer ---------
    constexpr int kNodes = 2;
    constexpr int kSteps = 4096;
    const double flightTime = kPi / 4.0;  // J=1, T=pi/4 -> cos=sin=1/sqrt(2)
    const double dt = flightTime / static_cast<double>(kSteps);

    Medium medium(kNodes);
    medium.transferEdge(1, 0, cd(1.0, 0.0));
    Stepper step;
    step.build(medium.h, dt);

    // Build the 2x2 transfer matrix U from the evolution itself.
    const Vec e0 = {cd(1, 0), cd(0, 0)};
    const Vec e1 = {cd(0, 0), cd(1, 0)};
    const Vec u0 = step.evolve(e0, kSteps);
    const Vec u1 = step.evolve(e1, kSteps);
    const Mat U = {{u0[0], u1[0]}, {u0[1], u1[1]}};

    std::printf("=====================================================================\n");
    std::printf(" GRAPH-WAVE SUPERPOSITION-AS-COMPUTATION CONTRACT TEST\n");
    std::printf(" Linear amplitude evolution + Born readout |psi|^2 on one edge.\n");
    std::printf("=====================================================================\n");
    std::printf("    hermiticity error=%.2e  |U00|^2=%.6f |U01|^2=%.6f\n",
                medium.hermiticityError(), std::norm(U[0][0]), std::norm(U[0][1]));

    int pass = 0;
    int total = 0;

    // [1] AMPLITUDE EVOLUTION IS LINEAR (the premise the power rests on) ------
    std::printf("\n[1] AMPLITUDE EVOLUTION IS LINEAR\n");
    {
        const cd alpha(0.7, -0.4), beta(-0.2, 0.9);
        const Vec a = {cd(0.3, 0.5), cd(-0.6, 0.2)};
        const Vec b = {cd(0.1, -0.8), cd(0.4, 0.7)};
        Vec mix(kNodes);
        for (int i = 0; i < kNodes; ++i) mix[i] = alpha * a[i] + beta * b[i];
        const Vec lhs = matvec(U, mix);
        const Vec ua = matvec(U, a);
        const Vec ub = matvec(U, b);
        double err = 0.0;
        for (int i = 0; i < kNodes; ++i)
            err = std::max(err, std::abs(lhs[i] - (alpha * ua[i] + beta * ub[i])));
        std::printf("    U(alpha a + beta b) - (alpha Ua + beta Ub) = %.2e\n", err);
        ++total; pass += report("amplitude evolution is not linear", err < 1e-10);
    }

    // [2] BORN READOUT IS NONLINEAR (interference cross term) -----------------
    std::printf("\n[2] BORN READOUT |psi|^2 IS NONLINEAR\n");
    {
        const Vec a = e0;                 // node-0 input
        const Vec b = {cd(0, 0), cd(0, 1)};  // i * node-1 input
        Vec ab(kNodes);
        for (int i = 0; i < kNodes; ++i) ab[i] = a[i] + b[i];
        const double pa = readP0(U, a);
        const double pb = readP0(U, b);
        const double pab = readP0(U, ab);
        const double discrepancy = pab - pa - pb;
        const cd ua0 = matvec(U, a)[0];
        const cd ub0 = matvec(U, b)[0];
        const double cross = 2.0 * std::real(std::conj(ua0) * ub0);
        std::printf("    P(a)=%.6f P(b)=%.6f P(a+b)=%.6f\n", pa, pb, pab);
        std::printf("    nonadditivity P(a+b)-P(a)-P(b)=%.6f  analytic cross=%.6f\n",
                    discrepancy, cross);
        const bool matches = std::abs(discrepancy - cross) < 1e-9;
        const bool nontrivial = std::abs(cross) > 0.1;
        ++total; pass += report("Born readout is additive (no interference term)",
                                matches && nontrivial);
    }

    // [3] COHERENT SUPERPOSITION COMPUTES WHAT INCOHERENT SUM CANNOT ----------
    std::printf("\n[3] COHERENT SUPERPOSITION CARRIES A RELATIVE-PHASE AGGREGATE\n");
    {
        const double incoherent = 0.5 * (readP0(U, e0) + readP0(U, e1));
        double pmin = 1e9, pmax = -1e9, maxDev = 0.0;
        constexpr int kPhases = 256;
        for (int s = 0; s < kPhases; ++s) {
            const double phi = 2.0 * kPi * static_cast<double>(s) / kPhases;
            const Vec sup = {cd(1.0 / std::sqrt(2.0), 0.0),
                             std::exp(cd(0, phi)) / std::sqrt(2.0)};
            const double p = readP0(U, sup);
            pmin = std::min(pmin, p);
            pmax = std::max(pmax, p);
            maxDev = std::max(maxDev, std::abs(p - incoherent));
        }
        const double visibility = (pmax - pmin) / (pmax + pmin);
        std::printf("    incoherent classical aggregate (phase-blind)=%.6f\n", incoherent);
        std::printf("    coherent P0 over relative phase: min=%.6f max=%.6f\n", pmin, pmax);
        std::printf("    max|coherent - incoherent|=%.6f  fringe visibility=%.6f\n",
                    maxDev, visibility);
        ++total; pass += report("superposition adds no phase-carried information",
                                maxDev > 0.3 && visibility > 0.9);
    }

    // [4] GLOBAL PHASE IS NOT A SIGNAL; RELATIVE PHASE IS ---------------------
    std::printf("\n[4] GLOBAL PHASE IS NOT A SIGNAL, RELATIVE PHASE IS\n");
    {
        const Vec s = {cd(1.0 / std::sqrt(2.0), 0.0),
                       std::exp(cd(0, 0.9)) / std::sqrt(2.0)};
        Vec sg(kNodes);
        const cd g = std::exp(cd(0, 0.7));
        for (int i = 0; i < kNodes; ++i) sg[i] = g * s[i];
        const double globalDiff = std::abs(readP0(U, s) - readP0(U, sg));
        // relative-phase sensitivity reuses the fringe from [3]
        const Vec sRel = {cd(1.0 / std::sqrt(2.0), 0.0),
                          std::exp(cd(0, 0.9 + kPi)) / std::sqrt(2.0)};
        const double relDiff = std::abs(readP0(U, s) - readP0(U, sRel));
        std::printf("    global-phase P0 diff=%.2e  relative-phase(+pi) P0 diff=%.6f\n",
                    globalDiff, relDiff);
        ++total; pass += report("global phase leaked into the readout / relative phase inert",
                                globalDiff < 1e-9 && relDiff > 0.3);
    }

    std::printf("\n RESULT : %d / %d verified\n", pass, total);
    return pass == total ? 0 : 1;
}
