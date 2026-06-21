// graph_wave_dynamic_bind_contract_test
// ----------------------------------------------------------------------------
// One question: is holographic / VSA BINDING actually PHYSICS on this substrate,
// or only hand-coded algebra? The holographic-memory contract bound vectors by
// elementwise phase multiplication (circular convolution). This test realizes
// the SAME binding as a real graph-wave time-evolution: light propagating
// through a translation-invariant ring medium (a phase grating / hologram),
// integrated by the suite's own Cayley `Stepper` in node space -- no FFT, no
// algebra shortcut.
//
//   * the operand is a dispersion omega_k on the ring's momentum modes;
//   * the medium is the Hermitian circulant H whose momentum eigenvalues are
//     omega_k (translation-invariant couplings = a recorded hologram);
//   * BIND   = evolve a under H for unit time   (==  e^{-iH}, momentum phase B_k=e^{-i omega_k});
//   * UNBIND = evolve under -H                  (the reverse propagation);
//   * position/order would be one hop of the same ring (shift), shown to commute.
//
// The Cayley integrator works purely in node space and CONVERGES to the
// algebraic holographic bind as it refines -- binding is physical propagation,
// honestly to the integrator's order. Substrate only, no V26, no trainer, no GA.
// ----------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <utility>
#include <vector>

namespace {

using cd = std::complex<double>;
using Vec = std::vector<cd>;
using Mat = std::vector<std::vector<cd>>;

constexpr int kN = 64;
constexpr double kTwoPi = 6.283185307179586476925286766559;

Vec dft(const Vec& x) {
    Vec X(kN, cd(0, 0));
    for (int k = 0; k < kN; ++k)
        for (int j = 0; j < kN; ++j)
            X[k] += x[j] * std::exp(cd(0, -kTwoPi * k * j / kN));
    return X;
}
Vec idft(const Vec& X) {
    Vec x(kN, cd(0, 0));
    for (int j = 0; j < kN; ++j) {
        for (int k = 0; k < kN; ++k) x[j] += X[k] * std::exp(cd(0, kTwoPi * k * j / kN));
        x[j] /= static_cast<double>(kN);
    }
    return x;
}

Vec matvec(const Mat& m, const Vec& z) {
    Vec out(kN, cd(0, 0));
    for (int i = 0; i < kN; ++i) { cd s(0, 0); for (int j = 0; j < kN; ++j) s += m[i][j] * z[j]; out[i] = s; }
    return out;
}
double norm2(const Vec& z) { double s = 0; for (auto& v : z) s += std::norm(v); return s; }
double maxDiff(const Vec& a, const Vec& b) { double e = 0; for (int i = 0; i < kN; ++i) e = std::max(e, std::abs(a[i] - b[i])); return e; }
Vec shift(const Vec& a, int k) { Vec c(kN); for (int i = 0; i < kN; ++i) c[i] = a[((i - k) % kN + kN) % kN]; return c; }

// --- Cayley / Crank-Nicolson unitary stepper (same idiom as the rest of suite)
struct LUC {
    int n = 0; Mat a; std::vector<int> piv;
    void factor(Mat in) {
        n = (int)in.size(); a = std::move(in); piv.resize(n);
        for (int i = 0; i < n; ++i) piv[i] = i;
        for (int k = 0; k < n; ++k) {
            double best = std::abs(a[k][k]); int row = k;
            for (int i = k + 1; i < n; ++i) { double v = std::abs(a[i][k]); if (v > best) { best = v; row = i; } }
            if (row != k) { std::swap(a[row], a[k]); std::swap(piv[row], piv[k]); }
            cd akk = a[k][k];
            for (int i = k + 1; i < n; ++i) { cd f = a[i][k] / akk; a[i][k] = f; for (int j = k + 1; j < n; ++j) a[i][j] -= f * a[k][j]; }
        }
    }
    Vec solve(const Vec& b) const {
        Vec x(n);
        for (int i = 0; i < n; ++i) x[i] = b[piv[i]];
        for (int i = 0; i < n; ++i) { cd s = x[i]; for (int j = 0; j < i; ++j) s -= a[i][j] * x[j]; x[i] = s; }
        for (int i = n - 1; i >= 0; --i) { cd s = x[i]; for (int j = i + 1; j < n; ++j) s -= a[i][j] * x[j]; x[i] = s / a[i][i]; }
        return x;
    }
};
struct Stepper {
    Mat am; LUC lu;
    void build(const Mat& h, double dt) {
        Mat ap(kN, Vec(kN)); am.assign(kN, Vec(kN)); const cd I(0, 1);
        for (int i = 0; i < kN; ++i) for (int j = 0; j < kN; ++j) {
            cd k = I * h[i][j] * (dt / 2.0); cd id = (i == j) ? cd(1, 0) : cd(0, 0);
            ap[i][j] = id + k; am[i][j] = id - k;
        }
        lu.factor(ap);
    }
    Vec evolve(Vec z, int steps) const { for (int t = 0; t < steps; ++t) z = lu.solve(matvec(am, z)); return z; }
};

double hermiticityError(const Mat& h) {
    double e = 0; for (int i = 0; i < kN; ++i) for (int j = 0; j < kN; ++j) e = std::max(e, std::abs(h[i][j] - std::conj(h[j][i]))); return e;
}

bool report(const char* name, bool ok) { std::printf("    => %s\n", ok ? "PASS" : "FAIL"); if (!ok) std::printf("    !! %s\n", name); return ok; }

}  // namespace

int main() {
    std::printf("=====================================================================\n");
    std::printf(" GRAPH-WAVE DYNAMIC BIND CONTRACT TEST  (ring N=%d)\n", kN);
    std::printf(" Holographic binding realized as real Cayley time-evolution.\n");
    std::printf("=====================================================================\n");

    std::mt19937_64 rng(0xB12DULL);
    std::uniform_real_distribution<double> uni(-3.14159, 3.14159);

    // operand = a momentum dispersion omega_k (the recorded hologram)
    Vec omega(kN);
    for (int k = 0; k < kN; ++k) omega[k] = cd(uni(rng), 0.0);

    // ring medium: Hermitian circulant H with momentum eigenvalues omega_k
    Vec c = idft(omega);                       // first column of the circulant
    Mat H(kN, Vec(kN));
    for (int i = 0; i < kN; ++i) for (int j = 0; j < kN; ++j) H[i][j] = c[((i - j) % kN + kN) % kN];

    // algebraic holographic bind: multiply momentum modes by B_k = e^{-i omega_k}
    auto bindAlg = [&](const Vec& a) {
        Vec A = dft(a);
        for (int k = 0; k < kN; ++k) A[k] *= std::exp(cd(0, -std::real(omega[k])));
        return idft(A);
    };

    // a random unit-phasor field to bind
    Vec a(kN);
    for (int i = 0; i < kN; ++i) { double p = uni(rng); a[i] = cd(std::cos(p), std::sin(p)); }

    int pass = 0, total = 0;

    // [1] the ring medium is a real (Hermitian) wave Hamiltonian -------------
    std::printf("\n[1] THE BINDING MEDIUM IS A HERMITIAN RING HAMILTONIAN\n");
    {
        double he = hermiticityError(H);
        std::printf("    hermiticity error = %.2e\n", he);
        ++total; pass += report("binding medium is not Hermitian", he < 1e-12);
    }

    // [2] Cayley evolution (node space, no FFT) CONVERGES to algebraic bind --
    std::printf("\n[2] DYNAMIC BIND CONVERGES TO THE HOLOGRAPHIC BIND (integrator refines)\n");
    {
        const Vec ref = bindAlg(a);
        double e256 = 0, e1024 = 0, e4096 = 0, drift = 0;
        for (int steps : {256, 1024, 4096}) {
            Stepper s; s.build(H, 1.0 / steps);
            const Vec dyn = s.evolve(a, steps);
            const double e = maxDiff(dyn, ref);
            if (steps == 256) e256 = e;
            if (steps == 1024) e1024 = e;
            if (steps == 4096) { e4096 = e; drift = std::abs(norm2(dyn) - norm2(a)); }
            std::printf("    steps=%5d  |dynamic - algebraic| = %.3e\n", steps, e);
        }
        std::printf("    norm drift (steps=4096) = %.2e   (2nd-order ratio 256->4096 ~ %.0f)\n",
                    drift, e256 / std::max(1e-30, e4096));
        ++total; pass += report("dynamic bind does not converge to the holographic bind",
                                e4096 < 1e-4 && e4096 < e256 / 50.0 && drift < 1e-9);
    }

    // [3] binding is a TRANSLATION-INVARIANT propagation (commutes with hop) -
    std::printf("\n[3] BINDING IS A TRANSLATION-INVARIANT PROPAGATION\n");
    {
        Stepper s; s.build(H, 1.0 / 4096);
        const double commErr = maxDiff(s.evolve(shift(a, 1), 4096), shift(s.evolve(a, 4096), 1));
        std::printf("    |bind(shift a) - shift(bind a)| = %.2e\n", commErr);
        ++total; pass += report("binding is not translation-invariant", commErr < 1e-9);
    }

    // [4] UNBIND = reverse propagation recovers the input --------------------
    std::printf("\n[4] UNBIND = REVERSE PROPAGATION RECOVERS THE INPUT\n");
    {
        Mat negH(kN, Vec(kN));
        for (int i = 0; i < kN; ++i) for (int j = 0; j < kN; ++j) negH[i][j] = -H[i][j];
        Stepper fwd; fwd.build(H, 1.0 / 4096);
        Stepper rev; rev.build(negH, 1.0 / 4096);
        const Vec recovered = rev.evolve(fwd.evolve(a, 4096), 4096);
        const double e = maxDiff(recovered, a);
        std::printf("    |unbind(bind(a)) - a| = %.2e\n", e);
        ++total; pass += report("reverse propagation does not invert binding", e < 1e-9);
    }

    // [5] it is a GENUINE binding: it hides its constituent ------------------
    std::printf("\n[5] BINDING HIDES ITS CONSTITUENT (VSA-consistent)\n");
    {
        const Vec bound = bindAlg(a);
        cd dot(0, 0); for (int i = 0; i < kN; ++i) dot += std::conj(bound[i]) * a[i];
        const double sim = std::abs(std::real(dot)) / std::sqrt(norm2(bound) * norm2(a));
        std::printf("    |cos(bind(a), a)| = %.4f   (1/sqrt(N) = %.4f)\n", sim, 1.0 / std::sqrt((double)kN));
        ++total; pass += report("binding did not hide its constituent", sim < 0.25);
    }

    std::printf("\n RESULT : %d / %d verified\n", pass, total);
    return pass == total ? 0 : 1;
}
