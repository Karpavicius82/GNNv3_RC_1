// graph_wave_resonator_cleanup_contract_test
// ----------------------------------------------------------------------------
// One question: is CLEANUP / associative recall -- the last piece of the
// holographic loop -- a physical RELAXATION on this substrate, with no argmax,
// no weights, and no trainer?
//
// Binding was shown to be unitary propagation (dynamic-bind contract). Cleanup
// is the complementary, NON-unitary half: the field settling onto the nearest
// stored attractor. This is the substrate's measurement/dissipation side -- the
// same place the absorber lives. The dynamics is a dense associative memory
// (modern-Hopfield) relaxation:
//
//   s_i = <atom_i | x>                          (overlap = interference readout)
//   x  <- normalize( sum_i s_i*|s_i|^2 * atom_i ) (sharpen + re-superpose)
//
// The only nonlinearity is the magnitude sharpening + renormalization (Born-like
// measurement). Stored atoms are fixed points; a corrupted probe relaxes onto
// the nearest one. No weight is stored or tuned. Together with dynamic binding
// this closes the holographic loop in dynamics: bind (unitary) + cleanup
// (dissipative relaxation).
//
// Substrate-only, deterministic. No V26 types, no trainer, no GA.
// ----------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>

namespace {

using cd = std::complex<double>;
using Vec = std::vector<cd>;

constexpr int kD = 512;
constexpr int kK = 24;              // stored atoms
constexpr double kTwoPi = 6.283185307179586476925286766559;

Vec makeAtom(std::mt19937_64& rng) {
    std::uniform_real_distribution<double> uni(0.0, kTwoPi);
    Vec a(kD);
    for (int i = 0; i < kD; ++i) { const double p = uni(rng); a[i] = cd(std::cos(p), std::sin(p)); }
    return a;
}

double norm(const Vec& a) { double s = 0; for (auto& v : a) s += std::norm(v); return std::sqrt(s); }

void normalizeInPlace(Vec& a) { const double n = norm(a); if (n > 0) for (auto& v : a) v /= n; }

cd inner(const Vec& a, const Vec& x) { cd s(0, 0); for (int i = 0; i < kD; ++i) s += std::conj(a[i]) * x[i]; return s; }

// phase-invariant similarity (global phase is not a signal)
double simAbs(const Vec& a, const Vec& x) {
    const double na = norm(a), nx = norm(x);
    return (na == 0 || nx == 0) ? 0.0 : std::abs(inner(a, x)) / (na * nx);
}

// one dense-associative-memory relaxation step
Vec relaxStep(const std::vector<Vec>& mem, const Vec& x) {
    Vec y(kD, cd(0, 0));
    for (int i = 0; i < kK; ++i) {
        const cd s = inner(mem[i], x);
        const cd w = s * std::norm(s);                 // sharpen by |s|^2, keep phase
        for (int k = 0; k < kD; ++k) y[k] += w * mem[i][k];
    }
    normalizeInPlace(y);
    return y;
}

int nearest(const std::vector<Vec>& mem, const Vec& x) {
    int best = -1; double bs = -1.0;
    for (int i = 0; i < kK; ++i) { const double s = simAbs(mem[i], x); if (s > bs) { bs = s; best = i; } }
    return best;
}

Vec corrupt(const Vec& clean, double alpha, std::mt19937_64& rng) {
    Vec noise = makeAtom(rng);
    Vec x(kD);
    for (int i = 0; i < kD; ++i) x[i] = clean[i] + alpha * noise[i];
    normalizeInPlace(x);
    return x;
}

bool report(const char* name, bool ok) { std::printf("    => %s\n", ok ? "PASS" : "FAIL"); if (!ok) std::printf("    !! %s\n", name); return ok; }

}  // namespace

int main() {
    std::printf("=====================================================================\n");
    std::printf(" GRAPH-WAVE RESONATOR CLEANUP CONTRACT TEST  (D=%d, K=%d atoms)\n", kD, kK);
    std::printf(" Recall = dissipative relaxation onto the nearest attractor. No weights.\n");
    std::printf("=====================================================================\n");

    std::mt19937_64 rng(0xC1EA11ULL);
    std::vector<Vec> mem(kK);
    for (auto& a : mem) a = makeAtom(rng);

    int pass = 0, total = 0;

    // [1] STORED ATOMS ARE FIXED POINTS (attractors) -------------------------
    std::printf("\n[1] STORED ATOMS ARE FIXED POINTS\n");
    {
        double minSim = 1.0;
        for (int i = 0; i < kK; ++i) minSim = std::min(minSim, simAbs(mem[i], relaxStep(mem, mem[i])));
        std::printf("    min sim(atom, relax(atom)) over %d atoms = %.6f\n", kK, minSim);
        ++total; pass += report("stored atoms are not fixed points", minSim > 0.999);
    }

    // [2] A CORRUPTED PROBE RELAXES ONTO THE NEAREST ATOM --------------------
    std::printf("\n[2] DISSIPATIVE RELAXATION CLEANS A CORRUPTED PROBE\n");
    {
        const int j = 7;
        Vec x = corrupt(mem[j], 1.2, rng);             // heavy corruption
        std::printf("    initial sim to target = %.4f\n", simAbs(mem[j], x));
        for (int t = 1; t <= 5; ++t) {
            x = relaxStep(mem, x);
            std::printf("    after relax %d: sim to target = %.6f\n", t, simAbs(mem[j], x));
        }
        const int got = nearest(mem, x);
        ++total; pass += report("relaxation did not recover the target",
                                simAbs(mem[j], x) > 0.999 && got == j);
    }

    // [3] BASIN: every heavily-corrupted probe recovers its atom -------------
    std::printf("\n[3] ASSOCIATIVE RECALL ACCURACY (all atoms, heavy noise)\n");
    {
        int correct = 0; double gainSum = 0.0;
        for (int j = 0; j < kK; ++j) {
            Vec x = corrupt(mem[j], 1.2, rng);
            const double before = simAbs(mem[j], x);
            for (int t = 0; t < 6; ++t) x = relaxStep(mem, x);
            const double after = simAbs(mem[j], x);
            gainSum += after - before;
            if (nearest(mem, x) == j) ++correct;
        }
        std::printf("    recovered %d/%d   mean cleanup gain (sim) = %.4f\n", correct, kK, gainSum / kK);
        ++total; pass += report("associative recall failed", correct == kK && gainSum / kK > 0.3);
    }

    // [4] HONEST BASIN BOUNDARY: accuracy vs corruption strength -------------
    std::printf("\n[4] HONEST BASIN BOUNDARY (accuracy vs corruption alpha)\n");
    {
        double accAt1 = 0.0;
        for (double alpha : {0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0}) {
            int correct = 0;
            for (int j = 0; j < kK; ++j) {
                Vec x = corrupt(mem[j], alpha, rng);
                for (int t = 0; t < 8; ++t) x = relaxStep(mem, x);
                if (nearest(mem, x) == j) ++correct;
            }
            const double acc = 100.0 * correct / kK;
            if (alpha == 1.0) accAt1 = acc;
            std::printf("    alpha=%.1f  init sim~%.3f  recall accuracy=%.1f%%\n",
                        alpha, 1.0 / std::sqrt(1.0 + alpha * alpha), acc);
        }
        ++total; pass += report("clean-regime recall not reliable", accAt1 > 99.0);
    }

    std::printf("\n RESULT : %d / %d verified\n", pass, total);
    return pass == total ? 0 : 1;
}
