// graph_wave_holographic_memory_contract_test
// ----------------------------------------------------------------------------
// One question: does the substrate's NATIVE algebra -- phase, interference,
// superposition -- store and recall associations WITHOUT any weights, any
// trainer, any GA? I.e. is the holographic / vector-symbolic (HRR/VSA) principle
// real on this medium, and where is its honest capacity limit?
//
// The field over D modes is a phasor vector (unit-magnitude complex per mode).
// The whole computation is the medium's own algebra:
//   * represent (a set)    = BUNDLE   = superposition (elementwise sum)
//   * bind (associate)     = BIND     = interference (elementwise complex
//                                       product == phase addition)
//   * recall an association = UNBIND   = interference with the conjugate phase
//   * similarity / cleanup  = overlap  = Hermitian dot, then nearest atom
//                                       (associative recall == amplitude
//                                        amplification onto the match)
//
// No weight is stored or tuned anywhere. "Learning" is one-shot RECORDING by
// superposition of bound pairs. The only real cost is holographic crosstalk,
// which this test measures and reports rather than hides.
//
// Substrate-only, deterministic (fixed seeds). No V26 types, no trainer, no GA.
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

constexpr int kD = 1024;          // field dimension (modes)
constexpr double kTwoPi = 6.283185307179586476925286766559;

Vec makeAtom(std::mt19937_64& rng) {
    std::uniform_real_distribution<double> uni(0.0, kTwoPi);
    Vec a(kD);
    for (int i = 0; i < kD; ++i) { const double p = uni(rng); a[i] = cd(std::cos(p), std::sin(p)); }
    return a;
}

Vec bind(const Vec& a, const Vec& b) {              // interference: phase addition
    Vec c(kD);
    for (int i = 0; i < kD; ++i) c[i] = a[i] * b[i];
    return c;
}

Vec unbind(const Vec& a, const Vec& b) {            // interference with conjugate
    Vec c(kD);
    for (int i = 0; i < kD; ++i) c[i] = a[i] * std::conj(b[i]);
    return c;
}

void accumulate(Vec& acc, const Vec& v) {           // bundle: superposition
    for (int i = 0; i < kD; ++i) acc[i] += v[i];
}

double norm(const Vec& a) {
    double s = 0.0;
    for (const auto& v : a) s += std::norm(v);
    return std::sqrt(s);
}

double cosine(const Vec& a, const Vec& b) {         // overlap (real part of <a|b>)
    cd dot(0, 0);
    for (int i = 0; i < kD; ++i) dot += std::conj(a[i]) * b[i];
    const double na = norm(a), nb = norm(b);
    return (na == 0 || nb == 0) ? 0.0 : std::real(dot) / (na * nb);
}

bool report(const char* name, const bool ok) {
    std::printf("    => %s\n", ok ? "PASS" : "FAIL");
    if (!ok) std::printf("    !! %s\n", name);
    return ok;
}

// Build P key/value pairs, bundle bound pairs into one holographic trace,
// recall every value by unbinding its key and cleaning up over the value atoms.
// Returns accuracy and the mean signal / mean top-distractor cosines.
struct Recall { double acc; double meanSignal; double meanDistractor; };

Recall holographicRecall(const int P, const std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<Vec> keys(P), vals(P);
    for (int p = 0; p < P; ++p) { keys[p] = makeAtom(rng); vals[p] = makeAtom(rng); }

    Vec mem(kD, cd(0, 0));
    for (int p = 0; p < P; ++p) accumulate(mem, bind(keys[p], vals[p]));

    int correct = 0;
    double sigSum = 0.0, distSum = 0.0;
    for (int p = 0; p < P; ++p) {
        const Vec probe = unbind(mem, keys[p]);     // val_p + crosstalk
        int best = -1; double bestSim = -2.0, secondSim = -2.0;
        for (int q = 0; q < P; ++q) {
            const double s = cosine(probe, vals[q]);
            if (s > bestSim) { secondSim = bestSim; bestSim = s; best = q; }
            else if (s > secondSim) secondSim = s;
        }
        if (best == p) ++correct;
        sigSum += cosine(probe, vals[p]);
        distSum += (best == p) ? secondSim : bestSim;  // strongest wrong match
    }
    return {static_cast<double>(correct) / P, sigSum / P, distSum / P};
}

}  // namespace

int main() {
    std::printf("=====================================================================\n");
    std::printf(" GRAPH-WAVE HOLOGRAPHIC / VSA MEMORY CONTRACT TEST  (D=%d)\n", kD);
    std::printf(" Phase + interference + superposition. No weights, no trainer, no GA.\n");
    std::printf("=====================================================================\n");

    int pass = 0, total = 0;

    // [1] RANDOM PHASOR ATOMS ARE QUASI-ORTHOGONAL (the codebook substrate) ---
    std::printf("\n[1] RANDOM PHASOR ATOMS ARE QUASI-ORTHOGONAL\n");
    {
        std::mt19937_64 rng(0xC0FFEEULL);
        std::vector<Vec> atoms(64);
        for (auto& a : atoms) a = makeAtom(rng);
        double selfErr = 0.0, sumAbs = 0.0, maxAbs = 0.0; int n = 0;
        for (int i = 0; i < 64; ++i) {
            selfErr = std::max(selfErr, std::abs(cosine(atoms[i], atoms[i]) - 1.0));
            for (int j = i + 1; j < 64; ++j) {
                const double c = std::abs(cosine(atoms[i], atoms[j]));
                sumAbs += c; maxAbs = std::max(maxAbs, c); ++n;
            }
        }
        const double meanAbs = sumAbs / n;
        std::printf("    self-similarity error=%.2e  mean|cross|=%.4f  max|cross|=%.4f  (1/sqrt(D)=%.4f)\n",
                    selfErr, meanAbs, maxAbs, 1.0 / std::sqrt((double)kD));
        ++total; pass += report("atoms not quasi-orthogonal", selfErr < 1e-12 && meanAbs < 0.06);
    }

    // [2] BUNDLING PRESERVES MEMBERSHIP (superposition == a set) --------------
    std::printf("\n[2] BUNDLING PRESERVES MEMBERSHIP\n");
    {
        std::mt19937_64 rng(0xB0BAULL);
        std::vector<Vec> members(8), outsiders(8);
        for (auto& a : members) a = makeAtom(rng);
        for (auto& a : outsiders) a = makeAtom(rng);
        Vec b(kD, cd(0, 0));
        for (const auto& m : members) accumulate(b, m);
        double minMember = 2.0, maxOutsider = -2.0;
        for (const auto& m : members) minMember = std::min(minMember, cosine(b, m));
        for (const auto& o : outsiders) maxOutsider = std::max(maxOutsider, cosine(b, o));
        std::printf("    min member sim=%.4f  max non-member sim=%.4f  (1/sqrt(8)=%.4f)\n",
                    minMember, maxOutsider, 1.0 / std::sqrt(8.0));
        ++total; pass += report("bundle does not separate members from non-members",
                                minMember > 5.0 * std::abs(maxOutsider));
    }

    // [3] BINDING IS INVERTIBLE AND HIDES ITS CONSTITUENTS --------------------
    std::printf("\n[3] BINDING IS INVERTIBLE (interference unbind recovers the factor)\n");
    {
        std::mt19937_64 rng(0xD1CEULL);
        const Vec a = makeAtom(rng), b = makeAtom(rng);
        const Vec c = bind(a, b);
        const double simCA = std::abs(cosine(c, a)), simCB = std::abs(cosine(c, b));
        const double recovered = cosine(unbind(c, b), a);
        std::printf("    sim(bind(a,b), a)=%.4f  sim(.,b)=%.4f   unbind->a sim=%.12f\n",
                    simCA, simCB, recovered);
        ++total; pass += report("binding is not an invertible, constituent-hiding operation",
                                simCA < 0.1 && simCB < 0.1 && recovered > 0.999999);
    }

    // [4] HOLOGRAPHIC KEY-VALUE MEMORY: STORE-BY-SUPERPOSITION, RECALL --------
    std::printf("\n[4] HOLOGRAPHIC KEY-VALUE MEMORY (P=8): one-shot record, recall by unbind+cleanup\n");
    {
        const Recall r = holographicRecall(8, 0x5EEDULL);
        std::printf("    accuracy=%.0f%%  mean signal sim=%.4f  mean top-distractor=%.4f\n",
                    100.0 * r.acc, r.meanSignal, r.meanDistractor);
        ++total; pass += report("holographic recall failed below capacity",
                                r.acc > 0.999 && r.meanSignal > 3.0 * std::abs(r.meanDistractor));
    }

    // [5] CAPACITY / CROSSTALK -- the honest limit, measured not hidden -------
    std::printf("\n[5] CAPACITY / CROSSTALK SCALING (honest limit)\n");
    {
        const int Ps[] = {4, 8, 16, 32, 64, 128, 256, 512};
        double accAt8 = 0.0, snrAt8 = 0.0, snrAtMax = 1e9;
        for (const int P : Ps) {
            const Recall r = holographicRecall(P, 0xA5A5ULL + P);
            const double snr = r.meanSignal / std::max(1e-9, std::abs(r.meanDistractor));
            std::printf("    P=%4d  accuracy=%5.1f%%  signal=%.4f  distractor=%.4f  SNR=%.2f\n",
                        P, 100.0 * r.acc, r.meanSignal, r.meanDistractor, snr);
            if (P == 8) { accAt8 = r.acc; snrAt8 = snr; }
            if (P == 512) snrAtMax = snr;
        }
        // The principle works below capacity AND its limit is real (crosstalk
        // grows with load): perfect recall at P=8, lower SNR at heavy load.
        ++total; pass += report("capacity behaviour not demonstrated",
                                accAt8 > 0.999 && snrAtMax < snrAt8);
    }

    std::printf("\n RESULT : %d / %d verified\n", pass, total);
    return pass == total ? 0 : 1;
}
