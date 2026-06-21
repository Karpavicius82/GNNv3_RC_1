// graph_wave_vsa_structure_contract_test
// ----------------------------------------------------------------------------
// One question: is the substrate's phase/interference/superposition algebra only
// a flat associative store, or can it hold STRUCTURE -- records, ordered
// sequences, and relational analogy -- still with no weights, no trainer, no GA?
//
// Same native operations as the holographic memory contract:
//   bundle = superposition, bind = interference (phase addition), unbind =
//   interference with conjugate phase, cleanup = overlap to nearest atom.
// Plus one more native primitive for order:
//   permute (cyclic shift) = a position / time operator (a graph automorphism).
//
// Demonstrated:
//   [1] role-filler record (a dictionary) queried by role
//   [2] ordered sequence via permutation -- order is encoded (a bag is not)
//   [3] holistic analogy "the dollar of Mexico is the peso", inferred from a
//       single mapping with no rules and no training.
//
// Substrate-only, deterministic. No V26 types, no trainer, no GA.
// ----------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <string_view>
#include <vector>

namespace {

using cd = std::complex<double>;
using Vec = std::vector<cd>;

constexpr int kD = 1024;
constexpr double kTwoPi = 6.283185307179586476925286766559;

struct Atom { const char* name; Vec v; };

Vec makeAtom(std::mt19937_64& rng) {
    std::uniform_real_distribution<double> uni(0.0, kTwoPi);
    Vec a(kD);
    for (int i = 0; i < kD; ++i) { const double p = uni(rng); a[i] = cd(std::cos(p), std::sin(p)); }
    return a;
}

Vec bind(const Vec& a, const Vec& b) { Vec c(kD); for (int i = 0; i < kD; ++i) c[i] = a[i] * b[i]; return c; }
Vec conjv(const Vec& a) { Vec c(kD); for (int i = 0; i < kD; ++i) c[i] = std::conj(a[i]); return c; }
Vec unbind(const Vec& a, const Vec& b) { return bind(a, conjv(b)); }

Vec shift(const Vec& a, int k) {                 // cyclic permutation = position op
    Vec c(kD);
    for (int i = 0; i < kD; ++i) { int j = ((i - k) % kD + kD) % kD; c[i] = a[j]; }
    return c;
}

void add(Vec& acc, const Vec& v) { for (int i = 0; i < kD; ++i) acc[i] += v[i]; }

double norm(const Vec& a) { double s = 0.0; for (const auto& v : a) s += std::norm(v); return std::sqrt(s); }

double cosine(const Vec& a, const Vec& b) {
    cd dot(0, 0);
    for (int i = 0; i < kD; ++i) dot += std::conj(a[i]) * b[i];
    const double na = norm(a), nb = norm(b);
    return (na == 0 || nb == 0) ? 0.0 : std::real(dot) / (na * nb);
}

// Nearest atom by overlap (cleanup memory == associative recall).
int cleanup(const Vec& probe, const std::vector<Atom>& mem, double& margin) {
    int best = -1; double b1 = -2.0, b2 = -2.0;
    for (int i = 0; i < (int)mem.size(); ++i) {
        const double s = cosine(probe, mem[i].v);
        if (s > b1) { b2 = b1; b1 = s; best = i; } else if (s > b2) b2 = s;
    }
    margin = b1 - b2;
    return best;
}

bool report(const char* name, const bool ok) {
    std::printf("    => %s\n", ok ? "PASS" : "FAIL");
    if (!ok) std::printf("    !! %s\n", name);
    return ok;
}

}  // namespace

int main() {
    std::printf("=====================================================================\n");
    std::printf(" GRAPH-WAVE VSA STRUCTURE CONTRACT TEST  (D=%d)\n", kD);
    std::printf(" Records, sequences and analogy by phase/interference. No weights.\n");
    std::printf("=====================================================================\n");

    std::mt19937_64 rng(0xC0FFEE57ULL);  // fixed seed
    int pass = 0, total = 0;

    // [1] ROLE-FILLER RECORD (a dictionary) ----------------------------------
    std::printf("\n[1] ROLE-FILLER RECORD QUERIED BY ROLE\n");
    {
        std::vector<Atom> roles = {{"name", {}}, {"age", {}}, {"city", {}}, {"job", {}}};
        std::vector<Atom> fillers = {{"alice", {}}, {"thirty", {}}, {"paris", {}}, {"pilot", {}},
                                     {"bob", {}}, {"forty", {}}, {"rome", {}}, {"medic", {}}};
        for (auto& a : roles) a.v = makeAtom(rng);
        for (auto& a : fillers) a.v = makeAtom(rng);
        Vec record(kD, cd(0, 0));
        for (int i = 0; i < 4; ++i) add(record, bind(roles[i].v, fillers[i].v));  // alice/30/paris/pilot
        int correct = 0; double minMargin = 1e9;
        for (int i = 0; i < 4; ++i) {
            double m; const int got = cleanup(unbind(record, roles[i].v), fillers, m);
            if (got == i) ++correct;
            minMargin = std::min(minMargin, m);
        }
        std::printf("    queried 4 roles -> %d/4 correct fillers  (min cleanup margin=%.4f)\n",
                    correct, minMargin);
        ++total; pass += report("record query by role failed", correct == 4 && minMargin > 0.05);
    }

    // [2] ORDERED SEQUENCE VIA PERMUTATION (order is encoded) -----------------
    std::printf("\n[2] ORDERED SEQUENCE VIA PERMUTATION\n");
    {
        std::vector<Atom> e = {{"e0", {}}, {"e1", {}}, {"e2", {}}, {"e3", {}}};
        for (auto& a : e) a.v = makeAtom(rng);
        Vec seq(kD, cd(0, 0));
        for (int k = 0; k < 4; ++k) add(seq, shift(e[k].v, k));         // e0@0,e1@1,e2@2,e3@3
        int correct = 0;
        for (int k = 0; k < 4; ++k) { double m; if (cleanup(shift(seq, -k), e, m) == k) ++correct; }

        Vec seqRev(kD, cd(0, 0));
        for (int k = 0; k < 4; ++k) add(seqRev, shift(e[k].v, 3 - k));  // reversed order
        const double seqOrderSim = cosine(seq, seqRev);

        Vec bagA(kD, cd(0, 0)), bagB(kD, cd(0, 0));                     // no position = order-blind
        for (int k = 0; k < 4; ++k) { add(bagA, e[k].v); add(bagB, e[3 - k].v); }
        const double bagSim = cosine(bagA, bagB);

        std::printf("    position recall %d/4   cos(seq, reversed)=%.4f   cos(bag, reversed bag)=%.4f\n",
                    correct, seqOrderSim, bagSim);
        ++total; pass += report("permutation did not encode order",
                                correct == 4 && seqOrderSim < 0.2 && bagSim > 0.999);
    }

    // [3] HOLISTIC ANALOGY: "the dollar of Mexico is the peso" ----------------
    std::printf("\n[3] HOLISTIC ANALOGY (inferred from one mapping, no rules, no training)\n");
    {
        Atom country{"country", makeAtom(rng)}, currency{"currency", makeAtom(rng)};
        std::vector<Atom> f = {{"usa", makeAtom(rng)}, {"mexico", makeAtom(rng)},
                               {"dollar", makeAtom(rng)}, {"peso", makeAtom(rng)}};
        auto F = [&](const char* n) { for (auto& a : f) if (std::string_view(a.name) == n) return a.v; return Vec(); };

        Vec USA(kD, cd(0, 0)); add(USA, bind(country.v, F("usa")));    add(USA, bind(currency.v, F("dollar")));
        Vec MEX(kD, cd(0, 0)); add(MEX, bind(country.v, F("mexico"))); add(MEX, bind(currency.v, F("peso")));

        const Vec map = unbind(MEX, USA);                  // usa->mexico + dollar->peso (+ crosstalk)
        double m1, m2;
        const int fwd = cleanup(bind(map, F("dollar")), f, m1);          // dollar of Mexico ?
        const int rev = cleanup(bind(conjv(map), F("peso")), f, m2);     // peso of USA ?
        const double baseline = cosine(F("dollar"), F("peso"));          // no transform: unrelated
        std::printf("    dollar -> %s (margin=%.4f)   peso -> %s (margin=%.4f)   raw cos(dollar,peso)=%.4f\n",
                    f[fwd].name, m1, f[rev].name, m2, baseline);
        const bool ok = (std::string_view(f[fwd].name) == "peso") &&
                        (std::string_view(f[rev].name) == "dollar") &&
                        m1 > 0.05 && std::abs(baseline) < 0.1;
        ++total; pass += report("analogy did not transfer dollar<->peso", ok);
    }

    std::printf("\n RESULT : %d / %d verified\n", pass, total);
    return pass == total ? 0 : 1;
}
