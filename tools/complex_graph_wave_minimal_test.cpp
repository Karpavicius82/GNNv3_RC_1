#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

namespace {

using cd = std::complex<double>;
using Mat = std::vector<std::vector<cd>>;
using Vec = std::vector<cd>;

constexpr double kPi = 3.141592653589793238462643383279502884;

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

Vec addScaled(const cd a, const Vec& x, const cd b, const Vec& y) {
 Vec out(x.size());
 for (std::size_t i = 0; i < x.size(); ++i) out[i] = a * x[i] + b * y[i];
 return out;
}

double maxDiff(const Vec& a, const Vec& b) {
 double err = 0.0;
 for (std::size_t i = 0; i < a.size(); ++i) err = std::max(err, std::abs(a[i] - b[i]));
 return err;
}

Graph diamond(const double flux) {
 Graph g(4);
 constexpr int s = 0;
 constexpr int upper = 1;
 constexpr int lower = 2;
 constexpr int target = 3;
 g.addEdge(s, upper, 1.0);
 g.addEdge(s, lower, 1.0);
 g.addEdge(upper, target, 1.0);
 g.addEdge(lower, target, 1.0, flux);
 return g;
}

double transferToTarget(const double flux) {
 const Graph g = diamond(flux);
 Stepper u;
 u.build(g.h, 0.10);
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

bool testUnitaryConservation() {
 std::printf("\n[1] HERMITIAN GRAPH -> UNITARY STEP\n");
 Graph g(5);
 g.addEdge(0, 1, 0.7, 0.2);
 g.addEdge(1, 2, 1.1, -0.4);
 g.addEdge(2, 3, 0.9, 1.3);
 g.addEdge(3, 4, 0.8, -0.7);
 g.addEdge(4, 0, 1.0, 0.5);
 Stepper u;
 u.build(g.h, 0.17);

 Vec z = {cd(0.3, 0.1), cd(-0.2, 0.4), cd(0.5, -0.7), cd(0.1, 0.2), cd(-0.4, 0.1)};
 const double inv = 1.0 / std::sqrt(norm2(z));
 for (auto& v : z) v *= inv;

 double max_drift = 0.0;
 for (int t = 0; t < 1000; ++t) {
 z = u.step(z);
 max_drift = std::max(max_drift, std::abs(norm2(z) - 1.0));
 }
 const bool pass = g.hermiticityError() < 1e-14 && max_drift < 1e-12;
 std::printf(" hermiticity error=%.2e max norm drift=%.2e\n",
 g.hermiticityError(), max_drift);
 std::printf(" => %s\n", pass ? "PASS" : "FAIL");
 return pass;
}

bool testSuperposition() {
 std::printf("\n[2] SUPERPOSITION / LINEARITY\n");
 const Graph g = diamond(0.37 * kPi);
 Stepper u;
 u.build(g.h, 0.12);

 Vec x = {cd(1, 0), cd(0, 0), cd(0, 0), cd(0, 0)};
 Vec y = {cd(0, 0), cd(0, 1), cd(0, 0), cd(0, 0)};
 const cd a(0.6, -0.2);
 const cd b(-0.1, 0.7);

 Vec lhs = addScaled(a, x, b, y);
 for (int i = 0; i < 20; ++i) lhs = u.step(lhs);

 for (int i = 0; i < 20; ++i) {
 x = u.step(x);
 y = u.step(y);
 }
 const Vec rhs = addScaled(a, x, b, y);
 const double err = maxDiff(lhs, rhs);
 const bool pass = err < 1e-12;
 std::printf(" max |U(a*x+b*y) - (a*Ux+b*Uy)| = %.2e\n", err);
 std::printf(" => %s\n", pass ? "PASS" : "FAIL");
 return pass;
}

bool testPhaseControlledInterference() {
 std::printf("\n[3] PHASE-CONTROLLED INTERFERENCE\n");
 const double constructive = transferToTarget(0.0);
 const double halfway = transferToTarget(kPi / 2.0);
 const double destructive = transferToTarget(kPi);
 const double ratio = constructive / std::max(destructive, 1e-30);
 const bool pass = constructive > 0.05 && destructive < 0.01 * constructive && halfway < constructive;
 std::printf(" <P_target>(flux=0) = %.6f\n", constructive);
 std::printf(" <P_target>(flux=pi/2) = %.6f\n", halfway);
 std::printf(" <P_target>(flux=pi) = %.6e\n", destructive);
 std::printf(" constructive/destructive ratio = %.2e\n", ratio);
 std::printf(" => %s\n", pass ? "PASS" : "FAIL");
 return pass;
}

bool testGlobalPhaseIsNotObservable() {
 std::printf("\n[4] GLOBAL PHASE INVARIANCE\n");
 const Graph g = diamond(0.41 * kPi);
 Stepper u;
 u.build(g.h, 0.08);

 Vec z = {cd(1, 0), cd(0, 0), cd(0, 0), cd(0, 0)};
 Vec shifted = z;
 const cd global = std::exp(cd(0, 1.234));
 for (auto& v : shifted) v *= global;

 double max_prob_err = 0.0;
 for (int t = 0; t < 100; ++t) {
 z = u.step(z);
 shifted = u.step(shifted);
 for (std::size_t i = 0; i < z.size(); ++i) {
 max_prob_err = std::max(max_prob_err, std::abs(std::norm(z[i]) - std::norm(shifted[i])));
 }
 }
 const bool pass = max_prob_err < 1e-12;
 std::printf(" max detector-probability error after global phase shift = %.2e\n", max_prob_err);
 std::printf(" => %s\n", pass ? "PASS" : "FAIL");
 return pass;
}

} // namespace

int main() {
 std::printf("=====================================================================\n");
 std::printf(" COMPLEX GRAPH-WAVE MINIMAL SUBSTRATE TEST\n");
 std::printf(" Checks only: unitary evolution, superposition, phase interference\n");
 std::printf("=====================================================================\n");

 int pass = 0;
 constexpr int total = 4;
 pass += testUnitaryConservation() ? 1 : 0;
 pass += testSuperposition() ? 1 : 0;
 pass += testPhaseControlledInterference() ? 1 : 0;
 pass += testGlobalPhaseIsNotObservable() ? 1 : 0;

 std::printf("\n=====================================================================\n");
 std::printf(" RESULT : %d / %d verified\n", pass, total);
 std::printf(" %s\n", pass == total
 ? "complex graph-wave is a valid minimal substrate candidate"
 : "claim not verified; inspect the failing invariant");
 std::printf("=====================================================================\n");
 return pass == total ? 0 : 1;
}
