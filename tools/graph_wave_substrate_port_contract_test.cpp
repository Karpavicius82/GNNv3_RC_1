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

Vec normalized(Vec z) {
 const double n = std::sqrt(norm2(z));
 if (n == 0.0) {
 z.assign(z.size(), cd(0, 0));
 z[0] = cd(1, 0);
 return z;
 }
 for (auto& v : z) v /= n;
 return z;
}

double maxProbabilityDiff(const Vec& a, const Vec& b) {
 double err = 0.0;
 for (std::size_t i = 0; i < a.size(); ++i) {
 err = std::max(err, std::abs(std::norm(a[i]) - std::norm(b[i])));
 }
 return err;
}

double maxStateDiff(const Vec& a, const Vec& b) {
 double err = 0.0;
 for (std::size_t i = 0; i < a.size(); ++i) err = std::max(err, std::abs(a[i] - b[i]));
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

struct WavePort {
 Mat ap;
 Mat am;
 LUC lu;

 void bind(const Medium& medium, const double dt) {
 const auto n = static_cast<int>(medium.h.size());
 ap.assign(n, Vec(n));
 am.assign(n, Vec(n));
 const cd imag(0, 1);
 for (int i = 0; i < n; ++i) {
 for (int j = 0; j < n; ++j) {
 const cd k = imag * medium.h[i][j] * (dt / 2.0);
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
};

ObservableFrame observe(const Vec& z) {
 ObservableFrame out;
 out.p.resize(z.size());
 for (std::size_t i = 0; i < z.size(); ++i) {
 out.p[i] = std::norm(z[i]);
 out.norm += out.p[i];
 }
 return out;
}

Medium gaugeLattice() {
 constexpr int w = 5;
 constexpr int h = 5;
 constexpr double alpha = 3.0 / 8.0;
 auto id = [](const int x, const int y) { return y * w + x; };

 Medium m(w * h);
 for (int y = 0; y < h; ++y) {
 for (int x = 0; x < w; ++x) {
 if (x + 1 < w) m.edge(id(x, y), id(x + 1, y), 1.0, 0.0);
 if (y + 1 < h) m.edge(id(x, y), id(x, y + 1), 1.0, 2.0 * kPi * alpha * x);
 }
 }
 return m;
}

Vec sourcePacket(const int n) {
 Vec z(n);
 for (int i = 0; i < n; ++i) {
 const double amp = 1.0 + static_cast<double>((i * 7) % 5);
 const double phase = 0.17 * static_cast<double>(i * i + 3 * i);
 z[i] = amp * std::exp(cd(0, phase));
 }
 return normalized(std::move(z));
}

std::vector<double> localGauge(const int n) {
 std::vector<double> chi(n);
 for (int i = 0; i < n; ++i) {
 chi[i] = 0.61 * std::sin(0.23 * static_cast<double>(i)) -
 0.37 * std::cos(0.11 * static_cast<double>(i * i + 5));
 }
 return chi;
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

Medium diamond(const double flux) {
 Medium m(4);
 m.edge(0, 1, 1.0);
 m.edge(1, 3, 1.0);
 m.edge(3, 2, 1.0);
 m.edge(2, 0, 1.0, flux);
 return m;
}

double targetTransfer(const double flux) {
 const Medium m = diamond(flux);
 WavePort port;
 port.bind(m, 0.10);
 Vec z(4, cd(0, 0));
 z[0] = cd(1, 0);
 double acc = 0.0;
 constexpr int steps = 400;
 for (int t = 0; t < steps; ++t) {
 z = port.step(z);
 acc += std::norm(z[3]);
 }
 return acc / static_cast<double>(steps);
}

bool testPurePortContract() {
 std::printf("\n[1] PURE SUBSTRATE PORT CONTRACT\n");
 const Medium m = gaugeLattice();
 WavePort port;
 port.bind(m, 0.07);
 const Vec z0 = sourcePacket(static_cast<int>(m.h.size()));
 const Vec z1 = port.evolve(z0, 80);
 const auto obs = observe(z1);
 const double drift = std::abs(obs.norm - 1.0);
 const bool pass = m.hermiticityError() < 1e-14 && std::abs(norm2(z0) - 1.0) < 1e-12 &&
 drift < 1e-12;
 std::printf(" hermiticity error=%.2e\n", m.hermiticityError());
 std::printf(" input norm=%.12f output norm=%.12f drift=%.2e\n", norm2(z0), obs.norm, drift);
 std::printf(" => %s\n", pass ? "PASS" : "FAIL");
 return pass;
}

bool testGaugeEquivalentPortsSameObservables() {
 std::printf("\n[2] GAUGE-EQUIVALENT PORTS GIVE SAME OBSERVABLES\n");
 Medium m = gaugeLattice();
 const auto chi = localGauge(static_cast<int>(m.h.size()));
 Medium mg(static_cast<int>(m.h.size()));
 mg.h = gaugeTransform(m.h, chi);

 WavePort p;
 WavePort pg;
 p.bind(m, 0.07);
 pg.bind(mg, 0.07);

 Vec z = sourcePacket(static_cast<int>(m.h.size()));
 Vec zg = gaugeTransform(z, chi);
 z = p.evolve(std::move(z), 80);
 zg = pg.evolve(std::move(zg), 80);

 const Vec expected = gaugeTransform(z, chi);
 const double state_err = maxStateDiff(zg, expected);
 const double prob_err = maxProbabilityDiff(zg, z);
 const bool pass = state_err < 1e-12 && prob_err < 1e-12;
 std::printf(" max gauge-covariant state error=%.2e\n", state_err);
 std::printf(" max observable probability diff=%.2e\n", prob_err);
 std::printf(" => %s\n", pass ? "PASS" : "FAIL");
 return pass;
}

bool testGlobalPhaseRejectedAtObservablePort() {
 std::printf("\n[3] GLOBAL PHASE DOES NOT CROSS THE OBSERVABLE PORT\n");
 const Medium m = gaugeLattice();
 WavePort port;
 port.bind(m, 0.07);
 Vec z = sourcePacket(static_cast<int>(m.h.size()));
 Vec zp = z;
 for (auto& v : zp) v *= std::exp(cd(0, 1.987));
 z = port.evolve(std::move(z), 80);
 zp = port.evolve(std::move(zp), 80);

 const double prob_err = maxProbabilityDiff(z, zp);
 const bool pass = prob_err < 1e-12;
 std::printf(" max probability diff after global phase=%.2e\n", prob_err);
 std::printf(" => %s\n", pass ? "PASS" : "FAIL");
 return pass;
}

bool testHolonomyIsPhysicalInput() {
 std::printf("\n[4] HOLONOMY CHANGES PHYSICAL OBSERVABLES\n");
 const double constructive = targetTransfer(0.0);
 const double half = targetTransfer(kPi / 2.0);
 const double destructive = targetTransfer(kPi);
 const bool pass = constructive > 0.05 && destructive < 0.01 * constructive &&
 half > destructive && half < constructive;
 std::printf(" <P_target>(flux=0) = %.6f\n", constructive);
 std::printf(" <P_target>(flux=pi/2) = %.6f\n", half);
 std::printf(" <P_target>(flux=pi) = %.6e\n", destructive);
 std::printf(" => %s\n", pass ? "PASS" : "FAIL");
 return pass;
}

} // namespace

int main() {
 std::printf("=====================================================================\n");
 std::printf(" GRAPH-WAVE SUBSTRATE PORT CONTRACT TEST\n");
 std::printf(" Pure new-engine boundary: complex packet + Hermitian medium -> observables\n");
 std::printf(" no measurement, no decision, no learning.\n");
 std::printf("=====================================================================\n");

 int pass = 0;
 constexpr int total = 4;
 pass += testPurePortContract() ? 1 : 0;
 pass += testGaugeEquivalentPortsSameObservables() ? 1 : 0;
 pass += testGlobalPhaseRejectedAtObservablePort() ? 1 : 0;
 pass += testHolonomyIsPhysicalInput() ? 1 : 0;

 std::printf("\n=====================================================================\n");
 std::printf(" RESULT : %d / %d verified\n", pass, total);
 std::printf(" %s\n", pass == total
 ? "pure graph-wave substrate port is usable without external assumptions"
 : "do not build on this port; inspect failing contract");
 std::printf("=====================================================================\n");
 return pass == total ? 0 : 1;
}
