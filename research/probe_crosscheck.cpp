// CROSS-CHECK: is the sparse Chebyshev engine (the basis of the whole scaling
// plan) actually correct, or did I fool myself again? Verify it against an
// INDEPENDENT exact ground truth: e^{-iHt}psi computed from the full
// eigendecomposition of the SAME operator. If they agree to ~1e-12 the engine is
// real. Also self-check the eigenbasis (reconstruct H, orthonormality) so the
// ground truth itself is trustworthy.
#include "../tools/graph_wave_substrate.hpp"
#include <cstdio>
#include <complex>
#include <vector>
#include <cmath>
#include <algorithm>
using namespace gw;

// --- the SAME sparse engine used in the scaling probe (copied verbatim) ---
struct Sparse { int N; std::vector<std::vector<std::pair<int, cd>>> adj; };
static std::vector<cd> matvec(const Sparse& g, const std::vector<cd>& x) {
 std::vector<cd> y(g.N, cd(0, 0));
 for (int i = 0; i < g.N; i++) { cd s(0, 0); for (auto& e : g.adj[i]) s += e.second * x[e.first]; y[i] = s; }
 return y;
}
static std::vector<cd> chebProp(const Sparse& g, std::vector<cd> psi, double t, double a, int& terms) {
 std::vector<cd> Tkm1 = psi, Tk = matvec(g, psi); for (auto& v : Tk) v /= a;
 auto J = [](int k, double x) { double h = x / 2.0, term = 1.0; for (int j = 1; j <= k; j++) term *= h / j;
 double s = term; for (int m = 1; m < 300; m++) { term *= -(h * h) / (double)(m * (m + k)); s += term; if (std::abs(term) < 1e-18) break; } return s; };
 std::vector<cd> out(g.N, cd(0, 0)); cd i0(0, -1); int K = (int)(a * t) + 25;
 for (int k = 0; k < K; k++) { double Jk = J(k, a * t); cd coef = (k == 0 ? 1.0 : 2.0) * std::pow(i0, k) * Jk;
 if (k <= 1) { const std::vector<cd>& T = (k == 0) ? psi : Tk; for (int n = 0; n < g.N; n++) out[n] += coef * T[n]; }
 else { std::vector<cd> Tp = matvec(g, Tk); for (auto& v : Tp) v /= a; for (int n = 0; n < g.N; n++) Tp[n] = 2.0 * Tp[n] - Tkm1[n]; Tkm1 = Tk; Tk = Tp; for (int n = 0; n < g.N; n++) out[n] += coef * Tk[n]; }
 if (k > a * t && std::abs(Jk) < 1e-14) { terms = k + 1; return out; } }
 terms = K; return out;
}

int main() {
 // dense Hermitian H = golden flux + small onsite disorder (lifts degeneracy so
 // the exact eigenbasis is clean). Build sparse adj from the SAME nonzeros.
 int w = 7, N = w * w;
 Graph G = golden(w, 8.0 / 13.0);
 std::mt19937 rng(99u); std::uniform_real_distribution<double> u(-0.5, 0.5);
 for (int i = 0; i < N; i++) G.onsite(i, u(rng));
 const CMat& H = G.h;
 Sparse sp; sp.N = N; sp.adj.assign(N, {});
 double a = 0;
 for (int i = 0; i < N; i++) { double row = 0; for (int j = 0; j < N; j++) { if (std::abs(H[i][j]) > 1e-15) { if (i != j) sp.adj[i].push_back({j, H[i][j]}); row += std::abs(H[i][j]); } } a = std::max(a, row); }
 // onsite (diagonal) terms: fold into adj as self-loops so matvec includes them
 for (int i = 0; i < N; i++) if (std::abs(H[i][i]) > 1e-15) sp.adj[i].push_back({i, H[i][i]});
 a *= 1.05;

 // exact ground truth: eigendecomposition (2N real embedding), take one complex eigvec per pair
 Eig e = jacobiV(realRep(H));
 std::vector<int> idx(2 * N); for (int i = 0; i < 2 * N; i++) idx[i] = i;
 std::sort(idx.begin(), idx.end(), [&](int p, int q) { return e.ev[p] < e.ev[q]; });
 std::vector<double> lam(N); std::vector<Vec> U(N);
 for (int c = 0; c < N; c++) { int col = idx[2 * c]; lam[c] = e.ev[col]; Vec v(N);
 for (int n = 0; n < N; n++) v[n] = cd(e.V[n][col], e.V[n + N][col]);
 double nn = std::sqrt(rinner(v, v).real()); for (auto& z : v) z /= nn; U[c] = v; }
 // self-check: reconstruct H and orthonormality
 double recErr = 0, orthErr = 0;
 for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) { cd s(0, 0); for (int c = 0; c < N; c++) s += lam[c] * U[c][i] * std::conj(U[c][j]); recErr = std::max(recErr, std::abs(s - H[i][j])); }
 for (int c = 0; c < N; c++) for (int d = 0; d < N; d++) { cd ov = rinner(U[c], U[d]); orthErr = std::max(orthErr, std::abs(ov - (c == d ? cd(1, 0) : cd(0, 0)))); }

 std::printf("=== CROSS-CHECK: sparse Chebyshev engine vs EXACT eigendecomposition ===\n");
 std::printf("eigenbasis self-check: reconstruct(H) err=%.2e orthonormality err=%.2e\n", recErr, orthErr);
 std::printf(" t | max|cheb - exact| cheb normDrift exact normDrift\n");
 for (double t : {0.5, 1.0, 2.0, 5.0}) {
 Vec psi(N, cd(0, 0)); psi[0] = cd(1, 0);
 int terms = 0; Vec ch = chebProp(sp, psi, t, a, terms);
 Vec ex(N, cd(0, 0));
 for (int c = 0; c < N; c++) { cd coef = std::exp(cd(0, -lam[c] * t)) * rinner(psi, U[c]); for (int n = 0; n < N; n++) ex[n] += coef * U[c][n]; }
 double diff = 0, dch = 0, dex = 0; for (int n = 0; n < N; n++) diff = std::max(diff, std::abs(ch[n] - ex[n]));
 dch = std::abs(power(ch) - 1.0); dex = std::abs(power(ex) - 1.0);
 std::printf(" %.1f | %.2e %.2e %.2e\n", t, diff, dch, dex);
 }
 std::printf("\nIf max|cheb-exact| ~ 1e-12 the scaling engine is VERIFIED against an\nindependent exact method -- not self-delusion. If not, the plan is unsafe.\n");
 return 0;
}
