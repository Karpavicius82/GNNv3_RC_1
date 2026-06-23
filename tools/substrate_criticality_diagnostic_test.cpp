// substrate_criticality_diagnostic_test
// ----------------------------------------------------------------------------
// DIAGNOSTIC, higher-quality substrate-candidate criticality measurement. Adds
// Anderson on-site disorder as the LOCALIZED control, giving the full triad:
// EXTENDED (plain), CRITICAL (golden-flux), LOCALIZED (strong disorder). The
// eigenstate participation ratio P ~ N^tau separates them: extended tau->1,
// localized tau->0, critical fractional in between. Pure eigenstate geometry; no
// measurement, decision, learning, or. The value is the data.
//
// Built on the shared substrate core (graph_wave_substrate.hpp): the lattice
// families and the eigensolver come from the single tested source.
// ----------------------------------------------------------------------------
#include <cstdio>
#include <numeric>
#include "graph_wave_substrate.hpp"
namespace {
using namespace gw;
double meanPR(const Graph& g, int N) {
 Eig e = jacobiV(realRep(g.h)); int M = 2 * N; std::vector<int> idx(M); std::iota(idx.begin(), idx.end(), 0);
 std::sort(idx.begin(), idx.end(), [&](int p, int q) { return e.ev[p] < e.ev[q]; });
 double sum = 0; int cnt = 0;
 for (int r = M / 4; r < 3 * M / 4; r++) { int k = idx[r]; double s2 = 0, s4 = 0;
 for (int i = 0; i < N; i++) { double pi = e.V[i][k] * e.V[i][k] + e.V[i + N][k] * e.V[i + N][k]; s2 += pi; s4 += pi * pi; }
 double pr = (s2 > 0) ? (s2 * s2) / s4 : 0; sum += pr; cnt++; }
 return cnt ? sum / cnt : 0;
}
}
int main() {
 std::printf("=====================================================================\n");
 std::printf(" DIAGNOSTIC: substrate CRITICALITY triad (extended / critical / localized)\n");
 std::printf(" P ~ N^tau: extended tau->1, localized tau->0, critical fractional between.\n");
 std::printf("=====================================================================\n");
 const double a = 8.0 / 13.0, Wdis = 20.0; int ws[3] = { 8, 12, 16 }; double Ns[3] = { 64, 144, 256 };
 const char* nm[3] = { "plain (extended)", "golden-flux (cand)", "disorder W=20 (loc)" };
 std::vector<std::vector<double>> P(3);
 std::printf(" substrate | P/N 8x8 | P/N 12x12| P/N 16x16\n");
 std::printf(" ---------------------+---------+----------+----------\n");
 for (int s = 0; s < 3; s++) {
 for (int si = 0; si < 3; si++) { int w = ws[si]; Graph g = (s == 0) ? plain(w) : (s == 1) ? golden(w, a) : disorder(w, Wdis); P[s].push_back(meanPR(g, w * w)); }
 std::printf(" %-20s | %7.3f | %8.3f | %8.3f (P/N = lattice fraction)\n", nm[s], P[s][0] / 64, P[s][1] / 144, P[s][2] / 256);
 }
 auto slope = [&](const std::vector<double>& p) { std::vector<double> x = { std::log(Ns[0]), std::log(Ns[1]), std::log(Ns[2]) }, y = { std::log(p[0]), std::log(p[1]), std::log(p[2]) };
 double mx = 0, my = 0; for (int i = 0; i < 3; i++) { mx += x[i]; my += y[i]; } mx /= 3; my /= 3; double num = 0, den = 0; for (int i = 0; i < 3; i++) { num += (x[i] - mx) * (y[i] - my); den += (x[i] - mx) * (x[i] - mx); } return num / den; };
 std::printf("\n participation scaling exponent tau:\n");
 double tp = slope(P[0]), tg = slope(P[1]), td = slope(P[2]);
 std::printf(" plain (extended) tau = %.3f\n golden-flux (cand) tau = %.3f\n disorder (localized) tau = %.3f\n", tp, tg, td);
 std::printf("\n DIAGNOSTIC FINDINGS: golden tau strictly between extended and localized? %s\n", (tg < tp - 0.02 && tg > td + 0.02) ? "YES -> critical/multifractal" : "not cleanly separated");
 std::printf("\n RESULT : DIAGNOSTIC (data gathered, not a pass/fail gate)\n");
 return 0;
}
