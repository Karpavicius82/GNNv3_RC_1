// DIFFRACTION MAP: the system as an interference/diffraction computer.
// The computational object is NOT a scalar readout -- it is the FULL field map
// psi(x,y), the superposition of all paths (a discrete Feynman path integral /
// the lattice Green's function). Fluxes/phases are phase masks. Computation =
// where the interference makes energy land. Weights-free, by construction.
// Demo 1: diffraction pattern (a delta source -> Huygens wavefront).
// Demo 2: PHASE STEERING -- relative source phase moves the focal centroid.
// superposition routes energy with NO weights.
// Demo 3: HOLOGRAPHIC refocus -- record a wavefront's phase, conjugate it,
// and the field refocuses to a point. Phase (not amplitude) carries it.
#include "../tools/graph_wave_substrate.hpp"
#include <cstdio>
#include <complex>
using namespace gw;

static void heat(const char* title, const Vec& psi, int w) {
 const char* ramp = " .:-=+*#%@";
 double mx = 0; for (auto& v : psi) mx = std::max(mx, std::norm(v));
 std::printf("%s\n", title);
 for (int y = w - 1; y >= 0; y--) { // top row printed first
 std::printf(" ");
 for (int x = 0; x < w; x++) {
 double I = std::norm(psi[y * w + x]) / (mx + 1e-300);
 int lvl = (int)(I * 9.0 + 0.5); if (lvl > 9) lvl = 9;
 std::printf("%c", ramp[lvl]);
 }
 std::printf("\n");
 }
}

int main() {
 int w = 28, N = w * w; double dt = 0.5;
 auto id = [w](int x, int y) { return y * w + x; };
 Stepper st; st.build(plain(w).h, dt);
 auto evolve = [&](Vec p, int K) { for (int k = 0; k < K; k++) p = st.step(p); return p; };

 // ---- Demo 1: diffraction from a point source ----
 Vec d(N, cd(0, 0)); d[id(w / 2, w / 2)] = cd(1, 0);
 heat("=== Demo 1: diffraction pattern, point source after 9 steps (Huygens) ===", evolve(d, 9), w);

 // ---- Demo 2: phase steering ----
 std::printf("\n=== Demo 2: PHASE STEERING -- two coherent sources, relative phase delta ===\n");
 int a = w / 3, b = 2 * w / 3, ysrc = 2, K = 13;
 std::printf("delta/pi intensity centroid x in the TOP band (energy lands where phase sends it)\n");
 Vec mapShow0, mapShowP;
 for (double f = 0.0; f <= 1.501; f += 0.5) {
 Vec p(N, cd(0, 0)); p[id(a, ysrc)] = cd(1, 0); p[id(b, ysrc)] = std::exp(cd(0, f * kPi));
 double nrm = std::sqrt(power(p)); for (auto& v : p) v /= nrm;
 Vec out = evolve(p, K);
 double sx = 0, sw = 0;
 for (int y = w - 8; y < w; y++) for (int x = 0; x < w; x++) { double I = std::norm(out[id(x, y)]); sx += I * x; sw += I; }
 std::printf(" %.2f x = %.2f\n", f, sx / sw);
 if (f == 0.0) mapShow0 = out; if (std::abs(f - 1.0) < 1e-9) mapShowP = out;
 }
 heat("\n field for delta=0 (top band):", mapShow0, w);
 heat(" field for delta=pi (top band) -- fringe pattern shifted:", mapShowP, w);

 // ---- Demo 3: holographic refocus (exact, unitary) ----
 std::printf("\n=== Demo 3: HOLOGRAPHIC refocus -- phase carries the focusing, not amplitude ===\n");
 int tx = w / 2, ty = w - 5, yb = 1, Kh = 12;
 Vec src(N, cd(0, 0)); src[id(tx, ty)] = cd(1, 0);
 Vec fwd = evolve(src, Kh); // forward field; its bottom row is the hologram
 Vec conj(N, cd(0, 0)), flat(N, cd(0, 0));
 for (int x = 0; x < w; x++) { cd h = fwd[id(x, yb)]; conj[id(x, yb)] = std::conj(h); flat[id(x, yb)] = std::abs(h); }
 double n1 = std::sqrt(power(conj)); for (auto& v : conj) v /= n1;
 double n2 = std::sqrt(power(flat)); for (auto& v : flat) v /= n2;
 Vec rc = evolve(conj, Kh), rf = evolve(flat, Kh);
 double Ic = std::norm(rc[id(tx, ty)]), If = std::norm(rf[id(tx, ty)]);
 std::printf(" intensity at target T after replay: phase-conjugate = %.4f amplitude-only(flat) = %.4f\n", Ic, If);
 std::printf(" refocus gain (phase / flat) = %.1fx -> the hologram is in the PHASE.\n", Ic / (If + 1e-300));
 heat(" refocused field (phase-conjugate replay) -- a bright spot reforms at T:", rc, w);
 return 0;
}
