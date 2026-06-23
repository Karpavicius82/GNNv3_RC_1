// Corrected steering: a PHASED ARRAY (beamforming). All source amplitudes equal;
// ONLY the phase ramp phi(x) = g*x varies. If the wavefront's landing position
// moves with g, then PHASE alone routes energy across the field -- the weights-
// free computational primitive (a "lens"/"router" made of pure phase).
#include "../tools/graph_wave_substrate.hpp"
#include <cstdio>
#include <complex>
using namespace gw;

int main() {
 int w = 36, N = w * w; double dt = 0.5; int K = 16;
 auto id = [w](int x, int y) { return y * w + x; };
 Stepper st; st.build(plain(w).h, dt);
 std::printf("=== phased-array steering: equal amplitudes, phase ramp phi(x)=g*x ===\n");
 std::printf(" g(rad/site) landing centroid x (top band) shift from g=0\n");
 double base = 0;
 for (double g = -0.6; g <= 0.601; g += 0.3) {
 Vec p(N, cd(0, 0));
 for (int x = w / 4; x < 3 * w / 4; x++) p[id(x, 2)] = std::exp(cd(0, g * x)); // central aperture
 double nrm = std::sqrt(power(p)); for (auto& v : p) v /= nrm;
 for (int k = 0; k < K; k++) p = st.step(p);
 double sx = 0, sw = 0;
 for (int y = w - 10; y < w; y++) for (int x = 0; x < w; x++) { double I = std::norm(p[id(x, y)]); sx += I * x; sw += I; }
 double cx = sx / sw; if (g == -0.6) base = cx;
 std::printf(" %+.2f x = %5.2f %+.2f\n", g, cx, cx - (w - 1) / 2.0);
 }
 std::printf("\nIf centroid moves monotonically with g, phase alone steers the beam:\nthe field is an interference map and phase is the (weights-free) control knob.\n");
 return 0;
}
