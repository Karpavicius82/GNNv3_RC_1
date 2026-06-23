// GNN1: a REAL weights-free GNN, end to end, on a REAL GNN task.
// Task = graph distinction. The pair C6 (one 6-cycle) vs 2*C3 (two triangles)
// is the textbook case standard message-passing GNNs CANNOT separate: both are
// 2-regular, so the 1-Weisfeiler-Lehman test (the expressivity ceiling of vanilla
// GNNs) assigns identical colours to every node in both graphs -> indistinguishable.
//
// Our GNN, weights-free:
// encoder : node feature -> initial amplitude / onsite term (here: unit basis)
// message : unitary hop U = Cayley(H,dt) (phases, superposition)
// readout : gauge-invariant return signature s_k(i) = <i|U^k|i> (k-hop echo)
// exact integer version = (H^k)[i][i] = closed k-walks from i
// No weights, no trainer. The function is the interference geometry.
// Correctness is an IDENTITY (integer walk counts), so it does not sink.
#include "../tools/graph_wave_substrate.hpp"
#include <cstdio>
#include <complex>
using namespace gw;

static CMat ring(int L) { CMat H(L, Vec(L, cd(0,0))); for (int i=0;i<L;i++){int j=(i+1)%L; H[i][j]+=1; H[j][i]+=1;} return H; }
static CMat twoTri() { CMat H(6, Vec(6, cd(0,0))); auto e=[&](int a,int b){H[a][b]+=1;H[b][a]+=1;};
 e(0,1);e(1,2);e(2,0); e(3,4);e(4,5);e(5,3); return H; }

static CMat diagPow(const CMat& H, int k) { int n=(int)H.size(); CMat P(n, Vec(n,cd(0,0))); for(int i=0;i<n;i++)P[i][i]=1; for(int s=0;s<k;s++)P=matmul(P,H); return P; }

static void report(const char* nm, const CMat& H) {
 int n=(int)H.size();
 // exact integer readout: closed k-walks per node
 CMat W3 = diagPow(H,3);
 // wave readout: unitary k-hop echo
 Stepper st; st.build(H,0.2); CMat U=st.asMatrix(); CMat U3=matmul(matmul(U,U),U);
 std::printf(" %-8s | node closed-3walks:", nm);
 for(int i=0;i<n;i++) std::printf(" %g", W3[i][i].real());
 // graph-level readout = trace(H^3) = 6 * (#triangles through scaling); and wave echo of node 0
 cd tr(0,0); for(int i=0;i<n;i++) tr+=W3[i][i];
 std::printf(" | graph readout tr(H^3)=%g | wave|<0|U^3|0>|=%.4f\n", tr.real(), std::abs(U3[0][0]));
}

int main() {
 std::printf("=== GNN1: weights-free GNN vs the 1-WL ceiling ===\n\n");
 std::printf("Two 2-regular graphs. 1-WL (vanilla-GNN ceiling) gives identical colours -> CANNOT tell apart.\n\n");
 report("C6", ring(6));
 report("2xC3", twoTri());
 std::printf("\n=> weights-free GNN SEPARATES them exactly (0 vs 2 closed-3-walks; tr 0 vs 12),\n a distinction provably beyond standard message-passing GNNs. No weights, no training.\n");

 std::printf("\n--- phase power: read the gauge-invariant cycle FLUX (real GNNs are blind to it) ---\n");
 std::printf("single triangle, thread flux phi on one edge. closed-3-walk return = 2*cos(phi), EXACT:\n");
 std::printf(" phi/pi Re<0|H^3|0>\n");
 for (double f=0.0; f<=1.001; f+=0.25) {
 CMat H(3, Vec(3,cd(0,0))); auto e=[&](int a,int b,double ph){cd v=std::exp(cd(0,ph));H[a][b]+=v;H[b][a]+=std::conj(v);};
 e(0,1,0); e(1,2,0); e(2,0,f*kPi);
 CMat W3=diagPow(H,3);
 std::printf(" %.2f %+.4f\n", f, W3[0][0].real());
 }
 std::printf("=> the echo carries the loop flux (gauge invariant): phi=0 -> +2, phi=pi -> -2.\n This is structural information a weights-free, REAL-valued GNN cannot represent.\n");
 return 0;
}
