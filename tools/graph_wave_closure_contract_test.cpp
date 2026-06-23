// graph_wave_closure_contract_test
// ----------------------------------------------------------------------------
// Fourth and final closure: the FEEDBACK / causal closure. The field's own
// readout (its decision: which community dominates) is fed back as its input.
// After a brief seed there is NO external input -- the loop runs on its own
// output. The field lives ONLY by feeding itself; cut the feedback wire and it
// dies. This is the causal closure that makes the system self-determining: one
// field whose next state depends only on its own readout, no external clock,
// input, or trainer.
// psi_i. = -gamma psi_i - |psi_i|^2 psi_i + (i in winning community ? rate : 0)
// winner = argmax(energy_A, energy_B) <- the readout fed back as input
// [1] after the seed the closed loop runs autonomously and stays alive.
// [2] the feedback is load-bearing: cut it and the field dies (causal closure).
// [3] self-determined identity: seed A -> it locks A; seed B -> it locks B.
// Substrate-only, deterministic., no weights, no trainer.
// ----------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>
namespace {
using cd = std::complex<double>; using Vec = std::vector<cd>;
constexpr int kN = 8; // community A={0..3}, B={4..7}
constexpr double kGamma = 0.2, kRate = 1.0, kDt = 0.02;
double e(const Vec& p,int lo,int hi){double s=0;for(int i=lo;i<hi;i++)s+=std::norm(p[i]);return s;}
// run the closed loop; seedA selects which community to seed; feedback toggles the closure
struct Out{ double eA, eB, minFree; };
Out run(bool seedA, bool feedback){
 Vec psi(kN, cd(0,0)); const int kSeed=400, kFree=4000; double minFree=1e9;
 for(int t=0;t<kSeed+kFree;t++){
 bool seeding = t<kSeed;
 int winner = (e(psi,0,4) >= e(psi,4,kN)) ? 0 : 1; // readout = the decision
 Vec nx(kN);
 for(int i=0;i<kN;i++){
 cd d = -kGamma*psi[i] - std::norm(psi[i])*psi[i];
 nx[i] = psi[i] + kDt*d;
 bool inGroupA = (i<4);
 if(seeding){ if(inGroupA==seedA) nx[i] += kDt*kRate; } // external seed
 else if(feedback){ if((inGroupA?0:1)==winner) nx[i] += kDt*kRate; } // self-feedback = own output
 }
 psi = nx;
 if(t>kSeed+200) minFree = std::min(minFree, e(psi,0,kN));
 }
 return { e(psi,0,4), e(psi,4,kN), minFree };
}
bool report(const char*n,bool ok){std::printf(" => %s\n",ok?"PASS":"FAIL");if(!ok)std::printf(" !! %s\n",n);return ok;}
}
int main(){
 std::printf("=====================================================================\n");
 std::printf(" GRAPH-WAVE CLOSURE CONTRACT TEST (output -> input, causal closure)\n");
 std::printf(" The field lives by feeding itself its own readout. No external input.\n");
 std::printf("=====================================================================\n");
 int pass=0,total=0;

 std::printf("\n[1] CLOSED LOOP RUNS AUTONOMOUSLY (no external input after the seed)\n");
 {Out o=run(true,true);std::printf(" seeded A -> free-phase min energy=%.4f final A=%.4f B=%.4f\n",o.minFree,o.eA,o.eB);
 ++total;pass+=report("closed loop did not sustain itself",o.minFree>0.3&&o.eA>0.3);}

 std::printf("\n[2] THE FEEDBACK IS LOAD-BEARING: cut it and the field DIES (causal closure)\n");
 {Out o=run(true,false);std::printf(" feedback OFF -> free-phase min energy=%.6f (-> 0 = death without the loop)\n",o.minFree);
 ++total;pass+=report("field survived without the feedback (closure not load-bearing)",o.minFree<0.02);}

 std::printf("\n[3] SELF-DETERMINED IDENTITY: seed A locks A; seed B locks B\n");
 {Out a=run(true,true),b=run(false,true);
 std::printf(" seed A -> A=%.4f B=%.4f | seed B -> A=%.4f B=%.4f\n",a.eA,a.eB,b.eA,b.eB);
 ++total;pass+=report("the closed loop has no self-determined attractors",a.eA>0.3&&a.eB<0.05&&b.eB>0.3&&b.eA<0.05);}

 std::printf("\n RESULT : %d / %d verified\n",pass,total);return pass==total?0:1;
}
