// graph_wave_closure_noise_diagnostic_test
// ----------------------------------------------------------------------------
// DIAGNOSTIC. Diagnostic 1 found the autopoietic kernel's self-organisation
// collapses at the slightest noise because the drive is a BLANKET gain (mu>0 on
// every node) over an unstable vacuum -> every node self-excites. Hypothesis:
// a COMPETITIVE / conditional drive (the closure's winner-take-all feedback,
// which has only decay + drive on the winning community) is robust to noise.
// This probe sweeps the same field noise on the CLOSURE loop and records whether
// the seeded community stays locked (eA vs eB), how often the winner flips, and
// whether it stays alive. Contrast against diagnostic 1 tells us the fix
// direction. The value is the DATA. Substrate-only, deterministic seeds.
// ----------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>
namespace {
using cd = std::complex<double>; using Vec = std::vector<cd>;
constexpr int kN = 8; constexpr double kGamma=0.2, kRate=1.0, kDt=0.02;
double e(const Vec&p,int lo,int hi){double s=0;for(int i=lo;i<hi;i++)s+=std::norm(p[i]);return s;}
struct R{double eA,eB,minE;int flips;};
R run(double sigma,std::mt19937_64&rng){
 Vec psi(kN,cd(0,0));std::normal_distribution<double>g(0,1);
 const int kSeed=400,kFree=4000;double minE=1e9;int flips=0,prevW=0;
 for(int t=0;t<kSeed+kFree;t++){
 bool seeding=t<kSeed;int w=(e(psi,0,4)>=e(psi,4,kN))?0:1;
 if(t>kSeed&&w!=prevW)flips++; prevW=w;
 Vec nx(kN);
 for(int i=0;i<kN;i++){
 cd d=-kGamma*psi[i]-std::norm(psi[i])*psi[i];
 nx[i]=psi[i]+kDt*d+sigma*std::sqrt(kDt)*cd(g(rng),g(rng));
 bool inA=(i<4);
 if(seeding){if(inA)nx[i]+=kDt*kRate;} // seed A
 else{if((inA?0:1)==w)nx[i]+=kDt*kRate;} // self-feedback to winner
 }
 psi=nx;
 if(t>kSeed+200)minE=std::min(minE,e(psi,0,kN));
 }
 return {e(psi,0,4),e(psi,4,kN),minE,flips};
}
}
int main(){
 std::printf("=====================================================================\n");
 std::printf(" DIAGNOSTIC: COMPETITIVE closure (winner-take-all) under field NOISE\n");
 std::printf(" Contrast vs blanket-drive autopoiesis: is competition the robust fix?\n");
 std::printf("=====================================================================\n");
 std::printf(" sigma | eA(seeded) | eB | min energy | winner-flips | verdict\n");
 std::printf(" ------+------------+--------+------------+--------------+--------\n");
 for(double s:{0.0,0.02,0.05,0.1,0.2,0.4,0.8}){
 std::mt19937_64 rng(7654321u+(unsigned)(s*1000));R r=run(s,rng);
 bool locked=r.eA>0.5&&r.eB<0.2, alive=r.minE>0.05;
 const char* v = !alive? "DEAD" : (locked? "A locked" : "lock lost");
 std::printf(" %5.2f | %10.4f | %6.4f | %10.4f | %12d | %s\n",s,r.eA,r.eB,r.minE,r.flips,v);
 }
 std::printf("\n DIAGNOSTIC FINDINGS: compare the noise level where A stays locked here\n");
 std::printf(" vs diagnostic 1 (autopoiesis broke at sigma=0.02). If competition\n");
 std::printf(" holds far higher, the fix is conditional/competitive drive, not blanket gain.\n");
 std::printf("\n RESULT : DIAGNOSTIC (data gathered, not a pass/fail gate)\n");
 return 0;
}
