// graph_wave_living_scale_diagnostic_test
// ----------------------------------------------------------------------------
// DIAGNOSTIC. Diagnostics 1-2 found the blanket-drive autopoietic kernel loses
// selectivity under the slightest noise (sigma~0.02) while the competitive
// closure is far more robust. Open question: does the blanket-drive fragility
// depend on SCALE (number of nodes)? This probe runs the same blanket-drive
// autopoiesis at N = 8,16,32,64 with a fixed small noise and records the
// community self-organisation ratio (within/cross) and aliveness. The value is
// the data. Substrate-only, deterministic seeds.
// ----------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>
namespace {
using cd = std::complex<double>; using Vec = std::vector<cd>;
constexpr double kMu=0.5,kKappa=0.04,kEta=0.02,kDecay=0.01,kDt=0.02;
struct R{double alive,win,cross;};
R run(int N,double sigma,std::mt19937_64&rng){
 int h=N/2; std::vector<std::vector<double>> H(N,std::vector<double>(N,0.0));
 Vec psi(N,cd(0,0)); std::normal_distribution<double> g(0,1);
 const int kIn=2500,kFree=2500; double minF=1e9;
 for(int t=0;t<kIn+kFree;t++){
 bool in=t<kIn; Vec nx(N);
 for(int i=0;i<N;i++){ cd c(0,0); for(int j=0;j<N;j++) c+=H[i][j]*psi[j];
 cd d=(kMu-std::norm(psi[i]))*psi[i]+kKappa*c;
 nx[i]=psi[i]+kDt*d+sigma*std::sqrt(kDt)*cd(g(rng),g(rng));
 if(in&&i<h) nx[i]+=0.05; }
 psi=nx;
 for(int i=0;i<N;i++) for(int j=i+1;j<N;j++){ double u=H[i][j]+kDt*(kEta*std::real(psi[i]*std::conj(psi[j]))-kDecay*H[i][j]); H[i][j]=u; H[j][i]=u; }
 if(!in&&t>kIn+150){ double e=0; for(auto&v:psi) e+=std::norm(v); minF=std::min(minF,e); }
 }
 double sw=0,sc=0; int nw=0,nc=0;
 for(int i=0;i<N;i++) for(int j=i+1;j<N;j++){ bool same=((i<h)==(j<h)); if(same){sw+=std::abs(H[i][j]);nw++;} else {sc+=std::abs(H[i][j]);nc++;} }
 return { minF, nw?sw/nw:0, nc?sc/nc:0 };
}
}
int main(){
 std::printf("=====================================================================\n");
 std::printf(" DIAGNOSTIC: blanket-drive autopoiesis vs SCALE (N) under small noise\n");
 std::printf("=====================================================================\n");
 std::printf(" N | sigma | alive(minE) | within | cross | ratio | verdict\n");
 std::printf(" ----+-------+-------------+--------+--------+-------+--------\n");
 for(int N:{8,16,32,64}){
 for(double s:{0.0,0.05}){
 std::mt19937_64 rng(424242u+(unsigned)N*100+(unsigned)(s*1000));
 R r=run(N,s,rng); double ratio=r.cross>1e-9?r.win/r.cross:1e9;
 const char* v = r.alive<0.1? "DEAD" : (ratio>5.0? "organised" : "no-org");
 std::printf(" %3d | %5.2f | %11.4f | %6.3f | %6.4f | %5.1f | %s\n",N,s,r.alive,r.win,r.cross,ratio,v);
 }
 }
 std::printf("\n DIAGNOSTIC FINDINGS: compare the noisy (sigma=0.05) rows across N --\n");
 std::printf(" does blanket-drive lose selectivity at every scale, or only small N?\n");
 std::printf("\n RESULT : DIAGNOSTIC (data gathered, not a pass/fail gate)\n");
 return 0;
}
