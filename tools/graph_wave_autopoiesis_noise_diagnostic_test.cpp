// graph_wave_autopoiesis_noise_diagnostic_test
// ----------------------------------------------------------------------------
// DIAGNOSTIC, not a gate. The autopoiesis contract passed in idealized, noiseless
// conditions (cross-coupling exactly 0). We do NOT yet know where / what / why
// the living closure breaks. This probe sweeps additive field NOISE and reports,
// for each level: does the field stay alive, does the medium still self-organise
// (within/cross coupling ratio), and is the learned pattern preserved (energy in
// the driven community A vs the other B). The deliverable is the DATA -- the
// breaking points -- not a verdict. Substrate-only, deterministic seeds.
// ----------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>
namespace {
using cd = std::complex<double>; using Vec = std::vector<cd>; using Mat = std::vector<std::vector<double>>;
constexpr int kN = 8;
constexpr double kMu=0.5, kKappa=0.04, kEta=0.02, kDecay=0.01, kDt=0.02;
double energy(const Vec&p,int lo,int hi){double s=0;for(int i=lo;i<hi;i++)s+=std::norm(p[i]);return s;}
double wc(const Mat&H,bool within){double s=0;int n=0;for(int i=0;i<kN;i++)for(int j=i+1;j<kN;j++){bool same=((i<4)==(j<4));if(same==within){s+=std::abs(H[i][j]);n++;}}return n?s/n:0;}
struct R{double alive,win,cross,eA,eB;};
R run(double sigma,std::mt19937_64&rng){
 Vec psi(kN,cd(0,0));Mat H(kN,std::vector<double>(kN,0.0));
 std::normal_distribution<double> g(0,1);const int kIn=3000,kFree=4000;double minFree=1e9;
 for(int t=0;t<kIn+kFree;t++){
 bool in=t<kIn;Vec nx(kN);
 for(int i=0;i<kN;i++){
 cd c(0,0);for(int j=0;j<kN;j++)c+=H[i][j]*psi[j];
 cd d=(kMu-std::norm(psi[i]))*psi[i]+kKappa*c;
 nx[i]=psi[i]+kDt*d+sigma*std::sqrt(kDt)*cd(g(rng),g(rng)); // additive field noise
 if(in&&i<4)nx[i]+=0.05;
 }
 psi=nx;
 for(int i=0;i<kN;i++)for(int j=i+1;j<kN;j++){double u=H[i][j]+kDt*(kEta*std::real(psi[i]*std::conj(psi[j]))-kDecay*H[i][j]);H[i][j]=u;H[j][i]=u;}
 if(!in&&t>kIn+200)minFree=std::min(minFree,energy(psi,0,kN));
 }
 return {minFree,wc(H,true),wc(H,false),energy(psi,0,4),energy(psi,4,kN)};
}
}
int main(){
 std::printf("=====================================================================\n");
 std::printf(" DIAGNOSTIC: autopoietic closure under field NOISE (where does it break?)\n");
 std::printf("=====================================================================\n");
 std::printf(" sigma | alive(minE) | within | cross | ratio | eA | eB | verdict\n");
 std::printf(" ------+-------------+--------+--------+---------+---------+---------+--------\n");
 double breakOrg=-1,breakPat=-1,breakDie=-1;
 for(double s:{0.0,0.02,0.05,0.1,0.2,0.35,0.6,1.0}){
 std::mt19937_64 rng(1234567u+(unsigned)(s*1000));
 R r=run(s,rng);double ratio=r.cross>1e-9?r.win/r.cross:1e9;
 const char* v="ok";
 bool alive=r.alive>0.1, org=ratio>5.0, pat=(r.eA>0.5&&r.eB<0.2);
 if(!alive){v="DEAD";if(breakDie<0)breakDie=s;}
 else if(!org){v="no-org";if(breakOrg<0)breakOrg=s;}
 else if(!pat){v="pattern-lost";if(breakPat<0)breakPat=s;}
 std::printf(" %5.2f | %11.4f | %6.3f | %6.4f | %7.1f | %7.4f | %7.4f | %s\n",
 s,r.alive,r.win,r.cross,ratio,r.eA,r.eB,v);
 }
 std::printf("\n DIAGNOSTIC FINDINGS:\n");
 std::printf(" self-organisation breaks (ratio<5) at sigma = %s\n", breakOrg<0?"not in range":std::to_string(breakOrg).c_str());
 std::printf(" learned pattern lost at sigma = %s\n", breakPat<0?"not in range":std::to_string(breakPat).c_str());
 std::printf(" field dies at sigma = %s\n", breakDie<0?"not in range":std::to_string(breakDie).c_str());
 std::printf("\n RESULT : DIAGNOSTIC (data gathered, not a pass/fail gate)\n");
 return 0; // diagnostic: always succeeds; the value is the printed map
}
