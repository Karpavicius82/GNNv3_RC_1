// graph_wave_bistable_drive_diagnostic_test
// ----------------------------------------------------------------------------
// ANTI-SELF-DECEPTION DIAGNOSTIC. Diagnostics 1-3 concluded the fix is a
// competitive / stable-vacuum drive. But that conclusion was EXTRAPOLATED from
// the closure loop, which has NO plasticity. The real proposed fix must be
// tested WITH the medium learning -- a competitive-drive autopoiesis could still
// fail if noise-activated nodes get coupled by plasticity before competition
// suppresses them. So here we build the candidate fix directly:
//   bistable drive: psi. = (-gamma + a|psi|^2 - b|psi|^4) psi + kappa H psi + in
//   (a STABLE vacuum: a node below threshold decays; input latches it high)
//   plus the same plasticity, and we sweep noise. If self-organisation holds far
//   above the blanket-drive break point (sigma=0.02), the conclusion survives.
//   If it breaks at the same noise, we were deceiving ourselves. The value is
//   the data. Substrate-only, deterministic seeds.
// ----------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>
namespace {
using cd = std::complex<double>; using Vec = std::vector<cd>;
constexpr int kN=8;
constexpr double kGamma=0.1,kA=1.0,kB=0.5,kKappa=0.04,kEta=0.02,kDecay=0.01,kDt=0.02;
double energy(const Vec&p,int lo,int hi){double s=0;for(int i=lo;i<hi;i++)s+=std::norm(p[i]);return s;}
double wc(const std::vector<std::vector<double>>&H,bool within){double s=0;int n=0;for(int i=0;i<kN;i++)for(int j=i+1;j<kN;j++){bool same=((i<4)==(j<4));if(same==within){s+=std::abs(H[i][j]);n++;}}return n?s/n:0;}
struct R{double alive,win,cross,eA,eB;};
R run(double sigma,std::mt19937_64&rng){
  Vec psi(kN,cd(0,0));std::vector<std::vector<double>>H(kN,std::vector<double>(kN,0.0));
  std::normal_distribution<double>g(0,1);const int kIn=3000,kFree=4000;double minF=1e9;
  for(int t=0;t<kIn+kFree;t++){
    bool in=t<kIn;Vec nx(kN);
    for(int i=0;i<kN;i++){
      double r2=std::norm(psi[i]); cd c(0,0);for(int j=0;j<kN;j++)c+=H[i][j]*psi[j];
      cd d=(-kGamma + kA*r2 - kB*r2*r2)*psi[i] + kKappa*c;     // BISTABLE drive (stable vacuum)
      nx[i]=psi[i]+kDt*d+sigma*std::sqrt(kDt)*cd(g(rng),g(rng));
      if(in&&i<4)nx[i]+=0.10;                                  // input latches group A above threshold
    }
    psi=nx;
    for(int i=0;i<kN;i++)for(int j=i+1;j<kN;j++){double u=H[i][j]+kDt*(kEta*std::real(psi[i]*std::conj(psi[j]))-kDecay*H[i][j]);H[i][j]=u;H[j][i]=u;}
    if(!in&&t>kIn+200)minF=std::min(minF,energy(psi,0,kN));
  }
  return {minF,wc(H,true),wc(H,false),energy(psi,0,4),energy(psi,4,kN)};
}
}
int main(){
  std::printf("=====================================================================\n");
  std::printf(" ANTI-SELF-DECEPTION: bistable (stable-vacuum) drive + PLASTICITY vs noise\n");
  std::printf(" Does the proposed fix actually hold WITH the medium learning?\n");
  std::printf("=====================================================================\n");
  std::printf("  sigma | alive | within | cross  | ratio | eA     | eB     | verdict\n");
  std::printf("  ------+-------+--------+--------+-------+--------+--------+--------\n");
  double breakOrg=-1;
  for(double s:{0.0,0.02,0.05,0.1,0.2,0.35,0.6}){
    std::mt19937_64 rng(987654u+(unsigned)(s*1000));R r=run(s,rng);double ratio=r.cross>1e-9?r.win/r.cross:1e9;
    bool org=ratio>5.0; if(!org&&breakOrg<0&&s>0)breakOrg=s;
    const char* v=r.alive<0.05?"DEAD":(org?"organised":"no-org");
    std::printf("  %5.2f | %5.2f | %6.3f | %6.4f | %5.1f | %6.3f | %6.3f | %s\n",s,r.alive,r.win,r.cross,ratio,r.eA,r.eB,v);
  }
  std::printf("\n VERDICT vs blanket-drive (broke at sigma=0.02):\n");
  std::printf("   bistable+plasticity self-organisation breaks at sigma = %s\n", breakOrg<0?"NOT in range (held throughout!)":std::to_string(breakOrg).c_str());
  std::printf("   -> if >> 0.02, the fix is real; if ~0.02, we were deceiving ourselves.\n");
  std::printf("\n RESULT : DIAGNOSTIC (data gathered, not a pass/fail gate)\n");
  return 0;
}
