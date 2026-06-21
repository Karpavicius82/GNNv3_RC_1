// graph_wave_flatband_memory_contract_test
// ----------------------------------------------------------------------------
// Consequence of the settling test: a stored superposition dephases on an
// evolving medium UNLESS it lives in a degenerate (flat-band) subspace. This
// contract puts holographic memory where it survives: the same key-value
// memory is embedded once in a FLAT BAND (all modes share one energy) and once
// in a SPREAD band (distinct energies), then evolved by the exact eigenmode
// dynamics. The flat-band register persists (only a global phase); the spread
// embedding dephases and recall collapses. Binding/recall is the same native
// phasor algebra, now in a register that actually survives time.
//
// Eigenmode evolution is exact here; Stepper-fidelity of flat-band persistence
// was already shown in graph_wave_settling_contract_test. No V26, no weights.
// ----------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>
namespace {
using cd = std::complex<double>; using Vec = std::vector<cd>;
constexpr int kF = 256;           // register dimension (modes); well above capacity ~F/4
constexpr int kVoc = 32; constexpr int kP = 8; constexpr double kTwoPi = 6.283185307179586;
constexpr double kEflat = 0.3;    // flat-band energy (degenerate)
std::vector<Vec> value(kVoc); std::vector<double> Espread(kF);
Vec atom(std::mt19937_64&r){std::uniform_real_distribution<double>u(0,kTwoPi);Vec a(kF);for(int i=0;i<kF;i++){double p=u(r);a[i]=cd(std::cos(p),std::sin(p));}return a;}
Vec bind(const Vec&a,const Vec&b){Vec c(kF);for(int i=0;i<kF;i++)c[i]=a[i]*b[i];return c;}
Vec unbind(const Vec&a,const Vec&b){Vec c(kF);for(int i=0;i<kF;i++)c[i]=a[i]*std::conj(b[i]);return c;}
cd inr(const Vec&a,const Vec&x){cd s(0,0);for(int i=0;i<kF;i++)s+=std::conj(a[i])*x[i];return s;}
double simAbs(const Vec&a,const Vec&b){double na=std::sqrt(std::real(inr(a,a))),nb=std::sqrt(std::real(inr(b,b)));return std::abs(inr(a,b))/(na*nb);}
int decode(const Vec&p){int b=-1;double bs=-1;for(int v=0;v<kVoc;v++){double s=simAbs(value[v],p);if(s>bs){bs=s;b=v;}}return b;}
// exact eigenmode evolution of a register: mode j rotates by its energy E[j]
Vec evolve(const Vec&M,const std::vector<double>&E,double t){Vec o(kF);for(int j=0;j<kF;j++)o[j]=M[j]*std::exp(cd(0,-E[j]*t));return o;}
bool report(const char*n,bool ok){std::printf("    => %s\n",ok?"PASS":"FAIL");if(!ok)std::printf("    !! %s\n",n);return ok;}
}
int main(){
  std::printf("=====================================================================\n");
  std::printf(" GRAPH-WAVE FLAT-BAND MEMORY CONTRACT TEST  (register F=%d)\n",kF);
  std::printf(" Holographic memory survives time only in a degenerate (flat) band.\n");
  std::printf("=====================================================================\n");
  std::mt19937_64 rng(0xF1A7ULL);std::uniform_real_distribution<double>u(-1.0,1.0);
  for(int v=0;v<kVoc;v++)value[v]=atom(rng);
  for(int j=0;j<kF;j++)Espread[j]=u(rng);                 // spread band: distinct energies
  std::vector<double>Eflat(kF,kEflat);                    // flat band: all equal
  std::vector<Vec>keys(kP);std::vector<int>vi(kP);std::uniform_int_distribution<int>vd(0,kVoc-1);
  Vec M(kF,cd(0,0));
  for(int i=0;i<kP;i++){keys[i]=atom(rng);vi[i]=vd(rng);Vec b=bind(keys[i],value[vi[i]]);for(int j=0;j<kF;j++)M[j]+=b[j];}
  auto acc=[&](const Vec&Mt){int ok=0;for(int i=0;i<kP;i++)if(decode(unbind(Mt,keys[i]))==vi[i])ok++;return 100.0*ok/kP;};

  int pass=0,total=0;
  const double ts[]={0.0,1.0,3.0,10.0,30.0};
  std::printf("\n[1] FLAT-BAND REGISTER PERSISTS (only a global phase)\n");
  {double minAcc=100,minFid=1;for(double t:ts){Vec Mt=evolve(M,Eflat,t);double a=acc(Mt),f=simAbs(Mt,M);
     std::printf("    t=%5.1f  recall=%5.1f%%  register fidelity=%.6f\n",t,a,f);minAcc=std::min(minAcc,a);minFid=std::min(minFid,f);}
   ++total;pass+=report("flat-band memory did not persist",minAcc>99.0&&minFid>0.999);}

  std::printf("\n[2] SAME MEMORY IN A SPREAD BAND DEPHASES (recall collapses)\n");
  {double accLast=100;for(double t:ts){Vec Mt=evolve(M,Espread,t);double a=acc(Mt),f=simAbs(Mt,M);
     std::printf("    t=%5.1f  recall=%5.1f%%  register fidelity=%.6f\n",t,a,f);accLast=a;}
   ++total;pass+=report("spread band did not dephase",accLast<70.0);}

  std::printf("\n[3] VERDICT: at t=30 the flat register still recalls; the spread one is gone\n");
  {double af=acc(evolve(M,Eflat,30.0)),as=acc(evolve(M,Espread,30.0));
   std::printf("    flat recall=%.1f%%   spread recall=%.1f%%   gap=%.1f points\n",af,as,af-as);
   ++total;pass+=report("flat-band advantage not demonstrated",af>99.0&&af-as>40.0);}

  std::printf("\n RESULT : %d / %d verified\n",pass,total);return pass==total?0:1;
}
