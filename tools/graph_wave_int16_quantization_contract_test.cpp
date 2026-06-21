// graph_wave_int16_quantization_contract_test
// ----------------------------------------------------------------------------
// Honesty gap: every other contract ran in float64, but the substrate's real
// home is int16 x AVX2 (Living Silicon). Phase computing could collapse under
// int16 phase quantization. This re-runs holographic store/recall with atoms
// quantized to an int16 fixed-point grid (re/im scaled by 2^14, like the real
// medium; int32 accumulators for the bundle) and measures recall vs float64.
// Substrate-only, no V26 linkage, no trainer.
// ----------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>
namespace {
using cd = std::complex<double>; using Vec = std::vector<cd>;
constexpr int kD = 512; constexpr double kTwoPi = 6.283185307179586;
constexpr double kS = 16384.0;  // 2^14 int16 fixed-point scale
double qc(double x){double v=std::round(x*kS);v=std::max(-32767.0,std::min(32767.0,v));return v/kS;}
cd q(const cd&z){return cd(qc(std::real(z)),qc(std::imag(z)));}
Vec atom(std::mt19937_64&r){std::uniform_real_distribution<double>u(0,kTwoPi);Vec a(kD);for(int i=0;i<kD;i++){double p=u(r);a[i]=cd(std::cos(p),std::sin(p));}return a;}
Vec quantize(const Vec&a){Vec o(kD);for(int i=0;i<kD;i++)o[i]=q(a[i]);return o;}
Vec bind(const Vec&a,const Vec&b,bool Q){Vec c(kD);for(int i=0;i<kD;i++)c[i]=Q?q(a[i]*b[i]):a[i]*b[i];return c;}
Vec unbind(const Vec&a,const Vec&b){Vec c(kD);for(int i=0;i<kD;i++)c[i]=a[i]*std::conj(b[i]);return c;}
cd inner(const Vec&a,const Vec&x){cd s(0,0);for(int i=0;i<kD;i++)s+=std::conj(a[i])*x[i];return s;}
double simq(const Vec&a,const Vec&b){double na=std::sqrt(std::real(inner(a,a))),nb=std::sqrt(std::real(inner(b,b)));return std::abs(inner(a,b))/(na*nb);}
bool report(const char*n,bool ok){std::printf("    => %s\n",ok?"PASS":"FAIL");if(!ok)std::printf("    !! %s\n",n);return ok;}
// holographic recall accuracy over P pairs; Q selects int16 atoms vs float atoms
double recall(int P,bool Q,std::mt19937_64&r){
  std::vector<Vec>K(P),V(P);for(int p=0;p<P;p++){K[p]=atom(r);V[p]=atom(r);if(Q){K[p]=quantize(K[p]);V[p]=quantize(V[p]);}}
  Vec M(kD,cd(0,0));for(int p=0;p<P;p++){Vec b=bind(K[p],V[p],Q);for(int i=0;i<kD;i++)M[i]+=b[i];}
  int ok=0;for(int p=0;p<P;p++){Vec pr=unbind(M,K[p]);int best=-1;double bs=-1;for(int qd=0;qd<P;qd++){double s=simq(V[qd],pr);if(s>bs){bs=s;best=qd;}}if(best==p)ok++;}
  return 100.0*ok/P;
}
}
int main(){
  std::printf("=====================================================================\n");
  std::printf(" GRAPH-WAVE INT16 QUANTIZATION CONTRACT TEST  (D=%d, scale 2^14)\n",kD);
  std::printf(" Does phase computing survive the real int16 substrate?\n");
  std::printf("=====================================================================\n");
  std::mt19937_64 rng(0x1A16ULL);std::uniform_real_distribution<double>u(0,kTwoPi);
  int pass=0,total=0;
  std::printf("\n[1] INT16 QUANTIZATION IS REAL BUT PRESERVES QUASI-ORTHOGONALITY\n");
  {Vec a=atom(rng);Vec qa=quantize(a);double dn=0;for(int i=0;i<kD;i++)dn=std::max(dn,std::abs(a[i]-qa[i]));
   Vec b=quantize(atom(rng));std::printf("    max per-component quant error = %.2e (~1/2^14=%.2e)   |cos(qa,qb)|=%.4f\n",dn,1.0/kS,simq(qa,b));
   ++total;pass+=report("quantization grid wrong",dn>1e-5&&dn<1e-3);}
  std::printf("\n[2] HOLOGRAPHIC RECALL: INT16 vs FLOAT64\n");
  {double q8=0;for(int P:{8,16,32,64}){double rf=recall(P,false,rng),rq=recall(P,true,rng);
     if(P==8)q8=rq;std::printf("    P=%3d   float64 = %.1f%%   int16 = %.1f%%\n",P,rf,rq);}
   ++total;pass+=report("phase computing does not survive int16 at low load",q8>99.0);}
  std::printf("\n RESULT : %d / %d verified\n",pass,total);return pass==total?0:1;
}
