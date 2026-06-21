// graph_wave_end_to_end_contract_test
// ----------------------------------------------------------------------------
// Gap #0 + #4: the organs must COMPOSE on ONE common field, and the noise budget
// must be measured end-to-end (per-organ capacities do not guarantee the chain).
// This wires a tiny task on a single D-dim field using only the native algebra:
//   encode (fractional power encoding) -> bind with a value -> bundle into one
//   memory -> query with a (possibly noisy) key -> unbind -> cleanup-decode.
// Reports the COMPOUND capacity (load) and key-noise robustness. No weights,
// no trainer, no V26.
// ----------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>
namespace {
using cd = std::complex<double>; using Vec = std::vector<cd>;
constexpr int kD = 1024; constexpr int kVoc = 32; constexpr double kTwoPi = 6.283185307179586;
std::vector<double> phi(kD); std::vector<Vec> value(kVoc);
Vec enc(double x){Vec v(kD);for(int k=0;k<kD;k++)v[k]=std::exp(cd(0,x*phi[k]));return v;}
Vec bind(const Vec&a,const Vec&b){Vec c(kD);for(int k=0;k<kD;k++)c[k]=a[k]*b[k];return c;}
Vec unbind(const Vec&a,const Vec&b){Vec c(kD);for(int k=0;k<kD;k++)c[k]=a[k]*std::conj(b[k]);return c;}
cd inner(const Vec&a,const Vec&x){cd s(0,0);for(int k=0;k<kD;k++)s+=std::conj(a[k])*x[k];return s;}
int decode(const Vec&p){int b=-1;double bs=-1;for(int v=0;v<kVoc;v++){double s=std::abs(inner(value[v],p));if(s>bs){bs=s;b=v;}}return b;}
bool report(const char*n,bool ok){std::printf("    => %s\n",ok?"PASS":"FAIL");if(!ok)std::printf("    !! %s\n",n);return ok;}
// build a memory of P (key,value) pairs on ONE field; return (M, value indices, keys)
struct DB{Vec M;std::vector<int>vi;std::vector<double>kx;};
DB build(int P,std::mt19937_64&r){DB db;db.M.assign(kD,cd(0,0));std::uniform_int_distribution<int>vd(0,kVoc-1);
  for(int i=0;i<P;i++){double x=0.7*i;int v=vd(r);Vec b=bind(enc(x),value[v]);for(int k=0;k<kD;k++)db.M[k]+=b[k];db.vi.push_back(v);db.kx.push_back(x);}return db;}
}
int main(){
  std::printf("=====================================================================\n");
  std::printf(" GRAPH-WAVE END-TO-END CONTRACT TEST  (one field, D=%d)\n",kD);
  std::printf(" encode(FPE) -> bind -> bundle -> query -> unbind -> cleanup-decode.\n");
  std::printf("=====================================================================\n");
  std::mt19937_64 rng(0xE2EULL);std::uniform_real_distribution<double>u(-3.14159,3.14159);
  for(int k=0;k<kD;k++)phi[k]=u(rng);
  for(int v=0;v<kVoc;v++){Vec a(kD);for(int k=0;k<kD;k++){double p=u(rng);a[k]=cd(std::cos(p),std::sin(p));}value[v]=a;}
  int pass=0,total=0;
  std::printf("\n[1] EXACT-KEY RECALL VS LOAD (compound capacity on one field)\n");
  {double acc8=0;for(int P:{8,16,32,64,128}){DB db=build(P,rng);int ok=0;
     for(int i=0;i<P;i++){if(decode(unbind(db.M,enc(db.kx[i])))==db.vi[i])ok++;}
     double a=100.0*ok/P;if(P==8)acc8=a;std::printf("    P=%4d  recall accuracy = %.1f%%\n",P,a);}
   ++total;pass+=report("end-to-end recall fails at low load",acc8>99.0);}
  std::printf("\n[2] NOISY-KEY RECALL (FPE generalization: jittered query still decodes)\n");
  {DB db=build(24,rng);for(double dx:{0.0,0.05,0.1,0.2,0.4}){int ok=0;
     for(int i=0;i<(int)db.kx.size();i++){if(decode(unbind(db.M,enc(db.kx[i]+dx)))==db.vi[i])ok++;}
     std::printf("    key jitter dx=%.2f  recall accuracy = %.1f%%\n",dx,100.0*ok/db.kx.size());}
   // robust for small jitter:
   DB db2=build(24,rng);int ok=0;for(int i=0;i<(int)db2.kx.size();i++)if(decode(unbind(db2.M,enc(db2.kx[i]+0.1)))==db2.vi[i])ok++;
   ++total;pass+=report("noisy-key recall not robust",100.0*ok/db2.kx.size()>90.0);}
  std::printf("\n RESULT : %d / %d verified\n",pass,total);return pass==total?0:1;
}
