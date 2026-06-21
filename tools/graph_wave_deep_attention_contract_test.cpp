// graph_wave_deep_attention_contract_test
// ----------------------------------------------------------------------------
// The depth advantage that proves attention does not collapse: TWO content-
// addressed attention layers perform a 2-hop relational lookup (pointer chasing)
// that a single layer -- and any linear/uniform diffusion -- provably cannot.
// Memory holds key->value links where some values are themselves keys
// (kA -> kB, kB -> vC). A query that matches kA, run through two attention
// layers, lands on vC: "the value of the key that is the value of my query."
// This is the native, weight-free realisation of deep, expressive GNN /
// transformer computation. Substrate-only, deterministic. No V26, no trainer.
// ----------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>
namespace {
using cd = std::complex<double>; using Vec = std::vector<cd>;
constexpr int kD = 1024; constexpr int kK = 4; constexpr double kTwoPi = 6.283185307179586;
Vec atom(std::mt19937_64&r){std::uniform_real_distribution<double>u(0,kTwoPi);Vec a(kD);for(int i=0;i<kD;i++){double p=u(r);a[i]=cd(std::cos(p),std::sin(p));}return a;}
double nrm(const Vec&a){double s=0;for(auto&v:a)s+=std::norm(v);return std::sqrt(s);}
cd inr(const Vec&a,const Vec&b){cd s(0,0);for(int i=0;i<kD;i++)s+=std::conj(a[i])*b[i];return s;}
double overlap(const Vec&a,const Vec&b){return std::abs(inr(a,b))/(nrm(a)*nrm(b));}
// content-addressed attention: weight each key by |<q,key>|^2, output sum of weighted values
Vec attnLayer(const Vec&q,const std::vector<Vec>&K,const std::vector<Vec>&V,double beta){
  std::vector<double>s(kK);for(int j=0;j<kK;j++){double o=overlap(q,K[j]);s[j]=o*o;}
  double mx=*std::max_element(s.begin(),s.end()),Z=0;std::vector<double>a(kK);
  for(int j=0;j<kK;j++){a[j]=std::exp(beta*(s[j]-mx));Z+=a[j];}for(auto&x:a)x/=Z;
  Vec o(kD,cd(0,0));for(int j=0;j<kK;j++)for(int d=0;d<kD;d++)o[d]+=a[j]*V[j][d];return o;}
bool report(const char*n,bool ok){std::printf("    => %s\n",ok?"PASS":"FAIL");if(!ok)std::printf("    !! %s\n",n);return ok;}
}
int main(){
  std::printf("=====================================================================\n");
  std::printf(" GRAPH-WAVE DEEP ATTENTION CONTRACT TEST  (D=%d)\n",kD);
  std::printf(" Two attention layers do a 2-hop lookup a single/linear layer cannot.\n");
  std::printf("=====================================================================\n");
  std::mt19937_64 rng(0xDEE9ULL);
  Vec kA=atom(rng),kB=atom(rng),vC=atom(rng),kd1=atom(rng),kd2=atom(rng),dv1=atom(rng),dv2=atom(rng);
  std::vector<Vec>K={kA,kB,kd1,kd2};
  std::vector<Vec>V={kB,vC,dv1,dv2};     // links: kA->kB, kB->vC  (chain)
  const double beta=10.0;
  int pass=0,total=0;

  std::printf("\n[1] ONE LAYER = 1-HOP LOOKUP: query kA retrieves its value kB\n");
  {Vec o1=attnLayer(kA,K,V,beta);std::printf("    overlap(layer1(kA), kB) = %.3f\n",overlap(o1,kB));
   ++total;pass+=report("single-layer 1-hop lookup failed",overlap(o1,kB)>0.9);}

  std::printf("\n[2] TWO LAYERS = 2-HOP LOOKUP (pointer chasing): kA -> kB -> vC\n");
  {Vec o2=attnLayer(attnLayer(kA,K,V,beta),K,V,beta);std::printf("    overlap(layer2(layer1(kA)), vC) = %.3f\n",overlap(o2,vC));
   ++total;pass+=report("two-layer 2-hop lookup failed",overlap(o2,vC)>0.9);}

  std::printf("\n[3] A SINGLE LAYER CANNOT REACH THE 2-HOP TARGET vC\n");
  {Vec o1=attnLayer(kA,K,V,beta);std::printf("    overlap(layer1(kA), vC) = %.3f (must be low)\n",overlap(o1,vC));
   ++total;pass+=report("single layer should not reach the 2-hop target",overlap(o1,vC)<0.2);}

  std::printf("\n[4] LINEAR/UNIFORM AGGREGATION CANNOT ROUTE AT ALL\n");
  {Vec u1=attnLayer(kA,K,V,0.0);Vec u2=attnLayer(u1,K,V,0.0);
   std::printf("    uniform(uniform(kA)) overlap with vC = %.3f (1/sqrt(K)=%.3f)\n",overlap(u2,vC),1.0/std::sqrt((double)kK));
   ++total;pass+=report("uniform aggregation unexpectedly routed",overlap(u2,vC)<0.6);}

  std::printf("\n RESULT : %d / %d verified  (depth advantage: 2 layers reach vC, 1 layer / linear cannot)\n",pass,total);
  return pass==total?0:1;
}
