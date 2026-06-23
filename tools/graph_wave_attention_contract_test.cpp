// graph_wave_attention_contract_test
// ----------------------------------------------------------------------------
// Fixes the miss in the message-passing layer: that aggregation was uniform
// (fixed graph diffusion), so it had no content-dependence and, being linear,
// its depth collapsed. The native fix is ATTENTION -- and it is native because
// the overlap (interference) <q|k> IS the relevance score, and the
// modern-Hopfield sharpening IS the softmax. Attention is also the nonlinearity
// that makes depth actually compute.
// [1] content-dependent: a node attends to the neighbour matching its query,
// not uniformly.
// [2] nonlinear in the query (softmax) -- so composed layers do not collapse.
// [3] the inverse-temperature beta interpolates uniform diffusion (beta->0)
// to winner-take-all hard attention (beta large = the cleanup limit).
// [4] selectivity enables content-addressed routing a uniform sum cannot do.
// Substrate-only, deterministic., no weights, no trainer.
// ----------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>
namespace {
using cd = std::complex<double>; using Vec = std::vector<cd>;
constexpr int kD = 1024; constexpr int kK = 6; constexpr double kTwoPi = 6.283185307179586;
Vec atom(std::mt19937_64&r){std::uniform_real_distribution<double>u(0,kTwoPi);Vec a(kD);for(int i=0;i<kD;i++){double p=u(r);a[i]=cd(std::cos(p),std::sin(p));}return a;}
double nrm(const Vec&a){double s=0;for(auto&v:a)s+=std::norm(v);return std::sqrt(s);}
cd inr(const Vec&a,const Vec&b){cd s(0,0);for(int i=0;i<kD;i++)s+=std::conj(a[i])*b[i];return s;}
double overlap(const Vec&a,const Vec&b){return std::abs(inr(a,b))/(nrm(a)*nrm(b));}
// attention weights of a query over the K neighbour keys; score = |<q,k>|^2 (interference)
std::vector<double> attn(const Vec&q,const std::vector<Vec>&K,double beta){
 std::vector<double>s(kK);for(int j=0;j<kK;j++){double o=overlap(q,K[j]);s[j]=o*o;}
 double mx=*std::max_element(s.begin(),s.end());double Z=0;std::vector<double>a(kK);
 for(int j=0;j<kK;j++){a[j]=std::exp(beta*(s[j]-mx));Z+=a[j];}for(auto&x:a)x/=Z;return a;}
Vec aggregate(const std::vector<double>&a,const std::vector<Vec>&V){Vec o(kD,cd(0,0));for(int j=0;j<kK;j++)for(int d=0;d<kD;d++)o[d]+=a[j]*V[j][d];return o;}
double entropy(const std::vector<double>&a){double h=0;for(double x:a)if(x>1e-12)h-=x*std::log(x);return h;}
bool report(const char*n,bool ok){std::printf(" => %s\n",ok?"PASS":"FAIL");if(!ok)std::printf(" !! %s\n",n);return ok;}
}
int main(){
 std::printf("=====================================================================\n");
 std::printf(" GRAPH-WAVE ATTENTION CONTRACT TEST (D=%d, %d neighbours)\n",kD,kK);
 std::printf(" Content-dependent overlap-weighted message passing (native attention).\n");
 std::printf("=====================================================================\n");
 std::mt19937_64 rng(0xA77EULL);
 std::vector<Vec>Kv(kK);for(auto&k:Kv)k=atom(rng); // neighbour keys = values
 const int target=4;Vec q=Kv[target]; // query matches neighbour 'target'
 int pass=0,total=0;

 std::printf("\n[1] CONTENT-DEPENDENT: the node attends to the MATCHING neighbour, not uniformly\n");
 {auto a=attn(q,Kv,8.0);double amax=0;int arg=-1;for(int j=0;j<kK;j++)if(a[j]>amax){amax=a[j];arg=j;}
 double others=0;for(int j=0;j<kK;j++)if(j!=target)others=std::max(others,a[j]);
 std::printf(" weights: target(%d)=%.3f max-other=%.3f uniform=%.3f\n",target,a[target],others,1.0/kK);
 ++total;pass+=report("attention is not content-dependent",arg==target&&a[target]>5.0*others);}

 std::printf("\n[2] NONLINEAR IN THE QUERY (softmax) -> composed layers do not collapse\n");
 {Vec q1=Kv[0],q2=Kv[2];Vec s(kD);for(int d=0;d<kD;d++)s[d]=q1[d]+q2[d];
 Vec o12=aggregate(attn(s,Kv,8.0),Kv);Vec o1=aggregate(attn(q1,Kv,8.0),Kv);Vec o2=aggregate(attn(q2,Kv,8.0),Kv);
 double diff=0;for(int d=0;d<kD;d++)diff=std::max(diff,std::abs(o12[d]-(o1[d]+o2[d])));
 std::printf(" max |attn(q1+q2) - (attn(q1)+attn(q2))| = %.3f (nonzero => nonlinear)\n",diff);
 ++total;pass+=report("attention is additive (linear)",diff>0.2);}

 std::printf("\n[3] BETA INTERPOLATES UNIFORM DIFFUSION <-> WINNER-TAKE-ALL\n");
 {double h0=entropy(attn(q,Kv,0.0)),h8=entropy(attn(q,Kv,12.0));
 std::printf(" weight entropy: beta=0 -> %.4f (=ln%d=%.4f, uniform) beta=12 -> %.4f (peaked)\n",h0,kK,std::log((double)kK),h8);
 ++total;pass+=report("beta does not interpolate diffusion<->attention",std::abs(h0-std::log((double)kK))<1e-6&&h8<0.1);}

 std::printf("\n[4] SELECTIVITY ENABLES CONTENT-ADDRESSED ROUTING (a uniform sum cannot)\n");
 {Vec hard=aggregate(attn(q,Kv,12.0),Kv);Vec unif=aggregate(attn(q,Kv,0.0),Kv);
 double oHard=overlap(hard,Kv[target]),oUnif=overlap(unif,Kv[target]);
 std::printf(" overlap with target value: attention=%.3f uniform=%.3f (1/sqrt(K)=%.3f)\n",oHard,oUnif,1.0/std::sqrt((double)kK));
 ++total;pass+=report("attention did not route to the matching neighbour",oHard>0.9&&oUnif<0.5);}

 std::printf("\n RESULT : %d / %d verified\n",pass,total);return pass==total?0:1;
}
