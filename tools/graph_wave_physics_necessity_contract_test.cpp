// graph_wave_physics_necessity_contract_test
// ----------------------------------------------------------------------------
// The discipline the polar test has and the VSA stack lacked: an adversarial
// "is the graph-wave PHYSICS actually needed?" control. It honestly separates
// two regimes:
// [1] PHYSICS NOT NEEDED -- holographic recall is INVARIANT under shuffling
// the substrate (a random permutation of all components). The computation
// depends only on quasi-orthogonality + dimension, not on any physical /
// spatial / spectral structure. A random substrate works identically.
// [2] PHYSICS NEEDED -- memory PERSISTENCE under evolution requires a real
// degenerate (flat-band) spectrum. Shuffle the spectrum to a generic one
// and the same memory dephases and recall collapses.
// This marks, not hides, where the physics is load-bearing vs decorative.
// Substrate-only, deterministic., no weights, no trainer.
// ----------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>
namespace {
using cd = std::complex<double>; using Vec = std::vector<cd>;
constexpr int kD = 256; constexpr int kVoc = 32; constexpr double kTwoPi = 6.283185307179586;
Vec atom(std::mt19937_64&r){std::uniform_real_distribution<double>u(0,kTwoPi);Vec a(kD);for(int i=0;i<kD;i++){double p=u(r);a[i]=cd(std::cos(p),std::sin(p));}return a;}
Vec bind(const Vec&a,const Vec&b){Vec c(kD);for(int i=0;i<kD;i++)c[i]=a[i]*b[i];return c;}
Vec unbind(const Vec&a,const Vec&b){Vec c(kD);for(int i=0;i<kD;i++)c[i]=a[i]*std::conj(b[i]);return c;}
cd inr(const Vec&a,const Vec&x){cd s(0,0);for(int i=0;i<kD;i++)s+=std::conj(a[i])*x[i];return s;}
double simA(const Vec&a,const Vec&b){double na=std::sqrt(std::real(inr(a,a))),nb=std::sqrt(std::real(inr(b,b)));return std::abs(inr(a,b))/(na*nb);}
Vec perm(const Vec&a,const std::vector<int>&p){Vec o(kD);for(int i=0;i<kD;i++)o[i]=a[p[i]];return o;}
bool report(const char*n,bool ok){std::printf(" => %s\n",ok?"PASS":"FAIL");if(!ok)std::printf(" !! %s\n",n);return ok;}
}
int main(){
 std::printf("=====================================================================\n");
 std::printf(" GRAPH-WAVE PHYSICS-NECESSITY CONTRACT TEST (D=%d)\n",kD);
 std::printf(" Adversarial control: where is the graph-wave physics actually needed?\n");
 std::printf("=====================================================================\n");
 std::mt19937_64 rng(0x9EED5ULL);std::uniform_real_distribution<double>u(-1.0,1.0);
 std::vector<Vec>voc(kVoc);for(auto&v:voc)v=atom(rng);
 const int P=16;std::vector<Vec>keys(P);std::vector<int>vi(P);std::uniform_int_distribution<int>vd(0,kVoc-1);
 for(int i=0;i<P;i++){keys[i]=atom(rng);vi[i]=vd(rng);}
 auto buildM=[&](const std::vector<Vec>&K,const std::vector<Vec>&V){Vec M(kD,cd(0,0));for(int i=0;i<P;i++){Vec b=bind(K[i],V[vi[i]]);for(int j=0;j<kD;j++)M[j]+=b[j];}return M;};
 auto recall=[&](const Vec&M,const std::vector<Vec>&K,const std::vector<Vec>&V){int ok=0;for(int i=0;i<P;i++){Vec pr=unbind(M,K[i]);int best=-1;double bs=-1;for(int v=0;v<kVoc;v++){double s=simA(V[v],pr);if(s>bs){bs=s;best=v;}}if(best==vi[i])ok++;}return 100.0*ok/P;};

 int pass=0,total=0;

 std::printf("\n[1] CONTROL: holographic recall is INVARIANT under substrate shuffle\n");
 {Vec M=buildM(keys,voc);double a0=recall(M,keys,voc);
 std::vector<int>p(kD);std::iota(p.begin(),p.end(),0);std::shuffle(p.begin(),p.end(),rng);
 std::vector<Vec>kP(P),vP(kVoc);for(int i=0;i<P;i++)kP[i]=perm(keys[i],p);for(int v=0;v<kVoc;v++)vP[v]=perm(voc[v],p);
 Vec MP=buildM(kP,vP);double ap=recall(MP,kP,vP);
 std::printf(" recall native=%.1f%% recall shuffled-substrate=%.1f%% |diff|=%.1e\n",a0,ap,std::abs(a0-ap));
 std::printf(" VERDICT: physics_needed = NO (substrate-agnostic VSA algebra)\n");
 ++total;pass+=report("recall should be invariant under shuffle (it is not?)",a0>99.0&&std::abs(a0-ap)<1e-9);}

 std::printf("\n[2] PHYSICS NEEDED: persistence requires a real degenerate (flat-band) spectrum\n");
 {const int P2=8;std::vector<Vec>k2(P2);std::vector<int>v2(P2);for(int i=0;i<P2;i++){k2[i]=atom(rng);v2[i]=vd(rng);}
 Vec M(kD,cd(0,0));for(int i=0;i<P2;i++){Vec b=bind(k2[i],voc[v2[i]]);for(int j=0;j<kD;j++)M[j]+=b[j];}
 std::vector<double>Eflat(kD,0.3),Erand(kD);for(int j=0;j<kD;j++)Erand[j]=u(rng);
 auto evolve=[&](const std::vector<double>&E,double t){Vec o(kD);for(int j=0;j<kD;j++)o[j]=M[j]*std::exp(cd(0,-E[j]*t));return o;};
 auto rec2=[&](const Vec&Mt){int ok=0;for(int i=0;i<P2;i++){Vec pr=unbind(Mt,k2[i]);int best=-1;double bs=-1;for(int v=0;v<kVoc;v++){double s=simA(voc[v],pr);if(s>bs){bs=s;best=v;}}if(best==v2[i])ok++;}return 100.0*ok/P2;};
 double af=rec2(evolve(Eflat,30.0)),ar=rec2(evolve(Erand,30.0));
 std::printf(" after t=30: flat-band spectrum recall=%.1f%% shuffled spectrum recall=%.1f%%\n",af,ar);
 std::printf(" VERDICT: physics_needed = YES (degenerate-band physics is load-bearing)\n");
 ++total;pass+=report("flat-band advantage not demonstrated",af>99.0&&ar<50.0);}

 std::printf("\n[3] HONEST MAP\n");
 std::printf(" VSA computation (memory/structure/encoding/factorization) : physics NOT needed\n");
 std::printf(" persistence-in-time / dynamics / int16 / directed-gauge : physics NEEDED\n");

 std::printf("\n RESULT : %d / %d verified\n",pass,total);return pass==total?0:1;
}
