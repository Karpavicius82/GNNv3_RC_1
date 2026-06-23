// graph_wave_resonator_factorization_contract_test
// ----------------------------------------------------------------------------
// Gap #3 (+ #2 iterated depth): decoding a bound STRUCTURE without knowing its
// factors. Auto-associative cleanup needs a clean key; this needs none. A
// resonator network (Frady/Kent/Olshausen/Sommer) factors a bound product
// c = a (x) b into a in codebook A and b in codebook B by iterated unbind +
// phasor cleanup -- the iterated, non-unitary depth loop in action. Substrate
// only, no weights, no trainer. Honest: resonators do not always converge, so
// the success rate is measured, not assumed.
// ----------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>
namespace {
using cd = std::complex<double>; using Vec = std::vector<cd>;
constexpr int kD = 512; constexpr int kK = 12; constexpr double kTwoPi = 6.283185307179586;
Vec atom(std::mt19937_64&r){std::uniform_real_distribution<double>u(0,kTwoPi);Vec a(kD);for(int i=0;i<kD;i++){double p=u(r);a[i]=cd(std::cos(p),std::sin(p));}return a;}
Vec bind(const Vec&a,const Vec&b){Vec c(kD);for(int i=0;i<kD;i++)c[i]=a[i]*b[i];return c;}
Vec unbind(const Vec&a,const Vec&b){Vec c(kD);for(int i=0;i<kD;i++)c[i]=a[i]*std::conj(b[i]);return c;}
cd inner(const Vec&a,const Vec&x){cd s(0,0);for(int i=0;i<kD;i++)s+=std::conj(a[i])*x[i];return s;}
Vec projUnit(const Vec&v){Vec o(kD);for(int i=0;i<kD;i++){double m=std::abs(v[i]);o[i]=(m>1e-12)?v[i]/m:cd(1,0);}return o;}
// one resonator estimate: project onto a codebook (weighted superposition) then phasor-clean
Vec clean(const std::vector<Vec>&cb,const Vec&v){Vec y(kD,cd(0,0));for(auto&a:cb){cd s=inner(a,v);for(int i=0;i<kD;i++)y[i]+=s*a[i];}return projUnit(y);}
int nearest(const std::vector<Vec>&cb,const Vec&x){int b=-1;double bs=-1;for(int m=0;m<(int)cb.size();m++){double s=std::abs(inner(cb[m],x));if(s>bs){bs=s;b=m;}}return b;}
bool report(const char*n,bool ok){std::printf(" => %s\n",ok?"PASS":"FAIL");if(!ok)std::printf(" !! %s\n",n);return ok;}
}
int main(){
 std::printf("=====================================================================\n");
 std::printf(" GRAPH-WAVE RESONATOR FACTORIZATION CONTRACT TEST (D=%d, %dx%d combos)\n",kD,kK,kK);
 std::printf(" Factor c = a (x) b into unknown factors by iterated unbind+cleanup.\n");
 std::printf("=====================================================================\n");
 std::mt19937_64 rng(0x12E50ULL);
 std::vector<Vec> A(kK),B(kK);for(auto&a:A)a=atom(rng);for(auto&b:B)b=atom(rng);
 int pass=0,total=0;
 const int trials=40,iters=50;int ok=0;long itSum=0;
 for(int t=0;t<trials;t++){
 int ia=(int)(rng()%kK),ib=(int)(rng()%kK);
 Vec c=bind(A[ia],B[ib]);
 Vec ah(kD,cd(0,0)),bh(kD,cd(0,0));for(auto&a:A)for(int i=0;i<kD;i++)ah[i]+=a[i];for(auto&b:B)for(int i=0;i<kD;i++)bh[i]+=b[i];
 ah=projUnit(ah);bh=projUnit(bh);int conv=iters;int pa=-1,pb=-1;
 for(int it=0;it<iters;it++){
 ah=clean(A,unbind(c,bh));
 bh=clean(B,unbind(c,ah));
 int na=nearest(A,ah),nb=nearest(B,bh);
 if(na==pa&&nb==pb){conv=it;break;}pa=na;pb=nb;
 }
 if(nearest(A,ah)==ia&&nearest(B,bh)==ib){ok++;itSum+=conv;}
 }
 double rate=(double)ok/trials;
 std::printf("\n[1] RESONATOR FACTORS THE BOUND PRODUCT WITHOUT KNOWING THE FACTORS\n");
 std::printf(" success %d/%d = %.0f%% mean iters-to-lock (successes) = %.1f\n",ok,trials,100*rate,ok?(double)itSum/ok:0.0);
 ++total;pass+=report("resonator factorization unreliable",rate>0.75);
 // sanity: a single clean unbind with the TRUE key trivially recovers the other factor
 std::printf("\n[2] SANITY: clean unbind with the true key recovers the partner exactly\n");
 {Vec c=bind(A[3],B[7]);double sim=std::abs(inner(B[7],unbind(c,A[3])))/(std::sqrt((double)kD)*std::sqrt((double)kD));
 std::printf(" sim(unbind(c, A3), B7) = %.6f\n",sim);++total;pass+=report("clean unbind failed",sim>0.999);}
 std::printf("\n RESULT : %d / %d verified\n",pass,total);return pass==total?0:1;
}
