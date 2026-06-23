// graph_wave_plasticity_contract_test
// ----------------------------------------------------------------------------
// Second small step toward the living, self-closing GNN: field<->medium
// plasticity (the Hdot equation). The medium (the couplings = the "weights")
// must reshape ITSELF from the field's own activity -- no trainer, no target,
// no gradient, no labels. A local Hebbian-style rule on field correlations:
// H_ij <- (1-decay) H_ij + eta * Re(a_i conj(a_j))
// Driven by a stream of co-activation patterns, the medium self-organises into
// the input's community structure, then ROUTES the field (the closure: field
// formed the medium; the medium now governs the field).
// [1] structured input -> the medium self-organises communities
// (within-group couplings >> cross-group).
// [2] control: unstructured (random) input -> no community structure forms.
// [3] the medium stays bounded (decay = homeostasis, no blow-up).
// [4] closure: the learned medium routes a probe within its own community.
// No trainer, no weights tuned by objective. Substrate-only, deterministic.
// ----------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <utility>
#include <vector>
namespace {
using cd = std::complex<double>; using Vec = std::vector<cd>; using Mat = std::vector<std::vector<cd>>;
constexpr int kN = 8; // groups A={0..3}, B={4..7}
constexpr double kEta = 0.01, kDecay = 0.002, kTwoPi = 6.283185307179586;
// run the plasticity stream; returns the self-organised real coupling matrix H
Mat learn(int T, bool structured, std::mt19937_64& r){
 Mat H(kN, Vec(kN, cd(0,0)));
 std::uniform_real_distribution<double> ph(0, kTwoPi), u01(0,1);
 std::uniform_int_distribution<int> grp(0,1);
 for(int t=0;t<T;t++){
 std::vector<cd> a(kN, cd(0,0)); double theta=ph(r);
 if(structured){ int g=grp(r); for(int i=0;i<kN;i++) if((i<4)==(g==0)) a[i]=std::exp(cd(0,theta)); }
 else { for(int i=0;i<kN;i++) if(u01(r)<0.5) a[i]=std::exp(cd(0,ph(r))); } // random, no group structure
 for(int i=0;i<kN;i++) for(int j=i+1;j<kN;j++){
 double upd=(1.0-kDecay)*std::real(H[i][j]) + kEta*std::real(a[i]*std::conj(a[j]));
 H[i][j]=cd(upd,0); H[j][i]=cd(upd,0);
 }
 }
 return H;
}
double groupStats(const Mat& H, bool within){
 double s=0; int n=0;
 for(int i=0;i<kN;i++) for(int j=i+1;j<kN;j++){ bool same=((i<4)==(j<4)); if(same==within){ s+=std::abs(H[i][j]); n++; } }
 return n? s/n : 0;
}
// LU + Cayley stepper to test the learned medium routing a probe
struct LUC{int n=0;Mat a;std::vector<int>piv;void factor(Mat in){n=(int)in.size();a=std::move(in);piv.resize(n);for(int i=0;i<n;i++)piv[i]=i;for(int k=0;k<n;k++){double b=std::abs(a[k][k]);int r=k;for(int i=k+1;i<n;i++){double v=std::abs(a[i][k]);if(v>b){b=v;r=i;}}if(r!=k){std::swap(a[r],a[k]);std::swap(piv[r],piv[k]);}cd akk=a[k][k];for(int i=k+1;i<n;i++){cd f=a[i][k]/akk;a[i][k]=f;for(int j=k+1;j<n;j++)a[i][j]-=f*a[k][j];}}}Vec solve(const Vec&b)const{Vec x(n);for(int i=0;i<n;i++)x[i]=b[piv[i]];for(int i=0;i<n;i++){cd s=x[i];for(int j=0;j<i;j++)s-=a[i][j]*x[j];x[i]=s;}for(int i=n-1;i>=0;i--){cd s=x[i];for(int j=i+1;j<n;j++)s-=a[i][j]*x[j];x[i]=s/a[i][i];}return x;}};
Vec mv(const Mat&m,const Vec&z){Vec o(kN,cd(0,0));for(int i=0;i<kN;i++){cd s(0,0);for(int j=0;j<kN;j++)s+=m[i][j]*z[j];o[i]=s;}return o;}
Vec evolve(const Mat&H,Vec z,int steps,double dt){Mat ap(kN,Vec(kN)),am(kN,Vec(kN));const cd I(0,1);for(int i=0;i<kN;i++)for(int j=0;j<kN;j++){cd k=I*H[i][j]*(dt/2.0);cd id=(i==j)?cd(1,0):cd(0,0);ap[i][j]=id+k;am[i][j]=id-k;}LUC lu;lu.factor(ap);for(int t=0;t<steps;t++)z=lu.solve(mv(am,z));return z;}
bool report(const char*n,bool ok){std::printf(" => %s\n",ok?"PASS":"FAIL");if(!ok)std::printf(" !! %s\n",n);return ok;}
}
int main(){
 std::printf("=====================================================================\n");
 std::printf(" GRAPH-WAVE PLASTICITY CONTRACT TEST (the medium reshapes itself)\n");
 std::printf(" Field activity self-organises the couplings -- no trainer, no labels.\n");
 std::printf("=====================================================================\n");
 std::mt19937_64 rng(0x9A57C0ULL);
 int pass=0,total=0;

 Mat H = learn(4000, true, rng);
 std::printf("\n[1] STRUCTURED INPUT -> THE MEDIUM SELF-ORGANISES COMMUNITIES\n");
 {double win=groupStats(H,true), cross=groupStats(H,false);
 std::printf(" mean coupling within-group=%.4f cross-group=%.4f ratio=%.1fx\n",win,cross,cross>1e-9?win/cross:1e9);
 ++total;pass+=report("medium did not self-organise the community structure",win>5.0*std::max(1e-6,cross));}

 std::printf("\n[2] CONTROL: UNSTRUCTURED INPUT -> NO COMMUNITY STRUCTURE\n");
 {std::mt19937_64 r2(0xC047ULL);Mat Hr=learn(4000,false,r2);double win=groupStats(Hr,true),cross=groupStats(Hr,false);
 std::printf(" random input: within-group=%.4f cross-group=%.4f ratio=%.2fx (~1 = no structure)\n",win,cross,win/std::max(1e-9,cross));
 ++total;pass+=report("spurious structure formed from random input",win<2.0*cross);}

 std::printf("\n[3] THE MEDIUM STAYS BOUNDED (decay = homeostasis)\n");
 {double mx=0;for(int i=0;i<kN;i++)for(int j=0;j<kN;j++)mx=std::max(mx,std::abs(H[i][j]));
 std::printf(" max |coupling| = %.4f (bounded, did not blow up)\n",mx);
 ++total;pass+=report("couplings blew up",mx<10.0&&mx>0.5);}

 std::printf("\n[4] CLOSURE: THE LEARNED MEDIUM ROUTES A PROBE WITHIN ITS COMMUNITY\n");
 {Vec e(kN,cd(0,0));e[0]=cd(1,0);Vec z=evolve(H,e,2000,0.5/2000*8); // probe node 0 (group A)
 double eA=0,eB=0;for(int i=0;i<4;i++)eA+=std::norm(z[i]);for(int i=4;i<kN;i++)eB+=std::norm(z[i]);
 std::printf(" probe at node 0 spreads -> energy in group A=%.4f group B=%.4f\n",eA,eB);
 ++total;pass+=report("learned medium did not route the field",eA>0.9&&eB<0.05);}

 std::printf("\n RESULT : %d / %d verified\n",pass,total);return pass==total?0:1;
}
