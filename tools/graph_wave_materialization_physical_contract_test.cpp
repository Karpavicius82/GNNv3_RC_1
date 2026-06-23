// graph_wave_materialization_physical_contract_test
// ----------------------------------------------------------------------------
// Closes the algebra->physics gap for the graph-layer materialization. The
// kernel W (orthonormal node-update rows + residual modes) is realised as a REAL
// Hermitian bipartite medium (input rails <-> output modes, coupling = W),
// evolved a quarter period by the Cayley stepper. The physical input->mode
// transfer then equals -i*W (same technique as the degree-2 / node-update
// contracts). So the materialisation is genuine unitary wave propagation, not
// just an algebraic matrix.
//
// Same 4-node graph (edges 0-1,0-2,0-3,1-2). 12 rails: 4 self + 8 oriented edges.
//
// [1] the bipartite medium is Hermitian
// [2] the physical transfer equals -i*W (the algebraic kernel) and the
// evolution is unitary (input norm fully transfers to the modes)
// [3] a SUPERPOSITION of a node's incoming messages materialises at that node
// by real propagation: aligned constructive, anti destructive, and local
// [4] residual modes conserve the superposition
// No decision, no learning, no trainer.
// ----------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <utility>
#include <vector>
namespace {
using cd = std::complex<double>; using Vec = std::vector<cd>; using Mat = std::vector<std::vector<cd>>;
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr int kInputs = 12, kModes = 12, kNodes = kInputs + kModes;
constexpr int kSteps = 512, kIn0 = 0, kOut0 = kInputs;
const double kSelfGain = 1.0, kInGain = 0.8;
double dt(){ return 2.0*std::tan(kPi/(4.0*kSteps)); }
double effectiveAngle(){ return kSteps*2.0*std::atan(dt()/2.0); }
cd transferScale(){ return cd(0,-std::sin(effectiveAngle())); } // -i at quarter period
cd rinner(const Vec&a,const Vec&b){cd s(0,0);for(size_t i=0;i<a.size();++i)s+=a[i]*std::conj(b[i]);return s;}
bool addOrthRow(Mat&rows,Vec c){for(const auto&b:rows){cd co=rinner(c,b);for(size_t i=0;i<c.size();++i)c[i]-=co*b[i];}double n=std::sqrt(std::max(0.0,rinner(c,c).real()));if(n<1e-10)return false;for(auto&v:c)v/=n;rows.push_back(c);return true;}
struct NodeRow{int self;std::vector<std::pair<int,double>>ins;};
Vec buildRow(const NodeRow&nr){double n=std::sqrt(kSelfGain*kSelfGain+nr.ins.size()*kInGain*kInGain);Vec row(kInputs,cd(0,0));row[nr.self]=cd(kSelfGain/n,0);for(auto&p:nr.ins)row[p.first]=(kInGain/n)*std::exp(cd(0,p.second));return row;}
double power(const Vec&z){double s=0;for(auto&v:z)s+=std::norm(v);return s;}
void norml(Vec&z){double n=std::sqrt(power(z));for(auto&v:z)v/=n;}
struct LUC{int n=0;Mat a;std::vector<int>piv;void factor(Mat in){n=(int)in.size();a=std::move(in);piv.resize(n);for(int i=0;i<n;i++)piv[i]=i;for(int k=0;k<n;k++){double b=std::abs(a[k][k]);int r=k;for(int i=k+1;i<n;i++){double v=std::abs(a[i][k]);if(v>b){b=v;r=i;}}if(r!=k){std::swap(a[r],a[k]);std::swap(piv[r],piv[k]);}cd akk=a[k][k];for(int i=k+1;i<n;i++){cd f=a[i][k]/akk;a[i][k]=f;for(int j=k+1;j<n;j++)a[i][j]-=f*a[k][j];}}}Vec solve(const Vec&b)const{Vec x(n);for(int i=0;i<n;i++)x[i]=b[piv[i]];for(int i=0;i<n;i++){cd s=x[i];for(int j=0;j<i;j++)s-=a[i][j]*x[j];x[i]=s;}for(int i=n-1;i>=0;i--){cd s=x[i];for(int j=i+1;j<n;j++)s-=a[i][j]*x[j];x[i]=s/a[i][i];}return x;}};
struct Stepper{Mat am;LUC lu;void build(const Mat&h){Mat ap(kNodes,Vec(kNodes)),m(kNodes,Vec(kNodes));const cd I(0,1);double sdt=dt();for(int i=0;i<kNodes;i++)for(int j=0;j<kNodes;j++){cd k=I*h[i][j]*(sdt/2.0);cd id=(i==j)?cd(1,0):cd(0,0);ap[i][j]=id+k;m[i][j]=id-k;}am=m;lu.factor(ap);}Vec mv(const Mat&M,const Vec&z)const{Vec o(kNodes,cd(0,0));for(int i=0;i<kNodes;i++){cd s(0,0);for(int j=0;j<kNodes;j++)s+=M[i][j]*z[j];o[i]=s;}return o;}Vec evolve(Vec z)const{for(int t=0;t<kSteps;t++)z=lu.solve(mv(am,z));return z;}};
Mat buildMedium(const Mat&W){Mat h(kNodes,Vec(kNodes,cd(0,0)));for(int mode=0;mode<kModes;mode++)for(int in=0;in<kInputs;in++){h[kOut0+mode][kIn0+in]+=W[mode][in];h[kIn0+in][kOut0+mode]+=std::conj(W[mode][in]);}return h;}
double hermErr(const Mat&h){double e=0;for(int i=0;i<kNodes;i++)for(int j=0;j<kNodes;j++)e=std::max(e,std::abs(h[i][j]-std::conj(h[j][i])));return e;}
Vec setInput(const Vec&in){Vec z(kNodes,cd(0,0));for(int i=0;i<kInputs;i++)z[kIn0+i]=in[i];return z;}
Vec outModes(const Vec&z){Vec o(kModes,cd(0,0));for(int m=0;m<kModes;m++)o[m]=z[kOut0+m];return o;}
bool report(const char*n,bool ok){std::printf(" => %s\n",ok?"PASS":"FAIL");if(!ok)std::printf(" !! %s\n",n);return ok;}
}
int main(){
 std::printf("=====================================================================\n");
 std::printf(" GRAPH-WAVE MATERIALIZATION PHYSICAL CONTRACT TEST (kernel = real propagation)\n");
 std::printf(" Bipartite Hermitian medium; quarter-period transfer = -i*W.\n");
 std::printf("=====================================================================\n");
 std::vector<NodeRow> nodes={ {0,{{4,0.11*kPi},{6,-0.27*kPi},{8,0.33*kPi}}},{1,{{5,0.19*kPi},{10,-0.13*kPi}}},{2,{{7,0.41*kPi},{11,-0.07*kPi}}},{3,{{9,0.23*kPi}}} };
 Mat W; for(auto&n:nodes)W.push_back(buildRow(n));
 for(int i=0;i<kInputs&&(int)W.size()<kModes;++i){Vec c(kInputs,cd(0,0));c[i]=cd(1,0);addOrthRow(W,c);}
 Mat H=buildMedium(W); Stepper s; s.build(H);
 int pass=0,total=0;

 std::printf("\n[1] THE BIPARTITE MEDIUM IS HERMITIAN\n");
 {double he=hermErr(H);std::printf(" hermiticity error = %.2e\n",he);++total;pass+=report("medium not hermitian",he<1e-12);}

 std::printf("\n[2] PHYSICAL TRANSFER = -i*W AND THE EVOLUTION IS UNITARY\n");
 {double terr=0,normErr=0;cd sc=transferScale();
 for(int in=0;in<kInputs;in++){Vec e(kInputs,cd(0,0));e[in]=cd(1,0);Vec z=s.evolve(setInput(e));Vec o=outModes(z);
 for(int m=0;m<kModes;m++)terr=std::max(terr,std::abs(o[m]-sc*W[m][in]));
 normErr=std::max(normErr,std::abs(power(z)-1.0));}
 std::printf(" max |T - (-i W)| = %.2e norm drift = %.2e (sin(angle)=%.6f)\n",terr,normErr,std::abs(transferScale()));
 ++total;pass+=report("physical transfer != -i W / not unitary",terr<1e-9&&normErr<1e-9);}

 std::printf("\n[3] SUPERPOSITION MATERIALISES AT ITS NODE BY REAL PROPAGATION (local)\n");
 {int rails[3]={4,6,8};double ph[3]={0.11*kPi,-0.27*kPi,0.33*kPi};
 Vec al(kInputs,cd(0,0)),an(kInputs,cd(0,0));for(int i=0;i<3;i++){al[rails[i]]=std::exp(cd(0,-ph[i]));an[rails[i]]=std::exp(cd(0,-ph[i]))*std::exp(cd(0,2.0*kPi*(i+1)/3.0));}norml(al);norml(an);
 Vec oa=outModes(s.evolve(setInput(al))), on=outModes(s.evolve(setInput(an)));
 double n0a=std::norm(oa[0]),n0n=std::norm(on[0]),leak=std::max({std::norm(oa[1]),std::norm(oa[2]),std::norm(oa[3])});
 std::printf(" node0 mode: aligned=%.6f anti=%.2e leak nodes 1-3=%.2e\n",n0a,n0n,leak);
 ++total;pass+=report("superposition did not materialise / not local",n0a>0.5&&n0n<1e-9&&leak<1e-9);}

 std::printf("\n[4] RESIDUAL MODES CONSERVE THE SUPERPOSITION\n");
 {int rails[3]={4,6,8};double ph[3]={0.11*kPi,-0.27*kPi,0.33*kPi};Vec an(kInputs,cd(0,0));for(int i=0;i<3;i++)an[rails[i]]=std::exp(cd(0,-ph[i]))*std::exp(cd(0,2.0*kPi*(i+1)/3.0));norml(an);
 Vec o=outModes(s.evolve(setInput(an)));double tot=power(o),node0=std::norm(o[0]);
 std::printf(" anti case: output mode power=%.6f node0=%.2e residual=%.6f\n",tot,node0,tot-node0);
 ++total;pass+=report("residual modes did not conserve",std::abs(tot-1.0)<1e-9&&(tot-node0)>0.99);}

 std::printf("\n RESULT : %d / %d verified\n",pass,total);return pass==total?0:1;
}
