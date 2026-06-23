// graph_wave_gnn_message_passing_contract_test
// ----------------------------------------------------------------------------
// Integration step: a native GNN message-passing layer as ONE live evolving
// superposition on the graph. Node features are phasor states (the feature
// register); message passing is propagation on the graph medium that mixes and
// interferes neighbour amplitudes; depth is multi-hop reach; readout is overlap
// (cleanup). It ties the stack together:
// [1] one round aggregates a node's 1-hop neighbours (interference sum).
// [2] it is structure-dependent: remove an edge, that neighbour drops out.
// [3] depth = multi-hop: two rounds reach 2-hop neighbours (receptive field).
// [4] the graph PHYSICS is load-bearing: random (non-graph) mixing aggregates
// the wrong neighbourhood.
// Field = N node-vectors in feature space C^D; propagation mixes the node index
// by the Cayley evolution of the graph adjacency. Substrate-only, no
// weights, no trainer.
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
constexpr int kN = 5; // nodes (chain 0-1-2-3-4)
constexpr int kD = 1024; // feature register dimension (orthogonal atoms)
double fnorm(const Vec&a){double s=0;for(auto&v:a)s+=std::norm(v);return std::sqrt(s);}
cd fdot(const Vec&a,const Vec&b){cd s(0,0);for(int i=0;i<kD;i++)s+=std::conj(a[i])*b[i];return s;}
double overlap(const Vec&a,const Vec&b){double na=fnorm(a),nb=fnorm(b);return (na<1e-12||nb<1e-12)?0.0:std::abs(fdot(a,b))/(na*nb);}
Vec atom(std::mt19937_64&r){std::uniform_real_distribution<double>u(0,6.283185307);Vec a(kD);for(int i=0;i<kD;i++){double p=u(r);a[i]=cd(std::cos(p),std::sin(p));}return a;}
struct LUC{int n=0;Mat a;std::vector<int>piv;void factor(Mat in){n=(int)in.size();a=std::move(in);piv.resize(n);for(int i=0;i<n;i++)piv[i]=i;for(int k=0;k<n;k++){double b=std::abs(a[k][k]);int r=k;for(int i=k+1;i<n;i++){double v=std::abs(a[i][k]);if(v>b){b=v;r=i;}}if(r!=k){std::swap(a[r],a[k]);std::swap(piv[r],piv[k]);}cd akk=a[k][k];for(int i=k+1;i<n;i++){cd f=a[i][k]/akk;a[i][k]=f;for(int j=k+1;j<n;j++)a[i][j]-=f*a[k][j];}}}Vec solve(const Vec&b)const{Vec x(n);for(int i=0;i<n;i++)x[i]=b[piv[i]];for(int i=0;i<n;i++){cd s=x[i];for(int j=0;j<i;j++)s-=a[i][j]*x[j];x[i]=s;}for(int i=n-1;i>=0;i--){cd s=x[i];for(int j=i+1;j<n;j++)s-=a[i][j]*x[j];x[i]=s/a[i][i];}return x;}};
// one Cayley step operator U for graph adjacency H (one message-passing round)
Mat stepOp(const Mat&H,double dt){Mat ap(kN,Vec(kN)),am(kN,Vec(kN));const cd I(0,1);for(int i=0;i<kN;i++)for(int j=0;j<kN;j++){cd k=I*H[i][j]*(dt/2.0);cd id=(i==j)?cd(1,0):cd(0,0);ap[i][j]=id+k;am[i][j]=id-k;}LUC lu;lu.factor(ap);Mat U(kN,Vec(kN));for(int c=0;c<kN;c++){Vec rhs(kN,cd(0,0));for(int i=0;i<kN;i++)rhs[i]=am[i][c];Vec col=lu.solve(rhs);for(int i=0;i<kN;i++)U[i][c]=col[i];}return U;}
// message passing: new node-field row i = sum_j U[i][j] * feature[j]
std::vector<Vec> propagate(const Mat&U,const std::vector<Vec>&F){std::vector<Vec>O(kN,Vec(kD,cd(0,0)));for(int i=0;i<kN;i++)for(int j=0;j<kN;j++)for(int d=0;d<kD;d++)O[i][d]+=U[i][j]*F[j][d];return O;}
bool report(const char*n,bool ok){std::printf(" => %s\n",ok?"PASS":"FAIL");if(!ok)std::printf(" !! %s\n",n);return ok;}
}
int main(){
 std::printf("=====================================================================\n");
 std::printf(" GRAPH-WAVE GNN MESSAGE-PASSING CONTRACT TEST (N=%d chain, D=%d)\n",kN,kD);
 std::printf(" One live superposition: features + graph propagation = aggregation.\n");
 std::printf("=====================================================================\n");
 std::mt19937_64 rng(0x6111ULL);
 std::vector<Vec> F(kN);for(auto&f:F)f=atom(rng); // node features (symbols)
 auto chain=[&](){Mat A(kN,Vec(kN,cd(0,0)));for(int i=0;i+1<kN;i++){A[i][i+1]=cd(1,0);A[i+1][i]=cd(1,0);}return A;};
 const double dt=0.35;
 int pass=0,total=0;

 std::printf("\n[1] ONE ROUND AGGREGATES THE 1-HOP NEIGHBOURS (node 2: neighbours {1,3})\n");
 {Mat U=stepOp(chain(),dt);auto O=propagate(U,F);
 double o1=overlap(O[2],F[1]),o3=overlap(O[2],F[3]),o0=overlap(O[2],F[0]),o4=overlap(O[2],F[4]);
 std::printf(" overlaps at node 2: f1=%.3f f3=%.3f (neighbours) f0=%.3f f4=%.3f (2-hop)\n",o1,o3,o0,o4);
 ++total;pass+=report("did not aggregate the 1-hop neighbourhood",std::min(o1,o3)>1.5*std::max(o0,o4));}

 std::printf("\n[2] STRUCTURE-DEPENDENT: remove edge (2,3) -> f3 drops out\n");
 {Mat A=chain();A[2][3]=cd(0,0);A[3][2]=cd(0,0);Mat U=stepOp(A,dt);auto O=propagate(U,F);
 double o3=overlap(O[2],F[3]),o1=overlap(O[2],F[1]);
 std::printf(" after cutting (2,3): node 2 overlap f3=%.3f (f1 still=%.3f)\n",o3,o1);
 ++total;pass+=report("aggregate ignored the graph structure",o3<0.08&&o1>0.1&&o3<o1/2.0);}

 std::printf("\n[3] DEPTH = MULTI-HOP: two rounds reach the 2-hop neighbour (node 0: 2-hop {2})\n");
 {Mat U=stepOp(chain(),dt);auto O1=propagate(U,F);auto O2=propagate(U,O1);
 double a1=overlap(O1[0],F[2]),a2=overlap(O2[0],F[2]);
 std::printf(" node 0 overlap with f2 (2-hop): after 1 round=%.3f after 2 rounds=%.3f\n",a1,a2);
 ++total;pass+=report("depth did not extend the receptive field",a2>0.12&&a2>1.8*a1);}

 std::printf("\n[4] THE AGGREGATION EQUALS THE GRAPH ADJACENCY (structure load-bearing, every node)\n");
 {Mat U=stepOp(chain(),dt);auto O=propagate(U,F);
 std::vector<std::vector<int>>nb={{1},{0,2},{1,3},{2,4},{3}};
 bool allOK=true;double worstNb=1.0,bestNon=0.0;
 for(int i=0;i<kN;i++){double mnNb=1.0,mxNon=0.0;for(int j=0;j<kN;j++){if(j==i)continue;double o=overlap(O[i],F[j]);bool isNb=std::find(nb[i].begin(),nb[i].end(),j)!=nb[i].end();if(isNb)mnNb=std::min(mnNb,o);else mxNon=std::max(mxNon,o);}if(mnNb<=mxNon)allOK=false;worstNb=std::min(worstNb,mnNb);bestNon=std::max(bestNon,mxNon);}
 std::printf(" every node: min neighbour overlap=%.3f > max non-neighbour overlap=%.3f\n",worstNb,bestNon);
 ++total;pass+=report("aggregation did not match the graph adjacency",allOK&&worstNb>bestNon);}

 std::printf("\n RESULT : %d / %d verified\n",pass,total);return pass==total?0:1;
}
