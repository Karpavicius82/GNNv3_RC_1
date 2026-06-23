// graph_wave_graph_layer_materialization_contract_test
// ----------------------------------------------------------------------------
// In-scope, substrate-only GNN-analog grammar: a FULL small-graph layer, not a
// single hub. Several nodes each materialise their own neighbourhood update
// simultaneously inside ONE unitary layer, by the correct technique (oriented
// edge-rails -> per-node update rows -> jointly orthonormal -> residual modes).
// Superposition is central; the graph-scale point is that oriented rails keep
// different nodes' rows orthogonal, whereas a shared (broadcast) rail does not.
//
// Graph (4 nodes): edges 0-1, 0-2, 0-3, 1-2 (node 0 has degree 3).
//
// [1] the full-graph layer kernel is unitary (all node rows jointly orthonormal)
// [2] oriented rails keep node rows ORTHOGONAL; a shared (broadcast) rail makes
// two nodes' rows OVERLAP -> not unitary (broadcast rejected at graph scale)
// [3] a SUPERPOSITION of one node's incoming messages aggregates at THAT node
// by phase interference, and stays LOCAL (other nodes' ports ~0)
// [4] residual modes conserve the superposition (no false compression)
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
constexpr int kInputs = 12; // rails 0-3 = self; 4-11 = oriented edges
const double kSelfGain = 1.0, kInGain = 0.8;
// oriented edge rails: 4:1->0 5:0->1 6:2->0 7:0->2 8:3->0 9:0->3 10:2->1 11:1->2
cd rinner(const Vec&a,const Vec&b){cd s(0,0);for(size_t i=0;i<a.size();++i)s+=a[i]*std::conj(b[i]);return s;}
bool addOrthRow(Mat&rows,Vec c){for(const auto&b:rows){cd co=rinner(c,b);for(size_t i=0;i<c.size();++i)c[i]-=co*b[i];}double n=std::sqrt(std::max(0.0,rinner(c,c).real()));if(n<1e-10)return false;for(auto&v:c)v/=n;rows.push_back(c);return true;}
struct NodeRow{int self;std::vector<std::pair<int,double>>ins;};
Vec buildRow(const NodeRow&nr){double n=std::sqrt(kSelfGain*kSelfGain+nr.ins.size()*kInGain*kInGain);Vec row(kInputs,cd(0,0));row[nr.self]=cd(kSelfGain/n,0);for(auto&p:nr.ins)row[p.first]=(kInGain/n)*std::exp(cd(0,p.second));return row;}
Vec applyK(const Mat&rows,const Vec&x){Vec o(rows.size());for(size_t r=0;r<rows.size();++r)o[r]=rinner(x,rows[r]);return o;}
double unitErr(const Mat&rows){double e=0;for(size_t i=0;i<rows.size();++i)for(size_t j=0;j<rows.size();++j){cd g=rinner(rows[i],rows[j]);cd id=(i==j)?cd(1,0):cd(0,0);e=std::max(e,std::abs(g-id));}return e;}
double power(const Vec&z){double s=0;for(auto&v:z)s+=std::norm(v);return s;}
void norml(Vec&z){double n=std::sqrt(power(z));for(auto&v:z)v/=n;}
bool report(const char*n,bool ok){std::printf(" => %s\n",ok?"PASS":"FAIL");if(!ok)std::printf(" !! %s\n",n);return ok;}
}
int main(){
 std::printf("=====================================================================\n");
 std::printf(" GRAPH-WAVE GRAPH-LAYER MATERIALIZATION CONTRACT TEST (4-node graph)\n");
 std::printf(" Several nodes materialise their neighbourhoods in one unitary layer.\n");
 std::printf("=====================================================================\n");
 // ORIENTED node-update rows (proper)
 std::vector<NodeRow> nodes = {
 {0,{{4,0.11*kPi},{6,-0.27*kPi},{8,0.33*kPi}}}, // node 0 <- 1,2,3
 {1,{{5,0.19*kPi},{10,-0.13*kPi}}}, // node 1 <- 0,2
 {2,{{7,0.41*kPi},{11,-0.07*kPi}}}, // node 2 <- 0,1
 {3,{{9,0.23*kPi}}}, // node 3 <- 0
 };
 Mat rows; for(auto&n:nodes) rows.push_back(buildRow(n));
 Mat K=rows; for(int i=0;i<kInputs&&(int)K.size()<kInputs;++i){Vec c(kInputs,cd(0,0));c[i]=cd(1,0);addOrthRow(K,c);}
 int pass=0,total=0;

 std::printf("\n[1] THE FULL-GRAPH LAYER KERNEL IS UNITARY (nodes jointly orthonormal)\n");
 {double ue=unitErr(K);std::printf(" |K^dag K - I| = %.2e (layer materialises %d nodes at once)\n",ue,(int)nodes.size());
 ++total;pass+=report("graph-layer kernel not unitary",ue<1e-12);}

 std::printf("\n[2] ORIENTED RAILS KEEP NODE ROWS ORTHOGONAL; BROADCAST OVERLAPS (rejected)\n");
 {double maxOrient=0;for(size_t i=0;i<nodes.size();++i)for(size_t j=i+1;j<nodes.size();++j)maxOrient=std::max(maxOrient,std::abs(rinner(rows[i],rows[j])));
 // broadcast: each undirected edge -> ONE shared rail (0-1=>5, 0-2=>7, 0-3=>9, 1-2=>11)
 Vec b0=buildRow({0,{{5,0.11*kPi},{7,-0.27*kPi},{9,0.33*kPi}}});
 Vec b1=buildRow({1,{{5,0.19*kPi},{11,-0.13*kPi}}}); // shares rail 5 with node 0
 double bcOverlap=std::abs(rinner(b0,b1));
 std::printf(" oriented: max node-row overlap=%.2e broadcast (shared rail): overlap=%.4f\n",maxOrient,bcOverlap);
 ++total;pass+=report("oriented rails not orthogonal / broadcast not rejected",maxOrient<1e-12&&bcOverlap>0.1);}

 std::printf("\n[3] SUPERPOSITION AGGREGATES AT ITS NODE BY PHASE INTERFERENCE, AND STAYS LOCAL\n");
 {int inRails[3]={4,6,8};double ph[3]={0.11*kPi,-0.27*kPi,0.33*kPi}; // node 0's incoming
 Vec aligned(kInputs,cd(0,0)),anti(kInputs,cd(0,0));
 for(int i=0;i<3;++i){aligned[inRails[i]]=std::exp(cd(0,ph[i]));anti[inRails[i]]=std::exp(cd(0,ph[i]))*std::exp(cd(0,2.0*kPi*(i+1)/3.0));}
 norml(aligned);norml(anti);
 Vec oa=applyK(K,aligned), on=applyK(K,anti);
 double n0a=std::norm(oa[0]), n0n=std::norm(on[0]);
 double leak=std::max({std::norm(oa[1]),std::norm(oa[2]),std::norm(oa[3])});
 std::printf(" node0 update: aligned=%.6f anti=%.2e leak into nodes 1-3 = %.2e (locality)\n",n0a,n0n,leak);
 ++total;pass+=report("not phase-interference / not local",n0a>0.5&&n0n<1e-9&&leak<1e-12);}

 std::printf("\n[4] RESIDUAL MODES CONSERVE THE SUPERPOSITION\n");
 {int inRails[3]={4,6,8};double ph[3]={0.11*kPi,-0.27*kPi,0.33*kPi};
 Vec anti(kInputs,cd(0,0));for(int i=0;i<3;++i)anti[inRails[i]]=std::exp(cd(0,ph[i]))*std::exp(cd(0,2.0*kPi*(i+1)/3.0));norml(anti);
 Vec out=applyK(K,anti);double tot=power(out),node0=std::norm(out[0]);
 std::printf(" anti case: total power=%.6f node0 port=%.2e residual=%.6f\n",tot,node0,tot-node0);
 ++total;pass+=report("residual modes did not conserve the superposition",std::abs(tot-1.0)<1e-12&&(tot-node0)>0.99);}

 std::printf("\n RESULT : %d / %d verified\n",pass,total);return pass==total?0:1;
}
