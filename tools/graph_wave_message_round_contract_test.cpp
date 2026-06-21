// graph_wave_message_round_contract_test
// ----------------------------------------------------------------------------
// A WORKING unitary GNN layer: one full message-passing round on a graph. Each
// node emits its state to its outgoing edges (fan-out), the edges carry the
// messages, and each node aggregates its incoming edges into a new state
// (fan-in). The whole round is ONE unitary acting on the node states; it
// computes "new node state = aggregate of neighbour states" with no broadcast
// and no cloning (both fan-out and fan-in are unitary with residual modes).
//
// Built by composing two materialization kernels: F_out = (out-edge
// materialization)^dagger emits, K_in (in-edge materialization) aggregates.
// Path graph 0-1-2. 7 rails: 3 self + 4 oriented edges (0->1,1->0,1->2,2->1).
//
//   [1] F_out, K_in, and the composed round are all unitary
//   [2] the round COMPUTES message passing: node j's new state draws only from
//       node j and its graph NEIGHBOURS (1-hop locality), matching the adjacency
//   [3] a 2-hop node contributes nothing in one round (locality is real)
//   [4] the round conserves the state (unitary; total power = 1)
// In-scope substrate grammar: no decision, no learning, no trainer, no V26.
// ----------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>
namespace {
using cd = std::complex<double>; using Vec = std::vector<cd>; using Mat = std::vector<std::vector<cd>>;
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr int kR = 7;   // rails: 0,1,2 self; 3=e(0->1) 4=e(1->0) 5=e(1->2) 6=e(2->1)
const double kSelf = 1.0, kIn = 0.8;
cd rinner(const Vec&a,const Vec&b){cd s(0,0);for(int i=0;i<kR;i++)s+=a[i]*std::conj(b[i]);return s;}
bool addOrth(Mat&rows,Vec c){for(const auto&b:rows){cd co=rinner(c,b);for(int i=0;i<kR;i++)c[i]-=co*b[i];}double n=std::sqrt(std::max(0.0,rinner(c,c).real()));if(n<1e-10)return false;for(auto&v:c)v/=n;rows.push_back(c);return true;}
Vec row(int self,std::vector<int> ins){double n=std::sqrt(kSelf*kSelf+ins.size()*kIn*kIn);Vec r(kR,cd(0,0));r[self]=cd(kSelf/n,0);double ph=0.13*kPi;for(int e:ins){r[e]=cd(kIn/n,0)*std::exp(cd(0,ph));ph+=0.11*kPi;}return r;}
// unitary matrix from orthonormal rows: A[r][c] = conj(row_r[c])  (out_r = <x,row_r>)
Mat kernel(const std::vector<Vec>& meaningful){Mat rows=meaningful;for(int i=0;i<kR&&(int)rows.size()<kR;i++){Vec c(kR,cd(0,0));c[i]=cd(1,0);addOrth(rows,c);}Mat A(kR,Vec(kR));for(int r=0;r<kR;r++)for(int c=0;c<kR;c++)A[r][c]=std::conj(rows[r][c]);return A;}
Mat dagger(const Mat&m){Mat o(kR,Vec(kR));for(int i=0;i<kR;i++)for(int j=0;j<kR;j++)o[j][i]=std::conj(m[i][j]);return o;}
Mat matmul(const Mat&a,const Mat&b){Mat o(kR,Vec(kR,cd(0,0)));for(int i=0;i<kR;i++)for(int k=0;k<kR;k++)for(int j=0;j<kR;j++)o[i][j]+=a[i][k]*b[k][j];return o;}
Vec matvec(const Mat&m,const Vec&x){Vec o(kR,cd(0,0));for(int i=0;i<kR;i++)for(int j=0;j<kR;j++)o[i]+=m[i][j]*x[j];return o;}
double unit(const Mat&m){Mat g=matmul(dagger(m),m);double e=0;for(int i=0;i<kR;i++)for(int j=0;j<kR;j++)e=std::max(e,std::abs(g[i][j]-(i==j?cd(1,0):cd(0,0))));return e;}
double power(const Vec&z){double s=0;for(auto&v:z)s+=std::norm(v);return s;}
bool report(const char*n,bool ok){std::printf("    => %s\n",ok?"PASS":"FAIL");if(!ok)std::printf("    !! %s\n",n);return ok;}
}
int main(){
  std::printf("=====================================================================\n");
  std::printf(" GRAPH-WAVE MESSAGE-ROUND CONTRACT TEST  (one working unitary GNN layer)\n");
  std::printf(" fan-out (emit) o fan-in (aggregate) on path 0-1-2; new state = neighbour aggregate.\n");
  std::printf("=====================================================================\n");
  // K_in: each node aggregates its INCOMING edges (0<-e10; 1<-e01,e21; 2<-e12)
  Mat Kin = kernel({ row(0,{4}), row(1,{3,6}), row(2,{5}) });
  // M_out: out-edge materialization (0->e01; 1->e10,e12; 2->e21); fan-out = its dagger
  Mat Mout = kernel({ row(0,{3}), row(1,{4,5}), row(2,{6}) });
  Mat Fout = dagger(Mout);
  Mat Round = matmul(Kin, Fout);
  int pass=0,total=0;

  std::printf("\n[1] F_out, K_in, AND THE COMPOSED ROUND ARE ALL UNITARY\n");
  {double a=unit(Fout),b=unit(Kin),c=unit(Round);std::printf("    |.|: F_out=%.2e  K_in=%.2e  Round=%.2e\n",a,b,c);
   ++total;pass+=report("a layer is not unitary",a<1e-12&&b<1e-12&&c<1e-12);}

  std::printf("\n[2] THE ROUND COMPUTES MESSAGE PASSING (node1 draws from neighbours 0 and 2)\n");
  {double w10=std::abs(Round[1][0]),w12=std::abs(Round[1][2]),w11=std::abs(Round[1][1]);
   std::printf("    new node1 from: self1=%.3f  node0(nbr)=%.3f  node2(nbr)=%.3f\n",w11,w10,w12);
   ++total;pass+=report("round did not aggregate node1's neighbours",w10>0.1&&w12>0.1);}

  std::printf("\n[3] LOCALITY: a 2-hop node contributes nothing in one round\n");
  {double w20=std::abs(Round[2][0]),w02=std::abs(Round[0][2]);   // node0 and node2 are 2-hop (not neighbours)
   std::printf("    new node2 from node0 (2-hop)=%.2e   new node0 from node2 (2-hop)=%.2e\n",w20,w02);
   ++total;pass+=report("2-hop leaked in one round",w20<1e-12&&w02<1e-12);}

  std::printf("\n[4] THE ROUND CONSERVES THE STATE (unitary)\n");
  {Vec x={cd(0.5,0.2),cd(-0.3,0.6),cd(0.4,-0.1),cd(0,0),cd(0,0),cd(0,0),cd(0,0)};double pin=power(x);Vec y=matvec(Round,x);
   std::printf("    input power=%.6f  output power=%.6f\n",pin,power(y));
   ++total;pass+=report("round did not conserve the state",std::abs(power(y)-pin)<1e-12);}

  std::printf("\n RESULT : %d / %d verified  (a working unitary message-passing layer)\n",pass,total);
  return pass==total?0:1;
}
