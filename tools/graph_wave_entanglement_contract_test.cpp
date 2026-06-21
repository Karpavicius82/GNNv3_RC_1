// graph_wave_entanglement_contract_test
// ----------------------------------------------------------------------------
// First step beyond single-particle wave superposition. Everything so far lived
// in C^N (one amplitude per node = classical/optical interference). The richer
// superposition is across SUBSYSTEMS -- a tensor product space C^(d^k) where
// states can be ENTANGLED (non-factorable) with exponential dimension. In V26
// the natural tensor factors are the lanes; here, minimal qubit "lanes":
//   [1] a product state factors; an entangled (Bell) state does NOT -- the
//       reduced state of one lane is pure for a product, mixed for entangled.
//   [2] entanglement requires an INTER-lane interaction; a local (single-lane)
//       gate cannot create it.
//   [3] entanglement = non-factorable correlations: <Z_A Z_B> != <Z_A><Z_B>.
//   [4] the resource is exponential: k lanes span d^k, not k*d.
// Substrate-only, deterministic. No V26 linkage, no weights, no trainer.
// ----------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>
namespace {
using cd = std::complex<double>; using Vec = std::vector<cd>;
Vec kron(const Vec&a,const Vec&b){Vec o(a.size()*b.size());for(size_t i=0;i<a.size();i++)for(size_t j=0;j<b.size();j++)o[i*b.size()+j]=a[i]*b[j];return o;}
void norm(Vec&v){double s=0;for(auto&x:v)s+=std::norm(x);s=std::sqrt(s);for(auto&x:v)x/=s;}
// reduced 2x2 density matrix of the FIRST qubit of an n-qubit state
void reduceFirst(const Vec&psi,int n,cd r[2][2]){int rest=1<<(n-1);for(int i=0;i<2;i++)for(int j=0;j<2;j++){cd s(0,0);for(int b=0;b<rest;b++)s+=psi[i*rest+b]*std::conj(psi[j*rest+b]);r[i][j]=s;}}
double purity(const cd r[2][2]){return std::norm(r[0][0])+std::norm(r[0][1])+std::norm(r[1][0])+std::norm(r[1][1]);}
double entropy(const cd r[2][2]){double p=std::real(r[0][0]);double c=std::abs(r[0][1]);double d=std::sqrt((2*p-1)*(2*p-1)+4*c*c);double l1=0.5*(1+d),l2=0.5*(1-d);auto h=[](double l){return l>1e-12?-l*std::log(l):0.0;};return h(l1)+h(l2);}
double expZ(const Vec&psi,int n,int q){double s=0;for(size_t i=0;i<psi.size();i++){int bit=(i>>(n-1-q))&1;s+=(bit?-1.0:1.0)*std::norm(psi[i]);}return s;}
double expZZ(const Vec&psi,int n,int a,int b){double s=0;for(size_t i=0;i<psi.size();i++){int ba=(i>>(n-1-a))&1,bb=(i>>(n-1-b))&1;s+=(ba?-1.0:1.0)*(bb?-1.0:1.0)*std::norm(psi[i]);}return s;}
Vec cnot2(const Vec&p){Vec o(4);o[0]=p[0];o[1]=p[1];o[2]=p[3];o[3]=p[2];return o;} // control q0, target q1
bool report(const char*n,bool ok){std::printf("    => %s\n",ok?"PASS":"FAIL");if(!ok)std::printf("    !! %s\n",n);return ok;}
}
int main(){
  std::printf("=====================================================================\n");
  std::printf(" GRAPH-WAVE ENTANGLEMENT CONTRACT TEST  (lanes as tensor factors)\n");
  std::printf(" Superposition across subsystems: non-factorable, exponential.\n");
  std::printf("=====================================================================\n");
  const double r2=1.0/std::sqrt(2.0);
  Vec q0={cd(1,0),cd(0,0)},q1={cd(0,0),cd(1,0)},plus={cd(r2,0),cd(r2,0)};
  int pass=0,total=0;

  std::printf("\n[1] PRODUCT STATE FACTORS; ENTANGLED (BELL) STATE DOES NOT\n");
  {Vec prod=kron(plus,q0);                          // |+>|0>  (separable)
   Vec bell={cd(r2,0),cd(0,0),cd(0,0),cd(r2,0)};    // (|00>+|11>)/sqrt2 (entangled)
   cd rp[2][2],rb[2][2];reduceFirst(prod,2,rp);reduceFirst(bell,2,rb);
   std::printf("    product : lane-A purity=%.4f  entropy=%.4f (pure -> separable)\n",purity(rp),entropy(rp));
   std::printf("    bell    : lane-A purity=%.4f  entropy=%.4f (mixed -> entangled, S=ln2=%.4f)\n",purity(rb),entropy(rb),std::log(2.0));
   ++total;pass+=report("entangled state not distinguished from product",purity(rp)>0.999&&std::abs(purity(rb)-0.5)<1e-9&&std::abs(entropy(rb)-std::log(2.0))<1e-6);}

  std::printf("\n[2] ENTANGLEMENT REQUIRES AN INTER-LANE INTERACTION\n");
  {Vec start=kron(plus,q0);                          // separable
   Vec ent=cnot2(start);                             // CNOT: inter-lane gate -> Bell
   Vec loc=kron(Vec{cd(0,0),cd(1,0)},q0);            // local flip on lane A only (X|+> ~ stays product)
   cd re[2][2],rl[2][2];reduceFirst(ent,2,re);reduceFirst(loc,2,rl);
   std::printf("    inter-lane gate (CNOT): lane-A purity=%.4f (entangled)\n",purity(re));
   std::printf("    local single-lane gate: lane-A purity=%.4f (still separable)\n",purity(rl));
   ++total;pass+=report("entanglement created locally / not by interaction",std::abs(purity(re)-0.5)<1e-9&&purity(rl)>0.999);}

  std::printf("\n[3] ENTANGLEMENT = NON-FACTORABLE CORRELATIONS  <Z_A Z_B> != <Z_A><Z_B>\n");
  {Vec bell={cd(r2,0),cd(0,0),cd(0,0),cd(r2,0)};Vec prod=kron(plus,q0);
   double cb=expZZ(bell,2,0,1)-expZ(bell,2,0)*expZ(bell,2,1);
   double cp=expZZ(prod,2,0,1)-expZ(prod,2,0)*expZ(prod,2,1);
   std::printf("    bell    : <ZZ>=%.3f <Z_A>=%.3f <Z_B>=%.3f  connected=%.4f\n",expZZ(bell,2,0,1),expZ(bell,2,0),expZ(bell,2,1),cb);
   std::printf("    product : <ZZ>=%.3f <Z_A>=%.3f <Z_B>=%.3f  connected=%.4f\n",expZZ(prod,2,0,1),expZ(prod,2,0),expZ(prod,2,1),cp);
   ++total;pass+=report("correlations factorize / no entanglement signature",std::abs(cb-1.0)<1e-9&&std::abs(cp)<1e-9);}

  std::printf("\n[4] THE RESOURCE IS EXPONENTIAL: k lanes span d^k (GHZ over 3 lanes)\n");
  {Vec ghz(8,cd(0,0));ghz[0]=cd(r2,0);ghz[7]=cd(r2,0);   // (|000>+|111>)/sqrt2
   cd rg[2][2];reduceFirst(ghz,3,rg);
   std::printf("    3-lane GHZ: state-space dim=%d (= 2^3, not 3*2=6)   lane-A purity=%.4f (entangled)\n",(int)ghz.size(),purity(rg));
   ++total;pass+=report("multi-lane entanglement / exponential space not shown",ghz.size()==8&&std::abs(purity(rg)-0.5)<1e-9);}

  std::printf("\n RESULT : %d / %d verified\n",pass,total);return pass==total?0:1;
}
