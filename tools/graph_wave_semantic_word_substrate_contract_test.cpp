// graph_wave_semantic_word_substrate_contract_test
// ----------------------------------------------------------------------------
// A real word -> substrate conversion. Words are injected as REAL impulses at
// different times (phase is BORN from unitary evolution, never assigned); each
// word carries MEANING via its co-occurrence pattern (not an arbitrary node id);
// the readout is a HIGH-DIM gauge-invariant fingerprint split into two channels:
//   magnitude |psi_i|^2  -> which words / semantics
//   edge current J_ij    -> order (the emergent phase)
//
// Sharpened over the order-only probe with two decisive checks:
//   [2] field-space RETRIEVAL: a phrase's nearest field is same-topic (3 topics)
//   [3] BAG-OF-WORDS CONTROL: a bag is EXACTLY order-blind (cos=1.000 vs reverse);
//       the substrate is not -> it adds order a bag fundamentally cannot represent,
//       while keeping the semantics (magnitude channel) nearly intact.
// ----------------------------------------------------------------------------
#include "graph_wave_substrate.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <string>
#include <vector>
using namespace gw;

namespace {
const int N = 12;  // 3 topics x 4 words: royalty(0-3) ocean(4-7) forest(8-11)
const std::vector<std::vector<int>> SENT = {
    {0,1,2},{1,2,3},{0,3,1},{2,0,3},
    {4,5,6},{5,6,7},{4,7,5},{6,4,7},
    {8,9,10},{9,10,11},{8,11,9},{10,8,11}};
int topic(int w){ return w/4; }

double cosv(const std::vector<double>&a,const std::vector<double>&b){
  double d=0,na=0,nb=0; for(size_t i=0;i<a.size();i++){d+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}
  return d/(std::sqrt(na*nb)+1e-15); }

std::vector<std::vector<double>> embeddings(){
  std::vector<std::vector<double>> M(N,std::vector<double>(N,0));
  for(auto&s:SENT)for(int a:s)for(int b:s)if(a!=b)M[a][b]+=1;
  for(auto&r:M){double n=0;for(double v:r)n+=v*v;n=std::sqrt(n)+1e-9;for(double&v:r)v/=n;} return M; }

Graph substrate(){ Graph g(N); for(int i=0;i<N;i++){g.addEdge(i,(i+1)%N,1.0);g.addEdge(i,(i+2)%N,0.5);} return g; }
double norm2(const Vec&z){double s=0;for(auto&v:z)s+=std::norm(v);return s;}
Vec normalized(Vec z){double n=std::sqrt(norm2(z))+1e-15;for(auto&v:z)v/=n;return z;}

Vec drivePhrase(const CMat&h,const std::vector<std::vector<double>>&emb,const std::vector<int>&w,double tau){
  Stepper st; st.build(h,tau); Vec psi(N,cd(0,0)); bool has=false;
  for(int x:w){ if(has)psi=st.step(psi); for(int i=0;i<N;i++)psi[i]+=cd(emb[x][i],0); has=true; }
  psi=st.step(psi); return normalized(psi); }

double edgeCurrent(const CMat&h,const Vec&psi,int i,int j){return 2.0*std::imag(std::conj(psi[i])*h[i][j]*psi[j]);}
struct FP{std::vector<double> mag,cur;};
FP fingerprint(const CMat&h,const Vec&psi){FP f;f.mag.resize(N);
  for(int i=0;i<N;i++)f.mag[i]=std::norm(psi[i]);
  for(int i=0;i<N;i++){f.cur.push_back(edgeCurrent(h,psi,i,(i+1)%N));f.cur.push_back(edgeCurrent(h,psi,i,(i+2)%N));}
  return f;}
std::vector<double> full(const FP&f){std::vector<double> v=f.mag;v.insert(v.end(),f.cur.begin(),f.cur.end());return v;}
// bag-of-words representation: order-independent sum of word embeddings
std::vector<double> bag(const std::vector<std::vector<double>>&emb,const std::vector<int>&w){
  std::vector<double> v(N,0); for(int x:w)for(int i=0;i<N;i++)v[i]+=emb[x][i]; return v; }

bool report(const char*n,bool ok){std::printf(" => %s\n",ok?"PASS":"FAIL");if(!ok)std::printf(" !! %s\n",n);return ok;}
}

int main(){
  std::printf("=====================================================================\n");
  std::printf(" SEMANTIC WORD -> SUBSTRATE CONTRACT  (words become a phase field)\n");
  std::printf(" 3 topics x 4 words; meaning from co-occurrence; order from emergent phase\n");
  std::printf("=====================================================================\n");
  auto emb=embeddings(); Graph g=substrate(); const CMat&h=g.h; double tau=0.45;
  // 2 phrases per topic
  std::vector<std::vector<int>> P={{0,1,2},{3,0,1}, {4,5,6},{7,4,5}, {8,9,10},{11,8,9}};
  auto phTopic=[&](int i){return topic(P[i][0]);};
  std::vector<FP> fp; for(auto&p:P) fp.push_back(fingerprint(h,drivePhrase(h,emb,p,tau)));
  int pass=0,total=0;

  std::printf("\n[1] EMBEDDING CARRIES MEANING: same-topic words closer than cross-topic\n");
  double intra=0,inter=0;int ni=0,nx=0;
  for(int a=0;a<N;a++)for(int b=a+1;b<N;b++){double c=cosv(emb[a],emb[b]);
    if(topic(a)==topic(b)){intra+=c;ni++;}else{inter+=c;nx++;}}
  intra/=ni;inter/=nx; std::printf(" mean cos intra=%.3f  inter=%.3f\n",intra,inter);
  ++total; pass+=report("embedding not semantic", intra>inter+0.15);

  std::printf("\n[2] FIELD-SPACE RETRIEVAL: each phrase's nearest field is the same topic\n");
  int hit=0; for(size_t i=0;i<P.size();i++){ int best=-1;double bs=-2;
    for(size_t j=0;j<P.size();j++)if(i!=j){double c=cosv(fp[i].mag,fp[j].mag);if(c>bs){bs=c;best=(int)j;}}
    bool ok=(phTopic((int)i)==phTopic(best)); hit+=ok;
    std::printf("  phrase %zu (topic %d) -> nearest %d (topic %d) %s\n",i,phTopic(i),best,phTopic(best),ok?"OK":"X"); }
  ++total; pass+=report("nearest field is wrong topic", hit==(int)P.size());

  std::printf("\n[3] BAG IS ORDER-BLIND, SUBSTRATE IS NOT (value-add over text bag)\n");
  double bagSim=0,curSim=0,magSim=0;
  for(auto&p:P){ std::vector<int> r(p.rbegin(),p.rend());
    bagSim+=cosv(bag(emb,p),bag(emb,r));
    FP a=fingerprint(h,drivePhrase(h,emb,p,tau)), b=fingerprint(h,drivePhrase(h,emb,r,tau));
    magSim+=cosv(a.mag,b.mag); curSim+=cosv(a.cur,b.cur); }
  bagSim/=P.size(); magSim/=P.size(); curSim/=P.size();
  std::printf(" forward vs reverse (avg):  bag cos=%.4f (blind)  substrate mag cos=%.3f  current cos=%+.3f\n",bagSim,magSim,curSim);
  ++total; pass+=report("substrate did not add order beyond the bag", bagSim>0.9999 && curSim<magSim-0.3 && magSim>0.85);

  std::printf("\n[4] INJECTIVE: distinct phrases give distinguishable fields\n");
  double maxsim=0; for(size_t i=0;i<P.size();i++)for(size_t j=i+1;j<P.size();j++)maxsim=std::max(maxsim,cosv(full(fp[i]),full(fp[j])));
  std::printf(" max pairwise fingerprint cos = %.4f\n",maxsim);
  ++total; pass+=report("two phrases collided", maxsim<0.999);

  std::printf("\n[5] GAUGE INVARIANCE: fingerprint is an observable, not absolute phase\n");
  Vec psi=drivePhrase(h,emb,P[0],tau); CMat hg=h; Vec pg=psi; double th[N];
  {std::mt19937 rr(7);std::uniform_real_distribution<double> u(-kPi,kPi);for(int i=0;i<N;i++)th[i]=u(rr);}
  for(int i=0;i<N;i++)pg[i]*=std::exp(cd(0,th[i]));
  for(int i=0;i<N;i++)for(int j=0;j<N;j++)hg[i][j]*=std::exp(cd(0,th[i]-th[j]));
  auto f0=full(fingerprint(h,psi)),f1=full(fingerprint(hg,pg));
  double gd=0;for(size_t i=0;i<f0.size();i++)gd=std::max(gd,std::abs(f0[i]-f1[i]));
  std::printf(" max fingerprint diff under local gauge = %.2e\n",gd);
  ++total; pass+=report("fingerprint changed under gauge", gd<1e-12);

  std::printf("\n RESULT : %d / %d verified\n",pass,total);
  return pass==total?0:1;
}
