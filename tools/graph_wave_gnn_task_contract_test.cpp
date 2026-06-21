// graph_wave_gnn_task_contract_test
// ----------------------------------------------------------------------------
// First WORKING end-to-end native GNN, with superposition kept central and the
// whole stack in play. Task: classify each node by the MAJORITY class of its
// neighbourhood -- a real graph task (a node's own feature is uninformative;
// you must aggregate neighbours). Native pipeline:
//   encode  : each node's class -> a phasor class-atom
//   message : aggregate neighbour atoms == a SUPERPOSITION of neighbour classes;
//             the majority class superposes with the largest amplitude
//   readout : cleanup over the class atoms == decide the majority (interference
//             picks the winner)
//   depth   : composing aggregation reaches the 2-hop neighbourhood
// Superposition does the computation: the aggregate's amplitude IS the class
// count. No weights, no trainer, no GA, no V26.
// ----------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>
namespace {
using cd = std::complex<double>; using Vec = std::vector<cd>;
constexpr int kN = 16; constexpr int kD = 1024; constexpr int kC = 2; constexpr double kTwoPi = 6.283185307179586;
Vec atom(std::mt19937_64&r){std::uniform_real_distribution<double>u(0,kTwoPi);Vec a(kD);for(int i=0;i<kD;i++){double p=u(r);a[i]=cd(std::cos(p),std::sin(p));}return a;}
double nrm(const Vec&a){double s=0;for(auto&v:a)s+=std::norm(v);return std::sqrt(s);}
cd inr(const Vec&a,const Vec&b){cd s(0,0);for(int i=0;i<kD;i++)s+=std::conj(a[i])*b[i];return s;}
double overlap(const Vec&a,const Vec&b){double na=nrm(a),nb=nrm(b);return(na<1e-12)?0.0:std::abs(inr(a,b))/(na*nb);}
bool report(const char*n,bool ok){std::printf("    => %s\n",ok?"PASS":"FAIL");if(!ok)std::printf("    !! %s\n",n);return ok;}
}
int main(){
  std::printf("=====================================================================\n");
  std::printf(" GRAPH-WAVE GNN TASK CONTRACT TEST  (N=%d nodes, classify by neighbourhood)\n",kN);
  std::printf(" Aggregation = superposition of neighbour classes; majority by amplitude.\n");
  std::printf("=====================================================================\n");
  std::mt19937_64 rng(0x6A54A5ULL);
  std::vector<Vec>classAtom(kC);for(auto&c:classAtom)c=atom(rng);
  std::vector<int>cls(kN);std::uniform_int_distribution<int>cd2(0,kC-1);for(int i=0;i<kN;i++)cls[i]=cd2(rng);
  // random graph (Erdos-Renyi)
  std::vector<std::vector<int>>adj(kN);std::uniform_real_distribution<double>u01(0,1);
  for(int i=0;i<kN;i++)for(int j=i+1;j<kN;j++)if(u01(rng)<0.35){adj[i].push_back(j);adj[j].push_back(i);}
  // node field = its class atom
  std::vector<Vec>F(kN);for(int i=0;i<kN;i++)F[i]=classAtom[cls[i]];
  auto aggregate=[&](const std::vector<Vec>&G){std::vector<Vec>O(kN,Vec(kD,cd(0,0)));for(int i=0;i<kN;i++)for(int j:adj[i])for(int d=0;d<kD;d++)O[i][d]+=G[j][d];return O;};
  auto decode=[&](const Vec&v){int b=-1;double bs=-1;for(int c=0;c<kC;c++){double o=overlap(v,classAtom[c]);if(o>bs){bs=o;b=c;}}return b;};
  // true 1-hop majority (skip ties / empty)
  auto trueMaj=[&](int i,const std::vector<std::vector<int>>& set){int cnt[kC]={0};for(int j:set[i])cnt[cls[j]]++;if(cnt[0]==cnt[1])return -1;return cnt[0]>cnt[1]?0:1;};
  int pass=0,total=0;

  std::printf("\n[1] NATIVE GNN CLASSIFIES NODES BY NEIGHBOURHOOD MAJORITY\n");
  {auto A1=aggregate(F);int ok=0,dec=0;for(int i=0;i<kN;i++){int t=trueMaj(i,adj);if(t<0)continue;dec++;if(decode(A1[i])==t)ok++;}
   std::printf("    accuracy on decidable nodes = %d/%d = %.1f%%\n",ok,dec,100.0*ok/std::max(1,dec));
   ++total;pass+=report("native GNN did not classify by neighbourhood",dec>=8&&ok==dec);}

  std::printf("\n[2] SUPERPOSITION DOES IT: aggregate amplitude reflects the class COUNT\n");
  {auto A1=aggregate(F);int node=-1;for(int i=0;i<kN;i++){if(trueMaj(i,adj)>=0&&adj[i].size()>=3){node=i;break;}}
   int cnt[kC]={0};for(int j:adj[node])cnt[cls[j]]++;
   double o0=overlap(A1[node],classAtom[0]),o1=overlap(A1[node],classAtom[1]);
   std::printf("    node %d neighbours: class0=%d class1=%d   overlaps: c0=%.3f c1=%.3f (amplitude ~ count)\n",node,cnt[0],cnt[1],o0,o1);
   ++total;pass+=report("amplitude does not reflect class count",(cnt[0]>cnt[1])==(o0>o1));}

  std::printf("\n[3] MESSAGE PASSING IS NECESSARY: a node's OWN feature is uninformative\n");
  {auto A1=aggregate(F);int okMP=0,okOwn=0,dec=0;for(int i=0;i<kN;i++){int t=trueMaj(i,adj);if(t<0)continue;dec++;if(decode(A1[i])==t)okMP++;if(cls[i]==t)okOwn++;}
   std::printf("    message-passing accuracy=%.1f%%   own-feature baseline=%.1f%% (~chance)\n",100.0*okMP/dec,100.0*okOwn/dec);
   ++total;pass+=report("own feature already solved it (task too easy)",okOwn<dec*0.75&&okMP>okOwn);}

  std::printf("\n[4] DEPTH: composing aggregation reaches the 2-hop neighbourhood\n");
  {std::vector<std::vector<int>>two(kN);for(int i=0;i<kN;i++){std::vector<int>m(kN,0);m[i]=1;for(int j:adj[i])m[j]=1;for(int j:adj[i])for(int k:adj[j])if(!m[k]){m[k]=2;}for(int k=0;k<kN;k++)if(m[k]==2)two[i].push_back(k);}
   auto A1=aggregate(F);auto A2=aggregate(A1);   // two rounds -> 2-hop reach
   int ok1=0,ok2=0,dec=0;for(int i=0;i<kN;i++){int t=trueMaj(i,two);if(t<0||two[i].size()<2)continue;dec++;if(decode(A1[i])==t)ok1++;if(decode(A2[i])==t)ok2++;}
   std::printf("    2-hop-majority task: 1-round acc=%.1f%%   2-round acc=%.1f%%  (decidable=%d)\n",dec?100.0*ok1/dec:0,dec?100.0*ok2/dec:0,dec);
   ++total;pass+=report("depth did not help reach 2-hop",dec>=4&&ok2>=ok1);}

  std::printf("\n RESULT : %d / %d verified\n",pass,total);return pass==total?0:1;
}
