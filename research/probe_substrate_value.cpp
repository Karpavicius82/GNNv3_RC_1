// The decisive cut: how much does the REAL semantic graph add, beyond a matched-
// random graph and beyond the PPMI embedding alone? Task = category classification
// where each word's own feature is CORRUPTED; FLOW (graph aggregation) must denoise
// it from its neighbours. On the real PPMI graph neighbours share topic -> recover;
// on a matched-random graph neighbours are unrelated -> no recovery. If real >>
// random and real recovers toward the clean PPMI ceiling, the SEMANTIC graph is
// load-bearing (not a generic unitary/phase effect).
#include "../tools/graph_wave_substrate.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
using namespace std;

int main(){
  const int TOK=1000000, VOCAB=4096, WIN=4, TOPK=32;
  // load text8
  ifstream in("data/text8"); if(!in){printf("no data/text8\n");return 2;}
  unordered_map<string,int> cnt; string w; int n=0;
  while(n<TOK && (in>>w)){cnt[w]++;n++;}
  vector<pair<string,int>> wv(cnt.begin(),cnt.end());
  sort(wv.begin(),wv.end(),[](auto&a,auto&b){return a.second!=b.second?a.second>b.second:a.first<b.first;});
  if((int)wv.size()>VOCAB)wv.resize(VOCAB);
  unordered_map<string,int> id; vector<string> word;
  for(auto&p:wv){id[p.first]=word.size();word.push_back(p.first);}
  int N=word.size();
  in.clear();in.seekg(0); vector<int> ids; int k=0;
  while(k<TOK&&(in>>w)){auto it=id.find(w);ids.push_back(it==id.end()?-1:it->second);k++;}
  // co-occurrence + PPMI embedding
  vector<float> co((size_t)N*N,0);
  for(size_t i=0;i<ids.size();i++){int a=ids[i];if(a<0)continue;
    for(int d=1;d<=WIN&&i+d<ids.size();d++){int b=ids[i+d];if(b<0||a==b)continue;float ww=1.0f/d;
      co[(size_t)a*N+b]+=ww;co[(size_t)b*N+a]+=ww;}}
  vector<double> rs(N,0);double tot=0;for(int i=0;i<N;i++)for(int j=0;j<N;j++){rs[i]+=co[(size_t)i*N+j];tot+=co[(size_t)i*N+j];}
  vector<float> emb((size_t)N*N,0);
  for(int i=0;i<N;i++){double nn=0;for(int j=0;j<N;j++){double c=co[(size_t)i*N+j];if(c<=0||rs[i]<=0||rs[j]<=0)continue;
    double pmi=log(c*tot/(rs[i]*rs[j]));if(pmi>0){emb[(size_t)i*N+j]=pmi;nn+=pmi*pmi;}}
    nn=sqrt(nn)+1e-12;for(int j=0;j<N;j++)emb[(size_t)i*N+j]/=nn;}
  // real semantic graph (top-k symmetric PPMI neighbours)
  vector<vector<int>> R(N), RND(N);
  for(int i=0;i<N;i++){vector<pair<float,int>> v;for(int j=0;j<N;j++){if(i==j)continue;
    float s=0.5f*(emb[(size_t)i*N+j]+emb[(size_t)j*N+i]);if(s>0)v.push_back({s,j});}
    sort(v.begin(),v.end(),[](auto&a,auto&b){return a.first>b.first;});
    for(int t=0;t<min(TOPK,(int)v.size());t++)R[i].push_back(v[t].second);}
  // matched-random graph: same degree per node, random endpoints
  mt19937 rng(7);uniform_int_distribution<int> rn(0,N-1);
  for(int i=0;i<N;i++)for(size_t e=0;e<R[i].size();e++){int j=rn(rng);if(j!=i)RND[i].push_back(j);}

  // categories (filtered to present words)
  vector<vector<string>> CATS={
    {"france","germany","spain","italy","russia","china","japan","england","india","greece"},
    {"war","army","military","soldiers","battle","navy","troops","forces","enemy","weapons"},
    {"church","god","jesus","christian","catholic","bible","religious","faith","holy","priest"},
    {"music","song","band","album","jazz","guitar","songs","pop","musical","rock"},
    {"theory","energy","physics","mathematics","chemistry","quantum","particles","equation","mass","atoms"},
    {"blood","heart","brain","bone","muscle","skin","cells","body","disease","organs"}};
  vector<vector<int>> cat; for(auto&c:CATS){vector<int> v;for(auto&s:c){auto it=id.find(s);if(it!=id.end())v.push_back(it->second);}if(v.size()>=6)cat.push_back(v);}
  int C=cat.size();
  printf("=== SUBSTRATE VALUE: does the SEMANTIC graph beat random/identity? ===\n");
  printf("tokens=%d nodes=%d categories=%d (words/cat: ",TOK,N,C);
  for(auto&v:cat)printf("%zu ",v.size());printf(")\n\n");

  auto emrow=[&](int wd){vector<double> r(N);for(int j=0;j<N;j++)r[j]=emb[(size_t)wd*N+j];return r;};
  auto cosd=[&](const vector<double>&a,const vector<double>&b){double d=0,na=0,nb=0;for(int i=0;i<N;i++){d+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}return d/(sqrt(na*nb)+1e-15);};
  uniform_real_distribution<double> U(0,1); normal_distribution<double> G(0,1);
  // FLOW: 2-hop neighbour-sum on graph A, starting from per-word features X
  auto flow=[&](const vector<vector<int>>&A, vector<vector<double>> X){
    for(int hop=0;hop<2;hop++){vector<vector<double>> Y(N,vector<double>(N,0));
      for(int i=0;i<N;i++){Y[i]=X[i];for(int j:A[i])for(int t=0;t<N;t++)Y[i][t]+=X[j][t];}
      X.swap(Y);} return X;};

  // build corrupted features for ALL nodes (mask 85% dims + noise), per a fixed seed
  auto corrupt=[&](){vector<vector<double>> X(N,vector<double>(N,0));
    for(int i=0;i<N;i++)for(int j=0;j<N;j++){double v=emb[(size_t)i*N+j];
      if(U(rng)<0.85)v=0; v+=0.02*G(rng); X[i][j]=v;} return X;};
  vector<vector<double>> Xc=corrupt();
  auto Freal=flow(R,Xc), Frand=flow(RND,Xc);

  // train prototype = mean CLEAN ppmi of first half of each category; test = second half
  auto eval=[&](function<vector<double>(int)> rep){
    int ok=0,tot=0;
    vector<vector<double>> proto(C,vector<double>(N,0));
    for(int c=0;c<C;c++){int h=cat[c].size()/2;for(int t=0;t<h;t++){auto r=emrow(cat[c][t]);for(int j=0;j<N;j++)proto[c][j]+=r[j];}}
    for(int c=0;c<C;c++){int h=cat[c].size()/2;
      for(int t=h;t<(int)cat[c].size();t++){int wd=cat[c][t];auto r=rep(wd);
        int best=0;double bs=-2;for(int cc=0;cc<C;cc++){double s=cosd(r,proto[cc]);if(s>bs){bs=s;best=cc;}}
        ok+=(best==c);tot++;}}
    return 100.0*ok/tot;};

  printf("method                     | category accuracy (corrupted features)\n");
  printf("  PPMI-only (clean, ceiling)| %.1f%%\n", eval([&](int wd){return emrow(wd);}));
  printf("  identity (corrupted)      | %.1f%%\n", eval([&](int wd){return Xc[wd];}));
  printf("  FLOW on MATCHED-RANDOM    | %.1f%%\n", eval([&](int wd){return Frand[wd];}));
  printf("  FLOW on REAL SEMANTIC     | %.1f%%\n", eval([&](int wd){return Freal[wd];}));
  printf("\nIf REAL >> RANDOM and REAL recovers toward the PPMI ceiling, the semantic\ngraph is load-bearing -- not a generic unitary/aggregation effect.\n");
  return 0;
}
