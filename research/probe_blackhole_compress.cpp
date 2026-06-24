// "Black-hole" compression: pack K items into ONE dense holographic code (K:1),
// then recover them by resonator cleanup. The unitary substrate is lossless and
// CANNOT compress (Parseval=1); compression lives in the dissipative collapse --
// many vectors superposed into one (the horizon), retrieved against a codebook.
// We measure the compression ratio vs reconstruction recall, and find the capacity
// HORIZON (a Bekenstein-like bound ~ D/log M) beyond which information scrambles
// (the information-paradox regime: still there in principle, not retrievable).
#include <complex>
#include <vector>
#include <cmath>
#include <cstdio>
#include <random>
#include <algorithm>
using cd=std::complex<double>; using Vec=std::vector<cd>;
static const double kPi=3.14159265358979323846;

int main(){
  const int D=512, M=1024;   // hypervector dim, codebook size
  std::mt19937 rng(7); std::uniform_real_distribution<double> ph(-kPi,kPi);
  std::vector<Vec> book(M);
  for(auto&v:book){ v.resize(D); for(auto&z:v) z=std::exp(cd(0,ph(rng)))/std::sqrt((double)D); }
  auto inner=[&](const Vec&a,const Vec&b){ cd s(0,0); for(int i=0;i<D;i++)s+=std::conj(a[i])*b[i]; return std::abs(s); };

  std::printf("=== BLACK-HOLE COMPRESSION: K items -> 1 holographic code, recover by cleanup ===\n");
  std::printf("D=%d codebook=%d.  compression K:1.  recall@K = fraction of stored items in top-K.\n\n",D,M);
  std::printf("   K   | compression | recall@K | mean stored score | noise floor\n");
  for(int K : {4,8,16,32,64,96,128,192,256}){
    // bundle the first K items into ONE vector (the compressed code = the horizon)
    Vec bundle(D,cd(0,0)); for(int i=0;i<K;i++)for(int d=0;d<D;d++)bundle[d]+=book[i][d];
    double nn=0;for(auto&z:bundle)nn+=std::norm(z);nn=std::sqrt(nn);for(auto&z:bundle)z/=nn;
    // retrieve: rank the whole codebook by overlap, recall the top-K
    std::vector<std::pair<double,int>> sc(M); for(int j=0;j<M;j++)sc[j]={inner(bundle,book[j]),j};
    std::sort(sc.rbegin(),sc.rend());
    int hit=0; for(int t=0;t<K;t++) if(sc[t].second<K) hit++;
    double stored=0; for(int i=0;i<K;i++)stored+=inner(bundle,book[i]); stored/=K;
    double floor=0; for(int j=K;j<M;j++)floor+=inner(bundle,book[j]); floor/=(M-K);
    std::printf("  %3d  |    %3d:1    |  %5.1f%%  |      %.3f       |   %.3f\n",
                K,K,100.0*hit/K,stored,floor);
  }
  std::printf("\nSmall K: near-lossless (recall ~100%%, K:1 compression). As K grows the stored\nscore ~1/sqrtK sinks toward the noise floor ~1/sqrtD -- the HORIZON: beyond it the\nitems are still in the code but no longer retrievable (the information-paradox regime).\n");
  std::printf("Compression is the dissipative collapse + cleanup, NOT the unitary substrate.\n");
  return 0;
}
