// graph_wave_learning_contract_test
// ----------------------------------------------------------------------------
// The last big piece: LEARNING without weights, without a trainer, without GA.
// A rule (here a threshold on a continuous feature) is acquired from labelled
// examples purely by holographic RECORDING:
//   - encode the feature with fractional-power encoding (similarity-preserving),
//   - record each labelled example by bundling it into its class prototype
//     (a superposition -- one-shot, online, additive),
//   - classify a new feature by nearest prototype (overlap); FPE similarity
//     makes it GENERALISE to unseen values, not just memorise.
// Learning = recording. No gradient, no optimiser, no parameters tuned.
//   [1] a learning curve: accuracy rises as more examples are recorded.
//   [2] it generalises to fresh, unseen features.
//   [3] a shuffled-label control collapses to chance (the structure is learned,
//       not faked).
// Substrate-only, deterministic. No V26, no weights, no trainer, no GA.
// ----------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>
namespace {
using cd = std::complex<double>; using Vec = std::vector<cd>;
constexpr int kD = 1024; constexpr int kCls = 2;
std::vector<double> phi(kD);
Vec enc(double x){Vec v(kD);for(int k=0;k<kD;k++)v[k]=std::exp(cd(0,x*phi[k]));return v;}
double nrm(const Vec&a){double s=0;for(auto&v:a)s+=std::norm(v);return std::sqrt(s);}
cd inr(const Vec&a,const Vec&b){cd s(0,0);for(int k=0;k<kD;k++)s+=std::conj(a[k])*b[k];return s;}
double overlap(const Vec&a,const Vec&b){double na=nrm(a),nb=nrm(b);return(na<1e-12)?0:std::abs(inr(a,b))/(na*nb);}
int label(double x){return x<5.0?0:1;}   // the rule to be learned (threshold)
bool report(const char*n,bool ok){std::printf("    => %s\n",ok?"PASS":"FAIL");if(!ok)std::printf("    !! %s\n",n);return ok;}
}
int main(){
  std::printf("=====================================================================\n");
  std::printf(" GRAPH-WAVE LEARNING CONTRACT TEST  (learn a rule by recording, D=%d)\n",kD);
  std::printf(" Prototypes = superposition of encoded examples. Learning = recording.\n");
  std::printf("=====================================================================\n");
  std::mt19937_64 rng(0x1EA59ULL);std::uniform_real_distribution<double>up(-3.14159,3.14159),ux(0,10);
  for(int k=0;k<kD;k++)phi[k]=up(rng);
  // fixed test set of fresh, unseen features
  const int kTest=200;std::vector<double>tx(kTest);for(auto&x:tx)x=ux(rng);
  auto trainEval=[&](int nTrain,bool shuffle){
    std::vector<Vec>proto(kCls,Vec(kD,cd(0,0)));std::vector<int>lab(nTrain);std::vector<double>xs(nTrain);
    for(int i=0;i<nTrain;i++){xs[i]=ux(rng);lab[i]=label(xs[i]);}
    if(shuffle){std::shuffle(lab.begin(),lab.end(),rng);}      // break the feature<->label structure
    for(int i=0;i<nTrain;i++){Vec e=enc(xs[i]);for(int k=0;k<kD;k++)proto[lab[i]][k]+=e[k];}
    int ok=0;for(double x:tx){Vec e=enc(x);int best=overlap(e,proto[0])>=overlap(e,proto[1])?0:1;if(best==label(x))ok++;}
    return 100.0*ok/kTest;
  };
  int pass=0,total=0;

  std::printf("\n[1] LEARNING CURVE: accuracy rises as more examples are recorded\n");
  {double a2=trainEval(4,false),a8=trainEval(8,false),a32=trainEval(32,false),a128=trainEval(128,false);
   std::printf("    examples=4 -> %.1f%%   8 -> %.1f%%   32 -> %.1f%%   128 -> %.1f%%\n",a2,a8,a32,a128);
   ++total;pass+=report("no learning curve",a128>a2&&a128>80.0);}

  std::printf("\n[2] GENERALISES TO FRESH, UNSEEN FEATURES\n");
  {double a=trainEval(128,false);
   std::printf("    accuracy on %d unseen test features = %.1f%%\n",kTest,a);
   ++total;pass+=report("did not generalise",a>85.0);}

  std::printf("\n[3] SHUFFLED-LABEL CONTROL COLLAPSES TO CHANCE (the rule is learned, not faked)\n");
  {double real=trainEval(128,false),shuf=trainEval(128,true);
   std::printf("    real labels=%.1f%%   shuffled labels=%.1f%% (~chance 50%%)\n",real,shuf);
   ++total;pass+=report("shuffled control did not collapse",real>85.0&&shuf<65.0);}

  std::printf("\n RESULT : %d / %d verified\n",pass,total);return pass==total?0:1;
}
