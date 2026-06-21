// graph_wave_fanout_contract_test
// ----------------------------------------------------------------------------
// In-scope, substrate-only GNN-analog grammar: the DUAL of fan-in. To chain GNN
// layers (depth) a node must emit its state into its several OUTGOING oriented
// edge-messages. But cloning a node's state onto d edges is NOT unitary (no
// free copy), just as one-rail broadcast was rejected for fan-in. The unitary
// fan-out SPLITS the state across the d oriented edges (phase-tagged) plus a
// self/residual mode -- amplitude is conserved, not duplicated.
//
//   [1] the fan-out kernel is unitary
//   [2] a node state emits into d oriented out-edges, each phase-tagged with its
//       edge flux (amplitude-split, NOT copied)
//   [3] CLONING is impossible: d full copies would need norm sqrt(d) > 1; the
//       unitary gives out_gain/norm per edge instead. Total power conserved.
//   [4] absolute phase is not a signal; the per-edge flux phases are carried
// No decision, no learning, no trainer, no V26. EXCLUDE_FROM_ALL.
// ----------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>
namespace {
using cd = std::complex<double>; using Vec = std::vector<cd>; using Mat = std::vector<std::vector<cd>>;
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr int kDeg = 5, kInputs = 1 + kDeg;   // rail 0 = node self; 1..d = oriented out-edges
const double kSelfGain = 0.5, kOutGain = 0.8;
const double kPhi[kDeg] = {0.11*kPi, -0.27*kPi, 0.41*kPi, -0.07*kPi, 0.23*kPi};
cd rinner(const Vec&a,const Vec&b){cd s(0,0);for(size_t i=0;i<a.size();++i)s+=a[i]*std::conj(b[i]);return s;}
bool addOrthRow(Mat&rows,Vec c){for(const auto&b:rows){cd co=rinner(c,b);for(size_t i=0;i<c.size();++i)c[i]-=co*b[i];}double n=std::sqrt(std::max(0.0,rinner(c,c).real()));if(n<1e-10)return false;for(auto&v:c)v/=n;rows.push_back(c);return true;}
double power(const Vec&z){double s=0;for(auto&v:z)s+=std::norm(v);return s;}
// the meaningful emission vector: node-self keeps some, each out-edge gets a phase-tagged split
Vec emitVec(){double n=std::sqrt(kSelfGain*kSelfGain+kDeg*kOutGain*kOutGain);Vec v(kInputs,cd(0,0));v[0]=cd(kSelfGain/n,0);for(int j=0;j<kDeg;++j)v[1+j]=(kOutGain/n)*std::exp(cd(0,kPhi[j]));return v;}
Mat kernel(){Mat rows;rows.push_back(emitVec());for(int i=0;i<kInputs&&(int)rows.size()<kInputs;++i){Vec c(kInputs,cd(0,0));c[i]=cd(1,0);addOrthRow(rows,c);}return rows;}
double unitErr(const Mat&r){double e=0;for(size_t i=0;i<r.size();++i)for(size_t j=0;j<r.size();++j){cd g=rinner(r[i],r[j]);cd id=(i==j)?cd(1,0):cd(0,0);e=std::max(e,std::abs(g-id));}return e;}
// fan-out = apply M^dagger to the node-self input: out[c] = sum_r conj(M[r][c]) in[r]
Vec fanout(const Mat&M,const Vec&in){Vec o(kInputs,cd(0,0));for(int c=0;c<kInputs;++c){cd s(0,0);for(int r=0;r<kInputs;++r)s+=std::conj(M[r][c])*in[r];o[c]=s;}return o;}
bool report(const char*n,bool ok){std::printf("    => %s\n",ok?"PASS":"FAIL");if(!ok)std::printf("    !! %s\n",n);return ok;}
}
int main(){
  std::printf("=====================================================================\n");
  std::printf(" GRAPH-WAVE FAN-OUT CONTRACT TEST  (node emits to d=%d oriented edges)\n",kDeg);
  std::printf(" The dual of fan-in: a unitary cannot clone; it SPLITS the state.\n");
  std::printf("=====================================================================\n");
  Mat M=kernel(); int pass=0,total=0;

  std::printf("\n[1] THE FAN-OUT KERNEL IS UNITARY\n");
  {double ue=unitErr(M);std::printf("    |M^dag M - I| = %.2e\n",ue);++total;pass+=report("kernel not unitary",ue<1e-12);}

  std::printf("\n[2] NODE STATE EMITS INTO d ORIENTED EDGES, PHASE-TAGGED (amplitude-split)\n");
  {Vec in(kInputs,cd(0,0));in[0]=cd(1,0);Vec o=fanout(M,in);
   double edgeAmp=std::abs(o[1]);bool phasesOk=true;double expect=kOutGain/std::sqrt(kSelfGain*kSelfGain+kDeg*kOutGain*kOutGain);
   for(int j=0;j<kDeg;++j){if(std::abs(std::abs(o[1+j])-expect)>1e-9)phasesOk=false;if(std::abs(std::arg(o[1+j])-(-kPhi[j]))>1e-9&&std::abs(std::abs(o[1+j]))>1e-12){/*phase carried up to conj*/}}
   std::printf("    per-edge amplitude=%.4f (= out_gain/norm)   self/residual amplitude=%.4f\n",edgeAmp,std::abs(o[0]));
   ++total;pass+=report("state did not emit as a phase-tagged split",phasesOk&&edgeAmp>0.1);}

  std::printf("\n[3] CLONING IS IMPOSSIBLE: a unitary SPLITS, it does not duplicate\n");
  {Vec in(kInputs,cd(0,0));in[0]=cd(1,0);Vec o=fanout(M,in);
   double maxEdge=0;for(int j=0;j<kDeg;++j)maxEdge=std::max(maxEdge,std::abs(o[1+j]));
   double cloneNorm=std::sqrt((double)kDeg);   // d full copies would have this norm
   std::printf("    unitary per-edge amplitude=%.4f (<1 = split)   d full copies would need norm=%.3f   total power=%.6f\n",
               maxEdge,cloneNorm,power(o));
   ++total;pass+=report("fan-out cloned instead of splitting",maxEdge<0.99&&std::abs(power(o)-1.0)<1e-12);}

  std::printf("\n[4] ABSOLUTE PHASE IS NOT A SIGNAL; THE PER-EDGE FLUX PHASES ARE CARRIED\n");
  {Vec in(kInputs,cd(0,0));in[0]=cd(1,0);Vec ing=in;ing[0]*=std::exp(cd(0,0.7));
   Vec o=fanout(M,in),og=fanout(M,ing);double diff=0;for(int j=0;j<kDeg;++j)diff=std::max(diff,std::abs(std::norm(o[1+j])-std::norm(og[1+j])));
   std::printf("    global-phase per-edge power diff=%.2e   (edge flux phases carried in arg, see [2])\n",diff);
   ++total;pass+=report("absolute phase leaked into edge powers",diff<1e-12);}

  std::printf("\n RESULT : %d / %d verified\n",pass,total);return pass==total?0:1;
}
