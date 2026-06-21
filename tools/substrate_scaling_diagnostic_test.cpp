// substrate_scaling_diagnostic_test
// ----------------------------------------------------------------------------
// DIAGNOSTIC for the sanctioned substrate-candidate question. substrate_spectral
// _comparison found golden-flux (alpha=8/13) is a distinct spectral candidate but
// DEFERRED the Cantor/fractal claim ("finite graph diagnostic only"). This probe
// does the missing finite-size SCALING: it measures the box-counting dimension of
// the spectrum and the large-gap density at growing lattice sizes for plain /
// random-phase / golden-flux, and asks whether the golden-flux signature is
// SCALE-STABLE and distinct (supporting a critical/fractal substrate) or drifts
// toward a control. Pure spectral geometry; no measurement, decision, learning,
// or V26. The value is the data. (Jacobi eigensolver reused from the spectral
// comparison contract.)
// ----------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
#include "graph_wave_substrate.hpp"
namespace {
using namespace gw;
std::vector<double> spectrum(const Graph&g){return eigvals(g.h);}
double boxDim(std::vector<double> ev){std::sort(ev.begin(),ev.end());double lo=ev.front(),hi=ev.back(),W=hi-lo;if(W<1e-12)return 0;std::vector<double>x,y;for(int k=2;k<=7;k++){double eps=W/std::pow(2.0,k);long prev=-1,cnt=0;for(double v:ev){long b=(long)std::floor((v-lo)/eps);if(b!=prev){cnt++;prev=b;}}x.push_back(std::log(1.0/eps));y.push_back(std::log((double)cnt));}
  double mx=0,my=0;for(size_t i=0;i<x.size();i++){mx+=x[i];my+=y[i];}mx/=x.size();my/=y.size();double num=0,den=0;for(size_t i=0;i<x.size();i++){num+=(x[i]-mx)*(y[i]-my);den+=(x[i]-mx)*(x[i]-mx);}return den>0?num/den:0;}
int bigGaps(std::vector<double> ev,double frac){std::sort(ev.begin(),ev.end());double W=ev.back()-ev.front();int c=0;for(size_t i=1;i<ev.size();i++)if(ev[i]-ev[i-1]>frac*W)c++;return c;}
}
int main(){
  std::printf("=====================================================================\n");
  std::printf(" DIAGNOSTIC: substrate-candidate SPECTRAL SCALING (resolve the deferred fractal claim)\n");
  std::printf(" Does the golden-flux signature stay scale-stable & distinct as N grows?\n");
  std::printf("=====================================================================\n");
  const double a=8.0/13.0;
  std::printf("  size | substrate     | N    | box-dim | gaps>2%% | gaps>1%%\n");
  std::printf("  -----+---------------+------+---------+---------+--------\n");
  for(int w:{8,12,16}){
    Graph gs[3]={plain(w),randomPhase(w),golden(w,a)}; const char* nm[3]={"plain","random-phase","golden-flux"};
    for(int s=0;s<3;s++){auto ev=spectrum(gs[s]);
      std::printf("  %2dx%-2d| %-13s | %4d | %7.3f | %7d | %7d\n",w,w,nm[s],w*w,boxDim(ev),bigGaps(ev,0.02),bigGaps(ev,0.01));}
    std::printf("  -----+---------------+------+---------+---------+--------\n");
  }
  std::printf("\n DIAGNOSTIC FINDINGS: read the golden-flux box-dim/gaps across sizes --\n");
  std::printf("   scale-stable & between plain (extended) and random (disordered) => critical/fractal candidate;\n");
  std::printf("   drifting to a control => the deferred fractal claim stays unresolved.\n");
  std::printf("\n RESULT : DIAGNOSTIC (data gathered, not a pass/fail gate)\n");
  return 0;
}
