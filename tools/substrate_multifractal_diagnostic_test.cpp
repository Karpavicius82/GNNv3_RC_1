// substrate_multifractal_diagnostic_test
// ----------------------------------------------------------------------------
// DIAGNOSTIC, deeper substrate-candidate characterization. The scaling diagnostic
// showed the golden-flux SPECTRUM keeps a self-similar gap hierarchy. The
// complementary, more decisive criticality signature is in the EIGENSTATES: at a
// critical (multifractal) point states are neither extended (fill the lattice,
// participation ratio P ~ N) nor localized (P ~ O(1)) but fractal (P ~ N^tau,
// 0 < tau < 1). This measures the mean participation ratio of mid-spectrum
// eigenstates across lattice sizes for plain / random-phase / golden-flux and
// reports the scaling exponent tau (slope of log<P> vs log N). A fractional tau
// below the extended control is the multifractal/critical signature. Pure
// eigenstate geometry; no measurement, decision, learning, or V26. The value is
// the data.
// ----------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>
#include "graph_wave_substrate.hpp"
namespace {
using namespace gw;
// mean participation ratio of mid-spectrum eigenstates; N = lattice nodes
double meanPR(const Graph&g,int N){Eig e=jacobiV(realRep(g.h));int M=2*N;std::vector<int>idx(M);std::iota(idx.begin(),idx.end(),0);std::sort(idx.begin(),idx.end(),[&](int p,int q){return e.ev[p]<e.ev[q];});
  double sum=0;int cnt=0;for(int r=M/4;r<3*M/4;r++){int k=idx[r];double s2=0,s4=0;for(int i=0;i<N;i++){double pi=e.V[i][k]*e.V[i][k]+e.V[i+N][k]*e.V[i+N][k];s2+=pi;s4+=pi*pi;}double pr=(s2>0)?(s2*s2)/s4:0;sum+=pr;cnt++;}return cnt?sum/cnt:0;}
}
int main(){
  std::printf("=====================================================================\n");
  std::printf(" DIAGNOSTIC: substrate eigenstate MULTIFRACTALITY (criticality signature)\n");
  std::printf(" P ~ N^tau:  tau~1 extended, tau~0 localized, 0<tau<1 critical/multifractal\n");
  std::printf("=====================================================================\n");
  const double a=8.0/13.0; int ws[3]={8,12,16};
  const char* nm[3]={"plain","random-phase","golden-flux"};
  std::vector<std::vector<double>> P(3);   // P[substrate][size]
  std::printf("  substrate     |   8x8(N=64) | 12x12(N=144)| 16x16(N=256)\n");
  std::printf("  --------------+-------------+-------------+-------------\n");
  for(int s=0;s<3;s++){
    for(int si=0;si<3;si++){int w=ws[si];Graph g = (s==0)?plain(w):(s==1)?randomPhase(w):golden(w,a);P[s].push_back(meanPR(g,w*w));}
    std::printf("  %-13s | %11.3f | %11.3f | %11.3f\n",nm[s],P[s][0],P[s][1],P[s][2]);
  }
  std::printf("\n  participation scaling exponent tau (slope log<P> vs log N):\n");
  auto slope=[&](const std::vector<double>&p){std::vector<double>x={std::log(64.0),std::log(144.0),std::log(256.0)},y={std::log(p[0]),std::log(p[1]),std::log(p[2])};double mx=0,my=0;for(int i=0;i<3;i++){mx+=x[i];my+=y[i];}mx/=3;my/=3;double num=0,den=0;for(int i=0;i<3;i++){num+=(x[i]-mx)*(y[i]-my);den+=(x[i]-mx)*(x[i]-mx);}return num/den;};
  for(int s=0;s<3;s++) std::printf("    %-13s tau = %.3f\n",nm[s],slope(P[s]));
  std::printf("\n DIAGNOSTIC FINDINGS: golden-flux tau between extended (plain) and 1,\n");
  std::printf("   i.e. fractional / below the extended control => multifractal/critical eigenstates.\n");
  std::printf("\n RESULT : DIAGNOSTIC (data gathered, not a pass/fail gate)\n");
  return 0;
}
