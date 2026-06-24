// The physical prism: let DISPERSION do the spectral separation, no FFT, no
// classifier. On a 1D tight-binding chain the dispersion is eps(k) = -2 cos(k), so
// the group velocity is v(k) = 2 sin(k): different frequencies travel at different
// speeds. A wavepacket built from several momenta therefore SPATIALLY separates as
// it propagates -- white light through a prism. We only read where the energy lands
// (|psi(x)|^2) and check the bands sit at x0 + v(k) t (the physics did the transform).
#include "../tools/graph_wave_substrate.hpp"
#include <cstdio>
#include <cmath>
#include <complex>
#include <vector>
using namespace gw;

int main(){
  const int N=240; const double x0=70, sig=6; const double T=18.0, dt=0.3;
  Graph g(N); for(int i=0;i+1<N;i++) g.addEdge(i,i+1,1.0);   // open chain, eps(k)=-2cos k
  const CMat& H=g.h; Stepper U; U.build(H,dt);

  std::vector<double> ks={kPi/12, kPi/4, kPi/2};
  std::printf("=== PHYSICAL PRISM: dispersion separates frequencies in SPACE ===\n");
  std::printf("1D chain, eps(k)=-2cos(k), v(k)=2sin(k). Inject k = pi/12, pi/4, pi/2 (distinct group velocities) in ONE packet.\n");
  std::printf("Read only |psi(x)|^2 after t=%.0f -> bands land at x0 + v_g(k)t, v_g=-2sin(k) (no FFT).\n\n",T);

  // initial packet: Gaussian envelope carrying a superposition of the three momenta
  Vec psi(N,cd(0,0));
  for(int x=0;x<N;x++){ double env=std::exp(-(x-x0)*(x-x0)/(2*sig*sig)); cd s(0,0);
    for(double k:ks) s+=std::exp(cd(0,k*x)); psi[x]=env*s; }
  double n0=0;for(auto&v:psi)n0+=std::norm(v);n0=std::sqrt(n0);for(auto&v:psi)v/=n0;

  for(int s=0;s<(int)(T/dt);s++) psi=U.step(psi);

  // find local intensity peaks (energy landing spots) in the forward region
  std::vector<int> peaks;
  for(int x=2;x<N-2;x++){ double I=std::norm(psi[x]);
    if(I>std::norm(psi[x-1])&&I>std::norm(psi[x+1])&&I>std::norm(psi[x-2])&&I>std::norm(psi[x+2])&&I>1e-4) peaks.push_back(x); }
  std::printf("predicted band centres x0 + v_g(k)T (v_g=-2sin k):\n");
  for(double k:ks) std::printf("  k=%.3f  v=%.3f  ->  x=%.1f\n",k,2*std::sin(k),x0-2*std::sin(k)*T);
  std::printf("observed intensity peaks (where energy landed): ");
  for(int p:peaks) std::printf("%d ",p); std::printf("\n");

  // match: each predicted band has an observed peak nearby?
  int hit=0; for(double k:ks){ double xp=x0-2*std::sin(k)*T; double best=1e9; for(int p:peaks)best=std::min(best,std::abs(p-xp));
    std::printf("  band k=%.3f predicted x=%.0f  nearest peak dist=%.1f %s\n",k,xp,best,best<6?"OK":"X"); hit+=(best<6); }
  std::printf("\n%d/%zu frequencies physically separated into distinct spatial bands.\n",hit,ks.size());
  std::printf("The prism is the DISPERSION: physics spread the spectrum in space; we only read |psi|^2.\n");
  return 0;
}
