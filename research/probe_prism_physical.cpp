// The physical prism with NO analytic harmonic input. We do not build plane
// waves and we do not predict with closed-form band formulas. We inject one IMPULSE
// (a delta -- one node set to 1), then STREAM the substrate's own propagation
// (driver in driver: outer = time steps, inner = the substrate step). A delta is
// broadband, so the substrate's dispersion fans it out in space all by itself.
// Everything is read FROM the streamed field: a finite aperture detector
// integrates probability current and density. One point is not a physical
// spectrometer for a delta impulse: within the fan many velocities interfere and
// density nodes make current/rho blow up. The aperture is the physical fix.
//   v_pos = (x - x0)/t                                 energy motion
//   v_cur = integral(current) / integral(rho)            from the phases only
// No analytic wave construction anywhere in the physics or the readout.
#include "../tools/graph_wave_substrate.hpp"
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <complex>
#include <vector>

int main(){
  const int N=300, x0=150; const double T=24.0, dt=0.3;
  gw::Graph g(N); for(int i=0;i+1<N;i++) g.addEdge(i,i+1,1.0);
  gw::Stepper U; U.build(g.h,dt);
  gw::Vec psi(N,gw::cd(0,0)); psi[x0]=gw::cd(1,0);     // IMPULSE -- no trig, no plane wave
  for(int s=0;s<(int)(T/dt);s++) psi=U.step(psi);      // STREAM the physics (driver in driver)

  std::printf("=== PHYSICAL PRISM from a streamed IMPULSE (no analytic wave input) ===\n");
  std::printf("delta injected, substrate propagated %g steps; spectrum read off the phases.\n\n",T/dt);
  std::printf("  aperture | v_pos=(x-x0)/t | v_cur=int(current)/int(rho) | match\n");
  // Sample finite detectors across the dispersed fan. The impulse response
  // carries many group velocities; a physical detector must average a small
  // local aperture instead of dividing by rho at one node.
  const int R=7;
  int ok=0,tot=0;
  for(int x=x0-44; x<=x0+44; x+=11){
    int lo=std::max(1,x-R), hi=std::min(N-2,x+R);
    double rho=0.0, cur=0.0, xpos=0.0;
    for(int y=lo;y<=hi;y++){
      double r=std::norm(psi[y]);
      rho+=r;
      xpos+=y*r;
      cur+=-2.0*std::imag(std::conj(psi[y])*psi[y+1]);
    }
    if(rho<1e-6) continue;
    double xc=xpos/rho;
    double vcom=(xc-x0)/T;
    double vcur=cur/(rho+1e-15);
    bool m=std::abs(vcom-vcur)<0.4; ok+=m; tot++;
    std::printf("  %4d..%-4d |    %+.3f      |        %+.3f             |  %s\n",lo,hi,vcom,vcur,m?"OK":"X");
  }
  std::printf("\n%d/%d aperture detectors: integrated phase-current velocity equals where the energy went.\n",ok,tot);
  std::printf("The substrate's own dispersion spread the impulse; we only read its phases.\n");
  std::printf("A delta carries all frequencies; the STREAM sorted them in space. No formula put in.\n");
  return (tot>=6 && ok==tot) ? 0 : 1;
}
