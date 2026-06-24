// BROAD wave-fidelity audit: does the substrate behave like a real wave across
// the FULL set of physical invariants, not just unitarity? Tested on plain, golden-
// flux, and directional-flux lattices:
//   1. norm conservation (unitarity)
//   2. ENERGY conservation  <H> constant under evolution
//   3. time reversal        U(-t)U(t) = I
//   4. CONTINUITY (local)   d|psi_i|^2/dt = - sum_j J_ij  (current conservation)
//   5. CAUSALITY / light cone: a delta spreads at bounded speed; beyond the front
//      the field is exponentially small. Also: dense Cayley (rational, has an
//      inverse) vs the strictly-local Chebyshev -- which respects the cone?
//   6. gauge invariance
#include "../tools/graph_wave_substrate.hpp"
#include <cstdio>
#include <cmath>
#include <complex>
#include <vector>
using namespace gw;

static double energy(const CMat& H, const Vec& z){ Vec Hz=matvec(H,z); cd e(0,0); for(size_t i=0;i<z.size();i++) e+=std::conj(z[i])*Hz[i]; return e.real(); }
static double pw(const Vec& z){ double s=0; for(auto&v:z)s+=std::norm(v); return s; }

int main(){
  std::printf("=== BROAD WAVE-FIDELITY AUDIT ===\n\n");
  struct Cfg{const char* nm; Graph g;};
  // directional-flux: Hermitian graph with asymmetric phase (like the text8 graph)
  auto fluxLat=[&](int w){ Graph g(w*w); auto id=[w](int x,int y){return y*w+x;};
    for(int y=0;y<w;y++)for(int x=0;x<w;x++){ if(x+1<w)g.addEdge(id(x,y),id(x+1,y),1.0,0.37); if(y+1<w)g.addEdge(id(x,y),id(x,y+1),1.0,-0.21);} return g; };
  std::vector<Cfg> cfgs={{"plain",plain(8)},{"golden",golden(8,8.0/13)},{"flux-dir",fluxLat(8)}};
  double dt=0.4;

  std::printf("%-10s | normDrift | energyDrift | timeRev | gaugeObs\n","substrate");
  for(auto&c:cfgs){ int N=c.g.size(); const CMat& H=c.g.h;
    Stepper Uf; Uf.build(H,dt); Stepper Ub; Ub.build(H,-dt);
    Vec psi(N,cd(0,0)); for(int i=0;i<N;i++) psi[i]=cd(std::sin(0.3*i+1),std::cos(0.2*i)); double n0=std::sqrt(pw(psi)); for(auto&v:psi)v/=n0;
    double E0=energy(H,psi); double ndrift=0,edrift=0; Vec p=psi;
    for(int s=0;s<200;s++){ p=Uf.step(p); ndrift=std::max(ndrift,std::abs(pw(p)-1.0)); edrift=std::max(edrift,std::abs(energy(H,p)-E0)); }
    // time reversal: forward 50 then backward 50
    Vec q=psi; for(int s=0;s<50;s++)q=Uf.step(q); for(int s=0;s<50;s++)q=Ub.step(q);
    double trev=0; for(int i=0;i<N;i++)trev=std::max(trev,std::abs(q[i]-psi[i]));
    // gauge invariance of |psi_i|^2 after evolution
    std::vector<double> th(N); for(int i=0;i<N;i++)th[i]=std::sin(0.11*i)*2.3;
    CMat Hg=H; for(int i=0;i<N;i++)for(int j=0;j<N;j++)Hg[i][j]*=std::exp(cd(0,th[i]-th[j]));
    Stepper Ug; Ug.build(Hg,dt); Vec pg=psi; for(int i=0;i<N;i++)pg[i]*=std::exp(cd(0,th[i]));
    Vec a=psi,b=pg; for(int s=0;s<50;s++){a=Uf.step(a);b=Ug.step(b);} double gd=0; for(int i=0;i<N;i++)gd=std::max(gd,std::abs(std::norm(a[i])-std::norm(b[i])));
    std::printf("%-10s | %.1e   | %.1e     | %.1e | %.1e\n",c.nm,ndrift,edrift,trev,gd);
  }

  // CONTINUITY: d p_i/dt = -sum_j J_ij  on a propagating packet (plain 2D)
  std::printf("\n[continuity] d|psi_i|^2/dt  vs  -div J  (should match):\n");
  { Graph g=plain(10); int N=g.size(); const CMat&H=g.h; auto id=[](int x,int y){return y*10+x;};
    Vec psi(N,cd(0,0)); for(int y=0;y<10;y++)for(int x=0;x<10;x++){double r2=(x-4.5)*(x-4.5)+(y-4.5)*(y-4.5);psi[id(x,y)]=std::exp(cd(-r2/3.0,0.6*x));}
    double n0=std::sqrt(pw(psi));for(auto&v:psi)v/=n0; double h=1e-4; Stepper Up;Up.build(H,h);Stepper Um;Um.build(H,-h);
    Vec p2=Up.step(psi),p0=Um.step(psi); double maxerr=0;  // central difference -> O(h^2)
    for(int i=0;i<N;i++){ double dpdt=(std::norm(p2[i])-std::norm(p0[i]))/(2*h);
      double sumJ=0; for(int j=0;j<N;j++) if(std::abs(H[i][j])>1e-15) sumJ+=2.0*std::imag(std::conj(psi[i])*H[i][j]*psi[j]);
      maxerr=std::max(maxerr,std::abs(dpdt-sumJ)); }   // continuity: dp_i/dt = sum_j 2 Im(conj psi_i H_ij psi_j)
    std::printf("  max | dp_i/dt - sum_j J_ij | = %.2e  (continuity)\n",maxerr);
  }

  // CAUSALITY / light cone on a 1D chain: delta at 0, who lights up after time t?
  std::printf("\n[causality] 1D chain N=60, delta at node 0, group velocity v~2:\n");
  { int N=60; Graph g(N); for(int i=0;i+1<N;i++)g.addEdge(i,i+1,1.0); const CMat&H=g.h;
    double T=5.0; // front should reach ~ v*T = 10
    // many small Cayley steps (good approx to e^{-iHt})
    Stepper Us; Us.build(H,0.05); Vec p(N,cd(0,0)); p[0]=cd(1,0); for(int s=0;s<(int)(T/0.05);s++)p=Us.step(p);
    // one large Cayley step of the same total time (rational approx -> nonlocal inverse)
    Stepper U1; U1.build(H,T); Vec q(N,cd(0,0)); q[0]=cd(1,0); q=U1.step(q);
    std::printf("  node:        d=8     d=12    d=20    d=40\n");
    std::printf("  many-step |psi|: %.2e %.2e %.2e %.2e\n",std::abs(p[8]),std::abs(p[12]),std::abs(p[20]),std::abs(p[40]));
    std::printf("  1-step    |psi|: %.2e %.2e %.2e %.2e\n",std::abs(q[8]),std::abs(q[12]),std::abs(q[20]),std::abs(q[40]));
    std::printf("  (front ~ d=10; beyond it should be tiny. 1-step inverse may leak instantly = causality)\n");
  }
  std::printf("\nFaithful wave iff: norm/energy conserved, time-reversible, continuity ~machine,\ngauge-invariant, AND a real light cone (no instantaneous far-field).\n");
  return 0;
}
