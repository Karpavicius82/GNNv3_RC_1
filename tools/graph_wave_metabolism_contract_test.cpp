// graph_wave_metabolism_contract_test
// ----------------------------------------------------------------------------
// First small step toward the living, self-closing GNN: the DRIVE (metabolism).
// The settling test proved that, left alone, a superposition dephases and dies
// (relaxes to equilibrium). A living physics stream must be a DISSIPATIVE
// STRUCTURE -- order sustained FAR from equilibrium by continuous drive
// (Prigogine; laser gain-loss; Stuart-Landau / Hopf limit cycle). Per-node
// saturable gain-loss: da/dt = (mu + i w) a - (1 + i c) |a|^2 a.
// [1] without drive (mu < 0) the field DIES (|a| -> 0).
// [2] with drive (mu > 0) the field SUSTAINS a steady nonzero amplitude
// indefinitely (alive, far from equilibrium).
// [3] HOMEOSTASIS: after a perturbation the amplitude returns to the same
// steady state -- self-regulating, self-maintaining.
// [4] the steady state is a LIVING oscillation (the phase keeps advancing;
// it is a limit cycle, not a dead fixed point).
// Substrate-only, deterministic., no weights, no trainer.
// ----------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
namespace {
using cd = std::complex<double>;
constexpr double kDt = 0.01, kW = 2.0, kC = 0.5;
// one Stuart-Landau step
cd sl(cd a, double mu){ return a + kDt*((cd(mu,kW))*a - cd(1.0,kC)*std::norm(a)*a); }
cd run(cd a, double mu, int steps){ for(int t=0;t<steps;t++) a=sl(a,mu); return a; }
bool report(const char*n,bool ok){ std::printf(" => %s\n", ok?"PASS":"FAIL"); if(!ok) std::printf(" !! %s\n",n); return ok; }
}
int main(){
 std::printf("=====================================================================\n");
 std::printf(" GRAPH-WAVE METABOLISM (DRIVE) CONTRACT TEST\n");
 std::printf(" A living stream is sustained far from equilibrium by drive, not dead.\n");
 std::printf("=====================================================================\n");
 int pass=0,total=0;

 std::printf("\n[1] WITHOUT DRIVE (mu<0) THE FIELD DIES\n");
 {cd a=run(cd(0.5,0.0),-0.3,4000);std::printf(" |a| after decay = %.6f (-> 0 = death)\n",std::abs(a));
 ++total;pass+=report("field did not die without drive",std::abs(a)<0.02);}

 std::printf("\n[2] WITH DRIVE (mu>0) THE FIELD SUSTAINS A STEADY STATE (alive)\n");
 {const double mu=0.5;cd a=run(cd(0.1,0.0),mu,6000);double amp=std::abs(a);
 cd a2=run(a,mu,6000);double amp2=std::abs(a2); // persists further
 std::printf(" |a| -> %.6f, still %.6f after 6000 more steps (continuum sqrt(mu)=%.4f; Euler-biased)\n",amp,amp2,std::sqrt(mu));
 ++total;pass+=report("field did not sustain a steady state",amp>0.5&&std::abs(amp2-amp)<1e-4);}

 std::printf("\n[3] HOMEOSTASIS: returns to its OWN steady state after a perturbation\n");
 {const double mu=0.5;cd a=run(cd(0.1,0.0),mu,6000);double steady=std::abs(a);
 cd down=run(a*0.2,mu,5000);double ad=std::abs(down); // kicked down
 cd up=run(a*3.0,mu,5000);double au=std::abs(up); // kicked up
 std::printf(" steady=%.6f after kick x0.2 -> %.6f after kick x3.0 -> %.6f\n",steady,ad,au);
 ++total;pass+=report("did not self-regulate back to steady state",std::abs(ad-steady)<1e-3&&std::abs(au-steady)<1e-3);}

 std::printf("\n[4] LIVING OSCILLATION: the steady state keeps turning (limit cycle, not dead)\n");
 {const double mu=0.5;cd a=run(cd(0.1,0.0),mu,6000);double amp0=std::abs(a);
 double phase=0;cd prev=a;for(int t=0;t<2000;t++){cd nx=sl(a,mu);phase+=std::arg(nx*std::conj(a));a=nx;}
 double amp1=std::abs(a);
 std::printf(" phase advanced %.2f rad over the window; |a| constant %.4f -> %.4f\n",phase,amp0,amp1);
 ++total;pass+=report("steady state is a dead fixed point, not a living cycle",std::abs(phase)>5.0&&std::abs(amp1-amp0)<0.01);}

 std::printf("\n RESULT : %d / %d verified\n",pass,total);return pass==total?0:1;
}
