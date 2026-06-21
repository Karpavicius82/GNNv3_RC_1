// graph_wave_autopoiesis_contract_test
// ----------------------------------------------------------------------------
// Third small step: the AUTOPOIETIC KERNEL -- drive and plasticity coupled on
// ONE persistent field, run continuously as the two-equation living system:
//   psi_i. = (mu - |psi_i|^2) psi_i  +  kappa * sum_j H_ij psi_j   (+ input)
//   H_ij.  = eta * Re(psi_i conj psi_j)  -  decay * H_ij
// The fast equation computes/sustains (metabolism + propagation), the slow one
// learns (the medium self-organises). After the input STOPS, the metabolism +
// the self-organised medium SUSTAIN the learned pattern with no input -- the
// system produces the conditions for its own continuation. That is autopoiesis.
//   [1] the field stays ALIVE throughout (bounded, nonzero) -- metabolism runs.
//   [2] the medium SELF-ORGANISES the driven community while alive.
//   [3] after the input stops the field STAYS ALIVE (self-sustaining, no input).
//   [4] it sustains the LEARNED pattern: the driven community stays active, the
//       rest stays dark -- the structure maintains itself.
// Substrate-only, deterministic. No V26, no weights, no trainer.
// ----------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>
namespace {
using cd = std::complex<double>; using Vec = std::vector<cd>; using Mat = std::vector<std::vector<double>>;
constexpr int kN = 8;            // group A = {0..3} driven, group B = {4..7}
constexpr double kMu = 0.5, kKappa = 0.04, kEta = 0.02, kDecay = 0.01, kDt = 0.02;
double energy(const Vec& p, int lo, int hi){ double s=0; for(int i=lo;i<hi;i++) s+=std::norm(p[i]); return s; }
double withinCross(const Mat& H, bool within){ double s=0;int n=0; for(int i=0;i<kN;i++)for(int j=i+1;j<kN;j++){bool same=((i<4)==(j<4)); if(same==within){s+=std::abs(H[i][j]);n++;}} return n?s/n:0; }
bool report(const char*n,bool ok){std::printf("    => %s\n",ok?"PASS":"FAIL");if(!ok)std::printf("    !! %s\n",n);return ok;}
}
int main(){
  std::printf("=====================================================================\n");
  std::printf(" GRAPH-WAVE AUTOPOIESIS CONTRACT TEST  (drive + plasticity, one field)\n");
  std::printf(" Two coupled equations: the field computes/lives, the medium learns.\n");
  std::printf("=====================================================================\n");
  Vec psi(kN, cd(0,0)); Mat H(kN, std::vector<double>(kN,0.0));
  const int kInput=3000, kFree=4000;
  double minAliveDuringInput=1e9, minAliveFree=1e9;
  for(int t=0; t<kInput+kFree; t++){
    bool inputting = (t<kInput);
    // fast equation: metabolism (saturable drive) + propagation through the medium
    Vec next(kN);
    for(int i=0;i<kN;i++){
      cd couple(0,0); for(int j=0;j<kN;j++) couple += H[i][j]*psi[j];
      cd d = (kMu - std::norm(psi[i]))*psi[i] + kKappa*couple;
      next[i] = psi[i] + kDt*d;
      if(inputting && i<4) next[i] += 0.05;            // drive only group A with input
    }
    psi = next;
    // slow equation: plasticity (the medium self-organises from field correlations)
    for(int i=0;i<kN;i++) for(int j=i+1;j<kN;j++){
      double upd = H[i][j] + kDt*(kEta*std::real(psi[i]*std::conj(psi[j])) - kDecay*H[i][j]);
      H[i][j]=upd; H[j][i]=upd;
    }
    double tot = energy(psi,0,kN);
    if(inputting){ if(t>400) minAliveDuringInput=std::min(minAliveDuringInput,tot); }  // skip birth/startup
    else if(t>kInput+200) minAliveFree=std::min(minAliveFree,tot);   // after transient
  }
  int pass=0,total=0;

  std::printf("\n[1] THE FIELD STAYS ALIVE WHILE LEARNING (metabolism runs)\n");
  {std::printf("    min total energy during input phase = %.4f (>0 = never died)\n",minAliveDuringInput);
   ++total;pass+=report("field died during the run",minAliveDuringInput>0.1);}

  std::printf("\n[2] THE MEDIUM SELF-ORGANISED THE DRIVEN COMMUNITY\n");
  {double win=withinCross(H,true),cross=withinCross(H,false);
   std::printf("    within-group coupling=%.4f   cross-group=%.4f   ratio=%.1fx\n",win,cross,cross>1e-9?win/cross:1e9);
   ++total;pass+=report("medium did not self-organise",win>5.0*std::max(1e-6,cross));}

  std::printf("\n[3] AFTER INPUT STOPS THE FIELD STAYS ALIVE (self-sustaining, no input)\n");
  {std::printf("    min total energy in the free (no-input) phase = %.4f\n",minAliveFree);
   ++total;pass+=report("field decayed once input stopped",minAliveFree>0.1);}

  std::printf("\n[4] IT SUSTAINS THE LEARNED PATTERN (driven community stays lit, rest dark)\n");
  {double eA=energy(psi,0,4),eB=energy(psi,4,kN);
   std::printf("    final energy: learned community A=%.4f   community B=%.4f\n",eA,eB);
   ++total;pass+=report("did not sustain the learned structure",eA>0.5&&eB<0.05);}

  std::printf("\n RESULT : %d / %d verified\n",pass,total);return pass==total?0:1;
}
