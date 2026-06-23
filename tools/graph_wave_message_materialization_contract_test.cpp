// graph_wave_message_materialization_contract_test
// ----------------------------------------------------------------------------
// Sanctioned, substrate-only GNN-analog GRAMMAR contract (per the checkpoint:
// "degree greater than one requires explicit edge-message rails or a separately
// tested unitary message-materialization contract"). Generalises the degree-2
// path to a hub node of arbitrary degree d, with SUPERPOSITION central: a
// superposition of the d oriented incoming edge-messages aggregates at the hub
// by PHASE-CONTROLLED INTERFERENCE, and the unitarity machinery (orthonormal
// rows + residual modes) exists only to SERVE that superposition (conserve it).
//
// Technique (same as degree2): each node-update is a normalised row over rails
// (self-rail + one oriented in-edge rail per neighbour); the rows must be
// mutually orthogonal (broadcast fails this); complete to a unitary with
// residual modes by Gram-Schmidt.
//
// [1] the materialisation kernel is unitary (a valid physical transfer)
// [2] a SUPERPOSITION of the d messages aggregates by phase interference:
// aligned -> constructive (hub port full), anti-aligned -> destructive (~0)
// [3] residual modes CONSERVE the superposition (total power = 1; the cancelled
// power goes to residual modes -- no false unitary compression)
// [4] absolute phase is not a signal; the gauge-invariant relative phases (the
// flux content) are what control the aggregation
// No decision, no learning, no trainer.
// ----------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>
namespace {
using cd = std::complex<double>; using Vec = std::vector<cd>; using Mat = std::vector<std::vector<cd>>;
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr int kDeg = 5; // hub degree
constexpr int kInputs = 1 + kDeg; // rail 0 = hub_self, rails 1..d = oriented in-edges
const double kSelfGain = 1.0, kInGain = 0.8;
const double kPhi[kDeg] = {0.17*kPi, -0.31*kPi, 0.43*kPi, -0.07*kPi, 0.23*kPi}; // edge flux phases

cd rinner(const Vec& a, const Vec& b){ cd s(0,0); for (size_t i=0;i<a.size();++i) s += a[i]*std::conj(b[i]); return s; }
bool addOrthogonalRow(Mat& rows, Vec cand){
 for (const auto& bvec : rows){ cd c = rinner(cand,bvec); for (size_t i=0;i<cand.size();++i) cand[i]-=c*bvec[i]; }
 double n = std::sqrt(std::max(0.0, rinner(cand,cand).real())); if (n<1e-10) return false;
 for (auto& v:cand) v/=n; rows.push_back(cand); return true;
}
Vec hubRow(){
 double n = std::sqrt(kSelfGain*kSelfGain + kDeg*kInGain*kInGain);
 Vec row(kInputs, cd(0,0)); row[0]=cd(kSelfGain/n,0);
 for (int i=0;i<kDeg;++i) row[1+i] = (kInGain/n)*std::exp(cd(0,kPhi[i]));
 return row;
}
Mat kernelRows(){
 Mat rows; rows.push_back(hubRow());
 for (int i=0;i<kInputs && (int)rows.size()<kInputs;++i){ Vec c(kInputs,cd(0,0)); c[i]=cd(1,0); addOrthogonalRow(rows,c); }
 return rows; // kInputs orthonormal rows
}
// output port r = <row_r | input>
Vec applyK(const Mat& rows, const Vec& x){ Vec o(rows.size()); for (size_t r=0;r<rows.size();++r) o[r]=rinner(x,rows[r]); return o; }
double unitarityErr(const Mat& rows){ double e=0; for (size_t i=0;i<rows.size();++i) for (size_t j=0;j<rows.size();++j){ cd g=rinner(rows[i],rows[j]); cd id=(i==j)?cd(1,0):cd(0,0); e=std::max(e,std::abs(g-id)); } return e; }
double power(const Vec& z){ double s=0; for (auto& v:z) s+=std::norm(v); return s; }
void normalize(Vec& z){ double n=std::sqrt(power(z)); for (auto& v:z) v/=n; }
bool report(const char*n,bool ok){std::printf(" => %s\n",ok?"PASS":"FAIL");if(!ok)std::printf(" !! %s\n",n);return ok;}
}
int main(){
 std::printf("=====================================================================\n");
 std::printf(" GRAPH-WAVE MESSAGE-MATERIALIZATION CONTRACT TEST (hub degree d=%d)\n",kDeg);
 std::printf(" A superposition of d messages aggregates by phase-controlled interference.\n");
 std::printf("=====================================================================\n");
 Mat K = kernelRows();
 int pass=0,total=0;

 std::printf("\n[1] THE MATERIALIZATION KERNEL IS UNITARY (valid physical transfer)\n");
 {double ue=unitarityErr(K);std::printf(" rows orthonormal, |K^dag K - I| = %.2e\n",ue);
 ++total;pass+=report("kernel not unitary",ue<1e-12);}

 std::printf("\n[2] A SUPERPOSITION OF MESSAGES AGGREGATES BY PHASE-CONTROLLED INTERFERENCE\n");
 {Vec aligned(kInputs,cd(0,0)), anti(kInputs,cd(0,0));
 for (int i=0;i<kDeg;++i){ aligned[1+i]=std::exp(cd(0,kPhi[i])); // matches the flux -> constructive
 anti[1+i] =std::exp(cd(0,kPhi[i]))*std::exp(cd(0,2.0*kPi*(i+1)/kDeg)); } // 5th-roots twist -> destructive
 normalize(aligned); normalize(anti);
 Vec oa=applyK(K,aligned), on=applyK(K,anti);
 double hubA=std::norm(oa[0]), hubN=std::norm(on[0]);
 std::printf(" hub port power: aligned superposition=%.6f anti-aligned=%.2e\n",hubA,hubN);
 ++total;pass+=report("superposition did not aggregate by phase interference",hubA>0.5&&hubN<1e-9);}

 std::printf("\n[3] RESIDUAL MODES CONSERVE THE SUPERPOSITION (no false compression)\n");
 {Vec anti(kInputs,cd(0,0)); for (int i=0;i<kDeg;++i) anti[1+i]=std::exp(cd(0,kPhi[i]))*std::exp(cd(0,2.0*kPi*(i+1)/kDeg)); normalize(anti);
 Vec out=applyK(K,anti); double tot=power(out), hub=std::norm(out[0]), resid=tot-hub;
 std::printf(" anti case: total power=%.6f hub port=%.2e residual modes=%.6f\n",tot,hub,resid);
 ++total;pass+=report("residual modes did not conserve the superposition",std::abs(tot-1.0)<1e-12&&resid>0.99);}

 std::printf("\n[4] ABSOLUTE PHASE IS NOT A SIGNAL; THE RELATIVE FLUX PHASES ARE\n");
 {Vec x(kInputs,cd(0,0)); for (int i=0;i<kDeg;++i) x[1+i]=std::exp(cd(0,kPhi[i])); normalize(x);
 Vec xg=x; for (auto& v:xg) v*=std::exp(cd(0,0.7)); // global phase
 Vec ox=applyK(K,x), oxg=applyK(K,xg);
 double diff=std::abs(std::norm(ox[0])-std::norm(oxg[0]));
 // relative-phase dependence already shown by [2] aligned != anti
 std::printf(" global-phase hub-port diff=%.2e (relative/flux phases set the aggregate: see [2])\n",diff);
 ++total;pass+=report("absolute phase leaked / relative phases inert",diff<1e-12);}

 std::printf("\n RESULT : %d / %d verified\n",pass,total);return pass==total?0:1;
}
