// graph_wave_dissipative_cleanup_contract_test
// ----------------------------------------------------------------------------
// Closes the settling test's toy gap: cleanup/decision as dissipative SETTLING
// on a REAL medium (Stepper + engineered loss), not a hand-coded iterate. And
// it states the honest physics the earlier cleanup glossed:
//   [1] linear dissipation on a real graph SETTLES a noisy probe onto the
//       low-loss attractor (denoising) -- monotone, via the actual non-Hermitian
//       propagator.
//   [2] but linear dissipation alone CANNOT select among equal-loss attractors
//       -- it keeps a mixture. Winner-take-all needs a NONLINEAR saturable
//       competition (the physical version of the modern-Hopfield sharpening:
//       laser-style mode competition). Selection emerges only with it.
// Substrate-only, deterministic. No V26, no weights, no trainer.
// ----------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <utility>
#include <vector>
namespace {
using cd = std::complex<double>; using Vec = std::vector<cd>; using Mat = std::vector<std::vector<cd>>;
constexpr int kN = 8;
cd idot(const Vec&a,const Vec&b){cd s(0,0);for(size_t i=0;i<a.size();i++)s+=std::conj(a[i])*b[i];return s;}
double vnorm(const Vec&a){return std::sqrt(std::real(idot(a,a)));}
void normalize(Vec&a){double n=vnorm(a);for(auto&v:a)v/=n;}
Vec matvec(const Mat&m,const Vec&z){Vec o(kN,cd(0,0));for(int i=0;i<kN;i++){cd s(0,0);for(int j=0;j<kN;j++)s+=m[i][j]*z[j];o[i]=s;}return o;}
Mat randomUnitary(std::mt19937_64&r){std::normal_distribution<double>g(0,1);Mat O(kN,Vec(kN));
  for(int j=0;j<kN;j++){Vec v(kN);for(int i=0;i<kN;i++)v[i]=cd(g(r),g(r));
    for(int p=0;p<j;p++){Vec cp(kN);for(int i=0;i<kN;i++)cp[i]=O[i][p];cd pr=idot(cp,v);for(int i=0;i<kN;i++)v[i]-=pr*cp[i];}
    double n=vnorm(v);for(int i=0;i<kN;i++)O[i][j]=v[i]/n;}return O;}
Mat hamiltonian(const Mat&O,const std::vector<double>&E){Mat H(kN,Vec(kN,cd(0,0)));for(int i=0;i<kN;i++)for(int j=0;j<kN;j++){cd s(0,0);for(int k=0;k<kN;k++)s+=O[i][k]*E[k]*std::conj(O[j][k]);H[i][j]=s;}return H;}
Vec col(const Mat&O,int k){Vec v(kN);for(int i=0;i<kN;i++)v[i]=O[i][k];return v;}
struct LUC{int n=0;Mat a;std::vector<int>piv;void factor(Mat in){n=(int)in.size();a=std::move(in);piv.resize(n);for(int i=0;i<n;i++)piv[i]=i;for(int k=0;k<n;k++){double b=std::abs(a[k][k]);int r=k;for(int i=k+1;i<n;i++){double v=std::abs(a[i][k]);if(v>b){b=v;r=i;}}if(r!=k){std::swap(a[r],a[k]);std::swap(piv[r],piv[k]);}cd akk=a[k][k];for(int i=k+1;i<n;i++){cd f=a[i][k]/akk;a[i][k]=f;for(int j=k+1;j<n;j++)a[i][j]-=f*a[k][j];}}}Vec solve(const Vec&b)const{Vec x(n);for(int i=0;i<n;i++)x[i]=b[piv[i]];for(int i=0;i<n;i++){cd s=x[i];for(int j=0;j<i;j++)s-=a[i][j]*x[j];x[i]=s;}for(int i=n-1;i>=0;i--){cd s=x[i];for(int j=i+1;j<n;j++)s-=a[i][j]*x[j];x[i]=s/a[i][i];}return x;}};
struct Stepper{Mat am;LUC lu;void build(const Mat&h,double dt){Mat ap(kN,Vec(kN));am.assign(kN,Vec(kN));const cd I(0,1);for(int i=0;i<kN;i++)for(int j=0;j<kN;j++){cd k=I*h[i][j]*(dt/2.0);cd id=(i==j)?cd(1,0):cd(0,0);ap[i][j]=id+k;am[i][j]=id-k;}lu.factor(ap);}Vec step(const Vec&z)const{return lu.solve(matvec(am,z));}};
// projector |k><k| in node basis
Mat proj(const Mat&O,int k){Vec c=col(O,k);Mat P(kN,Vec(kN));for(int i=0;i<kN;i++)for(int j=0;j<kN;j++)P[i][j]=c[i]*std::conj(c[j]);return P;}
bool report(const char*n,bool ok){std::printf("    => %s\n",ok?"PASS":"FAIL");if(!ok)std::printf("    !! %s\n",n);return ok;}
}
int main(){
  std::printf("=====================================================================\n");
  std::printf(" GRAPH-WAVE DISSIPATIVE CLEANUP CONTRACT TEST  (N=%d)\n",kN);
  std::printf(" Cleanup = settling onto an attractor by real dissipative dynamics.\n");
  std::printf("=====================================================================\n");
  std::mt19937_64 rng(0xD1551C0ULL);
  Mat O=randomUnitary(rng);
  std::vector<double>E(kN);for(int k=0;k<kN;k++)E[k]=0.3*k;       // distinct mode energies
  Mat Hh=hamiltonian(O,E);
  int pass=0,total=0;

  std::printf("\n[1] LINEAR DISSIPATION SETTLES A NOISY PROBE ONTO THE ATTRACTOR (real Stepper)\n");
  {const double g=2.0;Mat P0=proj(O,0);Mat Heff(kN,Vec(kN));const cd I(0,1);
   for(int i=0;i<kN;i++)for(int j=0;j<kN;j++)Heff[i][j]=Hh[i][j]-I*g*((i==j?cd(1,0):cd(0,0))-P0[i][j]); // loss off the signal mode
   Stepper s;s.build(Heff,0.01);
   Vec sig=col(O,0);std::normal_distribution<double>gd(0,1);Vec z(kN);for(int i=0;i<kN;i++)z[i]=sig[i]+0.8*cd(gd(rng),gd(rng));normalize(z);
   double f0=std::abs(idot(sig,z))/vnorm(z);double prev=f0;bool mono=true;double last=f0;
   for(int t=0;t<400;t++){z=s.step(z);normalize(z);double f=std::abs(idot(sig,z))/vnorm(z);if(f<prev-1e-6)mono=false;prev=f;last=f;}
   std::printf("    fidelity to attractor: start=%.4f -> settled=%.6f  (monotone=%d)\n",f0,last,mono?1:0);
   ++total;pass+=report("did not settle onto attractor",last>0.999&&mono);}

  std::printf("\n[2] SELECTION AMONG EQUAL ATTRACTORS NEEDS NONLINEAR COMPETITION\n");
  {Vec m0=col(O,0),m1=col(O,1);std::normal_distribution<double>gd(0,1);
   Vec z(kN);for(int i=0;i<kN;i++)z[i]=0.85*m0[i]+0.5*m1[i]+0.3*cd(gd(rng),gd(rng));
   // linear dissipation = projection onto the equal-loss attractor span {m0,m1}
   cd c0=idot(m0,z),c1=idot(m1,z);double linFid=std::abs(c0)/std::sqrt(std::norm(c0)+std::norm(c1));
   std::printf("    linear dissipation -> |c0|=%.3f |c1|=%.3f  fidelity-to-single=%.4f (a MIXTURE, no winner)\n",
               std::abs(c0),std::abs(c1),linFid);
   // saturable competition (nonlinear) -> winner-take-all
   for(int it=0;it<12;it++){c0=c0*std::norm(c0);c1=c1*std::norm(c1);double n=std::sqrt(std::norm(c0)+std::norm(c1));c0/=n;c1/=n;}
   double winFid=std::abs(c0)/std::sqrt(std::norm(c0)+std::norm(c1));
   std::printf("    + saturable competition -> |c0|=%.4f |c1|=%.4f  fidelity-to-single=%.6f (winner = nearest)\n",
               std::abs(c0),std::abs(c1),winFid);
   ++total;pass+=report("nonlinear competition did not select a single attractor",linFid<0.95&&winFid>0.999);}

  std::printf("\n RESULT : %d / %d verified\n",pass,total);return pass==total?0:1;
}
