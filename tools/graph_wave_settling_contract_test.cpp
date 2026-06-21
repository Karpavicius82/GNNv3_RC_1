// graph_wave_settling_contract_test
// ----------------------------------------------------------------------------
// Missed physics: a superposition is NOT a static bundle -- it evolves and
// SETTLES (sedimas). This contract tests the dynamical life I had frozen out:
//   [1] DEPHASING: a stored superposition on an evolving medium loses coherence
//       with T2 ~ 1/(spectral width) -- a hologram washes out over time.
//   [2] PERSISTENCE: a degenerate (flat-band) subspace / stationary eigenstate
//       does NOT dephase -> the stable memory register the static tests assumed.
//   [3] SETTLING vs RECURRENCE: with dissipation the field settles onto an
//       attractor (decision); pure unitary evolution never settles -- it recurs
//       (Poincare). Settling needs the non-unitary half.
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
// random unitary via modified Gram-Schmidt; columns are the eigenvectors
Mat randomUnitary(std::mt19937_64&r){std::normal_distribution<double>g(0,1);Mat O(kN,Vec(kN));
  for(int j=0;j<kN;j++){Vec v(kN);for(int i=0;i<kN;i++)v[i]=cd(g(r),g(r));
    for(int p=0;p<j;p++){Vec cp(kN);for(int i=0;i<kN;i++)cp[i]=O[i][p];cd pr=idot(cp,v);for(int i=0;i<kN;i++)v[i]-=pr*cp[i];}
    normalize(v);for(int i=0;i<kN;i++)O[i][j]=v[i];}return O;}
Mat hamiltonian(const Mat&O,const std::vector<double>&E){Mat H(kN,Vec(kN,cd(0,0)));
  for(int i=0;i<kN;i++)for(int j=0;j<kN;j++){cd s(0,0);for(int k=0;k<kN;k++)s+=O[i][k]*E[k]*std::conj(O[j][k]);H[i][j]=s;}return H;}
Vec col(const Mat&O,int k){Vec v(kN);for(int i=0;i<kN;i++)v[i]=O[i][k];return v;}
struct LUC{int n=0;Mat a;std::vector<int>piv;void factor(Mat in){n=(int)in.size();a=std::move(in);piv.resize(n);for(int i=0;i<n;i++)piv[i]=i;for(int k=0;k<n;k++){double b=std::abs(a[k][k]);int r=k;for(int i=k+1;i<n;i++){double v=std::abs(a[i][k]);if(v>b){b=v;r=i;}}if(r!=k){std::swap(a[r],a[k]);std::swap(piv[r],piv[k]);}cd akk=a[k][k];for(int i=k+1;i<n;i++){cd f=a[i][k]/akk;a[i][k]=f;for(int j=k+1;j<n;j++)a[i][j]-=f*a[k][j];}}}Vec solve(const Vec&b)const{Vec x(n);for(int i=0;i<n;i++)x[i]=b[piv[i]];for(int i=0;i<n;i++){cd s=x[i];for(int j=0;j<i;j++)s-=a[i][j]*x[j];x[i]=s;}for(int i=n-1;i>=0;i--){cd s=x[i];for(int j=i+1;j<n;j++)s-=a[i][j]*x[j];x[i]=s/a[i][i];}return x;}};
struct Stepper{Mat am;LUC lu;void build(const Mat&h,double dt){Mat ap(kN,Vec(kN));am.assign(kN,Vec(kN));const cd I(0,1);for(int i=0;i<kN;i++)for(int j=0;j<kN;j++){cd k=I*h[i][j]*(dt/2.0);cd id=(i==j)?cd(1,0):cd(0,0);ap[i][j]=id+k;am[i][j]=id-k;}lu.factor(ap);}Vec step(const Vec&z)const{return lu.solve(matvec(am,z));}};
bool report(const char*n,bool ok){std::printf("    => %s\n",ok?"PASS":"FAIL");if(!ok)std::printf("    !! %s\n",n);return ok;}
// return-probability half-time of |psi0> on H (spectral width set by E)
double tHalf(const Mat&H,const Vec&psi0,double dt,int maxSteps){Stepper s;s.build(H,dt);Vec z=psi0;
  for(int t=1;t<=maxSteps;t++){z=s.step(z);double ret=std::norm(idot(psi0,z));if(ret<0.5)return t*dt;}return maxSteps*dt;}
}
int main(){
  std::printf("=====================================================================\n");
  std::printf(" GRAPH-WAVE SETTLING (SEDIMAS) CONTRACT TEST  (N=%d modes)\n",kN);
  std::printf(" A superposition evolves and settles: dephasing, persistence, sedimas.\n");
  std::printf("=====================================================================\n");
  std::mt19937_64 rng(0x5ED1ULL);
  Mat O=randomUnitary(rng);
  int pass=0,total=0;

  std::printf("\n[1] DEPHASING: coherence half-time scales as 1/(spectral width)\n");
  {std::vector<double>E1(kN),E2(kN);for(int k=0;k<kN;k++){double e=2.0*k/(kN-1)-1.0;E1[k]=1.0*e;E2[k]=2.0*e;}
   Vec psi0(kN,cd(0,0));psi0[0]=cd(1,0);                    // localized = all modes
   double th1=tHalf(hamiltonian(O,E1),psi0,0.004,5000),th2=tHalf(hamiltonian(O,E2),psi0,0.004,5000);
   std::printf("    t_half(width W)=%.3f   t_half(width 2W)=%.3f   ratio=%.2f (expect ~2)\n",th1,th2,th1/th2);
   ++total;pass+=report("dephasing does not scale as 1/spectral-width",th1/th2>1.6&&th1/th2<2.5);}

  std::printf("\n[2] PERSISTENCE: a degenerate (flat-band) subspace does not dephase\n");
  {std::vector<double>E={0.5,0.5,0.5,-1.0,-0.4,0.3,0.8,1.2};  // 3-fold flat band at 0.5
   Mat H=hamiltonian(O,E);Stepper s;s.build(H,0.004);
   Vec flat(kN,cd(0,0));for(int k=0;k<3;k++){Vec c=col(O,k);for(int i=0;i<kN;i++)flat[i]+=c[i];}normalize(flat);
   Vec spread(kN,cd(0,0));for(int k:{0,4,7}){Vec c=col(O,k);for(int i=0;i<kN;i++)spread[i]+=c[i];}normalize(spread);
   Vec zf=flat,zs=spread;double minFlat=1.0,retSpread=1.0;
   for(int t=1;t<=500;t++){zf=s.step(zf);zs=s.step(zs);minFlat=std::min(minFlat,std::norm(idot(flat,zf)));retSpread=std::norm(idot(spread,zs));}
   std::printf("    flat-band return min=%.6f   spread-energy return @t=2.0 =%.4f\n",minFlat,retSpread);
   ++total;pass+=report("flat-band register dephases / spread does not",minFlat>0.999&&retSpread<0.85);}

  std::printf("\n[3] SETTLING vs RECURRENCE (dissipation settles; unitary recurs)\n");
  {const double Ea=0.0,Eb=1.0;const double dt=0.01;const int steps=2000;
   // unitary: return-to-initial oscillates (revivals), never settles
   cd a(1,0),b(1,0);double inv=1.0/std::sqrt(2.0);a*=inv;b*=inv;cd a0=a,b0=b;double mn=1,mx=0;
   for(int t=0;t<steps;t++){a*=std::exp(cd(0,-Ea*dt));b*=std::exp(cd(0,-Eb*dt));double ret=std::norm(std::conj(a0)*a+std::conj(b0)*b);mn=std::min(mn,ret);mx=std::max(mx,ret);}
   // dissipative: mode b decays faster -> field settles onto mode a
   cd ca(inv,0),cb(inv,0);double ga=0.05,gb=0.6;double popPrev=0;bool monotone=true;
   for(int t=0;t<steps;t++){ca*=std::exp(cd(0,-Ea*dt))*std::exp(-ga*dt);cb*=std::exp(cd(0,-Eb*dt))*std::exp(-gb*dt);
     double nrm=std::norm(ca)+std::norm(cb);double pop=std::norm(ca)/nrm;if(pop<popPrev-1e-9)monotone=false;popPrev=pop;}
   std::printf("    unitary return range [%.3f, %.3f] (oscillates=no settle)   dissipative pop_a -> %.6f (monotone=%d)\n",mn,mx,popPrev,monotone?1:0);
   ++total;pass+=report("settling/recurrence not demonstrated",(mx-mn)>0.9&&popPrev>0.999&&monotone);}

  std::printf("\n RESULT : %d / %d verified\n",pass,total);return pass==total?0:1;
}
