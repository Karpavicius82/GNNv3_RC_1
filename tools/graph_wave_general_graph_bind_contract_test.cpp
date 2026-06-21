// graph_wave_general_graph_bind_contract_test
// ----------------------------------------------------------------------------
// Gap A: propagation must work on a GENERAL graph, not only a ring. Shows the
// magnetic (gauge-phased) graph Hamiltonian gives a unitary, gauge-covariant
// propagation whose edge phases encode DIRECTION (broken reciprocity). Substrate
// capability only -- no weights, no trainer, no V26.
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
constexpr int kN = 4;  // complete graph K4 (has flux loops => chiral)
Vec matvec(const Mat& m, const Vec& z){Vec o(kN,cd(0,0));for(int i=0;i<kN;i++){cd s(0,0);for(int j=0;j<kN;j++)s+=m[i][j]*z[j];o[i]=s;}return o;}
Mat matmul(const Mat&a,const Mat&b){Mat o(kN,Vec(kN,cd(0,0)));for(int i=0;i<kN;i++)for(int k=0;k<kN;k++)for(int j=0;j<kN;j++)o[i][j]+=a[i][k]*b[k][j];return o;}
Mat dagger(const Mat&m){Mat o(kN,Vec(kN));for(int i=0;i<kN;i++)for(int j=0;j<kN;j++)o[j][i]=std::conj(m[i][j]);return o;}
double maxOff1(const Mat&m){double e=0;for(int i=0;i<kN;i++)for(int j=0;j<kN;j++)e=std::max(e,std::abs(m[i][j]-(i==j?cd(1,0):cd(0,0))));return e;}
double norm2(const Vec&z){double s=0;for(auto&v:z)s+=std::norm(v);return s;}
struct LUC{int n=0;Mat a;std::vector<int>piv;void factor(Mat in){n=(int)in.size();a=std::move(in);piv.resize(n);for(int i=0;i<n;i++)piv[i]=i;for(int k=0;k<n;k++){double b=std::abs(a[k][k]);int r=k;for(int i=k+1;i<n;i++){double v=std::abs(a[i][k]);if(v>b){b=v;r=i;}}if(r!=k){std::swap(a[r],a[k]);std::swap(piv[r],piv[k]);}cd akk=a[k][k];for(int i=k+1;i<n;i++){cd f=a[i][k]/akk;a[i][k]=f;for(int j=k+1;j<n;j++)a[i][j]-=f*a[k][j];}}}Vec solve(const Vec&b)const{Vec x(n);for(int i=0;i<n;i++)x[i]=b[piv[i]];for(int i=0;i<n;i++){cd s=x[i];for(int j=0;j<i;j++)s-=a[i][j]*x[j];x[i]=s;}for(int i=n-1;i>=0;i--){cd s=x[i];for(int j=i+1;j<n;j++)s-=a[i][j]*x[j];x[i]=s/a[i][i];}return x;}};
struct Stepper{Mat am;LUC lu;void build(const Mat&h,double dt){Mat ap(kN,Vec(kN));am.assign(kN,Vec(kN));const cd I(0,1);for(int i=0;i<kN;i++)for(int j=0;j<kN;j++){cd k=I*h[i][j]*(dt/2.0);cd id=(i==j)?cd(1,0):cd(0,0);ap[i][j]=id+k;am[i][j]=id-k;}lu.factor(ap);}Vec evolve(Vec z,int s)const{for(int t=0;t<s;t++)z=lu.solve(matvec(am,z));return z;}};
double hermErr(const Mat&h){double e=0;for(int i=0;i<kN;i++)for(int j=0;j<kN;j++)e=std::max(e,std::abs(h[i][j]-std::conj(h[j][i])));return e;}
Mat transfer(const Mat&H,int steps){Stepper s;s.build(H,1.0/steps);Mat U(kN,Vec(kN));for(int j=0;j<kN;j++){Vec e(kN,cd(0,0));e[j]=cd(1,0);Vec c=s.evolve(e,steps);for(int i=0;i<kN;i++)U[i][j]=c[i];}return U;}
bool report(const char*n,bool ok){std::printf("    => %s\n",ok?"PASS":"FAIL");if(!ok)std::printf("    !! %s\n",n);return ok;}
}
int main(){
  std::printf("=====================================================================\n");
  std::printf(" GRAPH-WAVE GENERAL-GRAPH PROPAGATION CONTRACT TEST  (K%d)\n",kN);
  std::printf(" Magnetic (gauge-phased) graph: unitary, gauge-covariant, directed.\n");
  std::printf("=====================================================================\n");
  std::mt19937_64 rng(0x6A6BULL);std::uniform_real_distribution<double>uni(-3.14159,3.14159);
  Mat H(kN,Vec(kN,cd(0,0)));
  for(int i=0;i<kN;i++)for(int j=i+1;j<kN;j++){double th=uni(rng);H[i][j]=std::exp(cd(0,th));H[j][i]=std::conj(H[i][j]);}
  int pass=0,total=0;
  std::printf("\n[1] MAGNETIC GRAPH HAMILTONIAN IS HERMITIAN\n");
  {double he=hermErr(H);std::printf("    hermiticity error = %.2e\n",he);++total;pass+=report("not hermitian",he<1e-12);}
  std::printf("\n[2] PROPAGATION ON THE GENERAL GRAPH IS UNITARY\n");
  Mat U=transfer(H,4096);
  {double ue=maxOff1(matmul(dagger(U),U));std::printf("    U^dag U - I = %.2e\n",ue);++total;pass+=report("not unitary",ue<1e-6);}
  std::printf("\n[3] PROPAGATION IS GAUGE-COVARIANT (node-phase gauge leaves |amplitude|^2)\n");
  {std::vector<double>g(kN);for(auto&x:g)x=uni(rng);Mat Hp(kN,Vec(kN));for(int i=0;i<kN;i++)for(int j=0;j<kN;j++)Hp[i][j]=std::exp(cd(0,g[i]-g[j]))*H[i][j];
   Vec z0(kN);for(int i=0;i<kN;i++){double p=uni(rng);z0[i]=cd(std::cos(p),std::sin(p));}Vec gz(kN);for(int i=0;i<kN;i++)gz[i]=std::exp(cd(0,g[i]))*z0[i];
   Stepper s1;s1.build(H,1.0/4096);Stepper s2;s2.build(Hp,1.0/4096);Vec a=s1.evolve(z0,4096),b=s2.evolve(gz,4096);
   double pe=0;for(int i=0;i<kN;i++)pe=std::max(pe,std::abs(std::norm(a[i])-std::norm(b[i])));
   std::printf("    max |prob_gauge - prob| = %.2e\n",pe);++total;pass+=report("not gauge-covariant",pe<1e-9);}
  std::printf("\n[4] EDGE PHASES ENCODE DIRECTION (flux breaks reciprocity)\n");
  {double asymFlux=0;for(int i=0;i<kN;i++)for(int j=0;j<kN;j++)asymFlux=std::max(asymFlux,std::abs(std::abs(U[i][j])-std::abs(U[j][i])));
   Mat H0(kN,Vec(kN,cd(0,0)));for(int i=0;i<kN;i++)for(int j=0;j<kN;j++)H0[i][j]=cd(std::abs(H[i][j]),0);Mat U0=transfer(H0,4096);
   double asymZero=0;for(int i=0;i<kN;i++)for(int j=0;j<kN;j++)asymZero=std::max(asymZero,std::abs(std::abs(U0[i][j])-std::abs(U0[j][i])));
   std::printf("    transfer asymmetry: with flux=%.4f   zero-phase=%.2e\n",asymFlux,asymZero);
   ++total;pass+=report("edge phases do not encode direction",asymFlux>0.05&&asymZero<1e-12);}
  std::printf("\n RESULT : %d / %d verified\n",pass,total);return pass==total?0:1;
}
