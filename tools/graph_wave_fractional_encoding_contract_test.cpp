// graph_wave_fractional_encoding_contract_test
// ----------------------------------------------------------------------------
// Gap #1 (generalization): random atoms are maximally orthogonal, so similar
// inputs map to dissimilar fields -> only exact lookup, no generalization. This
// contract adds a SIMILARITY-PRESERVING encoder for continuous quantities --
// fractional power encoding (Frady/Sommer, Komer/Eliasmith): enc(x)[k] =
// e^{i x phi_k}. Then bind = addition of values, and similarity is a smooth,
// shift-invariant kernel of (x - y). Substrate only -- no weights, no trainer.
// ----------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>
namespace {
using cd = std::complex<double>; using Vec = std::vector<cd>;
constexpr int kD = 2048; constexpr double kTwoPi = 6.283185307179586;
std::vector<double> phi(kD);
Vec enc(double x){Vec v(kD);for(int k=0;k<kD;k++)v[k]=std::exp(cd(0,x*phi[k]));return v;}
Vec bind(const Vec&a,const Vec&b){Vec c(kD);for(int k=0;k<kD;k++)c[k]=a[k]*b[k];return c;}
double norm(const Vec&a){double s=0;for(auto&v:a)s+=std::norm(v);return std::sqrt(s);}
double cosre(const Vec&a,const Vec&b){cd d(0,0);for(int k=0;k<kD;k++)d+=std::conj(a[k])*b[k];return std::real(d)/(norm(a)*norm(b));}
double maxdiff(const Vec&a,const Vec&b){double e=0;for(int k=0;k<kD;k++)e=std::max(e,std::abs(a[k]-b[k]));return e;}
bool report(const char*n,bool ok){std::printf("    => %s\n",ok?"PASS":"FAIL");if(!ok)std::printf("    !! %s\n",n);return ok;}
}
int main(){
  std::printf("=====================================================================\n");
  std::printf(" GRAPH-WAVE FRACTIONAL-POWER ENCODING CONTRACT TEST  (D=%d)\n",kD);
  std::printf(" Similarity-preserving continuous encoding: enc(x)[k]=e^{i x phi_k}.\n");
  std::printf("=====================================================================\n");
  std::mt19937_64 rng(0xF9ACULL);std::uniform_real_distribution<double>u(-3.14159,3.14159);
  for(int k=0;k<kD;k++)phi[k]=u(rng);
  int pass=0,total=0;
  std::printf("\n[1] enc(0) IS THE IDENTITY AND enc(x) IS A UNIT PHASOR FIELD\n");
  {Vec e0=enc(0.0);double d=0;for(int k=0;k<kD;k++)d=std::max(d,std::abs(e0[k]-cd(1,0)));
   std::printf("    max|enc(0)-1| = %.2e   |enc(1.7)| = %.6f\n",d,norm(enc(1.7))/std::sqrt((double)kD));
   ++total;pass+=report("enc(0) not identity / not unit",d<1e-12);}
  std::printf("\n[2] BINDING ADDS THE ENCODED VALUE: enc(x) (x) enc(y) == enc(x+y)\n");
  {double e=maxdiff(bind(enc(1.3),enc(-0.4)),enc(0.9));std::printf("    max|bind(enc x,enc y)-enc(x+y)| = %.2e\n",e);
   ++total;pass+=report("binding does not add values",e<1e-12);}
  std::printf("\n[3] SIMILARITY IS A SHIFT-INVARIANT KERNEL OF (x-y)\n");
  {double d1=cosre(enc(0.0),enc(0.3)),d2=cosre(enc(2.1),enc(2.4)),d3=cosre(enc(-1.0),enc(-0.7));
   std::printf("    k(0,0.3)=%.4f  k(2.1,2.4)=%.4f  k(-1.0,-0.7)=%.4f  (same gap -> same value)\n",d1,d2,d3);
   double spread=std::max({std::abs(d1-d2),std::abs(d2-d3),std::abs(d1-d3)});
   ++total;pass+=report("similarity not shift-invariant",spread<0.05);}
  std::printf("\n[4] SIMILARITY PRESERVED: near -> high, far -> low (generalization)\n");
  {double near=cosre(enc(0.0),enc(0.1)),mid=cosre(enc(0.0),enc(0.5)),far=cosre(enc(0.0),enc(3.0));
   std::printf("    k(dx=0.1)=%.4f  k(dx=0.5)=%.4f  k(dx=3.0)=%.4f\n",near,mid,far);
   ++total;pass+=report("encoding is not similarity-preserving",near>0.8&&near>mid&&mid>far&&far<0.2);}
  std::printf("\n RESULT : %d / %d verified\n",pass,total);return pass==total?0:1;
}
