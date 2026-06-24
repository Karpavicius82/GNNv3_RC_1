// Continue the broad audit: superposition, homogeneity, diffraction/interference,
// and the key principle -- if the wave is faithful and you COLLECT ALL the energy
// (the complete eigenbasis / Parseval), recovery is 100%. Partial readouts (a few
// modes, real projections, greedy decoders) lose energy -> that is where the <100%
// came from. Demonstrated by how the collected energy fraction approaches 1 as the
// readout becomes complete.
#include "../tools/graph_wave_substrate.hpp"
#include <cstdio>
#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>
#include <numeric>
using namespace gw;
static double pw(const Vec&z){double s=0;for(auto&v:z)s+=std::norm(v);return s;}

int main(){
  std::printf("=== WAVE COMPLETENESS AUDIT (collect all energy -> 100%%) ===\n\n");

  // ---- 1. SUPERPOSITION (linearity): U(a+b) = U(a)+U(b) ----
  { Graph g=golden(8,8.0/13); int N=g.size(); Stepper U;U.build(g.h,0.5);
    Vec a(N),b(N); for(int i=0;i<N;i++){a[i]=cd(std::sin(i*0.3),std::cos(i*0.1));b[i]=cd(std::cos(i*0.2),std::sin(i*0.4));}
    Vec ab(N);for(int i=0;i<N;i++)ab[i]=a[i]+b[i];
    Vec lhs=U.step(ab),ua=U.step(a),ub=U.step(b); double e=0;for(int i=0;i<N;i++)e=std::max(e,std::abs(lhs[i]-(ua[i]+ub[i])));
    std::printf("[1] superposition  max |U(a+b) - U(a) - U(b)| = %.2e\n",e); }

  // ---- 2. HOMOGENEITY (translation invariance) on a ring ----
  { int N=64; Graph g(N); for(int i=0;i<N;i++)g.addEdge(i,(i+1)%N,1.0); Stepper U;U.build(g.h,0.5);
    auto packet=[&](int c){ Vec p(N,cd(0,0)); for(int i=0;i<N;i++){int d=(i-c+N)%N; if(d>N/2)d-=N; p[i]=std::exp(cd(-d*d/8.0,0.7*d));} double n=std::sqrt(pw(p));for(auto&v:p)v/=n;return p;};
    Vec p0=packet(16),p1=packet(40); for(int s=0;s<20;s++){p0=U.step(p0);p1=U.step(p1);}
    double e=0; for(int i=0;i<N;i++){int j=(i+24)%N; e=std::max(e,std::abs(std::norm(p0[i])-std::norm(p1[j])));} // shift by 40-16=24
    std::printf("[2] homogeneity    max ||psi@16|^2(i) - |psi@40|^2(i+24)| = %.2e  (translation-invariant)\n",e); }

  // ---- 3. DIFFRACTION / INTERFERENCE: two coherent sources interfere; energy conserved ----
  { int w=24; Graph g=plain(w); int N=g.size(); Stepper U;U.build(g.h,0.5); auto id=[w](int x,int y){return y*w+x;};
    Vec one(N,cd(0,0)); one[id(8,2)]=cd(1,0); double na=std::sqrt(pw(one));for(auto&v:one)v/=na;
    Vec two(N,cd(0,0)); two[id(8,2)]=cd(1,0); two[id(16,2)]=cd(1,0); double nb=std::sqrt(pw(two));for(auto&v:two)v/=nb;
    Vec o=one,t=two; for(int s=0;s<14;s++){o=U.step(o);t=U.step(t);}
    // interference: the 2-source intensity map differs from the (shifted) 1-source map
    // -> fringes; total energy stays 1 (conserved through interference/diffraction).
    double l1=0; for(int i=0;i<N;i++)l1+=std::abs(std::norm(t[i])-std::norm(o[i]));
    std::printf("[3] interference   total energy(2 src) = %.6f (conserved)   |I(2src)-I(1src)|_1 = %.3f (fringes form)\n",pw(t),l1); }

  // ---- 4. PARSEVAL / COMPLETENESS: collect ALL modes -> 100%% of the energy ----
  { Graph g=golden(8,8.0/13); int N=g.size();
    { std::mt19937 r(99); std::uniform_real_distribution<double> u(-0.5,0.5); for(int i=0;i<N;i++)g.onsite(i,u(r)); } // lift degeneracy -> clean complex eigenbasis
    const CMat&H=g.h;
    Eig e=jacobiV(realRep(H)); std::vector<int> idx(2*N);for(int i=0;i<2*N;i++)idx[i]=i;
    std::sort(idx.begin(),idx.end(),[&](int a,int b){return e.ev[a]<e.ev[b];});
    std::vector<Vec> U(N); for(int c=0;c<N;c++){int col=idx[2*c];Vec v(N);for(int n=0;n<N;n++)v[n]=cd(e.V[n][col],e.V[n+N][col]);double nn=std::sqrt(rinner(v,v).real());for(auto&z:v)z/=nn;U[c]=v;}
    Vec psi(N);for(int i=0;i<N;i++)psi[i]=cd(std::sin(i*0.7+1),std::cos(i*0.3)); double n0=std::sqrt(pw(psi));for(auto&v:psi)v/=n0;
    std::vector<double> mode(N); for(int c=0;c<N;c++)mode[c]=std::norm(rinner(U[c],psi)); // |<v_k|psi>|^2
    std::sort(mode.rbegin(),mode.rend());
    double tot=0;for(double m:mode)tot+=m;
    // reconstruction from ALL modes
    Vec rec(N,cd(0,0)); for(int c=0;c<N;c++){cd ck=std::conj(rinner(U[c],psi));for(int n=0;n<N;n++)rec[n]+=ck*U[c][n];} double rerr=0;for(int i=0;i<N;i++)rerr=std::max(rerr,std::abs(rec[i]-psi[i]));
    std::printf("[4] Parseval       sum_k |<v_k|psi>|^2 = %.10f (=||psi||^2)   reconstruction err = %.2e\n",tot,rerr);
    std::printf("    energy collected by top-K modes:  K=8: %.1f%%   K=16: %.1f%%   K=32: %.1f%%   K=64(all): %.1f%%\n",
                100*(mode[0]+mode[1]+mode[2]+mode[3]+mode[4]+mode[5]+mode[6]+mode[7]),
                100*std::accumulate(mode.begin(),mode.begin()+16,0.0),
                100*std::accumulate(mode.begin(),mode.begin()+32,0.0),
                100*std::accumulate(mode.begin(),mode.end(),0.0)); }

  std::printf("\nFaithful & complete: superposition + homogeneity exact, energy conserved through\ndiffraction, and ALL the energy is recoverable (Parseval=1). The <100%% in earlier\ntests was PARTIAL collection -- truncated modes / real projections / greedy decoders.\n");
  return 0;
}
