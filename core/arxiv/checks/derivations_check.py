"""Symbolic and numerical checks of the derivations in appendix A of the extended (arXiv) version.

Зависит только от discrete.py, sympy и numpy: ни LaTeX, ни рисунки, ни файлы рукописи
для запуска не нужны. Код возврата 0 --- все проверки пройдены, 1 --- есть провалившиеся.
    python3 derivations_check.py [--root PATH]
"""
import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "paths.py").exists():
        sys.path.insert(0, str(_p)); break
else:
    sys.exit("не найден paths.py ни в одном родительском каталоге этого скрипта")
from paths import ROOT, require
require(ROOT / "discrete.py", what="модуль discrete.py")
import sympy as sp, numpy as np
from discrete import survival_disc, h_disc, ab_disc, stationary_disc
n,i,t,lam,Dd,mu,s2,k,M,m0,eta,x=sp.symbols('n i t lambda Ddot mu sigma2 kappa M m0 eta x',positive=True)
wp=lam*Dd+sp.Rational(1,2)*((s2-mu)*n+k*n**2); wm=sp.Rational(1,2)*((s2+mu)*n+k*n**2)
ok=[]
# A.1 generator identities
f1=sp.expand(wp*1+wm*(-1)); ok.append(sp.simplify(f1-(lam*Dd-mu*n))==0)
f2=sp.expand(wp*((n+1)**2-n**2)+wm*((n-1)**2-n**2)); ok.append(sp.simplify(f2-(2*n*(wp-wm)+wp+wm))==0)
m,n2=sp.symbols('m n2'); dn2=sp.expand(2*(lam*Dd*m-mu*n2)+lam*Dd+s2*m+k*n2)
dv=sp.expand(dn2-2*m*(lam*Dd-mu*m)); v=sp.symbols('v')
ok.append(sp.simplify(dv.subs(n2,v+m**2)-(-(2*mu-k)*v+lam*Dd+s2*m+k*m**2))==0)
# A.2 tail ratio
r=(wp.subs(n,i-1))/(wm.subs(n,i)); ser=sp.series(r.subs(i,1/x),x,0,2).removeO()
ok.append(sp.simplify(ser-(1-(2*mu/k+2)*x))==0)
# FP stationary: d/dn[ B P ]/2 = A P with B without shot noise
a=2*lam*Dd/s2; b=2*mu/k; q=k/s2
P=(q*n)**(a-1)*(1+q*n)**(-(a+b+1)); B=s2*n+k*n**2; A=lam*Dd-mu*n
ok.append(sp.simplify(sp.diff(B*P,n)/2-A*P)==0)
# A.3 scale density
s_prime=(s2+k*x)**(2*mu/k); ok.append(sp.simplify(sp.Rational(1,2)*(s2*x+k*x**2)*sp.diff(s_prime,x)+(-mu*x)*s_prime)==0)
# A.4 FP expansion 2/(e^y-1)
y=sp.symbols('y'); ok.append(sp.simplify(sp.series(2/(sp.exp(y)-1),y,0,3).removeO()-(2/y-1+y/6))==0)
# A.5 variance kinetics
m0_,C=sp.symbols('m0_ C'); vN=s2*m0/(mu-k)*sp.exp(-mu*t)-m0**2/M*sp.exp(-2*mu*t)+(m0-s2*m0/(mu-k)+m0**2/M)*sp.exp(-(2*mu-k)*t)
mN=m0*sp.exp(-mu*t)
ok.append(sp.simplify(sp.diff(vN,t)-(-(2*mu-k)*vN+s2*mN+k/M*mN**2))==0); ok.append(sp.simplify(vN.subs(t,0)-m0)==0)
F0=sp.simplify((vN/mN).subs(k,0)); ok.append(sp.simplify(F0-(s2/mu-(s2/mu-1)*sp.exp(-mu*t)))==0)
# A.6 thinning
vY=eta**2*vN+eta*(1-eta)*mN; mY=eta*mN; se=eta*s2+(1-eta)*(mu-k)
ok.append(sp.simplify(sp.diff(vY,t)-(-(2*mu-k)*vY+se*mY+k/M*mY**2))==0)
ok.append(sp.simplify(se-mu+k-eta*(s2-mu+k))==0)
NAMES=['A.1 среднее: генератор','A.1 второй момент: генератор','A.1 кинетика дисперсии',
       'A.2 отношение хвоста','FP: стационарное решение','A.3 плотность шкалы',
       'A.4 разложение 2/(e^y-1)','A.5 ОДУ для v_N(t)','A.5 начальное условие v_N(0)',
       'A.5 фактор Фано при kappa=0','A.6 прореживание: ОДУ для v_Y','A.6 прореживание: sigma^2_eff']
FAILED=[NAMES[i] if i<len(NAMES) else f'symbolic[{i}]' for i,v in enumerate(ok) if not v]
print('symbolic checks: %d/%d'%(sum(map(bool,ok)),len(ok)), 'OK' if all(ok) else 'FAILED: '+'; '.join(FAILED))
# A.4 numerical: alpha,beta vs finite differences of -ln S
rng=np.random.default_rng(3); worst=0
for _ in range(200):
    l=10**rng.uniform(-1,0.5); muv=1.0; s2v=muv*(1+10**rng.uniform(-2,1)); kv=10**rng.uniform(-2,1); nc=int(rng.integers(2,15)); Mv=int(rng.choice([1,2,4,10]))
    a_,b_=ab_disc(l,muv,s2v,kv,nc,Mv)
    hD=1e-3; Ds=np.array([hD,2*hD,3*hD]); g=-np.log(survival_disc(Ds,l,muv,s2v,kv,nc,Mv))
    # fit g = a D + b D^2 + c D^3 exactly through three points
    X=np.vstack([Ds,Ds**2,Ds**3]).T; c=np.linalg.solve(X,g)
    if a_>1e-3 and b_>1e-3: worst=max(worst,abs(c[0]-a_)/a_,abs(c[1]-b_)/b_); ncase=globals().get('ncase',0)+1; globals()['ncase']=ncase
    h=h_disc(nc,muv,s2v,kv); rho1=(s2v+muv+kv)/(s2v-muv+kv)
    assert abs(h[2]-(1+rho1)*h[1])<1e-12
    assert abs(a_/b_-2/(l*(rho1-1+h[1])))<1e-9*a_/b_
print('cases',ncase,'max relative deviation of alpha, beta from finite differences (h=1e-3):',worst)
# n_c=1: -ln S = M lambda D exactly (beta = 0)
nc1=bool(np.allclose(-np.log(survival_disc(np.array([.5,1,2]),0.7,1,2,0.3,1,3)),3*0.7*np.array([.5,1,2])))
print('n_c=1 check:',nc1)
# stationary: tail exponent from discrete pi_n
nn,p=stationary_disc(200000,2.0,1.0,1.5,0.4); slope=np.polyfit(np.log(nn[50000:]),np.log(p[50000:]),1)[0]
expected=-(2*1.0/0.4+2); print('tail slope',slope,'expected',expected)

# --- итог: код возврата для CI -------------------------------------------------
fails=[]
if not all(ok): fails += FAILED
if worst>=1e-2: fails.append(f'alpha,beta против конечных разностей: {worst:.2e} >= 1e-2')
if not nc1:     fails.append('n_c=1: -ln S не равно M*lambda*D')
if abs(slope-expected)>=1e-2: fails.append(f'наклон хвоста {slope:.4f} против ожидаемого {expected}')
print('\nИТОГ:', 'все проверки пройдены' if not fails else 'ПРОВАЛЕНО:\n  '+'\n  '.join(fails))
sys.exit(0 if not fails else 1)
