# Q4:
# 1) Free energy ψ(ε,T) for T in {280,300,330} K on the same plot
# 2) σ(ε,T) for the same T on the same plot
# 3) FRF x_RMS(ω): overlay A ∈ {2.5,5.0,9.81} at fixed T=300 K, Falk vs Linear (6 lines)
# 4) FRF x_RMS(ω): overlay T ∈ {280,300,330} K at fixed A=5.0 m/s², Falk vs Linear (6 lines)

import numpy as np, matplotlib.pyplot as plt, math, os
from dataclasses import dataclass

@dataclass
class FalkParams:
    zeta: float = 0.025
    a: float = 15.0
    b: float = 60e4
    Ta: float = 313.0
    Tm: float = 287.0
    steps_per_period: int = 300
    n_cycles: int = 50

def c5(p): return (p.b**2)/(24.0*p.a*(p.Ta - p.Tm))
def psi(eps, T, p): return 0.5*p.a*(T - p.Tm)*eps**2 - 0.25*p.b*eps**4 + (c5(p)/6.0)*eps**6
def sigma(eps, T, p): return p.a*(T - p.Tm)*eps - p.b*eps**3 + c5(p)*eps**5

def rk4_step(f,t,y,h):
    k1=f(t,y); k2=f(t+0.5*h,y+0.5*h*k1); k3=f(t+0.5*h,y+0.5*h*k2); k4=f(t+h,y+h*k3)
    return y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def x_rms_Falk(A, w, T, p: FalkParams):
    c=2.0*p.zeta; k1=p.a*(T - p.Tm); k3=-p.b; k5=c5(p)
    def f(t,y):
        x,xd=y; dd=-c*xd - (k1*x + k3*x**3 + k5*x**5) + A*math.sin(w*t)
        return np.array([xd, dd])
    h=min(2.0*np.pi/(p.steps_per_period*w), 1e-3)
    n=int(p.n_cycles*p.steps_per_period)
    x=0.0; xd=0.0; t=0.0
    xs=np.empty(n)
    for i in range(n):
        xs[i]=x
        x,xd = rk4_step(f,t,np.array([x,xd]),h)
        t+=h
    i0=int(2*n/3); return float(np.sqrt(np.mean(xs[i0:]**2)))

def x_rms_linear(A, w, T, p: FalkParams):
    c=2.0*p.zeta; k1=p.a*(T - p.Tm)
    X=A/np.sqrt((k1 - w**2)**2 + (c*w)**2)
    return float(X/np.sqrt(2.0))

p=FalkParams()
Temps=[280.0,300.0,330.0]
eps=np.linspace(-0.12,0.12,600)

# Fig 1: Free energy (3 curves)
plt.figure()
for T in Temps:
    plt.plot(eps, psi(eps,T,p), label=f'T={int(T)} K')
plt.xlabel(r'$\varepsilon$'); plt.ylabel(r'$\rho\psi(\varepsilon,T)$ [J/m$^3$]')
plt.title('Falk: energia livre')
plt.grid(True); plt.legend()
fig1='falk_panel_free_energy.png'
plt.savefig(fig1, dpi=160, bbox_inches='tight'); plt.close()

# Fig 2: Stress-strain (3 curves)
plt.figure()
for T in Temps:
    plt.plot(eps, sigma(eps,T,p), label=f'T={int(T)} K')
plt.xlabel(r'$\varepsilon$'); plt.ylabel(r'$\sigma(\varepsilon,T)$ [Pa] (escala relativa)')
plt.title('Falk: tensão-deformação')
plt.grid(True); plt.legend()
fig2='falk_panel_stress_strain.png'
plt.savefig(fig2, dpi=160, bbox_inches='tight'); plt.close()

# FRFs: define frequency vector
w_vec = np.linspace(5.0, 60.0, 100)

# Fig 3: FRF vs omega at T=300 K; overlay A=2.5,5.0,9.81; both Falk and linear.
plt.figure()
colors = ['blue', 'red', 'green']
for i, A in enumerate([2.5, 5.0, 9.81]):
    xr=[]; xl=[]
    for w in w_vec:
        xr.append(x_rms_Falk(A, w, 300.0, p))
        xl.append(x_rms_linear(A, w, 300.0, p))
    xr=np.array(xr); xl=np.array(xl)
    plt.plot(w_vec, xr, '-', color=colors[i], label=f'Falk A={A} m/s²')
    plt.plot(w_vec, xl, '--', color=colors[i], label=f'Linear A={A} m/s²')
plt.xlabel(r'$\omega$ [rad/s]'); plt.ylabel(r'$x_{\mathrm{RMS}}$ [m]')
plt.title('FRF — T=300 K (Falk vs Linear; A=2.5, 5.0, 9.81 m/s²)')
plt.grid(True); plt.legend()
fig3='falk_frf_T300_byA.png'
plt.savefig(fig3, dpi=160, bbox_inches='tight'); plt.close()

# Fig 4: FRF vs omega at A=5.0; overlay T=280,300,330; both Falk and linear.
plt.figure()
colors = ['blue', 'red', 'green']
Afix=5.0
for i, T in enumerate(Temps):
    xr=[]; xl=[]
    for w in w_vec:
        xr.append(x_rms_Falk(Afix, w, T, p))
        xl.append(x_rms_linear(Afix, w, T, p))
    xr=np.array(xr); xl=np.array(xl)
    plt.plot(w_vec, xr, '-', color=colors[i], label=f'Falk T={int(T)} K')
    plt.plot(w_vec, xl, '--', color=colors[i], label=f'Linear T={int(T)} K')
plt.xlabel(r'$\omega$ [rad/s]'); plt.ylabel(r'$x_{\mathrm{RMS}}$ [m]')
plt.title('FRF — A=5.0 m/s² (Falk vs Linear; T=280, 300, 330 K)')
plt.grid(True); plt.legend()
fig4='falk_frf_A5p0_byT.png'
plt.savefig(fig4, dpi=160, bbox_inches='tight'); plt.close()

[fig1, fig2, fig3, fig4]