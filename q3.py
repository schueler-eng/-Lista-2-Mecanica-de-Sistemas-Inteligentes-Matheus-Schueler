"""
q3.py: Coletores com modelos linear, biestável e com batente
----------------------------------------------------
Comparação de três coletores para colheita de energia:
Este script integra as EDOs via RK4 de passo fixo, varre frequências e calcula
a potência média Pm = v_RMS² / R no regime permanente. Gera gráficos Pm×ω
e um CSV com resumo (pico e largura de banda de 50%).

Observação prática: para evitar passo extremamente pequeno no modelo com
batente (ω_b muito alto), o valor padrão foi reduzido para ω_b=1500 rad/s.

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Tuple
import math

# -------------------------
# Parâmetros
# -------------------------
@dataclass
class GeneralParams:
    zeta: float = 0.025       # amortecimento [-]
    omega_n: float = 25.0     # frequência natural [rad/s]
    theta: float = 0.0045     # acoplamento eletromecânico [N/V] (por unidade de massa)
    C_p: float = 4.2e-8       # capacitância [F]
    R: float = 100e3          # resistência de carga [ohm]

@dataclass
class BistableParams:
    alpha: float = 1.0        # coef. linear (efeito "poço duplo")
    beta: float = 1.0e4       # coef. cúbico

@dataclass
class StopperParams:
    zeta_b: float = 0.025     # amortecimento adicional no contato
    omega_b: float = 1500.0   # frequência "de contato" (stiffness) [rad/s]
    g: float = 0.005          # gap [m]

# -------------------------
# Modelos (EDOs)
# -------------------------
def rhs_linear(t, y, p: GeneralParams, A: float, w: float):
    x, xd, v = y
    dd = -2*p.zeta*p.omega_n*xd - (p.omega_n**2)*x + p.theta*v + A*math.sin(w*t)
    dv = -(v/p.R + p.theta*xd)/p.C_p
    return np.array([xd, dd, dv])

def rhs_bistable(t, y, p: GeneralParams, bi: BistableParams, A: float, w: float):
    x, xd, v = y
    dd = -2*p.zeta*p.omega_n*xd + bi.alpha*x - bi.beta*(x**3) + p.theta*v + A*math.sin(w*t)
    dv = -(v/p.R + p.theta*xd)/p.C_p
    return np.array([xd, dd, dv])

def rhs_stopper(t, y, p: GeneralParams, st: StopperParams, A: float, w: float):
    x, xd, v = y
    if x < st.g:
        dd = -2*p.zeta*p.omega_n*xd - (p.omega_n**2)*x + p.theta*v + A*math.sin(w*t)
    else:
        dd = -2*(p.zeta*p.omega_n + st.zeta_b*st.omega_b)*xd \
             - (p.omega_n**2)*x - (st.omega_b**2)*(x - st.g) \
             + p.theta*v + A*math.sin(w*t)
    dv = -(v/p.R + p.theta*xd)/p.C_p
    return np.array([xd, dd, dv])

# -------------------------
# Integrador RK4 (passo fixo)
# -------------------------
def rk4(f: Callable, t0: float, tf: float, y0: np.ndarray, h: float):
    n = int(np.ceil((tf - t0)/h))
    ts = np.linspace(t0, t0 + n*h, n+1)
    Y = np.zeros((n+1, len(y0))); Y[0] = y0
    t = t0; y = y0.copy()
    for i in range(n):
        k1 = f(t, y)
        k2 = f(t + 0.5*h, y + 0.5*h*k1)
        k3 = f(t + 0.5*h, y + 0.5*h*k2)
        k4 = f(t + h, y + h*k3)
        y = y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        Y[i+1] = y; t = t + h
    return ts, Y

# -------------------------
# Simulação e varreduras
# -------------------------
def simulate(model: str, A: float, w: float,
             p: GeneralParams, bi: BistableParams, st: StopperParams,
             T: float = 6.0, dt: float = 5e-4):
    if model == 'linear':
        f = lambda t,y: rhs_linear(t, y, p, A, w)
    elif model == 'bistable':
        f = lambda t,y: rhs_bistable(t, y, p, bi, A, w)
    elif model == 'stopper':
        f = lambda t,y: rhs_stopper(t, y, p, st, A, w)
    else:
        raise ValueError('modelo desconhecido')
    ts, Y = rk4(f, 0.0, T, np.array([0.0,0.0,0.0]), dt)
    v = Y[:,2]; i0 = int(2*len(ts)/3)  # remover transiente
    v_rms = np.sqrt(np.mean(v[i0:]**2))
    Pm = (v_rms**2)/p.R
    return ts, Y, Pm

def sweep(model: str, A: float, omegas: np.ndarray,
          p: GeneralParams, bi: BistableParams, st: StopperParams,
          T: float = 6.0, dt_lin: float = 5e-4, dt_stop: float = 2e-4):
    P = []
    for w in omegas:
        dt = dt_stop if model == 'stopper' else dt_lin
        _,_,Pm = simulate(model, A, w, p, bi, st, T=T, dt=dt)
        P.append(Pm)
    return np.array(P)

# -------------------------
# Execução (gera gráficos e CSV)
# -------------------------
def main():
    p = GeneralParams(); bi = BistableParams(); st = StopperParams()
    omegas = np.linspace(5.0, 45.0, 100)
    A_cases = [2.5, 5, 9.81]

    summary = []
    for A in A_cases:
        P_lin = sweep('linear', A, omegas, p, bi, st, T=6.0)
        P_bis = sweep('bistable', A, omegas, p, bi, st, T=6.0)
        P_stp = sweep('stopper', A, omegas, p, bi, st, T=6.0)

        # Gráfico comparativo (um por A)
        plt.figure()
        plt.plot(omegas, P_lin, label='Linear')
        plt.plot(omegas, P_bis, label='Biestável')
        plt.plot(omegas, P_stp, label='Batente')
        plt.xlabel('Frequência, ω [rad/s]')
        plt.ylabel('Potência média, Pm [W]')
        plt.title(f'Pm × ω (A = {A} m/s², g = {st.g} m)')
        plt.grid(True); plt.legend()
        plt.savefig(f'harvest_Pm_vs_omega_A{str(A).replace(".","p")}.png', dpi=160, bbox_inches='tight')
        plt.close()

        # Métricas simples
        def metrics(P):
            im = int(np.argmax(P)); Pmax=P[im]; wpk=omegas[im]
            mask=P>=0.5*Pmax; bw=omegas[mask].max()-omegas[mask].min() if np.any(mask) else 0.0
            return Pmax, wpk, bw
        for name, arr in [('Linear', P_lin), ('Biestável', P_bis), ('Batente', P_stp)]:
            Pmax, wpk, bw = metrics(arr)
            summary.append({'Caso': f'A={A}', 'Sistema': name, 'Pmax[W]': Pmax, 'ω pico [rad/s]': wpk, 'Largura 50% [rad/s]': bw})

    # Efeito do gap g (mantendo A=5 m/s²) - todos na mesma figura
    plt.figure()
    for g in [0.001, 0.002, 0.005, 0.01]:
        P_g = sweep('stopper', 5.0, omegas, p, bi, StopperParams(g=g, omega_b=st.omega_b, zeta_b=st.zeta_b), T=6.0)
        plt.plot(omegas, P_g, label=f'g={g} m')
    plt.xlabel('Frequência, ω [rad/s]')
    plt.ylabel('Potência média, Pm [W]')
    plt.title('Pm × ω — Batente: Efeito do gap g (A = 5.0 m/s²)')
    plt.grid(True)
    plt.legend()
    plt.savefig('harvest_stop_Pm_vs_omega_all_gaps.png', dpi=160, bbox_inches='tight')
    plt.close()

    # Salvar CSV resumo
    df = pd.DataFrame(summary)
    df.to_csv('harvest_summary.csv', index=False)
    print('Feito. Arquivos: harvest_Pm_vs_omega_*.png, harvest_stop_Pm_vs_omega_all_gaps.png, harvest_summary.csv')

if __name__ == '__main__':
    main()
