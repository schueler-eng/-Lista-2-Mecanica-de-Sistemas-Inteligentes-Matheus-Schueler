# Full set of plots for Pseudoelastic (PE) and One-Way Shape Memory Effect (SME)
# Brinson (1993) 1D model with assumed transformation kinetics (ATK).
# Generates 12 figures (6 for PE, 6 for SME).

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from mpl_toolkits.mplot3d import Axes3D

@dataclass
class SMAParams:
    EA: float = 67e3
    EM: float = 26.3e3
    eps_L: float = 0.067
    Theta: float = 0.55
    Ms: float = 18.4
    Mf: float = 9.0
    As: float = 34.5
    Af: float = 49.0
    C_M: float = 8.0
    C_A: float = 13.8
    sigma_s_cr: float = 100.0
    sigma_f_cr: float = 170.0
    T_ref: float = 0.0

def E_of_beta(beta, p): return p.EA + beta*(p.EM - p.EA)
def strain_from_sigma_beta(sigma, beta, betaS, T, p):
    E = E_of_beta(beta, p)
    return sigma/E + p.eps_L*betaS - p.Theta*(T - p.T_ref)/E
def forward_thresholds(T, p):
    if T > p.Ms:
        return p.sigma_s_cr + p.C_M*(T - p.Ms), p.sigma_f_cr + p.C_M*(T - p.Ms)
    else:
        return p.sigma_s_cr, p.sigma_f_cr
def reverse_thresholds(T, p): return p.C_A*(T - p.As), p.C_A*(T - p.Af)
def betaS_forward_cosine(sigma, T, betaS0, p):
    ss, sf = forward_thresholds(T, p)
    if sigma <= ss: return betaS0
    if sigma >= sf: return 1.0
    xi = (sigma - ss)/(sf - ss)
    return betaS0 + (1.0 - betaS0)*0.5*(1 - np.cos(np.pi*xi))
def betaS_reverse_cosine(sigma, T, betaS_at_start, p):
    ss, sf = reverse_thresholds(T, p)
    if sigma >= ss: return betaS_at_start
    if sigma <= sf: return 0.0
    xi = (ss - sigma)/(ss - sf)
    return betaS_at_start*0.5*(1 + np.cos(np.pi*xi))
def thermal_reverse_betaS(T, betaS0, p):
    if T <= p.As: return betaS0
    if T >= p.Af: return 0.0
    xi = (T - p.As)/(p.Af - p.As)
    return betaS0*0.5*(1 + np.cos(np.pi*xi))
def thermal_forward_betaT(T, betaT0, p, heating=False):
    if heating: return betaT0
    if T >= p.Ms: return 0.0
    if T <= p.Mf: return 1.0
    xi = (p.Ms - T)/(p.Ms - p.Mf)
    return 0.5*(1 - np.cos(np.pi*xi))

def simulate_pseudoelastic_time(T_const=60.0, sigma_max=600.0, t_total=4.0, N=1401, p=SMAParams()):
    t = np.linspace(0.0, t_total, N)
    half = N // 2
    sigma = np.concatenate([np.linspace(0.0, sigma_max, half), np.linspace(sigma_max, 0.0, N-half)])
    T = np.full_like(t, T_const)

    betaS = 0.0; betaT = 0.0
    reverse_started = False; betaS_at_rev_start = None
    prev_sigma = sigma[0]

    eps_hist = np.zeros_like(t)
    betaS_hist = np.zeros_like(t)
    betaT_hist = np.zeros_like(t)
    beta_hist  = np.zeros_like(t)

    for i, (sig, Ti) in enumerate(zip(sigma, T)):
        loading = (sig >= prev_sigma)
        if loading:
            betaS_new = betaS_forward_cosine(sig, Ti, betaS, p)
            betaS = max(betaS, betaS_new)
        else:
            ss_rev, _ = reverse_thresholds(Ti, p)
            if not reverse_started and sig <= ss_rev and betaS > 0:
                reverse_started = True
                betaS_at_rev_start = betaS
            if reverse_started:
                betaS = betaS_reverse_cosine(sig, Ti, betaS_at_rev_start, p)

        beta = betaS + betaT
        eps_hist[i] = strain_from_sigma_beta(sig, beta, betaS, Ti, p)
        betaS_hist[i] = betaS; betaT_hist[i] = betaT; beta_hist[i] = beta
        prev_sigma = sig

    return t, sigma, T, eps_hist, betaS_hist, betaT_hist, beta_hist

def simulate_sme_time(T0=5.0, T1=60.0, sigma_max=200.0,
                      t_load=2.0, t_unload=2.0, t_heat=2.0, t_cool=2.0,
                      N=2401, p=SMAParams()):
    # time splits
    n1 = int(N * (t_load / (t_load + t_unload + t_heat + t_cool)))
    n2 = int(N * (t_unload / (t_load + t_unload + t_heat + t_cool)))
    n3 = int(N * (t_heat / (t_load + t_unload + t_heat + t_cool)))
    n4 = N - (n1 + n2 + n3)

    t1 = np.linspace(0.0, t_load, n1, endpoint=False)
    t2 = np.linspace(t_load, t_load + t_unload, n2, endpoint=False)
    t3 = np.linspace(t_load + t_unload, t_load + t_unload + t_heat, n3, endpoint=False)
    t4 = np.linspace(t_load + t_unload + t_heat, t_load + t_unload + t_heat + t_cool, n4)
    t = np.concatenate([t1, t2, t3, t4])

    sigma = np.concatenate([np.linspace(0.0, sigma_max, n1, endpoint=False),
                            np.linspace(sigma_max, 0.0, n2, endpoint=False),
                            np.zeros(n3), np.zeros(n4)])
    T = np.concatenate([np.full(n1, T0), np.full(n2, T0),
                        np.linspace(T0, T1, n3, endpoint=False),
                        np.linspace(T1, T0, n4)])

    # initial martensite fractions at low T0
    if T0 <= p.Mf: betaT = 1.0
    elif T0 >= p.Ms: betaT = 0.0
    else:
        xi = (p.Ms - T0)/(p.Ms - p.Mf); betaT = 0.5*(1 - np.cos(np.pi*xi))
    betaS = 0.0
    beta_total_mech = betaS + betaT  # ~1 for T0 < Mf

    eps_hist = np.zeros_like(t)
    betaS_hist = np.zeros_like(t)
    betaT_hist = np.zeros_like(t)
    beta_hist  = np.zeros_like(t)

    prev_sigma = sigma[0]; prev_T = T[0]
    reverse_started = False; betaS_at_rev_start = None

    for i, (sig, Ti) in enumerate(zip(sigma, T)):
        if i < n1:  # loading at low T: detwinning (betaT -> betaS)
            betaS = betaS_forward_cosine(sig, Ti, betaS, p)
            betaS = min(betaS, beta_total_mech)
            betaT = max(beta_total_mech - betaS, 0.0)
        elif i < n1 + n2:  # unloading at low T: no reverse detwinning
            pass
        elif i < n1 + n2 + n3:  # heating at σ≈0: M+ -> A
            betaS = thermal_reverse_betaS(Ti, betaS, p)
            betaT = thermal_forward_betaT(Ti, betaT, p, heating=True)
            beta_total_mech = betaS + betaT
        else:  # cooling at σ≈0: A -> M (twinned)
            betaS = 0.0
            betaT = thermal_forward_betaT(Ti, betaT, p, heating=False)
            beta_total_mech = betaS + betaT

        beta = betaS + betaT
        eps_hist[i] = strain_from_sigma_beta(sig, beta, betaS, Ti, p)
        betaS_hist[i] = betaS; betaT_hist[i] = betaT; beta_hist[i] = beta
        prev_sigma = sig; prev_T = Ti

    return t, sigma, T, eps_hist, betaS_hist, betaT_hist, beta_hist

def plot_applied_loading(t, sigma, T, title, fname):
    fig, ax1 = plt.subplots()
    line1 = ax1.plot(t, sigma, 'b-', linewidth=2, label='σ', color='black')
    ax1.set_xlabel('t [s]')
    ax1.set_ylabel('σ [MPa]', color='black')
    ax1.set_title(title)
    ax1.tick_params(axis='y', labelcolor='black')
    
    ax2 = ax1.twinx()
    line2 = ax2.plot(t, T, 'r-', linewidth=2, label='T')
    ax2.set_ylabel('T [°C]', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Add legend with both lines
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(fname, dpi=160, bbox_inches='tight')
    plt.close()

def plot_critical_diagram(Tmin, Tmax, path_T, path_sigma, title, fname, p=SMAParams()):
    Ts = np.linspace(Tmin, Tmax, 400)
    s_s_fwd = np.array([forward_thresholds(T, p)[0] for T in Ts])
    s_f_fwd = np.array([forward_thresholds(T, p)[1] for T in Ts])
    s_s_rev = p.C_A * (Ts - p.As); s_f_rev = p.C_A * (Ts - p.Af)
    
    plt.figure()
    # Plot threshold curves in black
    plt.plot(Ts, s_s_fwd, 'k-', linewidth=1.5)
    plt.plot(Ts, s_f_fwd, 'k-', linewidth=1.5)
    plt.plot(Ts, s_s_rev, 'k-', linewidth=1.5)
    plt.plot(Ts, s_f_rev, 'k-', linewidth=1.5)
    
    # Plot applied loading path in red
    plt.plot(path_T, path_sigma, 'r-', linewidth=2.5, label='Loading path')
    
    plt.xlabel('Temperature [°C]')
    plt.ylabel('Stress [MPa]')
    plt.title(title)
    plt.ylim(bottom=0)  # Set y-axis lower limit to 0
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(fname, dpi=160, bbox_inches='tight')
    plt.close()

def plot_beta_t(t, betaS, betaT, title, fname):
    plt.figure(); plt.plot(t, betaS, label='β_S'); plt.plot(t, betaT, label='β_T')
    plt.xlabel('t [s]'); plt.ylabel('β'); plt.title(title); plt.grid(True); plt.legend()
    plt.savefig(fname, dpi=160, bbox_inches='tight'); plt.close()

def plot_3d(T, sigma, eps, title, fname):
    fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
    ax.plot(T, eps, sigma); ax.set_xlabel('T [°C]'); ax.set_ylabel('ε [-]'); ax.set_zlabel('σ [MPa]')
    ax.set_title(title); plt.savefig(fname, dpi=160); plt.close()

def plot_sigma_eps(sigma, eps, title, fname):
    plt.figure(); plt.plot(eps, sigma); plt.xlabel('ε [-]'); plt.ylabel('σ [MPa]')
    plt.title(title); plt.grid(True); plt.savefig(fname, dpi=160, bbox_inches='tight'); plt.close()

def plot_T_eps(T, eps, title, fname):
    plt.figure(); plt.plot(eps, T); plt.xlabel('ε [-]'); plt.ylabel('T [°C]')
    plt.title(title); plt.grid(True); plt.savefig(fname, dpi=160, bbox_inches='tight'); plt.close()

def main():
    p = SMAParams()

    # --- PE ---
    t_pe, sigma_pe, T_pe, eps_pe, betaS_pe, betaT_pe, _ = simulate_pseudoelastic_time(p=p)
    plot_applied_loading(t_pe, sigma_pe, T_pe, 'Applied Loading (PE)', 'pe_applied_loading.png')
    plot_critical_diagram(0.0, 60.0, np.array([60.0, 60.0]), np.array([0.0, float(np.max(sigma_pe))]),
                          'Loading Path (PE) on Critical Diagram', 'pe_critical_diagram.png', p=p)
    plot_beta_t(t_pe, betaS_pe, betaT_pe, 'Martensite Volume Fractions (PE)', 'pe_beta_t.png')
    plot_3d(T_pe, sigma_pe, eps_pe, '3D Path (PE)', 'pe_3d_T_sigma_eps.png')
    plot_sigma_eps(sigma_pe, eps_pe, 'σ × ε (PE)', 'pe_sigma_eps.png')
    plot_T_eps(T_pe, eps_pe, 'T × ε (PE)', 'pe_T_eps.png')

    # --- SME (one-way) ---
    t_sme, sigma_sme, T_sme, eps_sme, betaS_sme, betaT_sme, _ = simulate_sme_time(p=p)
    plot_applied_loading(t_sme, sigma_sme, T_sme, 'Applied Loading (SME one-way)', 'sme_applied_loading.png')
    n_mech = np.count_nonzero(sigma_sme > 0)
    path_T = np.concatenate([np.full(n_mech, T_sme[0]), T_sme[n_mech:]])
    path_sigma = np.concatenate([sigma_sme[:n_mech], np.zeros_like(T_sme[n_mech:])])
    plot_critical_diagram(0.0, 60.0, path_T, path_sigma, 'Loading Path (SME) on Critical Diagram', 'sme_critical_diagram.png', p=p)
    plot_beta_t(t_sme, betaS_sme, betaT_sme, 'Martensite Volume Fractions (SME one-way)', 'sme_beta_t.png')
    plot_3d(T_sme, sigma_sme, eps_sme, '3D Path (SME one-way)', 'sme_3d_T_sigma_eps.png')
    plot_sigma_eps(sigma_sme, eps_sme, 'σ × ε (SME one-way)', 'sme_sigma_eps.png')
    plot_T_eps(T_sme, eps_sme, 'T × ε (SME one-way)', 'sme_T_eps.png')

if __name__ == '__main__':
    main()
