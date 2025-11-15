"""CAKE Ordinary Differential Equation (ODE) Solver"""

import copy
import numpy as np
from scipy.integrate import solve_ivp
from cake import rate_eqs
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


# General ODE calculator
def ode_calc(t, mol, t0, stoich, stoich_loc, conc_cont, dvol, dsub, vol, dtemp, temp, k, ord, ord_loc, rate_eq, i):
    mol = mol.copy()
    mol[mol < 0] = 0
    vol_curr = vol + dvol * (t - t0)
    rates = rate_eqs.rate_calc(k, mol / vol_curr, ord, ord_loc, temp[i] + dtemp[i] * (t - t0), rate_eq)
    dmol = [sum(([rates[m[i]] * stoich[m[i], j] for i in range(len(m))])) * vol_curr + conc_cont[j] + mol[j] *
            (dsub / vol_curr) for j, m in enumerate(stoich_loc)]
    return dmol


# ODE calculator if using fixed temperature values
def ode_calc_temp_col(t, mol, t0, stoich, stoich_loc, conc_cont, dvol, dsub, vol, t_temp, temp,
                      k, ord, ord_loc, rate_eq, i):
    mol = mol.copy()  # to prevent removal of poisoning
    mol[mol < 0] = 0   # to negate poisoned species
    vol_curr = vol + dvol * (t - t0)
    rates = rate_eqs.rate_calc(k, mol / vol_curr, ord, ord_loc, temp[t_temp <= t][-1], rate_eq)
    dmol = [sum(([rates[m[i]] * stoich[m[i], j] for i in range(len(m))])) * vol_curr + conc_cont[j] + mol[j] *
            (dsub / vol_curr) for j, m in enumerate(stoich_loc)]
    return dmol


# General kinetic simulator
def eq_sim_gen(t_split, exp_t_rows, mol0, stoich, stoich_loc, disc_event, cont_event, k, ord, ord_loc, var_locs,
               fit_param, fit_param_locs, rate_eq, ode_func, rate_method='Radau', rtol=1E-3, atol=1E-6):
    # Setup total k, ord, and pois
    k_adj = copy.deepcopy(k)  # added to prevent editing of variables
    ord_adj = copy.deepcopy(ord)  # added to prevent editing of variables
    var_k, var_ord, pois = [[fit_param[i] for i in loc] for loc in fit_param_locs]
    for i, (j, m) in enumerate(var_locs[0]):
        k_adj[j] = [var_k[i] if idx == m else val for idx, val in enumerate(k_adj[j])]
    for i, (j, m) in enumerate(var_locs[1]):
        ord_adj[j] = [var_ord[i] if idx == m else val for idx, val in enumerate(ord_adj[j])]

    # Set initial conditions
    t_tot = np.empty(0)
    mol_tot = np.empty((len(mol0), 1))
    mol_tot[:, 0] = mol0
    for i in range(len(var_locs[2])):
        mol_tot[var_locs[2][i]] -= pois[i]
    conc_tot = mol_tot / cont_event.vol[0]

    # Run ODE calculator through different discrete and continuous events
    for i in range(len(cont_event.t) - 1):
        # Run discrete event
        mol_post_disc = mol_tot[:, -1] + disc_event.dmol[i] + mol_tot[:, -1] * disc_event.dsub[i] / disc_event.vol[i]
        # Run continuous event
        sol = solve_ivp(ode_func, (cont_event.t[i], cont_event.t[i + 1]), mol_post_disc, t_eval=t_split[i],
                        args=(cont_event.t[i], stoich, stoich_loc, cont_event.dmol[i], cont_event.dvol[i],
                              cont_event.dsub, cont_event.vol[i], cont_event.t_dtemp, cont_event.temp,
                              k_adj, ord_adj, ord_loc, rate_eq, i), method=rate_method, rtol=rtol, atol=atol)
        # Store data
        t_tot = np.concatenate((t_tot[:-1], sol.t))
        mol_tot = np.concatenate((mol_tot[:, :-1], sol.y), axis=1)
        conc_tot = np.concatenate((conc_tot[:, :-1], sol.y / (cont_event.vol[i] + cont_event.dvol[i] *
                                                              (sol.t - cont_event.t[i]))), axis=1)
    # Sort concentrations
    conc_tot = np.concatenate((conc_tot, sol.y[:, -1:] / cont_event.vol[-1]), axis=1)
    conc_tot = np.transpose(conc_tot)
    conc_tot[conc_tot < 0] = 0
    conc_tot = conc_tot[exp_t_rows]
    return conc_tot.astype(float)


# Mapping to real temperature value function or not
temp_map = {
    'temp_norm': ode_calc,
    'temp_col': ode_calc_temp_col,
}
