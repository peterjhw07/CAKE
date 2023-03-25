"""CAKE Fitting Programme"""
# if __name__ == '__main__':
# import matplotlib
# matplotlib.use('Agg')
import math
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import pandas as pd
import io
import itertools
import base64
from scipy import optimize
import logging
import timeit  # to be removed
import warnings
from cake_5 import cake_prep_5 as cake_prep
from cake_5 import cake_plotting_5 as cake_plot

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


# rate equation - manipulate equation using k[x], conc[x] and ord[x] as required
#def rate_eq(k, conc, ord):
#    return k[0] * np.prod([conc[j] ** ord[j] for j in range(len(ord))])

def rate_eq(k, conc, ord):
    return k[0] * (conc[0] ** ord[0]) * conc[3] * (conc[1] / (k[1] + conc[1]))


# general kinetic simulator using Euler method
def eq_sim_gen(stoich, mol0, mol_end, add_pops, vol, vol_loss_rat, inc, k, ord, r_locs, p_locs, c_locs,
               var_k_locs, var_ord_locs, var_pois_locs, t_fit, fit_param, fit_param_locs):
    """
    Function Description

    Params
    ------

    Returns
    -------


    """
    var_k = [fit_param[i] for i in fit_param_locs[0]]
    for i, j in enumerate(var_k_locs):
        k[j] = var_k[i]
    var_ord = [fit_param[i] for i in fit_param_locs[1]]
    for i, j in enumerate(var_ord_locs):
        ord[j] = var_ord[i]
    pois = [fit_param[i] for i in fit_param_locs[2]]
    pops = np.zeros((len(t_fit), len(mol0)))
    rate = np.zeros(len(t_fit))
    pops[0] = mol0
    for i in range(len(var_pois_locs)):
        pops[:, var_pois_locs[i]] -= pois[i]
    i = 0
    for i in range(1, len(t_fit)):
        t_span = t_fit[i] - t_fit[i - 1]
        rate[i - 1] = rate_calc(i - 1, k, pops, vol, ord)
        pops[i, r_locs] = mol_calc(i, pops, vol_loss_rat, t_span, rate, stoich, vol, add_pops, -1, r_locs)
        pops[i, p_locs] = mol_calc(i, pops, vol_loss_rat, t_span, rate, stoich, vol, add_pops, 1, p_locs)
        pops[i, c_locs] = mol_calc(i, pops, vol_loss_rat, t_span, rate, stoich, vol, add_pops, 0, c_locs)
    rate[i] = rate_calc(i, k, pops, vol, ord)
    pops[pops < 0] = 0
    pops[:] = [pops[i, :] / vol[i] for i in range(0, len(t_fit))]
    exp_t_rows = list(range(0, len(t_fit), inc - 1))
    pops, rate = pops[exp_t_rows], rate[exp_t_rows]
    return [pops, rate]


def rate_calc(i, k, pops, vol, ord):
    conc = np.divide(pops[i, :], vol[i])
    conc[conc < 0] = 0
    return rate_eq(k, conc, ord)


def mol_calc(i, conc, vol_loss_rat, t_span, rate, stoich, vol, add_pops, spec_fac, spec_locs):
    return [conc[i - 1, j] * vol_loss_rat[i] + spec_fac * (t_span * rate[i - 1] * stoich[j]) * vol[i - 1]
                       + add_pops[i, j] for j in spec_locs]


def rk_rate_calc(i, k, pops, vol, ord, t_span):
    k1 = rate_calc(i, k, pops, vol, ord)
    k2 = rate_calc(i, k, np.add(pops, 0.5 * k1 * t_span), vol, ord)
    k3 = rate_calc(i, k, np.add(pops, 0.5 * k2 * t_span), vol, ord)
    k4 = rate_calc(i, k, np.add(pops, k3 * t_span), vol, ord)
    return (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# general kinetic simulator using runge-kutta method
def eq_sim_gen_rk(stoich, mol0, mol_end, add_pops, vol, vol_loss_rat, inc, k, ord, r_locs, p_locs, c_locs,
               var_k_locs, var_ord_locs, var_pois_locs, t_fit, fit_param, fit_param_locs):
    """
    Function Description

    Params
    ------

    Returns
    -------


    """
    k = fit_param[fit_param_locs[0]]
    var_ord = [fit_param[i] for i in fit_param_locs[1]]
    for i, j in enumerate(var_ord_locs):
        ord[j] = var_ord[i]
    pois = [fit_param[i] for i in fit_param_locs[2]]
    pops = np.zeros((len(t_fit), len(mol0)))
    rate = np.zeros(len(t_fit))
    pops[0] = mol0
    for i in range(len(var_pois_locs)):
        pops[:, var_pois_locs[i]] -= pois[i]
    i = 0
    for i in range(1, len(t_fit)):
        t_span = t_fit[i] - t_fit[i - 1]
        rate[i - 1] = rk_rate_calc(i - 1, k, pops, vol, ord, t_span)
        pops[i, r_locs] = mol_calc(i, pops, vol_loss_rat, t_span, rate, stoich, vol, add_pops, -1, r_locs)
        pops[i, p_locs] = mol_calc(i, pops, vol_loss_rat, t_span, rate, stoich, vol, add_pops, 1, p_locs)
        pops[i, c_locs] = mol_calc(i, pops, vol_loss_rat, t_span, rate, stoich, vol, add_pops, 0, c_locs)
    rate[i] = rk_rate_calc(i, k, pops, vol, ord, t_span)
    pops[pops < 0] = 0
    pops[:] = [pops[i, :] / vol[i] for i in range(0, len(t_fit))]
    exp_t_rows = list(range(0, len(t_fit), inc - 1))
    pops, rate = pops[exp_t_rows], rate[exp_t_rows]
    return [pops, rate]


# simulate CAKE experiments
def sim_cake(t, spec_type, react_vol_init, spec_name=None, stoich=1, mol0=None, mol_end=None, add_sol_conc=None,
             add_cont_rate=None, t_cont=None, add_one_shot=None, t_one_shot=None, add_col=None, sub_cont_rate=None,
             sub_aliq=None, t_aliq=None, sub_col=None, t_col=None, col=None, k_lim=None, ord_lim=None,
             pois_lim=None, fit_asp=None, win=1, inc=1):
    """
    Params
    ------
    t : numpy.array, list or tuple
        Time values to perform simulation with. Type numpy.array or list will use the exact values.
        Type tuple of the form (start, end, step size) will make time values using these parameters
    spec_type : str or list of str
        Type of each species: "r" for reactant, "p" for product, "c" for catalyst
    react_vol_init : float
        Initial reaction solution volume in volume_unit
    spec_name : str or list of str or None
        Name of each species. Species are given default names if none are provided
    stoich : list of int or None
        Stoichiometry of species, use "None" for catalyst. Default 1
    mol0 : list of float or None
        Initial moles of species in moles_unit or None if data do not need scaling
    mol_end : list of float or None
        Final moles of species in moles_unit or None if data do not need scaling
    add_sol_conc : list of float or None, optional
        Concentration of solution being added for each species in moles_unit volume_unit^-1.
        Default None (no addition solution for all species)
    add_cont_rate : list of float or list of tuple of float or None, optional
        Continuous addition rates of species in moles_unit volume_unit^-1 time_unit^-1.
        Default None (no continuous addition for all species)
    t_cont : list of tuple of float or None, optional
        Times at which continuous addition began for each species in time_unit^-1.
        Default None (no continuous addition for all species)
    add_one_shot : list of tuple of float or None, optional
        One shot additions in volume_unit for each species. Default None (no one shot additions for all species)
    t_one_shot : list of tuple of float or None, optional
        Times at which one shot additions occurred in time_unit^-1 for each species.
        Default None (no additions for all species)
    add_col : list of int or None, optional
        Index of addition column for each species, where addition column is in volume_unit.
        If not None, overwrites add_cont_rate, t_cont, add_one_shot and t_one_shot for each species.
        Default None (no add_col for all species)
    sub_cont_rate : float or None, optional
        Continuous addition rates of species in moles_unit volume_unit^-1 time_unit^-1.
        Default None (no continuous addition for all species)
    sub_aliq : float or list of float or None, optional
        Aliquot subtractions in volume_unit for each species. Default None (no aliquot subtractions)
    t_aliq : float or list of float or None, optional
        Times at which aliquot subtractions occurred in time_unit^-1.
        Default None (no aliquot subtractions)
    sub_col : list of int or None, optional
        Index of subtraction column, where subtraction column is in volume_unit.
        If not None, overwrites sub_cont_rate, sub_aliq and t_aliq.
        Default None (no sub_col)
    t_col : int
        Index of time column. Default 0
    col : list of int
        Index of species column. Default 1
    k_lim : float or tuple of float
        Estimated rate constant in (moles_unit volume_unit)^(sum of orders^-1 + 1) time_unit^-1.
        Can be specified as exact value for fixed variable or variable with bounds (estimate, factor difference) or
        (estimate, lower, upper). Default bounds set as (automated estimate, estimate * 1E-3, estimate * 1E3)
    ord_lim : float or list of tuple of float
        Species reaction order. Can be specified as exact value for fixed variable or
        variable with bounds (estimate, lower, upper) for each species. Default bounds set as (1, 0, 2) for "r" and "c" species and 0 for "p" species
    pois_lim : float, str or tuple of float or str, optional
        Moles of species poisoned in moles_unit. Can be specified as exact value for fixed variable,
        variable with bounds (estimate, lower, upper), or "max" with bounds (0, 0, max species concentration).
        Default assumes no poisoning occurs for all species
    fit_asp : list of str, optional
        Species to fit to: "y" to fit to species, "n" not to fit to species. Default "y"
    TIC_col : int, optional
        Index of TIC column or None if no TIC. Default None
    scale_avg_num : int, optional
        Number of data points from which to calculate mol0 and mol_end. Default 0 (no scaling)
    win : int, optional
        Smoothing window, default 1 if smoothing not required
    inc : int, optional
        Increments between adjacent points for improved simulation, default 1 for using raw time points
    """

    if type(t) is tuple:
        t = np.linspace(t[0], t[1], int((t[1] - t[0]) / t[2]) + 1)

    spec_name, num_spec, r_locs, p_locs, c_locs, stoich, mol0, mol_end, add_sol_conc, add_cont_rate, t_cont, \
    add_one_shot, t_one_shot, add_col, sub_aliq, t_aliq, t_col, col, k_lim, ord_lim, pois_lim, fit_asp, \
    var_k_locs, var_ord_locs, fix_pois_locs, var_pois_locs, fit_asp_locs, fit_param_locs, \
    inc = cake_prep.param_prep(spec_name, spec_type, stoich, mol0, mol_end, add_sol_conc, add_cont_rate, t_cont,
            add_one_shot, t_one_shot, add_col, sub_aliq, t_aliq, t_col, col, k_lim, ord_lim, pois_lim, fit_asp, inc)

    # Calculate iterative species additions and volumes
    add_pops, vol_data, vol_loss_rat = cake_prep.get_add_pops_vol(t, t, t, num_spec, react_vol_init,
                            add_sol_conc, add_cont_rate, t_cont, add_one_shot, t_one_shot, add_col,
                            sub_cont_rate, sub_aliq, t_aliq, sub_col, win=win)

    add_pops_new = np.zeros((len(t), num_spec))
    for i in range(1, len(add_pops)):
        add_pops_new[i] = add_pops[i] - add_pops[i - 1]
    add_pops = add_pops_new

    # Define initial values and lower and upper limits for parameters to fit: ord and pois
    for i in range(num_spec):
        if mol0[i] is None:
            mol0[i] = 0  # May cause issues
        if mol_end[i] is None:
            mol_end[i] = 0  # May cause issues
    for i in fix_pois_locs:
        mol0[i] -= pois_lim[i]
        mol_end[i] -= pois_lim[i]

    fit_pops_all, fit_rate_all = eq_sim_gen(stoich, mol0, mol_end, add_pops, vol_data, vol_loss_rat,
                                    inc, k_lim, ord_lim, r_locs, p_locs, c_locs, [], [], [], t, k_lim, [[], [], []])

    x_data_df = pd.DataFrame(t, columns=["Time / time_unit"])
    y_fit_conc_headers = [i + " fit conc. / moles_unit volume_unit$^{-1}$" for i in spec_name]
    y_fit_conc_df = pd.DataFrame(fit_pops_all, columns=y_fit_conc_headers)
    y_fit_rate_df = pd.DataFrame(np.reshape(fit_rate_all, (len(fit_rate_all), 1)),
                              columns=["Fit rate / moles_unit volume_unit$^{-1}$ time_unit$^{-1}$"])

    return x_data_df, y_fit_conc_df, y_fit_rate_df, ord_lim


# fit CAKE expeirments
def fit_cake(df, spec_type, react_vol_init, spec_name=None, stoich=1, mol0=None, mol_end=None, add_sol_conc=None,
             add_cont_rate=None, t_cont=None, add_one_shot=None, t_one_shot=None, add_col=None, sub_cont_rate=None,
             sub_aliq=None, t_aliq=None, sub_col=None, t_col=0, col=1, k_lim=None, ord_lim=None,
             pois_lim=None, fit_asp="y", TIC_col=None, scale_avg_num=0, win=1, inc=1):
    num_spec, spec_name, stoich, mol0, mol_end, col, k_lim, ord_lim, pois_lim, inc,\
    vol, add_pops_add, vol_data_add, vol_loss_rat_data_add, r_locs, p_locs, c_locs,\
    var_k_locs, var_ord_locs, fix_pois_locs, var_pois_locs, fit_asp_locs, fit_param_locs,\
    x_data, x_data_add, x_data_add_to_fit, y_data_to_fit, data_mod, col_ext, exp_rates = \
    pre_fit_cake(df, spec_type, react_vol_init, spec_name=spec_name, stoich=stoich, mol0=mol0, mol_end=mol_end,
                 add_sol_conc=add_sol_conc, add_cont_rate=add_cont_rate, t_cont=t_cont, add_one_shot=add_one_shot,
                 t_one_shot=t_one_shot, add_col=add_col, sub_cont_rate=sub_cont_rate, sub_aliq=sub_aliq,
                 t_aliq=t_aliq, sub_col=sub_col, t_col=t_col, col=col, k_lim=k_lim, ord_lim=ord_lim,
                 pois_lim=pois_lim, fit_asp=fit_asp, TIC_col=TIC_col, scale_avg_num=scale_avg_num, win=win, inc=inc)

    x_data, y_exp_conc, y_exp_rate, y_fit_conc, y_fit_rate, k_val, k_fit, k_fit_err, ord_fit, ord_fit_err,\
    pois_fit, pois_fit_err,fit_param_rss, fit_param_r_squared, col = \
    fitting_func(num_spec, spec_name, stoich, mol0, mol_end, col, k_lim, ord_lim, pois_lim, inc,
                 vol, add_pops_add, vol_data_add, vol_loss_rat_data_add, r_locs, p_locs, c_locs,
                 var_k_locs, var_ord_locs, fix_pois_locs, var_pois_locs, fit_asp_locs, fit_param_locs,
                 x_data, x_data_add, x_data_add_to_fit, y_data_to_fit, data_mod, col_ext, exp_rates)

    return x_data, y_exp_conc, y_exp_rate, y_fit_conc, y_fit_rate, k_val, k_fit, k_fit_err, ord_fit, ord_fit_err,\
           pois_fit, pois_fit_err, fit_param_rss, fit_param_r_squared, col, ord_lim


def pre_fit_cake(df, spec_type, react_vol_init, spec_name=None, stoich=1, mol0=None, mol_end=None, add_sol_conc=None,
             add_cont_rate=None, t_cont=None, add_one_shot=None, t_one_shot=None, add_col=None, sub_cont_rate=None,
             sub_aliq=None, t_aliq=None, sub_col=None, t_col=0, col=1, k_lim=None, ord_lim=None,
             pois_lim=None, fit_asp="y", TIC_col=None, scale_avg_num=0, win=1, inc=1):
    """
    Params
    ------
    df : pandas.DataFrame
        The reaction data
    spec_type : str or list of str
        Type of each species: "r" for reactant, "p" for product, "c" for catalyst
    react_vol_init : float
        Initial reaction solution volume in volume_unit
    spec_name : str or list of str or None
        Name of each species. Species are given default names if none are provided
    stoich : list of int or None
        Stoichiometry of species, use "None" for catalyst. Default 1
    mol0 : list of float or None
        Initial moles of species in moles_unit or None if data do not need scaling
    mol_end : list of float or None
        Final moles of species in moles_unit or None if data do not need scaling
    add_sol_conc : list of float or None, optional
        Concentration of solution being added for each species in moles_unit volume_unit^-1.
        Default None (no addition solution for all species)
    add_cont_rate : list of float or list of tuple of float or None, optional
        Continuous addition rates of species in moles_unit volume_unit^-1 time_unit^-1.
        Default None (no continuous addition for all species)
    t_cont : list of tuple of float or None, optional
        Times at which continuous addition began for each species in time_unit^-1.
        Default None (no continuous addition for all species)
    add_one_shot : list of tuple of float or None, optional
        One shot additions in volume_unit for each species. Default None (no one shot additions for all species)
    t_one_shot : list of tuple of float or None, optional
        Times at which one shot additions occurred in time_unit^-1 for each species.
        Default None (no additions for all species)
    add_col : list of int or None, optional
        Index of addition column for each species, where addition column is in volume_unit.
        If not None, overwrites add_cont_rate, t_cont, add_one_shot and t_one_shot for each species.
        Default None (no add_col for all species)
    sub_cont_rate : float or None, optional
        Continuous addition rates of species in moles_unit volume_unit^-1 time_unit^-1.
        Default None (no continuous addition for all species)
    sub_aliq : float or list of float or None, optional
        Aliquot subtractions in volume_unit for each species. Default None (no aliquot subtractions)
    t_aliq : float or list of float or None, optional
        Times at which aliquot subtractions occurred in time_unit^-1.
        Default None (no aliquot subtractions)
    sub_col : list of int or None, optional
        Index of subtraction column, where subtraction column is in volume_unit.
        If not None, overwrites sub_cont_rate, sub_aliq and t_aliq.
        Default None (no sub_col)
    t_col : int
        Index of time column. Default 0
    col : list of int
        Index of species column. Default 1
    k_lim : float or tuple of float
        Estimated rate constant in (moles_unit volume_unit)^(sum of orders^-1 + 1) time_unit^-1.
        Can be specified as exact value for fixed variable or variable with bounds (estimate, factor difference) or
        (estimate, lower, upper). Default bounds set as (automated estimate, estimate * 1E-3, estimate * 1E3)
    ord_lim : float or list of tuple of float
        Species reaction order. Can be specified as exact value for fixed variable or
        variable with bounds (estimate, lower, upper) for each species.
        Default bounds set as (1, 0, 2) for "r" and "c" species and 0 for "p" species
    pois_lim : float, str or tuple of float or str, optional
        Moles of species poisoned in moles_unit. Can be specified as exact value for fixed variable,
        variable with bounds (estimate, lower, upper), or "max" with bounds (0, 0, max species concentration).
        Default assumes no poisoning occurs for all species
    fit_asp : list of str, optional
        Species to fit to: "y" to fit to species, "n" not to fit to species. Default "y"
    TIC_col : int, optional
        Index of TIC column or None if no TIC. Default None
    scale_avg_num : int, optional
        Number of data points from which to calculate mol0 and mol_end. Default 0 (no scaling)
    win : int, optional
        Smoothing window, default 1 if smoothing not required
    inc : int, optional
        Increments between adjacent points for improved simulation, default 1 for using raw time points
    """

    def spec_err(spec_name, ord_min, ord_max, var_ord_locs, var_pois_locs, x_data_add_to_fit, y_data_to_fit, k_first_guess, pois_val, pois_min, pois_max):
        real_err_inc = 20  # enter number of increments between min and max orders
        real_err_inc += 1
        test_ord_pre = [np.linspace(ord_min[i], ord_max[i], real_err_inc).tolist() for i in range(len(var_ord_locs))]
        test_ord = list(itertools.product(*test_ord_pre))
        k_sec_guess = np.zeros([len(test_ord), len(var_ord_locs) + len(var_pois_locs) + 2])
        k_sec_guess[:] = [[*test_ord[i], 0, *np.zeros(len(var_pois_locs)), 0] for i in range(len(test_ord))]
        for i in range(len(k_sec_guess)):
            k_sec_guess_res = optimize.curve_fit(lambda x_data, k, *pois: eq_sim_fit(x_data, k,
                                                                                     *k_sec_guess[i,
                                                                                      :len(var_ord_locs)], *pois),
                                                 x_data_add_to_fit, y_data_to_fit,
                                                 [k_first_guess, *pois_val], maxfev=10000,
                                                 bounds=((k_first_guess * bound_adj,
                                                          *pois_min), (k_first_guess / bound_adj, *pois_max)))
            k_sec_guess[i, len(var_ord_locs):-1] = k_sec_guess_res[0]
            fit_guess = eq_sim_fit(x_data_add_to_fit, k_sec_guess[i, len(var_ord_locs)],
                                   *k_sec_guess[i, :len(var_ord_locs)],
                                   *k_sec_guess[i, len(var_ord_locs) + 1:-len(var_pois_locs) - 1])
            _, k_sec_guess[i, -1] = residuals(y_data_to_fit, fit_guess)
            eq_sim_gen(stoich, mol0, mol_end, add_pops_add, vol_data_add, vol_loss_rat_data_add,
                       inc, k_lim, ord_lim, r_locs, p_locs, c_locs, var_k_locs, var_ord_locs,
                       var_pois_locs, x_data_add, fit_param, fit_param_locs)
        real_err_calc_sort = k_sec_guess[k_sec_guess[:, -1].argsort()[::-1]]
        headers = [*[spec_name[i] + " order" for i in var_ord_locs], "k", *[spec_name[i] + " poisoning" for i in var_pois_locs], "r^2"]
        real_err_df = pd.DataFrame(real_err_calc_sort, columns=headers)
        return real_err_df

    spec_name, num_spec, r_locs, p_locs, c_locs, stoich, mol0, mol_end, add_sol_conc, add_cont_rate, t_cont, \
    add_one_shot, t_one_shot, add_col, sub_aliq, t_aliq, t_col, col, k_lim, ord_lim, pois_lim, fit_asp, \
    var_k_locs, var_ord_locs, fix_pois_locs, var_pois_locs, fit_asp_locs, fit_param_locs, \
    inc = cake_prep.param_prep(spec_name, spec_type, stoich, mol0, mol_end, add_sol_conc, add_cont_rate, t_cont, add_one_shot,
                     t_one_shot, add_col, sub_aliq, t_aliq, t_col, col, k_lim, ord_lim, pois_lim, fit_asp, inc)

    # Get x_data
    data_org = df.to_numpy()
    x_data = cake_prep.data_smooth(data_org, t_col, win)
    x_data_add = cake_prep.add_sim(np.reshape(x_data, (len(x_data))), inc)

    # Get TIC
    if TIC_col is not None:
        TIC = cake_prep.data_smooth(data_org, TIC_col, win)
    else:
        TIC = None

    # Calculate iterative species additions and volumes
    add_pops, vol, vol_loss_rat = cake_prep.get_add_pops_vol(data_org, data_org[:, t_col], x_data, num_spec, 
                                            react_vol_init, add_sol_conc, add_cont_rate, t_cont, add_one_shot, 
                                            t_one_shot, add_col, sub_cont_rate, sub_aliq, t_aliq, sub_col, win=win)

    add_pops_add = np.zeros((len(x_data_add), num_spec))
    for i in range(num_spec):
        add_pops_add[:, i] = cake_prep.add_sim(add_pops[:, i], inc)
    add_pops_add_new = np.zeros((len(x_data_add), num_spec))
    for i in range(1, len(add_pops_add)):
        add_pops_add_new[i] = add_pops_add[i] - add_pops_add[i - 1]
    add_pops_add = add_pops_add_new
    vol_data_add = cake_prep.add_sim(vol, inc)
    vol_loss_rat_data_add = cake_prep.add_sim(vol_loss_rat, inc)
    # Determine mol0, mol_end and scale data as required
    data_mod = np.empty((len(x_data), num_spec))
    col_ext = []
    for i in range(num_spec):
        if col[i] is not None:
            col_ext = [*col_ext, i]
            data_i = cake_prep.data_smooth(data_org, col[i], win)
            data_i = cake_prep.tic_norm(data_i, TIC)
            if mol0[i] is None and scale_avg_num == 0:
                mol0[i] = data_i[0] * vol[0]
            elif mol0[i] is None and scale_avg_num > 0:
                mol0[i] = np.mean([data_i[j] * vol[j] for j in range(scale_avg_num)])
            elif mol0[i] is not None and mol0[i] != 0 and scale_avg_num > 0 and (
                        mol_end[i] is None or mol0[i] >= mol_end[i]):
                data_scale = np.mean([data_i[j] / (mol0[i] / vol[j]) for j in range(scale_avg_num)])
                data_i = data_i / data_scale
            if mol_end[i] is None and scale_avg_num == 0:
                mol_end[i] = data_i[-1] * vol[-1]
            elif mol_end[i] is None and scale_avg_num > 0:
                mol_end[i] = np.mean([data_i[j] * vol[j] for j in range(-scale_avg_num, 0)])
            elif mol_end[i] is not None and mol_end[i] != 0 and scale_avg_num > 0 and (
                        mol0[i] is None or mol_end[i] >= mol0[i]):
                data_scale = np.mean([data_i[j] / (mol_end[i] / vol[j]) for j in range(-scale_avg_num, 0)])
                data_i = data_i / data_scale
            data_mod[:, i] = data_i
        if col[i] is None and mol0[i] is None:
            mol0[i] = 0  # May cause issues
        if col[i] is None and mol_end[i] is None:
            mol_end[i] = 0  # May cause issues

    warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
    exp_t_rows = list(range(0, len(x_data_add), inc - 1))
    exp_rates = np.zeros((len(exp_t_rows), num_spec))
    for i in range(num_spec):
        if col[i] is not None:
            exp_rates[:, i] = np.gradient(data_mod[:, i], x_data_add[exp_t_rows])

    # Manipulate data for fitting
    x_data_add_to_fit = np.empty(0)
    y_data_to_fit = np.empty(0)
    for i in range(len(fit_asp_locs)):
        x_data_add_to_fit = np.append(x_data_add_to_fit, x_data_add, axis=0)
        y_data_to_fit = np.append(y_data_to_fit, data_mod[:, fit_asp_locs[i]], axis=0)
    return num_spec, spec_name, stoich, mol0, mol_end, col, k_lim, ord_lim, pois_lim, inc,\
           vol, add_pops_add, vol_data_add, vol_loss_rat_data_add, r_locs, p_locs, c_locs,\
           var_k_locs, var_ord_locs, fix_pois_locs, var_pois_locs, fit_asp_locs, fit_param_locs,\
           x_data, x_data_add, x_data_add_to_fit, y_data_to_fit, data_mod, col_ext, exp_rates


def fitting_func(num_spec, spec_name, stoich, mol0, mol_end, col, k_lim, ord_lim, pois_lim, inc,
                 vol, add_pops_add, vol_data_add, vol_loss_rat_data_add, r_locs, p_locs, c_locs,
                 var_k_locs, var_ord_locs, fix_pois_locs, var_pois_locs, fit_asp_locs, fit_param_locs,
                 x_data, x_data_add, x_data_add_to_fit, y_data_to_fit, data_mod, col_ext, exp_rates):

    def eq_sim_fit(x_data_sim, *fit_param):
        x_data_fit = x_data_sim[:int(len(x_data_sim) / len(fit_asp_locs))]
        pops, rate = eq_sim_gen(stoich, mol0, mol_end, add_pops_add, vol_data_add, vol_loss_rat_data_add,
                                inc, k_lim, ord_lim, r_locs, p_locs, c_locs, var_k_locs,
                                var_ord_locs, var_pois_locs, x_data_fit, fit_param, fit_param_locs)
        pops_reshape = np.empty(0)
        for i in fit_asp_locs:
            pops_reshape = np.append(pops_reshape, pops[:, i], axis=0)
        return pops_reshape

    # Define initial values and lower and upper limits for parameters to fit: ord and pois
    bound_adj = 1E-3
    ord_val, ord_min, ord_max, pois_val, pois_min, pois_max = [], [], [], [], [], []
    for i in range(len(var_ord_locs)):
        unpack_ord_lim = ord_lim[var_ord_locs[i]]
        ord_val.append(unpack_ord_lim[0])
        ord_min.append(unpack_ord_lim[1])
        ord_max.append(unpack_ord_lim[2])
    for i in range(len(var_pois_locs)):
        unpack_pois_lim = pois_lim[var_pois_locs[i]]
        if "max" in unpack_pois_lim:
            pois_val.append(0)
            pois_min.append(0)
            pois_max.append(max(mol0[i], mol_end[i]))
        else:
            pois_val.append(unpack_pois_lim[0])
            pois_min.append(unpack_pois_lim[1])
            pois_max.append(unpack_pois_lim[2])
    for i in fix_pois_locs:
        mol0[i] -= pois_lim[i]
        mol_end[i] -= pois_lim[i]

    # Define initial values and lower and upper limits for parameters to fit: k
    k_val, k_min, k_max = [], [], []
    for j, i in enumerate(var_k_locs):
        if k_lim is None or (isinstance(k_lim, (int, float)) and k_lim == 0) or \
                (isinstance(k_lim, (tuple, list)) and isinstance(k_lim[0], (tuple, list)) and len(k_lim) == 1 and (k_lim[0][0] is None or k_lim[0][0] == 0)):  # need to find a better way to estimate k
            k_first_guess = np.zeros([len(range(-13, 13)), 2])
            k_first_guess[:] = [[10 ** i, 0] for i in range(-13, 13)]
            for i in range(len(k_first_guess)):
                fit_guess = eq_sim_fit(x_data_add_to_fit, k_first_guess[i, 0], *ord_val, *pois_val)
                _, k_first_guess[i, -1] = residuals(y_data_to_fit, fit_guess)
            index = np.where(k_first_guess[:, -1] == max(k_first_guess[:, -1]))
            k_first_guess = float(k_first_guess[index[0][0], 0])

            k_sec_guess_switch = "n"
            if "y" in k_sec_guess_switch:
                test_ord_pre = [list(range(round(ord_min[i]), round(ord_max[i]) + 1)) for i in range(len(var_ord_locs))]
                test_ord = list(itertools.product(*test_ord_pre))
                k_sec_guess = np.zeros([len(test_ord), len(var_ord_locs) + len(var_pois_locs) + 2])
                k_sec_guess[:] = [[*test_ord[i], 0, *np.zeros(len(var_pois_locs)), 0] for i in range(len(test_ord))]
                for i in range(len(k_sec_guess)):
                    k_sec_guess_res = optimize.curve_fit(lambda x_data, k, *pois: eq_sim_fit(x_data, k,
                                        *k_sec_guess[i, :len(var_ord_locs)], *pois), x_data_add_to_fit, y_data_to_fit,
                                        [k_first_guess, *pois_val], maxfev=10000, bounds=((k_first_guess *bound_adj,
                                        *pois_min), (k_first_guess / bound_adj, *pois_max)))
                    k_sec_guess[i, len(var_ord_locs):-1] = k_sec_guess_res[0]
                    fit_guess = eq_sim_fit(x_data_add_to_fit, k_sec_guess[i, len(var_ord_locs)],
                                           *k_sec_guess[i, :len(var_ord_locs)],
                                           *k_sec_guess[i, len(var_ord_locs) + 1:-len(var_pois_locs) - 1])
                    _, k_sec_guess[i, -1] = residuals(y_data_to_fit, fit_guess)
                index = np.where(k_sec_guess[:, -1] == max(k_sec_guess[:, -1]))
                k_val.append(float(k_sec_guess[index[0], -3]))
            else:
                k_val.append(k_first_guess)
            k_min.append(k_val[j] * bound_adj)
            k_max.append(k_val[j] / bound_adj)
        elif isinstance(k_lim[i], tuple) and len(k_lim[i]) == 2:
            k_val.append(k_lim[i][0])
            k_min.append(k_val[j] * k_lim[i][1])
            k_max.append(k_val[j] / k_lim[i][1])
        elif isinstance(k_lim[i], tuple) and len(k_lim[i]) == 3:
            k_val.append(k_lim[i][0])
            k_min.append(k_lim[i][1])
            k_max.append(k_lim[i][2])

    init_param = [*k_val, *ord_val, *pois_val]
    low_bounds = [*k_min, *ord_min, *pois_min]
    up_bounds = [*k_max, *ord_max, *pois_max]

    if not init_param or not low_bounds or not up_bounds:
        print("No parameters set to fit - no fitting applied.")
        x_data = pd.DataFrame(x_data, columns=["Time / time_unit"])
        y_exp_conc_headers = [spec_name[i] + " exp. conc. / moles_unit volume_unit$^{-1}$"
                              for i in range(num_spec) if col[i] is not None]
        y_exp_rate_headers = [spec_name[i] + " exp. rate / moles_unit volume_unit$^{-1}$ time_unit$^{-1}$"
                              for i in range(num_spec) if col[i] is not None]
        y_exp_conc = pd.DataFrame(data_mod[:, col_ext], columns=y_exp_conc_headers)
        y_exp_rate = pd.DataFrame(exp_rates[:, col_ext], columns=y_exp_rate_headers)
        return x_data, y_exp_conc, y_exp_rate, None, None, [], [], [],\
           [], [], [], [], [], [], col

    # Apply fittings, determine optimal parameters and determine resulting fits
    fit_param_res = optimize.curve_fit(eq_sim_fit, x_data_add_to_fit, y_data_to_fit, init_param, maxfev=10000,
                                       bounds=(low_bounds, up_bounds))
    fit_param = fit_param_res[0]
    k_fit = fit_param[fit_param_locs[0]]
    ord_fit = fit_param[fit_param_locs[1]]
    pois_fit = fit_param[fit_param_locs[2]]
    # print(k_fit, *ord_fit, *pois_fit)
    fit_pops_set = eq_sim_fit(x_data_add_to_fit, *k_fit, *ord_fit, *pois_fit)
    fit_pops_all, fit_rate_all = eq_sim_gen(stoich, mol0, mol_end, add_pops_add, vol_data_add, vol_loss_rat_data_add,
                                            inc, k_lim, ord_lim, r_locs, p_locs, c_locs, var_k_locs, var_ord_locs,
                                            var_pois_locs, x_data_add, fit_param, fit_param_locs)

    # Calculate residuals and errors
    fit_param_err = np.sqrt(np.diag(fit_param_res[1]))  # for 1SD
    k_fit_err = fit_param_err[fit_param_locs[0]]
    ord_fit_err = fit_param_err[fit_param_locs[1]]
    pois_fit_err = fit_param_err[fit_param_locs[2]]
    fit_param_rss, fit_param_r_squared = cake_prep.residuals(y_data_to_fit, fit_pops_set)
    fit_param_aic = len(y_data_to_fit) * math.log(fit_param_rss / len(y_data_to_fit)) + 2 * len(init_param)

    # Prepare data for output
    x_data = pd.DataFrame(x_data, columns=["Time / time_unit"])
    y_exp_conc_headers = [spec_name[i] + " exp. conc. / moles_unit volume_unit$^{-1}$"
                          for i in range(num_spec) if col[i] is not None]
    y_exp_rate_headers = [spec_name[i] + " exp. rate / moles_unit volume_unit$^{-1}$ time_unit$^{-1}$"
                          for i in range(num_spec) if col[i] is not None]
    y_fit_conc_headers = [i + " fit conc. / moles_unit volume_unit$^{-1}$" for i in spec_name]
    y_exp_conc = pd.DataFrame(data_mod[:, col_ext], columns=y_exp_conc_headers)
    y_exp_rate = pd.DataFrame(exp_rates[:, col_ext], columns=y_exp_rate_headers)
    y_fit_conc = pd.DataFrame(fit_pops_all, columns=y_fit_conc_headers)
    y_fit_rate = pd.DataFrame(fit_rate_all, columns=["Fit rate / moles_unit volume_unit$^{-1}$ time_unit$^{-1}$"])

    if not var_k_locs:
        k_val, k_fit, k_fit_err = "N/A", "N/A", "N/A"
    if not var_ord_locs:
        ord_fit, ord_fit_err = "N/A", "N/A"
    if not var_pois_locs:
        pois_fit, pois_fit_err = "N/A", "N/A"
        t_del_fit, t_del_fit_err = "N/A", "N/A"
    else:
        pois_fit, pois_fit_err = pois_fit / vol[0], pois_fit_err / vol[0]
        t_del_fit, t_del_fit_err = pois_fit * 1, pois_fit_err * 1  # need to make t_del work somehow

    return x_data, y_exp_conc, y_exp_rate, y_fit_conc, y_fit_rate, k_val, k_fit, k_fit_err,\
           ord_fit, ord_fit_err, pois_fit, pois_fit_err, fit_param_rss, fit_param_r_squared, col


def fit_err_real(df, spec_type, react_vol_init, spec_name=None, stoich=1, mol0=None, mol_end=None, add_sol_conc=None,
             add_cont_rate=None, t_cont=None, add_one_shot=None, t_one_shot=None, add_col=None, sub_cont_rate=None,
             sub_aliq=None, t_aliq=None, sub_col=None, t_col=0, col=1, k_lim=None, ord_lim=None,
             pois_lim=None, fit_asp="y", TIC_col=None, scale_avg_num=0, win=1, inc=1):

    x_data, y_exp_conc, y_exp_rate, y_fit_conc, y_fit_rate, k_val, k_fit, k_fit_err, ord_fit, ord_fit_err, \
    pois_fit, pois_fit_err, fit_param_rss, fit_param_r_squared, col, ord_lim = \
        fit_cake(df, spec_type, react_vol_init, spec_name=spec_name, stoich=stoich, mol0=mol0, mol_end=mol_end,
                 add_sol_conc=add_sol_conc, add_cont_rate=add_cont_rate, t_cont=t_cont, add_one_shot=add_one_shot,
                 t_one_shot=t_one_shot, add_col=add_col, sub_cont_rate=sub_cont_rate, sub_aliq=sub_aliq,
                 t_aliq=t_aliq, sub_col=sub_col, t_col=t_col, col=col, k_lim=k_lim, ord_lim=ord_lim,
                 pois_lim=pois_lim, fit_asp=fit_asp, TIC_col=TIC_col, scale_avg_num=scale_avg_num, win=win, inc=inc)
    bound_adj = 1E-3
    k_lim = [k_val, bound_adj]

    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ord_lim, _, _, _, _, _, _, _, _, _, _, _ = \
    cake_prep.param_prep(spec_name, spec_type, stoich, mol0, mol_end, add_sol_conc, add_cont_rate, t_cont,
               add_one_shot, t_one_shot, add_col, sub_aliq, t_aliq, t_col, col, k_lim, ord_lim, pois_lim, fit_asp, inc)

    real_err_inc = 10  # enter number of increments between min and max orders
    real_err_inc += 1
    test_ord_var_pre = [np.round(np.linspace(ord_lim[i][1], ord_lim[i][2], real_err_inc), 1).tolist() for i in range(len(ord_lim))
                        if isinstance(ord_lim[i], tuple)]
    test_ord_var = list(itertools.product(*test_ord_var_pre))
    test_ord_all_pre = [np.linspace(ord_lim[i][1], ord_lim[i][2], real_err_inc).tolist()
                        if isinstance(ord_lim[i], tuple) else [ord_lim[i]] for i in range(len(ord_lim))]
    test_ord_all = list(itertools.product(*test_ord_all_pre))

    real_err_fit = np.empty([1, 4], dtype=object)
    real_err_fit_y_fit_conc, real_err_fit_y_fit_rate = np.empty([1, 1], dtype=object), np.empty([1, 1], dtype=object)
    real_err_fit[0] = [ord_fit, k_fit, pois_fit, fit_param_r_squared]
    real_err_fit_y_fit_conc[0] = [y_fit_conc]
    real_err_fit_y_fit_rate[0] = [y_fit_rate]

    num_spec, spec_name, stoich, mol0, mol_end, col, k_lim, _, pois_lim, inc, \
    vol, add_pops_add, vol_data_add, vol_loss_rat_data_add, r_locs, p_locs, c_locs, \
    var_k_locs, var_ord_locs, fix_pois_locs, var_pois_locs, fit_asp_locs, fit_param_locs, \
    x_data, x_data_add, x_data_add_to_fit, y_data_to_fit, data_mod, col_ext, exp_rates = \
    pre_fit_cake(df, spec_type, react_vol_init, spec_name=spec_name, stoich=stoich, mol0=mol0, mol_end=mol_end,
                 add_sol_conc=add_sol_conc, add_cont_rate=add_cont_rate, t_cont=t_cont, add_one_shot=add_one_shot,
                 t_one_shot=t_one_shot, add_col=add_col, sub_cont_rate=sub_cont_rate, sub_aliq=sub_aliq,
                 t_aliq=t_aliq, sub_col=sub_col, t_col=t_col, col=col, k_lim=k_lim, ord_lim=list(test_ord_all[0]),
                 pois_lim=pois_lim, fit_asp=fit_asp, TIC_col=TIC_col, scale_avg_num=scale_avg_num, win=win, inc=inc)

    real_err = np.empty([len(test_ord_all), 4], dtype=object)
    real_err_y_fit_conc = np.empty([len(test_ord_all), 1], dtype=object)
    real_err_y_fit_rate = np.empty([len(test_ord_all), 1], dtype=object)
    for i in range(len(test_ord_all)):
        x_data, _, _, y_fit_conc, y_fit_rate, k_val, k_fit, k_fit_err, ord_fit, ord_fit_err, \
        pois_fit, pois_fit_err, fit_param_rss, fit_param_r_squared, col = \
        fitting_func(num_spec, spec_name, stoich, mol0, mol_end, col, k_lim, list(test_ord_all[i]), pois_lim, inc,
                     vol, add_pops_add, vol_data_add, vol_loss_rat_data_add, r_locs, p_locs, c_locs,
                     var_k_locs, var_ord_locs, fix_pois_locs, var_pois_locs, fit_asp_locs, fit_param_locs,
                     x_data, x_data_add, x_data_add_to_fit, y_data_to_fit, data_mod, col_ext, exp_rates)
        real_err[i] = [[*test_ord_var[i]], k_fit, pois_fit, fit_param_r_squared]
        real_err_y_fit_conc[i] = [y_fit_conc]
        real_err_y_fit_rate[i] = [y_fit_rate]

    real_err_sort = np.concatenate((real_err_fit, real_err[real_err[:, -1].argsort()[::-1]]))
    real_err_y_fit_conc_sort = np.concatenate((real_err_fit_y_fit_conc, real_err_y_fit_conc[real_err[:, -1].argsort()[::-1]]))
    real_err_y_fit_rate_sort = np.concatenate((real_err_fit_y_fit_rate, real_err_y_fit_rate[real_err[:, -1].argsort()[::-1]]))
    headers = ["Order", "k", "Poisoning", "r^2"]
    real_err_sort_df = pd.DataFrame(real_err_sort, columns=headers)
    print(real_err_sort_df)
    return x_data, y_exp_conc, y_exp_rate, real_err_y_fit_conc_sort, real_err_y_fit_rate_sort, real_err_sort_df, col, ord_lim


if __name__ == "__main__":
    spec_name = ["r1", "r2", "p1", "c1"]
    spec_type = ["r", "r", "p", "c"]
    react_vol_init = 0.1
    stoich = [1, 1, 1, None]  # insert stoichiometry of reactant, r
    mol0 = [0.1, 0.2, 0, 0]
    mol_end = [0, 0.1, 0.1, None]
    add_sol_conc = [None, None, None, 10]
    add_cont_rate = [None, None, None, 0.001]
    t_cont = [None, None, None, 1]
    t_col = 0
    col = [1, 2, 3, None]
    ord_lim = [(1, 0, 2), 1, 0, (1, 0, 2)]
    fit_asp = ["y", "n", "y", "n"]
    file_name = r'C:\Users\Peter\Documents\Postdoctorate\Work\CAKE\Case studies\CAKE preliminary trials.xlsx'
    sheet_name = r'Test_data'
    pic_save = r'/Users/bhenders/Desktop/CAKE/cake_app_test.png'
    xlsx_save = r'/Users/bhenders/Desktop/CAKE/fit_data.xlsx'

    df = cake_prep.read_data(file_name, sheet_name, t_col, col)
    output = fit_cake(df, spec_type, react_vol_init, spec_name=spec_name, stoich=stoich, mol0=mol0, mol_end=mol_end,
                      add_sol_conc=add_sol_conc, add_cont_rate=add_cont_rate, t_cont=t_cont,
                      t_col=t_col, col=col, ord_lim=ord_lim, fit_asp=fit_asp)
    x_data_df, y_exp_df, y_fit_conc_df, y_fit_rate_df, k_val_est, k_fit, k_fit_err, \
    ord_fit, ord_fit_err, pois_fit, pois_fit_err, ss_res, r_squared, col = output

    if not isinstance(col, (tuple, list)): col = [col]

    html = cake_plot.plot_fit_results(x_data_df, y_exp_df, y_fit_conc_df, col,
                            f_format='svg', return_image=False, save_disk=True, save_to=pic_save)

    param_dict = cake_app.make_param_dict(spec_type, react_vol_init, stoich=stoich, mol0=mol0, mol_end=mol_end,
                                 add_sol_conc=add_sol_conc, add_cont_rate=add_cont_rate, t_cont=t_cont,
                                 t_col=t_col, col=col, ord_lim=None, fit_asp=fit_asp)

    # write_fit_data(xlsx_save, df, param_dict, t, r, p, fit_p, fit_r, res_val, res_err, ss_res, r_squared, cat_pois)
    file, _ = cake_app.write_fit_data_temp(df, param_dict, x_data_df, y_exp_df, y_fit_conc_df, y_fit_rate_df,
                                  k_fit, k_fit_err, ord_fit, ord_fit_err, pois_fit, pois_fit_err, ss_res, r_squared)
    file.seek(0)
    with open(xlsx_save, "wb") as f:
        f.write(file.getbuffer())
