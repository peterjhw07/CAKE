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


# define additional t values for data sets with few data points
def add_sim(s, inc):
    """
    Function Description

    Params
    ------

    Returns
    -------


    """
    s_fit = np.zeros((((len(s) - 1) * (inc - 1)) + 1))
    for i in range(len(s) - 1):
        new_s_i = np.linspace(s[i], s[i + 1], num=inc)[:-1]
        s_fit[i * len(new_s_i):(i * len(new_s_i)) + len(new_s_i)] = new_s_i
    s_fit[-1] = s[-1]
    return s_fit


# smooth data (if required)
def data_smooth(arr, d_col, win=1):
    """
    Function Description

    Params
    ------

    Returns
    -------


    """
    if win <= 1:
        d_ra = arr[:, d_col]
    else:
        ret = np.cumsum(arr[:, d_col], dtype=float)
        ret[win:] = ret[win:] - ret[:-win]
        d_ra = ret[win - 1:] / win
    return d_ra


# manipulate to TIC values (for MS only)
def tic_norm(data, tic=None):
    """
    Function Description

    Params
    ------

    Returns
    -------


    """
    if tic is not None:
        data = data / tic
    else:
        data = data
    return data


# calculate residuals
def residuals(y_data, fit):
    """
    Function Description

    Params
    ------

    Returns
    -------


    """
    rss = np.sum((y_data - fit) ** 2)
    r_squared = 1 - (rss / np.sum((y_data - np.mean(y_data)) ** 2))
    return [rss, r_squared]


# find nearest value
def find_nearest(array, value):
    """
        Find nearest element to value in array

        Params
        ------

        Returns
        -------


    """
    array = np.asarray(array)
    index = (np.abs(array - value)).argmin()
    return index


# return all None to ensure correct list lengths for iterative purposes
def return_all_nones(s, num_spec):
    if s is None: s = [None] * num_spec
    return s


# convert non-list to list
def type_to_list(s):
    if not isinstance(s, list):
        s = [s]
    return s


# convert int and float into lists inside tuples
def tuple_of_lists_from_tuple_of_int_float(s):
    s_list = []
    for i in range(len(s)):
        if isinstance(s[i], (int, float)):
            s_list = [*s_list, [s[i]]]
        else:
            s_list = [*s_list, s[i]]
    return s_list


# read imported data
def read_data(file_name, sheet_name, t_col, col, add_col, sub_col):
    """
    Read in data from excel filename

    Params
    ------

    Returns
    -------


    """
    df = pd.read_excel(file_name, sheet_name=sheet_name, engine='openpyxl', dtype=str)
    headers = list(pd.read_excel(file_name, sheet_name=sheet_name, engine='openpyxl').columns)
    if isinstance(col, int): col = [col]
    if add_col is None: add_col = [None]
    elif isinstance(add_col, int): add_col = [add_col]
    conv_col = [i for i in [t_col, *col, *add_col, sub_col] if i is not None]
    try:
        for i in conv_col:
            df[headers[i]] = pd.to_numeric(df[headers[i]], downcast="float")
        return df
    except ValueError:
        pass
    try:
        for i in conv_col:
            df[headers[i]] = pd.to_numeric(df[headers[i]], downcast="float")
        return df
    except ValueError:
        raise ValueError("Excel file must contain data rows (i.e. col specified) of numerical input with at most 1 header row.")


# prepare parameters
def param_prep(spec_name, spec_type, stoich, mol0, mol_end, add_sol_conc, add_cont_rate, t_cont, add_one_shot,
               t_one_shot, add_col, sub_aliq, t_aliq, t_col, col, k_lim, ord_lim, pois_lim, fit_asp, inc):
    spec_type = type_to_list(spec_type)
    num_spec = len(spec_type)
    r_locs = [i for i in range(num_spec) if 'r' in spec_type[i]]
    p_locs = [i for i in range(num_spec) if 'p' in spec_type[i]]
    c_locs = [i for i in range(num_spec) if 'c' in spec_type[i]]

    if stoich is None: stoich = [1] * num_spec
    elif isinstance(stoich, int): stoich = [stoich]
    else: stoich = [i if i is not None else 0 for i in stoich]
    spec_name, mol0, mol_end, add_sol_conc, add_cont_rate, t_cont, add_one_shot, t_one_shot, add_col, \
    fit_asp = map(return_all_nones, [spec_name, mol0, mol_end, add_sol_conc, add_cont_rate, t_cont,
                                     add_one_shot, t_one_shot, add_col, fit_asp], [num_spec] * 10)
    for i in range(num_spec):
        if spec_name[i] is None:
            spec_name[i] = "Species " + str(i + 1)
    if k_lim is None: k_lim = [(None, None, None)]
    elif isinstance(k_lim, (int, float)): k_lim = [k_lim]
    else:
        for i in range(len(k_lim)):
            if k_lim[i] is None:
                k_lim[i] = (None, None, None)
    if ord_lim is None:
        ord_lim = []
        for i in spec_type:
            if 'r' in i or 'c' in i:
                ord_lim.append((1, 0, 2))
            elif 'p' in i:
                ord_lim.append(0)
    elif p_locs:
        for i in p_locs:
            if ord_lim[i] is None: ord_lim[i] = 0
    if pois_lim is None: pois_lim = [0] * num_spec

    spec_name, stoich, mol0, mol_end, add_sol_conc, add_cont_rate, t_cont, add_one_shot, t_one_shot, add_col, \
    sub_aliq, t_aliq, t_col, col, ord_lim, pois_lim, fit_asp = map(type_to_list, [spec_name,
    stoich, mol0, mol_end, add_sol_conc, add_cont_rate, t_cont, add_one_shot, t_one_shot, add_col,
    sub_aliq, t_aliq, t_col, col, ord_lim, pois_lim, fit_asp])
    add_cont_rate, t_cont, add_one_shot, t_one_shot = map(tuple_of_lists_from_tuple_of_int_float,
                                            [add_cont_rate, t_cont, add_one_shot, t_one_shot])
    # print(spec_name, spec_type, stoich, mol0, mol_end, add_sol_conc, add_cont_rate, t_cont,
    #      add_one_shot, t_one_shot, add_col, sub_aliq, t_aliq, t_col, col,
    #      ord_lim, pois_lim, fit_asp)
    var_k_locs = [i for i in range(len(k_lim)) if (isinstance(k_lim[i], (tuple, list)) and len(k_lim[i]) > 1)]
    var_ord_locs = [i for i in range(num_spec) if (isinstance(ord_lim[i], (tuple, list)) and len(ord_lim[i]) > 1)]
    fix_pois_locs = [i for i in range(num_spec) if (isinstance(pois_lim[i], (int, float))
                    or (isinstance(pois_lim[i], (tuple, list)) and len(pois_lim[i]) == 1))]
    var_pois_locs = [i for i in range(num_spec) if (isinstance(pois_lim[i], (tuple, list, str))
                                                    and len(pois_lim[i]) > 1)]
    fit_asp_locs = [i for i in range(num_spec) if fit_asp[i] is not None and 'y' in fit_asp[i]]
    fit_param_locs = [range(0, len(var_k_locs)), range(len(var_k_locs), len(var_k_locs) + len(var_ord_locs)),
                      range(len(var_k_locs) + len(var_ord_locs),
                            len(var_k_locs) + len(var_ord_locs) + len(var_pois_locs))]
    inc += 1

    return spec_name, num_spec, r_locs, p_locs, c_locs, stoich, mol0, mol_end, add_sol_conc, add_cont_rate, t_cont, \
           add_one_shot, t_one_shot, add_col, sub_aliq, t_aliq, t_col, col, k_lim, ord_lim, pois_lim, fit_asp,\
           var_k_locs, var_ord_locs, fix_pois_locs, var_pois_locs, fit_asp_locs, fit_param_locs, inc


# calculate additions and subtractions of species
def get_add_pops_vol(data_org, x_data_org, x_data_new, num_spec, react_vol_init, add_sol_conc, add_cont_rate, t_cont,
                     add_one_shot, t_one_shot, add_col, sub_cont_rate, sub_aliq, t_aliq, sub_col, win=1):
    add_pops = np.zeros((len(x_data_new), num_spec))
    vol = np.ones(len(x_data_new)) * react_vol_init
    for i in range(num_spec):
        if add_col[i] is not None:
            add_pops[:, i] = data_smooth(data_org, add_col[i], win)
        else:
            add_pops_i = np.zeros((len(x_data_org), 1))
            if add_cont_rate[i] is not None and add_cont_rate[i] != 0:
                for j in range(len(add_cont_rate[i])):
                    index = find_nearest(x_data_org, t_cont[i][j])
                    for k in range(index + 1, len(x_data_org)):
                        add_pops_i[k] = add_pops_i[k - 1] + add_cont_rate[i][j] * \
                                        (x_data_org[k] - x_data_org[k - 1])
            if add_one_shot[i] is not None and add_one_shot[i] != 0:
                for j in range(len(add_one_shot[i])):
                    index = find_nearest(x_data_org, t_one_shot[i][j])
                    add_pops_i[index:] += add_one_shot[i][j]
            add_pops[:, i] = data_smooth(add_pops_i, 0, win)
    vol += add_pops.sum(axis=1)
    for i in range(num_spec):
        if add_sol_conc[i] is not None: add_pops[:, i] = add_pops[:, i] * add_sol_conc[i]

    if sub_col is not None:
        vol_loss = data_smooth(data_org, sub_col, win)
    else:
        vol_loss_i = np.zeros((len(x_data_org), 1))
        if sub_cont_rate is not None and sub_cont_rate != 0:
            for i in range(1, len(x_data_org)):
                vol_loss_i[i] = vol_loss_i[i - 1] + sub_cont_rate * \
                                (x_data_org[i] - x_data_org[i - 1])
        if sub_aliq[0] is not None and sub_aliq[0] != 0:
            for i in range(len(sub_aliq)):
                index = find_nearest(x_data_org, t_aliq[i])
                vol_loss_i[index:] += sub_aliq[i]
        vol_loss = data_smooth(vol_loss_i, 0, win)
    vol_loss = np.reshape(vol_loss, len(vol_loss))
    vol -= [np.float64(vol_loss[i]) for i in range(len(vol_loss))]
    vol_loss_rat = [1.0] + [1 - ((vol_loss[i] - vol_loss[i - 1]) / vol[i - 1]) for i in range(1, len(vol_loss))]
    return add_pops, vol, vol_loss_rat


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
    inc = param_prep(spec_name, spec_type, stoich, mol0, mol_end, add_sol_conc, add_cont_rate, t_cont,
            add_one_shot, t_one_shot, add_col, sub_aliq, t_aliq, t_col, col, k_lim, ord_lim, pois_lim, fit_asp, inc)

    # Calculate iterative species additions and volumes
    add_pops, vol_data, vol_loss_rat = get_add_pops_vol(t, t, t, num_spec, react_vol_init,
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
    inc = param_prep(spec_name, spec_type, stoich, mol0, mol_end, add_sol_conc, add_cont_rate, t_cont, add_one_shot,
                     t_one_shot, add_col, sub_aliq, t_aliq, t_col, col, k_lim, ord_lim, pois_lim, fit_asp, inc)

    # Get x_data
    data_org = df.to_numpy()
    x_data = data_smooth(data_org, t_col, win)
    x_data_add = add_sim(np.reshape(x_data, (len(x_data))), inc)

    # Get TIC
    if TIC_col is not None:
        TIC = data_smooth(data_org, TIC_col, win)
    else:
        TIC = None

    # Calculate iterative species additions and volumes
    add_pops, vol, vol_loss_rat = get_add_pops_vol(data_org, data_org[:, t_col], x_data, num_spec, react_vol_init,
                                                   add_sol_conc, add_cont_rate, t_cont, add_one_shot, t_one_shot,
                                                   add_col, sub_cont_rate, sub_aliq, t_aliq, sub_col, win=win)

    add_pops_add = np.zeros((len(x_data_add), num_spec))
    for i in range(num_spec):
        add_pops_add[:, i] = add_sim(add_pops[:, i], inc)
    add_pops_add_new = np.zeros((len(x_data_add), num_spec))
    for i in range(1, len(add_pops_add)):
        add_pops_add_new[i] = add_pops_add[i] - add_pops_add[i - 1]
    add_pops_add = add_pops_add_new
    vol_data_add = add_sim(vol, inc)
    vol_loss_rat_data_add = add_sim(vol_loss_rat, inc)
    # Determine mol0, mol_end and scale data as required
    data_mod = np.empty((len(x_data), num_spec))
    col_ext = []
    for i in range(num_spec):
        if col[i] is not None:
            col_ext = [*col_ext, i]
            data_i = data_smooth(data_org, col[i], win)
            data_i = tic_norm(data_i, TIC)
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
    fit_param_rss, fit_param_r_squared = residuals(y_data_to_fit, fit_pops_set)
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
    param_prep(spec_name, spec_type, stoich, mol0, mol_end, add_sol_conc, add_cont_rate, t_cont,
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


def write_sim_data(filename, df, param_dict, x_data_df, y_fit_conc_df, y_fit_rate_df):
    param_dict = {key: np.array([value]) for (key, value) in param_dict.items()}
    df_params = pd.DataFrame.from_dict(param_dict)
    df[list(x_data_df.columns)] = x_data_df
    df[list(y_fit_conc_df.columns)] = y_fit_conc_df
    df[list(y_fit_rate_df.columns)] = y_fit_rate_df

    writer = pd.ExcelWriter(filename, engine='openpyxl')
    df.to_excel(writer, sheet_name='FitData')
    df_params.to_excel(writer, sheet_name='InputParams')
    writer.save()


def write_sim_data_temp(df, param_dict, x_data_df, y_fit_conc_df, y_fit_rate_df):
    tmp_file = io.BytesIO()
    write_fit_data(tmp_file, df, param_dict, x_data_df, y_fit_conc_df, y_fit_rate_df)

    return tmp_file, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'


def write_fit_data(filename, df, param_dict, x_data_df, y_exp_df, y_fit_conc_df, y_fit_rate_df,
                   k_fit, k_fit_err, ord_fit, ord_fit_err, pois_fit, pois_fit_err, ss_res, r_squared):
    param_dict = {key: np.array([value]) for (key, value) in param_dict.items()}
    out_dict = {"Rate Constant": [k_fit],
                  "Reaction Orders": [ord_fit],
                  "Species Poisoning": [pois_fit],
                  "Rate Constant Error": [k_fit_err],
                  "Reaction Order Errors": [ord_fit_err],
                  "Species Poisoning Errors": [pois_fit_err],
                  "RSS": [ss_res],
                  "R2": [r_squared]}
    df_outputs = pd.DataFrame.from_dict(out_dict)
    df_params = pd.DataFrame.from_dict(param_dict)
    df[list(x_data_df.columns)] = x_data_df
    df[list(y_exp_df.columns)] = y_exp_df
    df[list(y_fit_conc_df.columns)] = y_fit_conc_df
    df[list(y_fit_rate_df.columns)] = y_fit_rate_df

    writer = pd.ExcelWriter(filename, engine='openpyxl')
    df.to_excel(writer, sheet_name='FitData')
    df_outputs.to_excel(writer, sheet_name='Outputs')
    df_params.to_excel(writer, sheet_name='InputParams')
    writer.save()


def write_fit_data_temp(df, param_dict, x_data_df, y_exp_df, y_fit_conc_df, y_fit_rate_df,
                   k_fit, k_fit_err, ord_fit, ord_fit_err, pois_fit, pois_fit_err, ss_res, r_squared):
    tmp_file = io.BytesIO()
    write_fit_data(tmp_file, df, param_dict, x_data_df, y_exp_df, y_fit_conc_df, y_fit_rate_df,
                   k_fit, k_fit_err, ord_fit, ord_fit_err, pois_fit, pois_fit_err, ss_res, r_squared)

    return tmp_file, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'


def make_param_dict(spec_type, react_vol_init, stoich=1, mol0=None, mol_end=None, add_sol_conc=None, add_cont_rate=None,
                    t_cont=None, add_one_shot=None, t_one_shot=None, add_col=None,
                    sub_cont_rate=None, sub_aliq=None, t_aliq=None, sub_col=None, t_col=0, col=1, k_lim=None,
                    ord_lim=None, pois_lim=None, fit_asp="y", TIC_col=None, scale_avg_num=0, win=1, inc=1):
    param_dict = {'Species types': spec_type,
                  'Initial reaction solution volume': react_vol_init,
                  'Stoichiometries': stoich,
                  'Initial moles': mol0,
                  'Final moles': mol_end,
                  'Addition solution concentrations': add_sol_conc,
                  'Continuous addition rates': add_cont_rate,
                  'Continuous addition start times': t_cont,
                  'One shot additions': add_one_shot,
                  'One shot addition start times': t_one_shot,
                  'Addition columns': add_col,
                  'Continuous subtraction rate': sub_cont_rate,
                  'Subtracted aliquot volumes': sub_aliq,
                  'Subtracted aliquot start times': t_aliq,
                  'Subtraction columns': sub_col,
                  'Time column': t_col,
                  'Species columns': col,
                  'Rate constant limits': k_lim,
                  'Reaction order limits': ord_lim,
                  'Poisoning limits': pois_lim,
                  'Species to fit': fit_asp,
                  'Total ion count column': TIC_col,
                  'Concentration calibration points': scale_avg_num,
                  'Smoothing window': win,
                  'Interpolation multiplier': inc
     }
    if len(k_lim) == 1:
        param_dict['Rate constant starting estimate'] = k_lim[0]
        param_dict['Rate constant minimum'] = k_lim[0] - (1E3 * k_lim[0])
        param_dict['Rate constant maximum'] = k_lim[0] + (1E3 * k_lim[0])
    else:
        param_dict['Rate constant starting estimate'], param_dict['Rate constant minimum'], \
        param_dict['Rate constant maximum'] = k_lim

    if len(ord_lim) == 1:
        param_dict['Reaction order starting estimates'] = ord_lim[0]
        param_dict['Reaction order minima'] = r_ord_lim[0] - 1E6
        param_dict['Reaction order maxima'] = r_ord_lim[0] + 1E6
    else:
        param_dict['Reaction order starting estimates'], param_dict['Reaction order minima'], \
        param_dict['Reaction order maxima'] = ord_lim

    if len(pois_lim) == 1:
        param_dict['Poisoning starting estimates'] = pois_lim[0]
        param_dict['Poisoning starting minima'] = pois_lim[0] - 1E6
        param_dict['Poisoning starting maxima'] = pois_lim[0] + 1E6
    else:
        param_dict['Poisoning starting estimates'], param_dict['Poisoning starting minima'], \
        param_dict['Poisoning starting maxima'] = pois_lim

    return param_dict


def calc_x_lim(x_data, edge_adj):
    return [float(min(x_data) - (edge_adj * max(x_data))), float(max(x_data) * (1 + edge_adj))]


def calc_y_lim(y_exp, y_fit, edge_adj):
    return [float(min(np.min(y_exp), np.min(y_fit)) - edge_adj * max(np.max(y_exp), np.max(y_fit))),
            float(max(np.max(y_exp), np.max(y_fit)) * (1 + edge_adj))]


# processes plotted data
def plot_process(return_fig, fig, f_format, save_disk, save_to, transparent):
    if return_fig:
        return fig, fig.get_axes()

    # correct mimetype based on filetype (for displaying in browser)
    if f_format == 'svg':
        mimetype = 'image/svg+xml'
    elif f_format == 'png':
        mimetype = 'image/png'
    elif f_format == 'jpg':
        mimetype = 'image/jpg'
    elif f_format == 'pdf':
        mimetype = 'application/pdf'
    elif f_format == 'eps':
        mimetype = 'application/postscript'
    else:
        raise ValueError('Image format {} not supported.'.format(format))

    # save to disk if desired
    if save_disk:
        plt.savefig(save_to, transparent=transparent)

    # save the figure to the temporary file-like object
    img = io.BytesIO()  # file-like object to hold image
    plt.savefig(img, format=f_format, transparent=transparent)
    plt.close()
    img.seek(0)
    return img, mimetype


# plot time vs conc
def plot_time_vs_conc(x_data_df, y_exp_conc_df=None, y_fit_conc_df=None, col=None, show_asp=None, method="lone", f_format='svg', return_image=False, save_disk=False,
                      save_to='cake_fit.svg', return_fig=False, transparent=False):
    # methods
    x_data = pd.DataFrame.to_numpy(x_data_df)

    if y_exp_conc_df is not None:
        y_exp_conc_headers = [i.replace(' conc. / moles_unit volume_unit$^{-1}$', '') for i in list(y_exp_conc_df.columns)]
        y_exp_conc = pd.DataFrame.to_numpy(y_exp_conc_df)
    else:
        y_exp_conc_headers = []
        y_exp_conc = pd.DataFrame.to_numpy(y_fit_conc_df)
    if y_fit_conc_df is not None:
        y_fit_conc_headers = [i.replace(' conc. / moles_unit volume_unit$^{-1}$', '') for i in list(y_fit_conc_df.columns)]
        y_fit_conc = pd.DataFrame.to_numpy(y_fit_conc_df)
    else:
        y_fit_conc_headers = []
        y_fit_conc = y_exp_conc
        y_fit_col = []

    if col is not None and show_asp is None:
        y_fit_col = [i for i in range(len(col)) if col[i] is not None]
        non_y_fit_col = [i for i in range(len(col)) if col[i] is None]
    if "lone all" in method and show_asp is None and y_fit_conc_df is not None:
        show_asp = ["y"] * len(y_fit_conc_headers)
    if show_asp is not None:
        y_fit_col = [i for i in range(len(show_asp)) if 'y' in show_asp[i]]
        non_y_fit_col = [i for i in range(len(show_asp)) if 'n' in show_asp[i]]
    if "comp" in method and (len(non_y_fit_col) == 0 or y_fit_conc_df is None):
        method = "lone"
    if show_asp is not None and 'y' not in show_asp:
        print("If used, show_asp must contain at least one 'y'. Plot time_vs_conc has been skipped.")
        return

    # graph results
    x_ax_scale = 1
    y_ax_scale = 1
    edge_adj = 0.02
    std_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

    x_data_adj = x_data * x_ax_scale
    y_exp_conc_adj = y_exp_conc * y_ax_scale
    y_fit_conc_adj = y_fit_conc * y_ax_scale

    x_label_text = list(x_data_df.columns)[0]
    y_label_text = "Concentration / moles_unit volume_unit$^{-1}$"

    cur_exp = 0
    cur_clr = 0
    if "lone" in method:  # lone plots a single figure containing all exps and fits as specified
        fig = plt.figure(figsize=(5, 5))
        #plt.rcParams.update({'font.size': 15})
        plt.xlabel(x_label_text)
        plt.ylabel(y_label_text)
        for i in range(len(y_exp_conc_headers)):
            if len(x_data_adj) <= 50:
                plt.scatter(x_data_adj, y_exp_conc_adj[:, i], label=y_exp_conc_headers[i])
            else:
                plt.plot(x_data_adj, y_exp_conc_adj[:, i], label=y_exp_conc_headers[i])
        for i in y_fit_col:
            plt.plot(x_data_adj, y_fit_conc_adj[:, i], label=y_fit_conc_headers[i])
        if len(y_fit_col) == 0: y_fit_col = range(len(y_exp_conc_headers))
        plt.xlim(calc_x_lim(x_data_adj, edge_adj))
        plt.ylim(calc_y_lim(y_exp_conc_adj, y_fit_conc_adj[:, y_fit_col], edge_adj))
        plt.legend(prop={'size': 10}, frameon=False)
    elif "comp" in method:  # plots two figures, with the first containing show_asp (or col if show_asp not specified) and the second containing all fits
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        #plt.rcParams.update({'font.size': 15})
        for i in range(len(y_exp_conc_headers)):
            if len(x_data_adj) <= 50:
                ax1.scatter(x_data_adj, y_exp_conc_adj[:, i], color=std_colours[cur_clr],
                            label=y_exp_conc_headers[i])
                ax2.scatter(x_data_adj, y_exp_conc_adj[:, i], color=std_colours[cur_clr],
                            label=y_exp_conc_headers[i])
                cur_clr += 1
            else:
                ax1.plot(x_data_adj, y_exp_conc_adj[:, i], color=std_colours[cur_clr],
                            label=y_exp_conc_headers[i])
                ax2.plot(x_data_adj, y_exp_conc_adj[:, i], color=std_colours[cur_clr],
                            label=y_exp_conc_headers[i])
                cur_clr += 1
        for i in y_fit_col:
            ax1.plot(x_data_adj, y_fit_conc_adj[:, i], color=std_colours[cur_clr],
                            label=y_fit_conc_headers[i])
            ax2.plot(x_data_adj, y_fit_conc_adj[:, i], color=std_colours[cur_clr],
                            label=y_fit_conc_headers[i])
            cur_clr += 1
        for i in non_y_fit_col:
            ax2.plot(x_data_adj, y_fit_conc_adj[:, i], color=std_colours[cur_clr],
                            label=y_fit_conc_headers[i])
            cur_clr += 1

        ax1.set_xlim(calc_x_lim(x_data_adj, edge_adj))
        ax1.set_ylim(calc_y_lim(y_exp_conc_adj, y_fit_conc_adj[:, y_fit_col], edge_adj))
        ax2.set_xlim(calc_x_lim(x_data_adj, edge_adj))
        ax2.set_ylim(calc_y_lim(y_exp_conc_adj, y_fit_conc_adj, edge_adj))

        ax1.set_xlabel(x_label_text)
        ax1.set_ylabel(y_label_text)
        ax2.set_xlabel(x_label_text)
        ax2.set_ylabel(y_label_text)
        ax1.legend(prop={'size': 10}, frameon=False)
        ax2.legend(prop={'size': 10}, frameon=False)

    elif "sep" in method:
        num_spec = max([len(y_exp_conc_headers), len(y_fit_conc_headers)])
        grid_shape = (int(round(np.sqrt(num_spec))), int(math.ceil(np.sqrt(num_spec))))
        fig = plt.figure(figsize=(grid_shape[0] * 6, grid_shape[1] * 5))
        # plt.subplots_adjust(hspace=0.4, wspace=0.4)
        for i in range(num_spec):
            ax = plt.subplot(grid_shape[0], grid_shape[1], i + 1)
            if col is not None and col[i] is not None and y_exp_conc_df is not None:
                if len(x_data_adj) <= 50:
                    ax.scatter(x_data_adj, y_exp_conc_adj[:, cur_exp], color=std_colours[cur_clr], label=y_exp_conc_headers[cur_exp])
                else:
                    ax.plot(x_data_adj, y_exp_conc_adj[:, cur_exp], color=std_colours[cur_clr], label=y_exp_conc_headers[cur_exp])
                ax.set_ylim(calc_y_lim(y_exp_conc_adj[:, cur_exp], y_fit_conc_adj[:, i], edge_adj))
                cur_exp += 1
                cur_clr += 1
            else:
                ax.set_ylim(calc_y_lim(y_fit_conc_adj[:, i], y_fit_conc_adj[:, i], edge_adj))
            ax.plot(x_data_adj, y_fit_conc_adj[:, i], color=std_colours[cur_clr], label=y_fit_conc_headers[i])
            cur_clr += 1

            ax.set_xlim(calc_x_lim(x_data_adj, edge_adj))
            ax.set_xlabel(x_label_text)
            ax.set_ylabel(y_label_text)
            plt.legend(prop={'size': 10}, frameon=False)

    else:
        print("Invalid method inputted. Please enter appropriate method or remove method argument.")
        return

    # plt.show()
    img, mimetype = plot_process(return_fig, fig, f_format, save_disk, save_to, transparent)
    if not return_image:
        graph_url = base64.b64encode(img.getvalue()).decode()
        return 'data:{};base64,{}'.format(mimetype, graph_url)
    else:
        return img, mimetype


# plot conc vs rate
def plot_conc_vs_rate(x_data_df, y_fit_conc_df, y_fit_rate_df, orders, y_exp_conc_df=None, y_exp_rate_df=None,
                      f_format='svg', return_image=False, save_disk=False,
                     save_to='cake_conc_vs_rate.svg', return_fig=False, transparent=False):
    x_data, y_fit_conc, y_fit_rate = map(pd.DataFrame.to_numpy, [x_data_df, y_fit_conc_df, y_fit_rate_df])
    if y_exp_conc_df is not None:
        pd.DataFrame.to_numpy(y_exp_conc_df)
    if y_exp_rate_df is not None:
        pd.DataFrame.to_numpy(y_exp_rate_df)

    num_spec = len(orders)
    # y_exp_conc_headers = list(y_exp_conc_df.columns)
    y_fit_conc_headers = list(y_fit_conc_df.columns)
    # y_exp_rate_adj_headers = [i.replace('fit conc. / moles_unit volume_unit$^{-1}$', 'exp.') for i in list(y_fit_conc_df.columns)]
    y_fit_rate_adj_headers = [i.replace(' conc. / moles_unit volume_unit$^{-1}$', '') for i in list(y_fit_conc_df.columns)]

    # y_exp_rate_adj = np.empty((len(y_exp_rate), num_spec))
    y_fit_rate_adj = np.empty((len(y_fit_rate), num_spec))
    for i in range(num_spec):
        # y_exp_rate_adj[:, i] = np.divide(y_exp_rate.reshape(len(y_exp_rate)), np.product([y_fit_conc[:, j] ** orders[j] for j in range(num_spec) if i != j], axis=0))
        y_fit_rate_adj[:, i] = np.divide(y_fit_rate.reshape(len(y_fit_rate)), np.product([y_fit_conc[:, j] ** orders[j] for j in range(num_spec) if i != j], axis=0))
    # y_exp_rate_adj_df = pd.DataFrame(y_exp_rate_adj, columns=y_exp_rate_adj_headers)
    y_fit_rate_adj_df = pd.DataFrame(y_fit_rate_adj, columns=y_fit_rate_adj_headers)

    grid_shape = (int(round(np.sqrt(num_spec))), int(math.ceil(np.sqrt(num_spec))))

    x_ax_scale = 1
    y_ax_scale = 1
    edge_adj = 0.02
    fig = plt.figure(figsize=(grid_shape[0] * 5, grid_shape[1] * 5))
    # plt.subplots_adjust(hspace=0.5)
    std_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    y_label_text = "Rate / moles_unit volume_unit$^{-1}$ time_unit$^{-1}$"
    for i in range(num_spec):
        x_label_text = y_fit_conc_headers[i]
        ax = plt.subplot(grid_shape[0], grid_shape[1], i + 1)
        ax.set_xlabel(x_label_text)
        ax.set_ylabel(y_label_text)
        # ax.scatter(y_fit_conc[:, i] * x_ax_scale, y_exp_rate_adj[:, i] * y_ax_scale, color=std_colours[i])
        ax.plot(y_fit_conc[:, i] * x_ax_scale, y_fit_rate_adj[:, i] * y_ax_scale, color=std_colours[i])
        ax.set_xlim([float(min(y_fit_conc[:, i] * x_ax_scale) - (edge_adj * max(y_fit_conc[:, i] * x_ax_scale))),
                float(max(y_fit_conc[:, i] * x_ax_scale) * (1 + edge_adj))])
        # ax.set_ylim([float(np.nanmin(y_fit_rate_adj[:, i]) - edge_adj * np.nanmax(y_fit_rate_adj[:, i]) * y_ax_scale),
        #        float(np.nanmax(y_fit_rate_adj[:, i])) * y_ax_scale * (1 + edge_adj)])
    # plt.show()
    save_to_replace = save_to.replace('.png', '_rates.png')
    img, mimetype = plot_process(return_fig, fig, f_format, save_disk, save_to_replace, transparent)
    if not return_image:
        graph_url = base64.b64encode(img.getvalue()).decode()
        return 'data:{};base64,{}'.format(mimetype, graph_url)
    else:
        return img, mimetype


# plot other fits in 2D
def plot_other_fits_2D(x_data_df, y_exp_conc_df, y_fit_conc_df_arr, real_err_df, col, cutoff=1, f_format='svg', return_image=False,
                       save_disk=False, save_to='cake_other_fits.svg', return_fig=False, transparent=False):
    num_spec = len(col)
    x_data, y_exp_conc, real_err = map(pd.DataFrame.to_numpy, [x_data_df, y_exp_conc_df, real_err_df])
    # np.savetxt(r"C:\Users\Peter\Desktop\real_err.csv", real_err, delimiter="\t", fmt='%s')
    cut_thresh = cutoff * real_err[0, -1]
    rows_cut = [i for i, x in enumerate(real_err[:, -1] > cut_thresh) if x]
    cur_clr = 0

    col_ext = [i for i in range(len(col)) if col[i] is not None]
    grid_shape = (int(round(np.sqrt(len(col_ext)))), int(math.ceil(np.sqrt(len(col_ext)))))

    x_ax_scale = 1
    y_ax_scale = 1
    edge_adj = 0.02

    x_data_adj = x_data * x_ax_scale
    fig = plt.figure(figsize=(grid_shape[0] * 5, grid_shape[1] * 5))
    # plt.subplots_adjust(hspace=0.5)
    std_colours = plt.rcParams['axes.prop_cycle'].by_key()['color'] * 100

    x_label_text = "Time / time_unit"
    y_label_text = "Concentration / moles_unit volume_unit$^{-1}$"
    for i in range(len(col_ext)):
        ax = plt.subplot(grid_shape[0], grid_shape[1], i + 1)
        ax.plot(x_data * x_ax_scale, y_exp_conc[:, i] * y_ax_scale, '-o', color="black", label="Exp")
        for j in rows_cut:
            y_fit_conc_df = y_fit_conc_df_arr[j][0]
            y_fit_conc = y_fit_conc_df.to_numpy()
            ax.plot(x_data_adj, y_fit_conc[:, col_ext[i]] * y_ax_scale, label=real_err[j, 0])
            #color=std_colours[j]
        ax.set_xlim(calc_x_lim(x_data_adj, edge_adj))
        #ax.set_ylim([float(np.nanmin(y_fit_rate_adj[:, i]) - edge_adj * np.nanmax(y_fit_rate_adj[:, i]) * y_ax_scale),
        #    float(np.nanmax(y_fit_rate_adj[:, i])) * y_ax_scale * (1 + edge_adj)])

        ax.set_xlabel(x_label_text)
        ax.set_ylabel(y_label_text)
        ax.legend(prop={'size': 10}, frameon=False)

    save_to_replace = save_to.replace('.png', '_other_fits_2D.png')
    img, mimetype = plot_process(return_fig, fig, f_format, save_disk, save_to_replace, transparent)
    # plt.show()

    for i in range(len(col_ext)):
        grid_shape = (int(round(np.sqrt(len(rows_cut)))), int(math.ceil(np.sqrt(len(rows_cut)))))
        fig = plt.figure(figsize=(grid_shape[0] * 5, grid_shape[1] * 5))
        plt.subplots_adjust(hspace=0.2, wspace=0.08)
        for j in rows_cut:
            ax = plt.subplot(grid_shape[0], grid_shape[1], j + 1)
            ax.plot(x_data * x_ax_scale, y_exp_conc[:, i] * y_ax_scale, '-o', color="black", label="Exp")
            y_fit_conc_df = y_fit_conc_df_arr[j][0]
            y_fit_conc = y_fit_conc_df.to_numpy()
            ax.plot(x_data * x_ax_scale, y_fit_conc[:, col_ext[i]] * y_ax_scale, color=std_colours[j], label=real_err[j, 0])
            # color=std_colours[j]
            ax.set_xlim(calc_x_lim(x_data_adj, edge_adj))
            # ax.set_ylim([float(np.nanmin(y_fit_rate_adj[:, i]) - edge_adj * np.nanmax(y_fit_rate_adj[:, i]) * y_ax_scale),
            #    float(np.nanmax(y_fit_rate_adj[:, i])) * y_ax_scale * (1 + edge_adj)])

            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.tick_params(length=0, width=0)
            # ax.set_xlabel(x_label_text)
            # ax.set_ylabel(y_label_text)

    plt.show()

    if not return_image:
        graph_url = base64.b64encode(img.getvalue()).decode()
        return 'data:{};base64,{}'.format(mimetype, graph_url)
    else:
        return img, mimetype


# plot other fits in 3D (contour map and 3D projection)
def plot_other_fits_3D(real_err_df, cutoff=1, f_format='svg', return_image=False, save_disk=False,
                     save_to='cake_other_fits.svg', return_fig=False, transparent=False):
    real_err_arr = pd.DataFrame.to_numpy(real_err_df)
    real_err_arr_cut = real_err_arr[real_err_arr[:, -1] > cutoff, :]
    cont_x_org = [real_err_arr_cut[i, 0][0] for i in range(len(real_err_arr_cut))]
    cont_y_org = [real_err_arr_cut[i, 0][1] for i in range(len(real_err_arr_cut))]
    cont_z_org = real_err_arr_cut[:, -1]
    cont_x_add, cont_y_add = np.linspace(min(cont_x_org), max(cont_x_org), 1000), \
                             np.linspace(min(cont_y_org), max(cont_y_org), 1000)
    cont_x_plot, cont_y_plot = np.meshgrid(cont_x_add, cont_y_add)
    cont_z_plot = interpolate.griddata((cont_x_org, cont_y_org), cont_z_org, (cont_x_plot, cont_y_plot), method='linear')
    # rbf = scipy.interpolate.Rbf(cont_x_org, cont_y_org, cont_z_org, function='linear')
    # cont_z_plot = rbf(cont_x_plot, cont_y_plot)

    cont_fig = plt.imshow(cont_z_plot, vmin=cont_z_org.min(), vmax=cont_z_org.max(), origin='lower', cmap='coolwarm',
               extent=[min(cont_x_org), max(cont_x_org), min(cont_y_org), max(cont_y_org)], aspect='auto')
    # plt.scatter(cont_x_org, cont_y_org, c=cont_z_org, cmap='coolwarm')
    plt.xlabel('Order 1'), plt.ylabel('Order 2')
    plt.colorbar()
    img, mimetype = plot_process(return_fig, cont_fig, f_format, save_disk, save_to.replace('.png', '_other_fits_contour.png'), transparent)

    fig_3D = plt.axes(projection='3d')
    fig_3D.plot_surface(cont_x_plot, cont_y_plot, cont_z_plot, cmap='coolwarm')  # rstride=1, cstride=1
    fig_3D.set_xlabel('Order 1'), fig_3D.set_ylabel('Order 2'), fig_3D.set_zlabel('r^2')
    img, mimetype = plot_process(return_fig, fig_3D, f_format, save_disk, save_to.replace('.png', '_other_fits_3D.png'), transparent)
    if not return_image:
        graph_url = base64.b64encode(img.getvalue()).decode()
        return 'data:{};base64,{}'.format(mimetype, graph_url)
    else:
        return img, mimetype


# print CAKE results (webapp only)
def pprint_cake(k_fit, k_fit_err, ord_fit, ord_fit_err, pois_fit, pois_fit_err, ss_res, r_squared):
    result = f"""|               | Rate Constant (k) | Reaction Orders |
    |---------------|-------------------|----------------|
    |  Opt. Values  | {k_fit: 17.6E} | {ord_fit[1]: 14.6f} |
    | Est. Error +- | {k_fit_err: 17.6E} | {ord_fit_err: 14.6f} |

    |               | Species Poisoning
    |---------------|----------------|
    |  Opt. Values  | {pois_fit: 14.6f} |
    | Est. Error +- | {pois_fit_err: 14.6f} |

    Residual Sum of Squares for Optimization: {ss_res: 8.6f}.

    R^2 Value of Fit: {r_squared: 8.6f}.
    """

    return result


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

    df = read_data(file_name, sheet_name, t_col, col)
    output = fit_cake(df, spec_type, react_vol_init, spec_name=spec_name, stoich=stoich, mol0=mol0, mol_end=mol_end,
                      add_sol_conc=add_sol_conc, add_cont_rate=add_cont_rate, t_cont=t_cont,
                      t_col=t_col, col=col, ord_lim=ord_lim, fit_asp=fit_asp)
    x_data_df, y_exp_df, y_fit_conc_df, y_fit_rate_df, k_val_est, k_fit, k_fit_err, \
    ord_fit, ord_fit_err, pois_fit, pois_fit_err, ss_res, r_squared, col = output

    if not isinstance(col, (tuple, list)): col = [col]

    html = plot_fit_results(x_data_df, y_exp_df, y_fit_conc_df, col,
                            f_format='svg', return_image=False, save_disk=True, save_to=pic_save)

    param_dict = make_param_dict(spec_type, react_vol_init, stoich=stoich, mol0=mol0, mol_end=mol_end,
                                 add_sol_conc=add_sol_conc, add_cont_rate=add_cont_rate, t_cont=t_cont,
                                 t_col=t_col, col=col, ord_lim=None, fit_asp=fit_asp)

    # write_fit_data(xlsx_save, df, param_dict, t, r, p, fit_p, fit_r, res_val, res_err, ss_res, r_squared, cat_pois)
    file, _ = write_fit_data_temp(df, param_dict, x_data_df, y_exp_df, y_fit_conc_df, y_fit_rate_df,
                                  k_fit, k_fit_err, ord_fit, ord_fit_err, pois_fit, pois_fit_err, ss_res, r_squared)
    file.seek(0)
    with open(xlsx_save, "wb") as f:
        f.write(file.getbuffer())
