"""CAKE Fitting Programme"""
# if __name__ == '__main__':
# import matplotlib
# matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
import itertools
import base64
from scipy import optimize
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


# general kinetic simulator
def eq_sim_gen(stoich, mol0, mol_end, add_pops, vol, vol_loss_rat, inc, ord_lim, r_locs, p_locs, c_locs,
               fix_ord_locs, var_ord_locs, var_pois_locs, t_fit, fit_param, fit_param_locs):
    """
    Function Description

    Params
    ------

    Returns
    -------


    """
    k = fit_param[fit_param_locs[0]]
    ord = [fit_param[i] for i in fit_param_locs[1]]
    pois = [fit_param[i] for i in fit_param_locs[2]]
    pops = np.zeros((len(t_fit), len(mol0)))
    rate = np.zeros(len(t_fit))
    pops[0] = mol0
    for i in range(len(var_pois_locs)):
        pops[:, var_pois_locs[i]] -= pois[i]
    i = 0
    rate[i] = k * np.prod([(max(0, pops[i, j]) / vol[i]) ** ord_lim[j] for j in fix_ord_locs]) * np.prod([(max(0, pops[i, var_ord_locs[j]]) / vol[i]) ** ord[j] for j in range(len(var_ord_locs))])
    for i in range(1, len(t_fit)):
        t_span = t_fit[i] - t_fit[i - 1]
        pops[i, r_locs] = [pops[i - 1, j] * vol_loss_rat[i] - (t_span * rate[i - 1] * stoich[j]) * vol[i - 1]
                           + add_pops[i, j] for j in r_locs]
        pops[i, p_locs] = [pops[i - 1, j] * vol_loss_rat[i] + (t_span * rate[i - 1] * stoich[j]) * vol[i - 1]
                           + add_pops[i, j] for j in p_locs]
        pops[i, c_locs] = [pops[i - 1, j] * vol_loss_rat[i] + add_pops[i, j] for j in c_locs]
        rate[i] = k * np.prod([(max(0, pops[i, j]) / vol[i]) ** ord_lim[j] for j in fix_ord_locs]) * np.prod([(max(0, pops[i, var_ord_locs[j]]) / vol[i]) ** ord[j] for j in range(len(var_ord_locs))])
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


def residuals(y_data, fit):
    """
    Function Description

    Params
    ------

    Returns
    -------


    """
    res = y_data - fit
    ss_res = np.sum(res ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return [ss_res, r_squared]


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


def return_all_nones(s, num_spec):
    if s is None: s = [None] * num_spec
    return s


def type_to_list(s):
    if not isinstance(s, list):
        s = [s]
    return s


def tuple_of_lists_from_tuple_of_int_float(s):
    s_list = []
    for i in range(len(s)):
        if isinstance(s[i], (int, float)):
            s_list = [*s_list, [s[i]]]
        else:
            s_list = [*s_list, s[i]]
    return s_list


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


def fit_cake(df, spec_type, react_vol_init, stoich=1, mol0=None, mol_end=None, add_sol_conc=None, add_cont_rate=None,
             t_cont=None, add_one_shot=None, t_one_shot=None, add_col=None, sub_cont_rate=None,
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

    def eq_sim_fit(x_data_sim, *fit_param):
        x_data_fit = x_data_sim[:int(len(x_data_sim) / len(fit_asp_locs))]
        pops, rate = eq_sim_gen(stoich, mol0, mol_end, add_pops_add, vol_data_add, vol_loss_rat_data_add,
                                inc, ord_lim, r_locs, p_locs, c_locs,
                                fix_ord_locs, var_ord_locs, var_pois_locs, x_data_fit, fit_param, fit_param_locs)
        pops_reshape = np.empty(0)
        for i in fit_asp_locs:
            pops_reshape = np.append(pops_reshape, pops[:, i], axis=0)
        return pops_reshape

    spec_type = type_to_list(spec_type)
    num_spec = len(spec_type)
    r_locs = [i for i in range(num_spec) if 'r' in spec_type[i]]
    p_locs = [i for i in range(num_spec) if 'p' in spec_type[i]]
    c_locs = [i for i in range(num_spec) if 'c' in spec_type[i]]

    if stoich is None: stoich = [1] * num_spec
    mol0 = return_all_nones(mol0, num_spec)
    mol_end = return_all_nones(mol_end, num_spec)
    add_sol_conc = return_all_nones(add_sol_conc, num_spec)
    add_cont_rate = return_all_nones(add_cont_rate, num_spec)
    t_cont = return_all_nones(t_cont, num_spec)
    add_one_shot = return_all_nones(add_one_shot, num_spec)
    t_one_shot = return_all_nones(t_one_shot, num_spec)
    add_col = return_all_nones(add_col, num_spec)
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

    stoich, mol0, mol_end, add_sol_conc, add_cont_rate, t_cont, add_one_shot, t_one_shot, add_col, sub_aliq, t_aliq, \
    t_col, col, ord_lim, pois_lim, fit_asp = map(type_to_list, [stoich, mol0, mol_end, add_sol_conc,
    add_cont_rate, t_cont, add_one_shot, t_one_shot, add_col, sub_aliq, t_aliq, t_col, col, ord_lim, pois_lim, fit_asp])
    add_cont_rate, t_cont, add_one_shot, t_one_shot = map(tuple_of_lists_from_tuple_of_int_float,
                                            [add_cont_rate, t_cont, add_one_shot, t_one_shot])
    print(spec_type, react_vol_init, stoich, mol0, mol_end, add_sol_conc, add_cont_rate, t_cont, add_one_shot,
          t_one_shot, add_col, sub_cont_rate, sub_aliq, t_aliq, sub_col, t_col, col, k_lim, ord_lim, pois_lim,
          fit_asp, TIC_col, scale_avg_num, win, inc)

    fix_ord_locs = [i for i in range(num_spec) if (isinstance(ord_lim[i], (int, float))
                    or (isinstance(ord_lim[i], (tuple, list)) and len(ord_lim[i]) == 1))]
    var_ord_locs = [i for i in range(num_spec) if (isinstance(ord_lim[i], (tuple, list)) and len(ord_lim[i]) > 1)]
    fix_pois_locs = [i for i in range(num_spec) if (isinstance(pois_lim[i], (int, float))
                    or (isinstance(pois_lim[i], (tuple, list)) and len(pois_lim[i]) == 1))]
    var_pois_locs = [i for i in range(num_spec) if (isinstance(pois_lim[i], (tuple, list, str)) and len(pois_lim[i]) > 1)]
    fit_asp_locs = [i for i in range(num_spec) if 'y' in fit_asp[i]]
    fit_param_locs = [0, range(1, 1 + len(var_ord_locs)),
                      range(1 + len(var_ord_locs), 1 + len(var_ord_locs) + len(var_pois_locs))]
    inc += 1

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

    # Manipulate data for fitting
    x_data_add_to_fit = np.empty(0)
    y_data_to_fit = np.empty(0)
    for i in range(len(fit_asp_locs)):
        x_data_add_to_fit = np.append(x_data_add_to_fit, x_data_add, axis=0)
        y_data_to_fit = np.append(y_data_to_fit, data_mod[:, fit_asp_locs[i]], axis=0)

    # Define initial values and lower and upper limits for parameters to fit: ord and pois
    bound_adj = 1E-3
    ord_val, ord_min, ord_max, pois_val, pois_min, pois_max = [], [], [], [], [], []
    for i in range(len(var_ord_locs)):
        unpack_ord_lim = ord_lim[var_ord_locs[i]]
        ord_val = [*ord_val, unpack_ord_lim[0]]
        ord_min = [*ord_min, unpack_ord_lim[1]]
        ord_max = [*ord_max, unpack_ord_lim[2]]
    for i in range(len(var_pois_locs)):
        unpack_pois_lim = pois_lim[var_pois_locs[i]]
        if "max" in unpack_pois_lim:
            pois_val = [*pois_val, 0]
            pois_min = [*pois_min, 0]
            pois_max = [*pois_max, max(mol0[i], mol_end[i])]
        else:
            pois_val = [*pois_val, unpack_pois_lim[0]]
            pois_min = [*pois_min, unpack_pois_lim[1]]
            pois_max = [*pois_max, unpack_pois_lim[2]]
    for i in fix_pois_locs:
        mol0[i] -= pois_lim[i]
        mol_end[i] -= pois_lim[i]

    # Define initial values and lower and upper limits for parameters to fit: k
    if k_lim is None or (isinstance(k_lim, (int, float)) and k_lim == 0) or \
            (isinstance(k_lim, (tuple, list)) and len(k_lim) == 1 and (k_lim[0] is None or k_lim[0] == 0)):  # need to find a better way to estimate k
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
                                    [k_first_guess, *pois_val], maxfev=10000, bounds=((k_first_guess *
                                    bound_adj, *pois_min), (k_first_guess / bound_adj, *pois_max)))
                k_sec_guess[i, len(var_ord_locs):-1] = k_sec_guess_res[0]
                fit_guess = eq_sim_fit(x_data_add_to_fit, k_sec_guess[i, len(var_ord_locs)],
                                       *k_sec_guess[i, :len(var_ord_locs)],
                                       *k_sec_guess[i, len(var_ord_locs) + 1:-len(var_pois_locs) - 1])
                _, k_sec_guess[i, -1] = residuals(y_data_to_fit, fit_guess)
            index = np.where(k_sec_guess[:, -1] == max(k_sec_guess[:, -1]))
            k_val = float(k_sec_guess[index[0], -3])
        else:
            k_val = k_first_guess
        k_min = k_val * bound_adj
        k_max = k_val / bound_adj
        k_lim = [k_val, k_min, k_max]
    elif isinstance(k_lim, (int, float)) or (isinstance(k_lim, (tuple, list)) and len(k_lim) == 1):
        if isinstance(k_lim, (int, float)): k_val = k_lim
        else: k_val = k_lim[0]
        k_min = k_val - (bound_adj * k_val)
        k_max = k_val + (bound_adj * k_val)
    elif len(k_lim) == 2:
        k_val = k_lim[0]
        k_min = k_val * k_lim[1]
        k_max = k_val / k_lim[1]
    elif len(k_lim) == 3:
        k_val = k_lim[0]
        k_min = k_lim[1]
        k_max = k_lim[2]

    init_param = [k_val, *ord_val, *pois_val]
    low_bounds = [k_min, *ord_min, *pois_min]
    up_bounds = [k_max, *ord_max, *pois_max]

    # Apply fittings, determine optimal parameters and determine resulting fits
    fit_param_res = optimize.curve_fit(eq_sim_fit, x_data_add_to_fit, y_data_to_fit, init_param, maxfev=10000,
                                       bounds=(low_bounds, up_bounds))
    fit_param = fit_param_res[0]
    k_fit = fit_param[fit_param_locs[0]]
    ord_fit = fit_param[fit_param_locs[1]]
    pois_fit = fit_param[fit_param_locs[2]]
    # print(k_fit, *ord_fit, *pois_fit)
    fit_pops_set = eq_sim_fit(x_data_add_to_fit, k_fit, *ord_fit, *pois_fit)
    fit_pops_all, fit_rate_all = eq_sim_gen(stoich, mol0, mol_end, add_pops_add, vol_data_add, vol_loss_rat_data_add,
                                            inc, ord_lim, r_locs, p_locs, c_locs, fix_ord_locs, var_ord_locs,
                                            var_pois_locs, x_data_add, fit_param, fit_param_locs)

    # Calculate residuals and errors
    fit_param_err = np.sqrt(np.diag(fit_param_res[1]))  # for 1SD
    k_fit_err = fit_param_err[fit_param_locs[0]]
    ord_fit_err = fit_param_err[fit_param_locs[1]]
    pois_fit_err = fit_param_err[fit_param_locs[2]]
    fit_param_ss, fit_param_r_squared = residuals(y_data_to_fit, fit_pops_set)

    # "N/A" to non-fitted parameters
    if k_lim is not None and k_lim != 0 and isinstance(k_lim, (int, float) or len(k_lim) == 1):
        k_fit, k_fit_err = "N/A", "N/A"
    if len(var_ord_locs) == 0:
        ord_fit, ord_fit_err = "N/A", "N/A"
    if len(var_pois_locs) == 0:
        pois_fit, pois_fit_err = "N/A", "N/A"
        t_del_fit, t_del_fit_err = "N/A", "N/A"
    else:
        pois_fit, pois_fit_err = pois_fit / vol[0], pois_fit_err / vol[0]
        t_del_fit, t_del_fit_err = pois_fit * 1, pois_fit_err * 1  # need to make t_del work somehow

    return np.reshape(x_data, (len(x_data), 1)), data_mod[:, col_ext], fit_pops_all, \
           np.reshape(fit_rate_all, (len(fit_rate_all), 1)), k_val, k_fit, k_fit_err, \
           ord_fit, ord_fit_err, pois_fit, pois_fit_err, fit_param_ss, fit_param_r_squared, col


def write_fit_data(filename, df, param_dict, x_data, y_data, fit,
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
    if t is not None:
        df['Time Transformed'] = t
    if r is not None:
        df['R Transformed'] = r
    if p is not None:
        df['P Transformed'] = p
    if fit_r is not None:
        df['Fit R'] = fit_r
    if fit_p is not None:
        df['Fit P'] = fit_p

    writer = pd.ExcelWriter(filename, engine='openpyxl')
    df.to_excel(writer, sheet_name='FitData')
    df_outputs.to_excel(writer, sheet_name='Outputs')
    df_params.to_excel(writer, sheet_name='InputParams')
    writer.save()


def write_fit_data_temp(df, param_dict, x_data, y_data, fit,
                   k_fit, k_fit_err, ord_fit, ord_fit_err, pois_fit, pois_fit_err, ss_res, r_squared):
    tmp_file = io.BytesIO()
    write_fit_data(tmp_file, df, param_dict, x_data, y_data, fit,
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


def plot_cake_results(x_data, y_data, fit, col, exp_headers, f_format='svg', return_image=False, save_disk=False,
                      save_to='cake.svg', return_fig=False, transparent=False):

    data_fit_col = [i for i in range(len(col)) if col[i] is not None]
    non_data_fit_col = [i for i in range(len(col)) if col[i] is None]
    data_headers = [exp_headers[i] for i in range(1, len(data_fit_col) + 1)]
    fit_headers = [exp_headers[i] for i in range(len(data_fit_col) + 1, len(exp_headers) - 1)]
    # graph results
    x_ax_scale = 1
    y_ax_scale = 1
    edge_adj = 0.02
    x_label_text = "Time / time_unit"
    y_label_text = "Concentration / moles_unit volume_unit$^{-1}$"
    std_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    cur_clr = 0
    if len(data_headers) == len(fit_headers):
        fig = plt.figure(figsize=(5, 5))
        #plt.rcParams.update({'font.size': 15})
        plt.xlabel(x_label_text)
        plt.ylabel(y_label_text)
        for i in range(len(data_headers)):
            if len(x_data) <= 50:
                plt.scatter(x_data * x_ax_scale, y_data[:, i] * y_ax_scale, label=data_headers[i])
            else:
                plt.plot(x_data * x_ax_scale, y_data[:, i] * y_ax_scale, label=data_headers[i])
        for i in range(len(fit_headers)):
            plt.plot(x_data, fit[:, i] * y_ax_scale, label=fit_headers[i])
        plt.xlim([float(min(x_data * x_ax_scale) - (edge_adj * max(x_data * x_ax_scale))), float(max(x_data * x_ax_scale) * (1 + edge_adj))])
        plt.ylim([float(min(np.min(y_data), np.min(fit)) - edge_adj * max(np.max(y_data), np.max(fit)) * x_ax_scale), float(max(np.max(y_data), np.max(fit)) * x_ax_scale * (1 + edge_adj))])
        plt.legend(prop={'size': 10}, frameon=False)
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        #plt.rcParams.update({'font.size': 15})
        ax1.set_xlabel(x_label_text)
        ax1.set_ylabel(y_label_text)
        ax2.set_xlabel(x_label_text)
        ax2.set_ylabel(y_label_text)
        for i in range(len(data_headers)):
            if len(x_data) <= 50:
                line, = ax1.scatter(x_data * x_ax_scale, y_data[:, i] * y_ax_scale, color=std_colours[cur_clr], label=data_headers[i])
                ax2.scatter(x_data * x_ax_scale, y_data[:, i] * y_ax_scale, color=std_colours[cur_clr], label=data_headers[i])
                cur_clr += 1
            else:
                line, = ax1.plot(x_data * x_ax_scale, y_data[:, i] * y_ax_scale, color=std_colours[cur_clr], label=data_headers[i])
                ax2.plot(x_data * x_ax_scale, y_data[:, i] * y_ax_scale, color=std_colours[cur_clr], label=data_headers[i])
                cur_clr += 1
        for i in data_fit_col:
            line, = ax1.plot(x_data * x_ax_scale, fit[:, i] * y_ax_scale, color=std_colours[cur_clr], label=fit_headers[i])
            ax2.plot(x_data * x_ax_scale, fit[:, i] * y_ax_scale, color=std_colours[cur_clr], label=fit_headers[i])
            cur_clr += 1
        for i in non_data_fit_col:
            ax2.plot(x_data * x_ax_scale, fit[:, i] * y_ax_scale, color=std_colours[cur_clr], label=fit_headers[i])
            cur_clr += 1

        ax1.set_xlim([float(min(x_data) - edge_adj * max(x_data) * x_ax_scale), float(max(x_data) * x_ax_scale * (1 + edge_adj))])
        ax1.set_ylim([float(min(np.min(y_data), np.min(fit[:, data_fit_col])) - edge_adj * max(np.max(y_data), np.max(fit[:, data_fit_col])) * x_ax_scale), float(max(np.max(y_data), np.max(fit[:, data_fit_col])) * x_ax_scale * (1 + edge_adj))])
        ax2.set_xlim([float(min(x_data) - edge_adj * max(x_data) * x_ax_scale), float(max(x_data) * x_ax_scale * (1 + edge_adj))])
        ax2.set_ylim([float(min(np.min(y_data), np.min(fit)) - edge_adj * max(np.max(y_data), np.max(fit)) * x_ax_scale), float(max(np.max(y_data), np.max(fit)) * x_ax_scale * (1 + edge_adj))])
        ax1.legend(prop={'size': 10}, frameon=False)
        ax2.legend(prop={'size': 10}, frameon=False)

    #plt.show()

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
    if not return_image:
        graph_url = base64.b64encode(img.getvalue()).decode()
        return 'data:{};base64,{}'.format(mimetype, graph_url)
    else:
        return img, mimetype


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
    output = fit_cake(df, spec_type, react_vol_init, stoich=stoich, mol0=mol0, mol_end=mol_end,
                        add_sol_conc=add_sol_conc, add_cont_rate=add_cont_rate, t_cont=t_cont,
                        t_col=t_col, col=col, ord_lim=ord_lim, fit_asp=fit_asp)
    x_data, y_data, fit, fit_rate, k_val_est, k_fit, k_fit_err, ord_fit, ord_fit_err, pois_fit, pois_fit_err, ss_res, r_squared, col = output

    imp_headers = list(df.columns)
    fit_headers=[]
    if not isinstance(col, (tuple, list)): col = [col]
    for j in range(len(col)):
        if col[j] is not None:
            fit_headers = [*fit_headers, 'Fit '+imp_headers[col[j]]]
        else:
            fit_headers = [*fit_headers, 'Fit species ' + str(j + 1)]
    exp_headers = [imp_headers[t_col], *['Exp '+imp_headers[i] for i in col if i is not None], *fit_headers, 'Fit rate']

    html = plot_cake_results(x_data, y_data, fit, col, exp_headers,
                             f_format='svg', return_image=False, save_disk=True, save_to=pic_save)

    param_dict = make_param_dict(spec_type, react_vol_init, stoich=stoich, mol0=mol0, mol_end=mol_end,
                                 add_sol_conc=add_sol_conc, add_cont_rate=add_cont_rate, t_cont=t_cont,
                                 t_col=t_col, col=col, ord_lim=None, fit_asp=fit_asp)

    # write_fit_data(xlsx_save, df, param_dict, t, r, p, fit_p, fit_r, res_val, res_err, ss_res, r_squared, cat_pois)
    file, _ = write_fit_data_temp(df, param_dict, x_data, y_data, fit,
                   k_fit, k_fit_err, ord_fit, ord_fit_err, pois_fit, pois_fit_err, ss_res, r_squared)
    file.seek(0)
    with open(xlsx_save, "wb") as f:
        f.write(file.getbuffer())
