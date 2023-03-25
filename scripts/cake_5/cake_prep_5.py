"""CAKE Miscellaneous Functions"""
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
