"""CAKE Miscellaneous Functions"""

import numpy as np
import logging
from cake.prep.get_rxns import get_rxns

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


# Defines additional t values for data sets with few data points
def add_sim(s, inc):
    if inc != 1:
        s = np.array(s, dtype=np.float64)
        num_segments = len(s) - 1
        num_points = int(num_segments * max(1, inc)) + 1
        s_fit = np.interp(np.linspace(0, num_segments, num_points), np.arange(num_segments + 1), s)
    else:
        s_fit = s
    return s_fit


# Smooths data
def data_smooth(arr, d_col, win=1, inc=1):
    if win <= 1:
        d_ra = arr[:, d_col]
    else:
        ret = np.cumsum(arr[:, d_col], dtype=float)
        ret[win:] = ret[win:] - ret[:-win]
        d_ra = ret[win - 1:] / win
    if inc < 1:
        exp_rows = np.linspace(0, len(d_ra) - 1, num=int(((len(d_ra) - 1) * inc))).astype(int)
        d_ra = d_ra[exp_rows]
    return d_ra


# Manipulates to TIC values (for MS only)
def tic_norm(data, tic=None):
    if tic is not None:
        data = data / tic
    else:
        data = data
    return data


# Finds nearest value
def find_nearest(array, value):
    array = np.asarray(array)
    index = (np.abs(array - value)).argmin()
    return index


# Returns all None to ensure correct list lengths for iterative purposes
def return_all_nones(s, num_spec):
    if s is None: s = [None] * num_spec
    return s


# Rearranges species in correct order
def rearrange_list(list, indices, replacement):
    return [list[i] if i is not None else replacement for i in indices]


# Inserts empty value
def insert_empty(s, ex_len, replacement):
    return s.append([replacement] * ex_len)


# Converts non-list to list
def type_to_list(s):
    if not isinstance(s, list):
        s = [s]
    return s


# Converts int and float into lists inside tuples
def tuple_of_lists_from_tuple_of_int_float(s):
    s_list = []
    for i in range(len(s)):
        if isinstance(s[i], (int, float)):
            s_list = [*s_list, [s[i]]]
        else:
            s_list = [*s_list, s[i]]
    return s_list


# Replaces single None with all None
def replace_none_with_nones(item):
    if isinstance(item, list):
        return [replace_none_with_nones(sub_item) if sub_item is not None else (None, None, None) for sub_item in item]
    else:
        return item if item is not None else (None, None, None)


# Prepares parameters
def param_prep(spec_name, spec_type, stoich_inp, rxns, t_col, col, fit_asp, mol0, mol_end, add_sol_conc,
               cont_add_rate, t_cont_add, disc_add_vol, t_disc_add, disc_sub_vol, t_disc_sub, cont_temp_rate,
               t_cont_temp, temp_col, rate_eq_type, k_lim, ord_lim, pois_lim, inc):

    # If using rxns
    if rxns:
        spec, stoich, stoich_loc, ord_lim, ord_loc = get_rxns(rxns)
        num_spec = len(spec)

        spec, fit_asp, mol0, mol_end, add_sol_conc, cont_add_rate, t_cont_add, disc_add_vol, t_disc_add = \
            map(return_all_nones, [spec, fit_asp, mol0, mol_end, add_sol_conc, cont_add_rate, t_cont_add,
                                   disc_add_vol, t_disc_add], [num_spec] * 10)

        if spec_name is None:
            spec_name = spec
        else:
            spec_locs = [spec_name.index(i) if i in spec_name else None for i in spec]
            spec_name = spec
            mol0 = rearrange_list(mol0, spec_locs, 0)
            mol_end, add_sol_conc, cont_add_rate, t_cont_add, disc_add_vol, t_disc_add = \
                map(lambda s: rearrange_list(s, spec_locs, None), [mol_end, add_sol_conc, cont_add_rate,
                                                                   t_cont_add, disc_add_vol, t_disc_add])
            fit_asp = rearrange_list(fit_asp, spec_locs, None)
            if col is not None:
                col = rearrange_list(col, spec_locs, None)
            if pois_lim is None:
                pois_lim = [0] * num_spec
            else:
                pois_lim = rearrange_list(pois_lim, spec_locs, 0)

    # If not using rxns
    elif spec_type:
        spec_type = type_to_list(spec_type)
        spec_type = [spec.lower() for spec in spec_type]
        if spec_name is None:
            spec_name = spec_type
        num_spec = len(spec_name)

        # Setup stoich
        if stoich_inp is None:
            stoich = np.empty((1, len(spec_type)))
            for j, i in enumerate(spec_type):
                if 'p' in i.lower(): stoich[:, j] = [1]
                elif 'r' in i.lower(): stoich[:, j] = [-1]
                elif 'c' in i.lower(): stoich[:, j] = [0]

        else:
            stoich = np.empty((1, len(spec_type)))
            if isinstance(stoich_inp, (int, float)): stoich_inp = [stoich_inp]
            for j, i in enumerate(spec_type):
                if stoich_inp[j] is None:
                    if 'p' in i.lower(): stoich[:, j] = 1
                    elif 'r' in i.lower(): stoich[:, j] = -1
                    elif 'c' in i.lower(): stoich[:, j] = 0
                else:
                    if 'p' in i.lower(): stoich[:, j] = stoich_inp[j]
                    elif 'r' in i.lower(): stoich[:, j] = abs(stoich_inp[j]) * -1
                    elif 'c' in i.lower(): stoich[:, j] = stoich_inp[j] * 0

        spec_name, fit_asp, mol0, mol_end, add_sol_conc, cont_add_rate, t_cont_add, disc_add_vol, t_disc_add = \
        map(return_all_nones, [spec_name, fit_asp, mol0, mol_end, add_sol_conc, cont_add_rate, t_cont_add,
                               disc_add_vol, t_disc_add], [num_spec] * 10)
        for i in range(num_spec):
            if spec_name[i] is None:
                spec_name[i] = 'Species ' + str(i + 1)

        # Setup ord_lim for fitting
        if ord_lim is None:
            ord_lim = np.empty((1, num_spec), dtype=object)
            for j, i in enumerate(spec_type):
                if 'r' in i or 'c' in i: ord_lim[0, j] = (1, 0, 2)
                elif 'p' in i: ord_lim[0, j] = 0
        else:
            if isinstance(ord_lim, (int, float)): ord_lim = [ord_lim]
            for j, i in enumerate(spec_type):
                if 'p' in i and ord_lim[j] is None: ord_lim[j] = 0
            ord_lim = np.array(ord_lim, dtype=object).reshape(1, -1)
        stoich_loc = [(0,)] * num_spec
        ord_loc = [tuple(range(num_spec))]

        if isinstance(ord_lim[0], np.ndarray):
            for j, i in enumerate(ord_lim[0]):
                if i is None:
                    if 'p' in spec_type[j]: ord_lim[0, j] = 0
                    elif 'r' in spec_type[j] or 'c' in spec_type[j]: ord_lim[0, j] = (1, 0, 2)

        if pois_lim is None: pois_lim = [0] * num_spec

    if temp_col: cont_temp_rate, t_cont_temp = [], []

    # Setup k_lim for fitting
    if k_lim is None:
        if 'standard' in rate_eq_type.lower():
            k_lim = [[(None, None, None)]] * len(ord_loc)
        elif 'michaelis-menten' or 'arrhenius' or 'eyring' in rate_eq_type: k_lim = [[(None, None, None),
                                                                        (None, None, None)]] * len(ord_loc)
    elif isinstance(k_lim, (int, float)): k_lim = [k_lim]
    else:
        k_lim = replace_none_with_nones(k_lim)
    if not (isinstance(k_lim, list) and all(isinstance(sublist, list) for sublist in k_lim)):
        k_lim = [k_lim]

    # Convert tuple of int or float into tuple of lists
    spec_name, t_col, col, fit_asp, mol0, mol_end, add_sol_conc, cont_add_rate, t_cont_add, disc_add_vol, t_disc_add, \
    disc_sub_vol, t_disc_sub, cont_temp_rate, t_cont_temp, pois_lim = map(type_to_list, [spec_name, t_col, col, fit_asp,
                                mol0, mol_end, add_sol_conc, cont_add_rate, t_cont_add, disc_add_vol, t_disc_add,
                                disc_sub_vol, t_disc_sub, cont_temp_rate, t_cont_temp, pois_lim])
    cont_add_rate, t_cont_add, disc_add_vol, t_disc_add = map(tuple_of_lists_from_tuple_of_int_float,
                                                              [cont_add_rate, t_cont_add, disc_add_vol, t_disc_add])

    if fit_asp[0] != 'y' and fit_asp[0] != 'n': fit_asp = ['y' if i in fit_asp else 'n' for i in spec_name]

    # Get location of variable parameters
    fix_pois_locs, var_locs, fit_asp_locs, fit_param_locs = get_var_locs(spec_type, num_spec,
                                                                         k_lim, ord_lim, pois_lim, fit_asp)

    return spec_name, num_spec, stoich, t_col, col, fit_asp, mol0, mol_end, add_sol_conc, cont_add_rate, t_cont_add, \
           disc_add_vol, t_disc_add, disc_sub_vol, t_disc_sub, cont_temp_rate, t_cont_temp, stoich_loc, ord_loc, \
           k_lim, ord_lim, pois_lim, fix_pois_locs, var_locs, fit_asp_locs, fit_param_locs, inc


# Gets location of variable parameters
def get_var_locs(spec_type, num_spec, k_lim, ord_lim, pois_lim, fit_asp):
    var_k_locs = [(i, j) for i in range(len(k_lim)) for j in range(len(k_lim[i])) if (isinstance(k_lim[i][j], (tuple, list)) and len(k_lim[i][j]) > 1)]
    if spec_type:
        var_ord_locs = [(i, j) for i in range(len(ord_lim)) for j in range(len(ord_lim[i])) if (isinstance(ord_lim[i][j], (tuple, list)) and len(ord_lim[i][j]) > 1)]
    else:
        var_ord_locs = []
    fix_pois_locs = [i for i in range(num_spec) if (isinstance(pois_lim[i], (int, float))
                    or (isinstance(pois_lim[i], (tuple, list)) and len(pois_lim[i]) == 1))]
    var_pois_locs = [i for i in range(num_spec) if (isinstance(pois_lim[i], (tuple, list, str))
                                                    and len(pois_lim[i]) > 1)]
    var_locs = [var_k_locs, var_ord_locs, var_pois_locs]

    fit_asp_locs = [i for i in range(num_spec) if fit_asp[i] is not None and 'y' in fit_asp[i]]
    fit_param_locs = [range(0, len(var_k_locs)), range(len(var_k_locs), len(var_k_locs) + len(var_ord_locs)),
                      range(len(var_k_locs) + len(var_ord_locs),
                            len(var_k_locs) + len(var_ord_locs) + len(var_pois_locs))]
    return fix_pois_locs, var_locs, fit_asp_locs, fit_param_locs


# Calculates temperature throughout
def get_temp_data(t, y_data_org, cont_event, temp_col, win=1, inc=1):
    if temp_col is not None:
        cont_event.t_dtemp = t
        temp = data_smooth(y_data_org, temp_col, win, inc)
        temp = np.reshape(temp, len(temp))
        cont_event.temp = temp
    else:
        temp = np.empty(len(t))
        for i in range(1, len(cont_event.t)):
            row_get = (t >= cont_event.t[i - 1]) & (t < cont_event.t[i])
            temp[row_get] = cont_event.temp[i - 1] + cont_event.t_dtemp[i - 1] * (t[row_get] - t[row_get][0])
        temp[-1] = cont_event.temp[-1]
    return temp, cont_event


# Gets volume
def get_vol(t, cont_event):
    vol = np.empty(len(t))
    for i in range(len(cont_event.t) - 1):
        row_get = (t >= cont_event.t[i]) & (t < cont_event.t[i + 1])
        vol[row_get] = cont_event.vol[i] + cont_event.dvol[i] * (t[row_get] - t[row_get][0])
    vol[-1] = cont_event.vol[-1]
    return vol
