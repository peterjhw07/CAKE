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
def eq_sim_gen(stoich, conc0, conc_end, add_rate, t_inj, inc, ord_lim, r_locs, p_locs, c_locs, fix_ord_locs, var_ord_locs, t_fit, k, ord, t_del):
    """
    Function Description

    Params
    ------

    Returns
    -------


    """
    spec_calc = np.zeros((len(t_fit), len(conc0)))
    rate_calc = np.zeros(len(t_fit))
    spec_calc[0] = conc0

    r_c_locs = *r_locs, *c_locs
    recon_ord = [None] * len(r_c_locs)
    for it in range(len(r_c_locs)):
        if r_c_locs[it] in fix_ord_locs:
            recon_ord[it] = ord_lim[fix_ord_locs[fix_ord_locs.index(r_c_locs[it])]]
        elif r_c_locs[it] in var_ord_locs:
            recon_ord[it] = ord[var_ord_locs.index(r_c_locs[it])]

    rate_calc[0] = k * np.prod([spec_calc[0, r_c_locs[i]] ** recon_ord[i] for i in range(len(r_c_locs))])
    for it in range(1, len(t_fit)):
        time_span = t_fit[it] - t_fit[it - 1]
        t_start_adj = max(0, t_fit[it] - t_inj - t_del)
        spec_calc[it, r_locs] = [max(0, spec_calc[it - 1, i] - (time_span * rate_calc[it - 1] * stoich[i]) + (t_start_adj * add_rate[i])) for i in r_locs]
        spec_calc[it, p_locs] = [max(0, spec_calc[it - 1, i] + (time_span * rate_calc[it - 1] * stoich[i]) + (t_start_adj * add_rate[i])) for i in p_locs]
        spec_calc[it, c_locs] = [(t_start_adj * add_rate[i]) for i in c_locs]
        rate_calc[it] = k * np.prod([spec_calc[it, r_c_locs[i]] ** recon_ord[i] for i in range(len(r_c_locs))])
    exp_t_rows = list(range(0, len(t_fit), inc - 1))
    spec_calc, rate_calc = spec_calc[exp_t_rows], rate_calc[exp_t_rows]
    return [spec_calc, rate_calc]


def eq_sim_multi(stoich, conc0, conc_end, add_rate, t_inj, inc, ord_lim, r_locs, p_locs, c_locs, fix_ord_locs, var_ord_locs, t_fit, k, ord, t_del):
    """
    Function Description

    Params
    ------

    Returns
    -------


    """
    [spec_calc, rate_it] = eq_sim_gen(stoich, conc0, conc_end, add_rate, t_inj, inc, ord_lim, r_locs, p_locs, c_locs, fix_ord_locs, var_ord_locs, t_fit, k, ord, t_del)
    return spec_calc


# define additional t values for data sets with few data points
def add_sim_t(t, inc):
    """
    Function Description

    Params
    ------

    Returns
    -------


    """
    t_fit = np.zeros(((len(t) - 1) * (inc - 1)) + 1)
    for it in range(len(t) - 1):
        new_t_it = np.linspace(t[it], t[it + 1], inc)[0:-1]
        t_fit[it * len(new_t_it):(it * len(new_t_it)) + len(new_t_it)] = new_t_it
    t_fit[-1] = t[-1]
    return t_fit


# to estimate a k value
def half_life_calc(y_data0, x_data, y_data):
    """
    Function Description

    Params
    ------

    Returns
    -------


    """
    half_y = 0.5 * y_data0
    index = np.where(abs(y_data - half_y) == min(abs(y_data - half_y)))
    print(index)
    half_life = float(x_data[index])
    return half_life


def est_k_order(r_ord, cat_ord, t_del, r0, cat_add_rate, t_inj, half_life):
    """
    Function Description

    Params
    ------

    Returns
    -------


    """
    k_guess = (((2 ** (r_ord - 1)) - 1) * (r0 ** (1 - r_ord)) * (cat_ord + 1)) / (((half_life - t_inj - t_del)
                                                        ** (cat_ord + 1)) * (cat_add_rate ** cat_ord) * (r_ord - 1))
    return k_guess


def est_k_first_order(cat_ord, t_del, cat_add_rate, t_inj, half_life):
    """
    Function Description

    Params
    ------

    Returns
    -------


    """
    k_guess = ((cat_ord + 1) * np.log(2)) / (((half_life - t_inj - t_del) ** (cat_ord + 1)) * (cat_add_rate ** cat_ord))
    return k_guess


# smooth data (if required)
def data_smooth(df, d_col, win=1):
    """
    Function Description

    Params
    ------

    Returns
    -------


    """
    d_raw = df.iloc[:, d_col].values
    if win > 1:
        d_ra = df.iloc[:, d_col].rolling(win).mean().values
        d_manip = d_ra[np.logical_not(np.isnan(d_ra))]
    else:
        d_manip = d_raw
    return d_manip


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


def read_data(file_name, sheet_name):
    """
    Read in data from excel filename

    Params
    ------

    Returns
    -------


    """
    try:
        df = pd.read_excel(file_name, sheet_name=sheet_name, engine='openpyxl', header=None, dtype=np.float64)
        return df
    except ValueError:
        pass
    try:
        df = pd.read_excel(file_name, sheet_name=sheet_name, engine='openpyxl', dtype=np.float64)
        return df
    except ValueError:
        raise ValueError("Excel file must contain all numerical input with at most 1 header row.")


def type_sort(substrate):
    """
    Read in data from excel filename

    Params
    ------

    Returns
    -------


    """



def get_cat_add_rate(cat_sol_conc, inject_rate, react_vol_init):
    """
        Compute the approximate catalyst addition rate in units of concentration / time.

        Params
        ------

        Returns
        -------


    """
    return [i * j for i, j in zip(cat_sol_conc, inject_rate)] / react_vol_init

def fit_cake(df, spec_type, stoich, conc0, conc_end, add_rate, t_inj, k_lim, ord_lim, t_del_lim,
             t_col, TIC_col, col, scale_avg_num=0, win=1, inc=1, fit_asp='y'):
    """
    Params
    ------
    df : pandas.DataFrame
        The reaction data
    stoich_r : int
        Stoichiometry of reactant, r
    stoich_p : int
        Stoichiometry of product, p
    r0 : float
        Value of r0 in M dm^-3 or None if data are given in M dm^-3
    p0 : float
        Value of p0 in M dm^-3 or None if data are given in M dm^-3
    p_end : float
        End value of product in M dm^-3, r0 if equal to start r0 value, or None if data are given in M dm^-3
    cat_add_rate : float
        Catalyst addition rate in M time_unit^-1
    t_inj : float
        Time at which injection began in time_unit^-1
    k_lim : list of float
        Estimated rate constant in (M dm^-3)^? time_unit^-1. Can be specified as [exact value] for fixed variable or
        variable with bounds [estimate, factor difference] or [estimate, lower, upper]. If None, bounds set
        automatically as [estimate from half life, estimate from half life * 1E-3, estimate from half life * 1E3]
    r_ord_lim : list of int
        Reactant order. Can be specified as [exact value] for fixed variable or
        variable with bounds [estimate, lower, upper]. If None, bounds set automatically as [1, 0, 2]
    cat_ord_lim : list of int
        Catalyst order. Can be specified as [exact value] for fixed variable or
        variable with bounds [estimate, lower, upper]. If None, bounds set automatically as [1, 0, 2]
    t_del_lim : list of float
        Time after t_inj when reaction began in time_unit^-1. Can be specified as [exact value] for fixed variable or
        variable with bounds [estimate, lower, upper]. If None, bounds set automatically as [0, 0, final time point]
    t_col : int
        Index of time column.
    TIC_col : int or str
        Index of TIC column or None if no TIC
    r_col : int or str
        Index of reactant column or None if no reactant
    p_col : int or str
        Index of product column or None if no product
    scale_avg_num : int, optional
        Number of data points from which to calculate r0 and p_end. Default 0 (no scaling)
    win : int, optional
        Smoothing window, default 1 if smoothing not required
    inc : int, optional
        Increments between adjacent points for improved simulation, default 1 for using raw time points
    fit_asp : str
        Aspect you want to fit to: 'r' for reactant, 'p' for product or 'rp' for both
    """

    def eq_sim_multi_fit(t_sim, k, t_del, *ord):
        t_fit = t_sim[:int(len(t_sim) / len(fit_asp_locs))]
        spec_calc = eq_sim_multi(stoich, conc0, conc_end, add_rate, t_inj, inc, ord_lim, r_locs, p_locs, c_locs, fix_ord_locs,
                     var_ord_locs, t_fit, k, ord, t_del)
        total_calc = np.empty(0)
        for i in fit_asp_locs:
            total_calc = np.append(total_calc, spec_calc[:, i], axis=0)
        return total_calc

    r_locs = [i for i in range(len(spec_type)) if 'r' in spec_type[i]]
    p_locs = [i for i in range(len(spec_type)) if 'p' in spec_type[i]]
    c_locs = [i for i in range(len(spec_type)) if 'c' in spec_type[i]]
    fix_ord_locs = [i for i in range(len(ord_lim)) if isinstance(ord_lim[i], (int or float))]
    var_ord_locs = [i for i in range(len(ord_lim)) if isinstance(ord_lim[i], tuple)]
    fit_asp_locs = [i for i in range(len(fit_asp)) if 'y' in fit_asp[i]]

    inc += 1
    data_org = df.to_numpy().T
    data_mod = data_org
    t = data_smooth(df, t_col, win)
    data_mod[t_col] = t
    TIC = None
    if TIC_col is not None:
        TIC = data_smooth(df, TIC_col, win)
    data_it = None
    col_ext = []
    for it in range(len(col)):
        if col[it] is not None:
            col_ext = [*col_ext, col[it]]
            data_it = data_smooth(df, col[it], win)
            data_it = tic_norm(data_it, TIC)
            if conc0[it] is None and scale_avg_num == 0:
                conc0_it = data_it[0]
            elif conc0[it] is None and scale_avg_num > 0:
                conc0_it = np.mean(data_it[0:scale_avg_num])
            elif conc0[it] is not None and scale_avg_num > 0 and (conc_end[it] is None or conc0[it] >= conc_end[it]):
                data_scale = np.mean(data_it[0:scale_avg_num]) / conc0[it]
                data_it = data_it / data_scale
            if conc_end[it] is None and scale_avg_num == 0:
                conc_end_it = data_it[-1]
            elif conc_end[it] is None and scale_avg_num > 0:
                conc_end_it = np.mean(data_it[-scale_avg_num:])
            elif conc_end[it] is not None and scale_avg_num > 0 and (conc0[it] is None or conc_end[it] >= conc0[it]):
                data_scale = np.mean(data_it[-scale_avg_num:]) / conc_end[it]
                data_it = data_it / data_scale
            data_mod[col[it]] = data_it

    # define half lives for different fit aspects
    x_data_to_fit = np.empty(0)
    y_data_to_fit = np.empty(0)
    half_life_est = 0
    #for it in range(len(col)):
    #    x_data = np.append(x_data, t, axis=0)
    #    y_data = np.append(y_data, data_mod[it], axis=0)
    for it in range(len(fit_asp_locs)):
        x_data_to_fit = np.append(x_data_to_fit, t, axis=0)
        y_data_to_fit = np.append(y_data_to_fit, data_mod[col[fit_asp_locs[it]]], axis=0)
        half_life_est += half_life_calc(max(conc0[fit_asp_locs[it]], conc_end[fit_asp_locs[it]]), t, data_mod[col[fit_asp_locs[it]]])
    half_life = half_life_est / len(fit_asp_locs)
    x_data_to_fit_add = add_sim_t(x_data_to_fit, inc)

    # define initial values and lower and upper limits for parameters to fit: k, r_ord, cat_ord and t_del
    bound_adj = 1E-3
    ord_val, ord_min, ord_max = [], [], []
    for it in range(len(var_ord_locs)):
        unpack_ord_lim = ord_lim[var_ord_locs[it]]
        ord_val = [*ord_val, unpack_ord_lim[0]]
        ord_min = [*ord_min, unpack_ord_lim[1]]
        ord_max = [*ord_max, unpack_ord_lim[2]]
    if t_del_lim is None:
        t_del_val = 0
        t_del_min = 0
        t_del_max = t[-1]
    elif len(t_del_lim) == 1:
        t_del_val = t_del_lim[0]
        t_del_min = max(0, t_del_val - bound_adj)
        t_del_max = t_del_val + bound_adj
    elif len(t_del_lim) > 1:
        t_del_val = t_del_lim[0]
        t_del_min = t_del_lim[1]
        t_del_max = t_del_lim[2]
    if k_lim is None or k_lim == 0:
        test_ord = [list(range(round(ord_min[i]), round(ord_max[i]) + 1)) for i in range(len(var_ord_locs))]  # section currently not working
        test_ord_combi = list(itertools.product(*test_ord))
        print(test_ord_combi)
        k_guess = np.zeros([len(test_ord_combi), len(var_ord_locs) + 2])
        print([*test_ord_combi[0], 0, 0])
        k_guess[:] = [[*test_ord_combi[i], 0, 0] for i in range(len(test_ord_combi))]
        for it in range(len(k_guess)):
            if k_guess[it, 0] != 1:
                k_guess[it, 2] = est_k_order(k_guess[it, :-2], t_del_val, r0[0],
                                             cat_add_rate, t_inj, half_life)  # changed r0[0]
            else:
                k_guess[it, 2] = est_k_first_order(k_guess[it, 1], t_del_val, cat_add_rate, t_inj, half_life)
            if fit_asp == 'r':
                fit_guess = eq_sim_r(x_data_to_fit_add, k_guess[it, 2], k_guess[it, 0], k_guess[it, 1], t_del_val)
            elif fit_asp == 'p':
                fit_guess = eq_sim_p(x_data_to_fit_add, k_guess[it, 2], k_guess[it, 0], k_guess[it, 1], t_del_val)
            elif 'r' in fit_asp and 'p' in fit_asp:
                fit_guess = eq_sim_multi_fit(x_data_to_fit_add, k_guess[it, 2], k_guess[it, 0], k_guess[it, 1], t_del_val)
            _, k_guess[it, 3] = residuals(y_data_to_fit, fit_guess)
        index = np.where(k_guess == max(k_guess[:, 3]))
        index_first = index[0]
        k_val = float(k_guess[index_first, 2])
        k_min = k_val / 1E3
        k_max = k_val * 1E3
    elif len(k_lim) == 1:
        k_val = k_lim[0]
        k_min = k_val - (bound_adj * k_val)
        k_max = k_val + (bound_adj * k_val)
    elif len(k_lim) == 2:
        k_val = k_lim[0]
        k_min = k_val / k_lim[1]
        k_max = k_val * k_lim[1]
    elif len(k_lim) == 3:
        k_val = k_lim[0]
        k_min = k_lim[1]
        k_max = k_lim[2]

    init_param = [k_val, t_del_val, *ord_val]

    # apply fittings, determine optimal parameters and determine resulting fits
    fit, fit_p, fit_r, cat_pois = None, None, None, None
    res = optimize.curve_fit(eq_sim_multi_fit, x_data_to_fit_add, y_data_to_fit, init_param, maxfev=10000,
                             bounds=((k_min, t_del_min, *ord_min), (k_max, t_del_max, *ord_max)))
    kf, t_delf, *ordf = res[0]
    print(res[0])
    fit = eq_sim_multi_fit(x_data_to_fit_add, kf, t_delf, *ordf)
    fit_data_outcome = eq_sim_multi(stoich, conc0, conc_end, add_rate, t_inj, inc, ord_lim, r_locs, p_locs, c_locs, fix_ord_locs, var_ord_locs, t, kf, ordf, t_delf)
    # fit_r = fit[:int(len(x_data) / 2)]
    # fit_p = fit[int(len(x_data) / 2):]
    # fit_rate = eq_sim_gen(stoich, conc0, conc_end, cat_add_rate, t_inj, inc, x_data_add, kf, ordf, t_delf)[-1]

    # calculate residuals and errors
    res_val = res[0]
    res_err = np.sqrt(np.diag(res[1]))  # for 1SD
    ss_res, r_squared = residuals(y_data_to_fit, fit)

    # calculate catalyst poisoning, if any
    if t_del_lim is None or len(t_del_lim) > 1:
        cat_pois = t_delf * add_rate
        cat_pois_err = res_err[3] * add_rate
    else:
        cat_pois = 0
        cat_pois_err = 0

    if ord_lim is not None and len(ord_lim) == 1:
        res_val[2:], res_err[2:] = [ord_val, 0]
    if t_del_lim is not None and len(t_del_lim) == 1:
        res_val[1], res_err[1] = [t_del_val, 0]
    if k_lim is not None and len(k_lim) == 1:
        res_val[0], res_err[0] = [k_val, 0]

    return np.reshape(t, (len(t), 1)), data_mod[col_ext].T, fit_data_outcome, k_val, res_val, res_err, ss_res, r_squared, cat_pois, cat_pois_err


def write_fit_data(filename, df, param_dict, t, r, p, fit_p, fit_r, res_val, res_err, ss_res, r_squared, cat_pois,
                   cat_pois_err):
    param_dict = {key: np.array([value]) for (key, value) in param_dict.items()}
    out_dict = {"Rate Constant": [res_val[0]],
                  "Reactant Order": [res_val[1]],
                  "Catalyst Order": [res_val[2]],
                  "Zero Time": [res_val[3]],
                  "Catalyst Poisoning": [cat_pois],
                  "Rate Constant Error": [res_err[0]],
                  "Reactant Order Error": [res_err[1]],
                  "Catalyst Order Error": [res_err[2]],
                  "Zero Time Error": [res_err[3]],
                  "Catalyst Poisoning Error": [cat_pois_err],
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


def write_fit_data_temp(df, param_dict, t, r, p, fit_p, fit_r, res_val, res_err, ss_res, r_squared, cat_pois,
                        cat_pois_err):
    tmp_file = io.BytesIO()
    write_fit_data(tmp_file, df, param_dict, t, r, p, fit_p, fit_r, res_val, res_err, ss_res, r_squared, cat_pois,
                   cat_pois_err)

    return tmp_file, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'


def make_param_dict(stoich_r, stoich_p, r0, p0, p_end, cat_add_rate, t_inj, k_lim, r_ord_lim, cat_ord_lim,
                    t_del_lim, t_col, TIC_col, r_col, p_col, scale_avg_num, win, inc, fit_asp):
    param_dict = {'Stoich R': stoich_r,
     'Stoich P': stoich_p,
     'R_0': r0,
     'P_0': p0,
     'P_end': p_end,
     'Cat Add Rate': cat_add_rate,
     'Total Ion Count col': TIC_col,
     'R col': r_col,
     'P col': p_col,
     'Time col': t_col,
     'Concentration Calibration Points': scale_avg_num,
     'Smoothing Window': win,
     'Interpolation Multiplier': inc,
     'Fitting Aspect': fit_asp
     }
    if len(k_lim) == 1:
        param_dict['k Estimate'] = k_lim[0]
        param_dict['k Minimum'] = k_lim[0] - (1E6 * k_lim[0])
        param_dict['k Maximum'] = k_lim[0] + (1E6 * k_lim[0])
    else:
        param_dict['k Estimate'], param_dict['k Minimum'], param_dict['k Maximum'] = k_lim

    if len(r_ord_lim) == 1:
        param_dict['R Order Estimate'] = r_ord_lim[0]
        param_dict['R Order Minimum'] = r_ord_lim[0] - 1E6
        param_dict['R Order Maximum'] = r_ord_lim[0] + 1E6
    else:
        param_dict['R Order Estimate'], param_dict['R Order Minimum'], param_dict['R Order Maximum'] = r_ord_lim

    if len(r_ord_lim) == 1:
        param_dict['Cat Order Estimate'] = cat_ord_lim[0]
        param_dict['Cat Order Minimum'] = cat_ord_lim[0] - 1E6
        param_dict['Cat Order Maximum'] = cat_ord_lim[0] + 1E6
    else:
        param_dict['Cat Order Estimate'], param_dict['Cat Order Minimum'], param_dict['Cat Order Maximum'] = cat_ord_lim

    if len(t_del_lim) == 1:
        param_dict['Start Time Estimate'] = t_del_lim[0]
        param_dict['Start Time Minimum'] = t_del_lim[0] - 1E6
        param_dict['Start Time Maximum'] = t_del_lim[0] + 1E6
    else:
        param_dict['Start Time Estimate'], param_dict['Start Time Minimum'], param_dict['Start Time Maximum'] = t_del_lim

    return param_dict


def plot_cake_results(t, r, p, fit, fit_p, fit_r, r_col, p_col, f_format='svg', return_image=False, save_disk=False,
                      save_to='cake.svg', return_fig=False):
    # graph results
    x_ax_scale = 1
    y_ax_scale = 1
    edge_adj = 0.02
    x_label_text = "Time"
    y_label_text = ""
    if r_col is not None and p_col is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    if r_col is not None:
        if p_col is None:
            fig = plt.figure(figsize=(6, 6))
            plt.rcParams.update({'font.size': 15})
            if len(t) <= 50:
                plt.scatter(t * x_ax_scale, r * y_ax_scale, color='k')  # plt.plot(t, r * 1E6, color='k')
            else:
                plt.plot(t * x_ax_scale, r * y_ax_scale, color='k')  # plt.plot(t, r * 1E6, color='k')
            plt.plot(t, fit * y_ax_scale, color='r')
            # plt.title("Raw")
            plt.xlim([min(t * x_ax_scale) - (edge_adj * max(t * x_ax_scale)), max(t * x_ax_scale) * (1 + edge_adj)])
            plt.ylim([min(t * y_ax_scale) - (edge_adj * max(r * y_ax_scale)), max(r * y_ax_scale) * (1 + edge_adj)])
            plt.xlabel(x_label_text)
            plt.ylabel("[R]")
            # plt.savefig(pic_save)
            # plt.show()
        else:
            if len(t) <= 50:
                ax1.scatter(t*x_ax_scale, r * y_ax_scale, color='k')
            else:
                ax1.plot(t * x_ax_scale, r * y_ax_scale, color='k')
            ax1.plot(t * x_ax_scale, fit_r * y_ax_scale, color='r')
            ax1.set_xlim([min(t * x_ax_scale) - (edge_adj * max(t * x_ax_scale)), max(t * x_ax_scale) * (1 + edge_adj)])
            # ax1.set_ylim([min(r * x_ax_scale) - (edge_adj * max(r * x_ax_scale)), max(r * x_ax_scale) * (1 + edge_adj)])
            ax1.set_xlabel(x_label_text)
            ax1.set_ylabel("[R]")
    if p_col is not None:
        if r_col is None:
            fig = plt.figure(figsize=(6, 6))
            plt.rcParams.update({'font.size': 15})
            if len(t) <= 50:
                plt.scatter(t * x_ax_scale, p * y_ax_scale, color='k')
            else:
                plt.plot(t * x_ax_scale, p * y_ax_scale, color='k')
            plt.plot(t * x_ax_scale, fit * y_ax_scale, color='r')
            plt.xlim([min(t * x_ax_scale) - (edge_adj * max(t * x_ax_scale)), max(t * x_ax_scale) * (1 + edge_adj)])
            plt.ylim([min(t * y_ax_scale) - (edge_adj * max(p * y_ax_scale)), max(p * y_ax_scale) * (1 + edge_adj)])
            plt.xlabel(x_label_text)
            plt.ylabel("[P]")
        else:
            if len(t) <= 50:
                ax2.scatter(t * x_ax_scale, p * y_ax_scale, color='k')
            else:
                ax2.plot(t * x_ax_scale, p * y_ax_scale, color='k')
            print(len(t))
            print(len(fit_p))
            ax2.plot(t * x_ax_scale, fit_p * y_ax_scale, color='r')
            ax2.set_xlim([min(t * x_ax_scale) - (edge_adj * max(t * x_ax_scale)), max(t * x_ax_scale) * (1 + edge_adj)])
            ax2.set_ylim([min(p * x_ax_scale) - (edge_adj * max(p * x_ax_scale)), max(p * x_ax_scale) * (1 + edge_adj)])
            ax2.set_xlabel(x_label_text)
            ax2.set_ylabel("[P]")

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
        plt.savefig(save_to, transparent=True)

    # save the figure to the temporary file-like object
    img = io.BytesIO()  # file-like object to hold image
    plt.savefig(img, format=f_format, transparent=True)
    plt.close()
    img.seek(0)
    if not return_image:
        graph_url = base64.b64encode(img.getvalue()).decode()
        return 'data:{};base64,{}'.format(mimetype, graph_url)
    else:
        return img, mimetype


def pprint_cake(res_val, res_err, ss_res, r_squared, cat_pois, cat_pois_err):
    if cat_pois is None:
        cat_pois = "N/A"
    else:
        cat_pois = f"{cat_pois: 8.6E}"
    result = f"""|               | Rate Constant (k) | Reactant Order |
|---------------|-------------------|----------------|
|  Opt. Values  | {res_val[0]: 17.6E} | {res_val[1]: 14.6f} |
| Est. Error +- | {res_err[0]: 17.6E} | {res_err[1]: 14.6f} |

|               | Catalyst Order | Start Time |
|---------------|----------------|------------|
|  Opt. Values  | {res_val[2]: 14.6f} | {res_val[3]: 10.6f} |
| Est. Error +- | {res_err[2]: 14.6f} | {res_err[3]: 10.6f} |

Residual Sum of Squares for Optimization: {ss_res: 8.6f}.

R^2 Value of Fit: {r_squared: 8.6f}.

Catalyst Poisoning (if applicable): {cat_pois}
Catalyst Poisoning Error: {cat_pois_err}
"""

    return result


if __name__ == "__main__":
    stoich_r = 1  # insert stoichiometry of reactant, r
    stoich_p = 1  # insert stoichiometry of product, p
    r0 = 2.5  # enter value of r0 in M dm^-3 or None if data are given in M dm^-3
    p0 = 0  # enter value of p0 in M dm^-3 or None if data are given in M dm^-3
    p_end = r0  # enter end value of product in M dm^-3, r0 if equal to start r0 value, or None if data are given in M dm^-3
    cat_add_rate = .000102  # enter catalyst addition rate in M time_unit^-1
    t_inj = 60
    win = 1  # enter smoothing window (1 if smoothing not required)

    # Parameter fitting
    # Enter None for any order, [exact value] for fixed variable or variable with bounds [estimate, factor difference] or [estimate, lower, upper]
    inc = 1  # enter increments between adjacent points for improved simulation, None or 1 for using raw time points
    k_lim = [1E-1, 1E-4, 1E2]  # enter rate constant in (M dm^-3)^? time_unit^-1
    r_ord_lim = [1, 0, 3]  # enter r order
    cat_ord_lim = [1, 0, 3]  # enter cat order
    t_del_lim = [0, 0, 300]  # enter time at which injection began in time_unit^-1

    # Experimental data location
    file_name = r'/Users/bhenders/Desktop/CAKE/WM_220317_Light_Intensity.xlsx'  # enter filename as r'file_name'
    file_name = r'/Users/bhenders/Downloads/PJHW_22040802.xlsx'  # enter filename as r'file_name'

    sheet_name = 'Sheet2'  # enter sheet name as 'sheet_name'
    t_col = 0  # enter time column
    TIC_col = None  # enter TIC column or None if no TIC
    r_col = None  # enter [r] column or None if no r
    p_col = 1  # enter [p] column or None if no p
    scale_avg_num = 5  # enter number of data points from which to calculate r0 and p_end

    fit_asp = 'p'  # enter aspect you want to fit to: 'r' for reactant, 'p' for product or 'rp' for both

    pic_save = r'/Users/bhenders/Desktop/CAKE/cake_app_test.png'
    xlsx_save = r'/Users/bhenders/Desktop/CAKE/fit_data.xlsx'

    df = read_data(file_name, sheet_name)
    CAKE = fit_cake(df, stoich_r, stoich_p, r0, p0, p_end, cat_add_rate, t_inj, k_lim, r_ord_lim, cat_ord_lim,
                    t_del_lim, t_col, TIC_col, r_col, p_col, scale_avg_num, win, inc, fit_asp)
    t, r, p, fit, fit_p, fit_r, k_val, res_val, res_err, ss_res, r_squared, cat_pois, cat_pois_err = CAKE

    html = plot_cake_results(t, r, p, fit, fit_p, fit_r, r_col, p_col, f_format='svg', return_image=False,
                             save_disk=True, save_to=pic_save)

    param_dict = make_param_dict(stoich_r, stoich_p, r0, p0, p_end, cat_add_rate, t_inj, k_lim, r_ord_lim, cat_ord_lim,
                                 t_del_lim, t_col, TIC_col, r_col, p_col, scale_avg_num, win, inc, fit_asp)

    # write_fit_data(xlsx_save, df, param_dict, t, r, p, fit_p, fit_r, res_val, res_err, ss_res, r_squared, cat_pois)
    file, _ = write_fit_data_temp(df, param_dict, t, r, p, fit_p, fit_r, res_val, res_err, ss_res, r_squared, cat_pois,
                                  cat_pois_err)
    file.seek(0)
    with open(xlsx_save, "wb") as f:
        f.write(file.getbuffer())
