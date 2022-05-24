"""CAKE Fitting Programme"""
# if __name__ == '__main__':
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
from scipy import optimize
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


# general kinetic simulator
def eq_sim_gen(stoich_r, stoich_p, r0, p0, cat_add_rate, t_fit, k, x, y, t0):
    """
    Function Description

    Params
    ------

    Returns
    -------


    """
    r_calc = np.zeros(len(t_fit))
    p_calc = np.zeros(len(t_fit))
    rate_it = np.zeros(len(t_fit))
    r_calc[0] = r0
    p_calc[0] = p0
    rate_it[0] = k * (r_calc[0] ** x) * ((max(0, t_fit[0] - t0) * cat_add_rate) ** y)
    for it in range(1, len(t_fit)):
        time_span = t_fit[it] - t_fit[it - 1]
        r_calc[it] = max(0, r_calc[it - 1] - (time_span * rate_it[it - 1] * stoich_r))
        p_calc[it] = max(0, p_calc[it - 1] + (time_span * rate_it[it - 1] * stoich_p))
        rate_it[it] = k * (r_calc[it] ** x) * ((max(0, t_fit[it] - t0) * cat_add_rate) ** y)
    return [r_calc, p_calc, rate_it]


def eq_sim_multi(stoich_r, stoich_p, r0, p0, cat_add_rate, t, k, x, y, t0):
    """
    Function Description

    Params
    ------

    Returns
    -------


    """
    t_fit = t[:int(len(t) / 2)]
    [r_calc, p_calc, rate_it] = eq_sim_gen(stoich_r, stoich_p, r0, p0, cat_add_rate, t_fit, k, x, y, t0)
    total_calc = np.append(r_calc, p_calc)
    return total_calc


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
    for it in range(0, len(t) - 1):
        new_t_it = np.linspace(t[it], t[it + 1], inc)[0:-1]
        t_fit[it * len(new_t_it):(it * len(new_t_it)) + len(new_t_it)] = new_t_it
    t_fit[-1] = t[-1]
    return t_fit


# find the location of original t values
def find_exp_t(t_fit, out, inc):
    """
    Function Description

    Params
    ------

    Returns
    -------


    """
    out_calc_inc = out[list(range(0, len(t_fit), inc - 1))]
    return out_calc_inc


# equivalent to previous simulation functions but with more increments (x4)
def eq_sim_gen_inc(stoich_r, stoich_p, r0, p0, cat_add_rate, inc, t, k, x, y, t0):
    """
    Function Description

    Params
    ------

    Returns
    -------


    """
    t_fit = add_sim_t(t, inc)
    [r_calc, p_calc, rate_it] = eq_sim_gen(stoich_r, stoich_p, r0, p0, cat_add_rate, t_fit, k, x, y, t0)
    r_calc_inc = find_exp_t(t_fit, r_calc, inc)
    p_calc_inc = find_exp_t(t_fit, p_calc, inc)
    rate_it_inc = find_exp_t(t_fit, rate_it, inc)
    return [r_calc_inc, p_calc_inc, rate_it_inc]


def eq_sim_multi_inc(stoich_r, stoich_p, r0, p0, cat_add_rate, inc, t, k, x, y, t0):
    """
    Function Description

    Params
    ------

    Returns
    -------


    """
    t_fit = add_sim_t(t[:int(len(t) / 2)], inc)
    r_calc, p_calc, _ = eq_sim_gen(stoich_r, stoich_p, r0, p0, cat_add_rate, t_fit, k, x, y, t0)
    r_calc_inc = find_exp_t(t_fit, r_calc, inc)
    p_calc_inc = find_exp_t(t_fit, p_calc, inc)
    total_calc_inc = np.append(r_calc_inc, p_calc_inc)
    return total_calc_inc


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
    half_life = float(x_data[index])
    return half_life


def est_k_order(x, y, t0, r0, cat_add_rate, half_life):
    """
    Function Description

    Params
    ------

    Returns
    -------


    """
    k_guess = (((2 ** (x - 1)) - 1) * (r0 ** (1 - x)) * (y + 1)) / (((half_life - t0) ** (y + 1)) * (cat_add_rate ** y) * (x - 1))
    return k_guess


def est_k_first_order(y, t0, cat_add_rate, half_life):
    """
    Function Description

    Params
    ------

    Returns
    -------


    """
    k_guess = ((y + 1) * np.log(2)) / (((half_life - t0) ** (y + 1)) * (cat_add_rate ** y))
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
        df = pd.read_excel(file_name, sheet_name=sheet_name, engine='openpyxl')
        return df
    except ValueError as e:
        return str(e)


def get_cat_add_rate(cat_sol_conc, inject_rate, react_vol_init):
    """
        Compute the approximate catalyst addition rate in units of concentration / time.

        Params
        ------

        Returns
        -------


    """
    return (cat_sol_conc * inject_rate) / react_vol_init

def fit_cake(df, stoich_r, stoich_p, r0, p0, p_end, cat_add_rate, k_est, r_ord, cat_ord,
             t0_est, t_col, TIC_col, r_col, p_col, max_order=3, scale_avg_num=1, win=1, inc=1, fit_asp='r'):
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
    k_est : list of float
        Estimated rate constant in (M dm^-3)^? time_unit^-1. Can be specified as [exact value] for fixed variable or
        variable with bounds [estimate, factor difference] or [estimate, lower, upper].
    r_ord : list of int
        Reactant order. Can be specified as [exact value] for fixed variable or
        variable with bounds [estimate, lower, upper]. If None, bounds set automatically as [1, 0, `max_order`]
    cat_ord : list of int
        Catalyst order. Can be specified as [exact value] for fixed variable or
        variable with bounds [estimate, lower, upper]. If None, bounds set automatically as [1, 0, `max_order`]
    t0_est : list of float
        Time at which injection began in time_unit^-1. Can be specified as [exact value] for fixed variable or
        variable with bounds [estimate, lower, upper]
    t_col : int
        Index of time column.
    TIC_col : int or str
        Index of TIC column or None if no TIC
    r_col : int or str
        Index of reactant column or None if no reactant
    p_col : int or str
        Index of product column or None if no product
    max_order : int, optional
        Maximum possible order for species. Default 3.
    scale_avg_num : int, optional
        Number of data points from which to calculate r0 and p_end. Default 1
    win : int, optional
        Smoothing window, default 1 if smoothing not required
    inc : int, optional
        Increments between adjacent points for improved simulation, default 1 for using raw time points
    fit_asp : str
        Aspect you want to fit to: 'r' for reactant, 'p' for product or 'rp' for both
    """
    inc += 1
    t = data_smooth(df, t_col, win)
    TIC = None
    if TIC_col is not None:
        TIC = data_smooth(df, TIC_col, win)
    r = None
    if r_col is not None:
        r = data_smooth(df, r_col, win)
        r = tic_norm(r, TIC)
        if r0 is None:
            r0 = np.mean(r[:scale_avg_num])
        else:
            r_scale = np.mean(r[:scale_avg_num]) / r0
            r = r / r_scale
    p = None
    if p_col is not None:
        p = data_smooth(df, p_col, win)
        p = tic_norm(p, TIC)
        if p_end is None:
            p_end = np.mean(p[-scale_avg_num:])
        else:
            p_scale = np.mean(p[-scale_avg_num:]) / p_end
            p = p / p_scale

    # define half lives for different fit aspects
    if fit_asp == 'r':
        x_data = t
        y_data = r
        half_life = half_life_calc(r0, x_data, y_data)
    elif fit_asp == 'p':
        x_data = t
        y_data = p
        half_life = half_life_calc(p_end, x_data, y_data)
    elif 'r' in fit_asp and 'p' in fit_asp:
        x_data = np.append(t, t)
        y_data = np.append(r, p)
        half_life = (half_life_calc(r0, t, r) + half_life_calc(p_end, t, p)) / 2

    # define inital values and lower and upper limits for parameters to fit: k, x, y (orders wrt. r and p) and t0
    bound_adj = 1E-6
    if r_ord is None:
        r_val = 1
        r_min = 0
        r_max = max_order
    elif len(r_ord) == 1:
        r_val = r_ord[0]
        r_min = r_ord[0] - bound_adj
        r_max = r_ord[0] + bound_adj
    elif len(r_ord) > 1:
        r_val = r_ord[0]
        r_min = r_ord[1]
        r_max = r_ord[2]
    if cat_ord is None:
        cat_val = 1
        cat_min = 0
        cat_max = max_order
    elif len(cat_ord) == 1:
        cat_val = cat_ord[0]
        cat_min = cat_ord[0] - bound_adj
        cat_max = cat_ord[0] + bound_adj
    elif len(cat_ord) > 1:
        cat_val = cat_ord[0]
        cat_min = cat_ord[1]
        cat_max = cat_ord[2]
    if len(t0_est) == 1:
        t0_val = t0_est[0]
        t0_min = t0_val - bound_adj
        t0_max = t0_val + bound_adj
    elif len(t0_est) > 1:
        t0_val = t0_est[0]
        t0_min = t0_est[1]
        t0_max = t0_est[2]
    if k_est is None or k_est == 0:
        k_guess = np.zeros([16, 4])
        k_guess[:] = [[x, y, 0, 0] for x in [0, 1, 2, 3] for y in [0, 1, 2, 3]]
        for it in range(0, len(k_guess)):
            if k_guess[it, 0] != 1:
                k_guess[it, 2] = est_k_order(k_guess[it, 0], k_guess[it, 1], t0_val, r0, cat_add_rate, half_life)
            else:
                k_guess[it, 2] = est_k_first_order(k_guess[it, 1], t0_val, cat_add_rate, half_life)
            fit_guess = eq_sim_gen(stoich_r, stoich_p, r0, p0, cat_add_rate, x_data, k_guess[it, 2], k_guess[it, 0],
                                   k_guess[it, 1], t0_val)[0]
            _, k_guess[it, 3] = residuals(y_data, fit_guess)
        index = np.where(k_guess == max(k_guess[:, 3]))
        index_first = index[0]
        k_val = float(k_guess[index_first, 2])
        k_min = k_val / 1E3
        k_max = k_val * 1E3
    elif len(k_est) == 1:
        k_val = k_est[0]
        k_min = k_val - (bound_adj * k_val)
        k_max = k_val + (bound_adj * k_val)
    elif len(k_est) == 2:
        k_val = k_est[0]
        k_min = k_val / k_est[1]
        k_max = k_val * k_est[1]
    elif len(k_est) == 3:
        k_val = k_est[0]
        k_min = k_est[1]
        k_max = k_est[2]

    init_param = [k_val, r_val, cat_val, t0_val]

    # apply fittings, determine optimal parameters and determine resulting fits
    fit, fit_p, fit_r, cat_pois = None, None, None, None
    if fit_asp == 'r':

        def eq_sim_r(tsim, k, x, y, t0):
            return eq_sim_gen(stoich_r, stoich_p, r0, p0, cat_add_rate, tsim, k, x, y, t0)[0]

        def eq_sim_r_inc(tsim, k, x, y, t0):
            return eq_sim_gen_inc(stoich_r, stoich_p, r0, p0, cat_add_rate, inc, tsim, k, x, y, t0)[0]

        if inc is None or inc == 1:
            res = optimize.curve_fit(eq_sim_r, x_data, y_data, init_param, maxfev=10000,
                                     bounds=((k_min, r_min, cat_min, t0_min), (k_max, r_max, cat_max, t0_max)))
        else:
            res = optimize.curve_fit(eq_sim_r_inc, x_data, y_data, init_param, maxfev=10000,
                                     bounds=((k_min, r_min, cat_min, t0_min), (k_max, r_max, cat_max, t0_max)))
        kf, xf, yf, t0f = res[0]
        if p_col is None:
            fit = eq_sim_r(x_data, kf, xf, yf, t0f)
            fit_r = fit
        else:
            x_data = np.append(t, t)
            y_data = np.append(r, p)
            if inc is None or inc == 1:
                fit = eq_sim_multi(stoich_r, stoich_p, r0, p0, cat_add_rate, x_data, kf, xf, yf, t0f)
            else:
                fit = eq_sim_multi_inc(stoich_r, stoich_p, r0, p0, cat_add_rate, inc, x_data, kf, xf, yf, t0f)
            fit_r = fit[:int(len(x_data) / 2)]
            fit_p = fit[int(len(x_data) / 2):]
    elif fit_asp == 'p':

        def eq_sim_p(tsim, k, x, y, t0):
            return eq_sim_gen(stoich_r, stoich_p, r0, p0, cat_add_rate, tsim, k, x, y, t0)[1]

        def eq_sim_p_inc(tsim, k, x, y, t0):
            return eq_sim_gen_inc(stoich_r, stoich_p, r0, p0, cat_add_rate, inc, tsim, k, x, y, t0)[1]

        if inc is None or inc == 1:
            res = optimize.curve_fit(eq_sim_p, x_data, y_data, init_param, maxfev=10000,
                                     bounds=((k_min, r_min, cat_min, t0_min), (k_max, r_max, cat_max, t0_max)))
        else:
            res = optimize.curve_fit(eq_sim_p_inc, x_data, y_data, init_param, maxfev=10000,
                                     bounds=((k_min, r_min, cat_min, t0_min), (k_max, r_max, cat_max, t0_max)))
        kf, xf, yf, t0f = res[0]
        if r_col is None:
            if inc is None or inc == 1:
                fit = eq_sim_p(x_data, kf, xf, yf, t0f)
            else:
                fit = eq_sim_p_inc(x_data, kf, xf, yf, t0f)
            fit_p = fit
        else:
            x_data = np.append(t, t)
            y_data = np.append(r, p)
            if inc is None or inc == 1:
                fit = eq_sim_multi(stoich_r, stoich_p, r0, p0, cat_add_rate, x_data, kf, xf, yf, t0f)
            else:
                fit = eq_sim_multi_inc(stoich_r, stoich_p, r0, p0, cat_add_rate, inc, x_data, kf, xf, yf, t0f)
            fit_r = fit[:int(len(x_data) / 2)]
            fit_p = fit[int(len(x_data) / 2):]
    elif 'r' in fit_asp and 'p' in fit_asp:
        if inc is None or inc == 1:
            def eq_sim_multi_fit(tsim, k, x, y, t0):
                return eq_sim_multi(stoich_r, stoich_p, r0, p0, cat_add_rate, tsim, k, x, y, t0)
            res = optimize.curve_fit(eq_sim_multi_fit, x_data, y_data, init_param, maxfev=10000,
                                     bounds=((k_min, r_min, cat_min, t0_min), (k_max, r_max, cat_max, t0_max)))
            kf, xf, yf, t0f = res[0]
            fit = eq_sim_multi(stoich_r, stoich_p, r0, p0, cat_add_rate, x_data, kf, xf, yf, t0f)
        else:
            def eq_sim_multi_inc_fit(t, k, r, c, t0):
                return eq_sim_multi_inc(stoich_r, stoich_p, r0, p0, cat_add_rate, inc, t, k, r, c, t0)
            res = optimize.curve_fit(eq_sim_multi_inc_fit, x_data, y_data, init_param, maxfev=10000,
                                     bounds=((k_min, r_min, cat_min, t0_min), (k_max, r_max, cat_max, t0_max)))
            kf, xf, yf, t0f = res[0]
            fit = eq_sim_multi_inc(stoich_r, stoich_p, r0, p0, cat_add_rate, inc, x_data, kf, xf, yf, t0f)
        fit_r = fit[:int(len(x_data) / 2)]
        fit_p = fit[int(len(x_data) / 2):]
    if inc is None or inc == 1:
        fit_rate = eq_sim_gen(stoich_r, stoich_p, r0, p0, cat_add_rate, x_data, kf, xf, yf, t0f)[2]
    else:
        fit_rate = eq_sim_gen_inc(stoich_r, stoich_p, r0, p0, cat_add_rate, inc, x_data, kf, xf, yf, t0f)

    # calculate residuals and errors
    res_val = res[0]
    res_err = np.sqrt(np.diag(res[1]))
    ss_res, r_squared = residuals(y_data, fit)

    # calculate catalyst poisoning, if any
    if len(t0_est) > 1:
        cat_pois = max(0, (t0f - t0_val) * cat_add_rate)
        if t0f >= t0_val:
            cat_pois_err = res_err[3] * cat_add_rate
        else:
            cat_pois_err = max(0, ((t0f - t0_val) + res_err[3]) * cat_add_rate)
    else:
        cat_pois = 0
        cat_pois_err = 0

    return t, r, p, fit, fit_p, fit_r, k_val, res_val, res_err, ss_res, r_squared, cat_pois, cat_pois_err


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


def make_param_dict(stoich_r, stoich_p, r0, p0, p_end, cat_add_rate, k_est, r_ord, cat_ord,
                    t0_est, t_col, TIC_col, r_col, p_col, max_order, scale_avg_num, win, inc, fit_asp):
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
     'Max Order': max_order,
     'Concentration Calibration Points': scale_avg_num,
     'Smoothing Window': win,
     'Interpolation Multiplier': inc,
     'Fitting Aspect': fit_asp
     }
    if len(k_est) == 1:
        param_dict['k Estimate'] = k_est[0]
        param_dict['k Minimum'] = k_est[0] - (0.001 * k_est[0])
        param_dict['k Maximum'] = k_est[0] + (0.001 * k_est[0])
    else:
        param_dict['k Estimate'], param_dict['k Minimum'], param_dict['k Maximum'] = k_est

    if len(r_ord) == 1:
        param_dict['R Order Estimate'] = r_ord[0]
        param_dict['R Order Minimum'] = r_ord[0] - 0.001
        param_dict['R Order Maximum'] = r_ord[0] + 0.001
    else:
        param_dict['R Order Estimate'], param_dict['R Order Minimum'], param_dict['R Order Maximum'] = r_ord

    if len(r_ord) == 1:
        param_dict['Cat Order Estimate'] = cat_ord[0]
        param_dict['Cat Order Minimum'] = cat_ord[0] - 0.001
        param_dict['Cat Order Maximum'] = cat_ord[0] + 0.001
    else:
        param_dict['Cat Order Estimate'], param_dict['Cat Order Minimum'], param_dict['Cat Order Maximum'] = cat_ord

    if len(t0_est) == 1:
        param_dict['Start Time Estimate'] = t0_est[0]
        param_dict['Start Time Minimum'] = t0_est[0] - 0.001
        param_dict['Start Time Maximum'] = t0_est[0] + 0.001
    else:
        param_dict['Start Time Estimate'], param_dict['Start Time Minimum'], param_dict['Start Time Maximum'] = t0_est

    return param_dict


def plot_cake_results(t, r, p, fit, fit_p, fit_r, r_col, p_col, f_format='svg', return_image=False, save_disk=False,
                      save_to='cake.svg', return_fig=False):
    # graph results
    x_ax_scale = 1
    y_ax_scale = 1
    edge_adj = 0.02
    x_label_text = "Time / min"
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
            plt.ylabel("[R] / mM")
            # plt.savefig(pic_save)
            # plt.show()
        else:
            if len(t) <= 50:
                ax1.scatter(t*x_ax_scale, r * y_ax_scale, color='k')
            else:
                ax1.plot(t * x_ax_scale, r * y_ax_scale, color='k')
            ax1.plot(t * x_ax_scale, fit_r * y_ax_scale, color='r')
            ax1.set_xlim([min(t * x_ax_scale) - (edge_adj * max(t * x_ax_scale)), max(t * x_ax_scale) * (1 + edge_adj)])
            ax1.set_ylim([min(r * x_ax_scale) - (edge_adj * max(r * x_ax_scale)), max(r * x_ax_scale) * (1 + edge_adj)])
            ax1.set_xlabel(x_label_text)
            ax1.set_ylabel("[R] / $\mathregular{10^{-6}}$ M")
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
            plt.ylabel("[P] / mM")
        else:
            if len(t) <= 50:
                ax2.scatter(t * x_ax_scale, p * y_ax_scale, color='k')
            else:
                ax2.plot(t * x_ax_scale, p * y_ax_scale, color='k')
            ax2.plot(t * x_ax_scale, fit_p * y_ax_scale, color='r')
            ax2.set_xlim([min(t * x_ax_scale) - (edge_adj * max(t * x_ax_scale)), max(t * x_ax_scale) * (1 + edge_adj)])
            ax2.set_ylim([min(p * x_ax_scale) - (edge_adj * max(p * x_ax_scale)), max(p * x_ax_scale) * (1 + edge_adj)])
            ax2.set_xlabel(x_label_text)
            ax2.set_ylabel("[P] / $\mathregular{10^{-6}}$ M")

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
    win = 1  # enter smoothing window (1 if smoothing not required)

    # Parameter fitting
    # Enter None for any order, [exact value] for fixed variable or variable with bounds [estimate, factor difference] or [estimate, lower, upper]
    inc = 1  # enter increments between adjacent points for improved simulation, None or 1 for using raw time points
    k_est = [1E-1, 1E-4, 1E2]  # enter rate constant in (M dm^-3)^? time_unit^-1
    r_ord = [1, 0, 3]  # enter r order
    cat_ord = [1, 0, 3]  # enter cat order
    t0_est = [60, 60, 300]  # enter time at which injection began in time_unit^-1
    max_order = 3  # enter maximum possible order for species

    # stoich_r = 1  # insert stoichiometry of reactant, r
    # stoich_p = 1  # insert stoichiometry of product, p
    # r0 = 3.18  # enter value of r0 in M dm^-3 or None if data are given in M dm^-3
    # p0 = 0  # enter value of p0 in M dm^-3 or None if data are given in M dm^-3
    # p_end = r0  # enter end value of product in M dm^-3, r0 if equal to start r0 value, or None if data are given in M dm^-3
    # cat_add_rate = 1.57  # enter catalyst addition rate in M time_unit^-1
    # win = 1  # enter smoothing window (1 if smoothing not required)
    #
    # # Parameter fitting
    # # Enter None for any order, [exact value] for fixed variable or variable with bounds [estimate, factor difference] or [estimate, lower, upper]
    # inc = 2  # enter increments between adjacent points for improved simulation, None or 1 for using raw time points
    # k_est = [1E-2, 1E3]  # enter rate constant in (M dm^-3)^? time_unit^-1
    # r_ord = [1, 0, 3]  # enter r order
    # cat_ord = [1, 0, 3]  # enter cat order
    # t0_est = [0.167]  # enter time at which injection began in time_unit^-1
    # max_order = 3  # enter maximum possible order for species

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
    CAKE = fit_cake(df, stoich_r, stoich_p, r0, p0, p_end, cat_add_rate, k_est, r_ord, cat_ord,
                    t0_est, t_col, TIC_col, r_col, p_col, max_order, scale_avg_num, win, inc, fit_asp)
    t, r, p, fit, fit_p, fit_r, k_val, res_val, res_err, ss_res, r_squared, cat_pois, cat_pois_err = CAKE

    html = plot_cake_results(t, r, p, fit, fit_p, fit_r, r_col, p_col, f_format='svg', return_image=False,
                             save_disk=True, save_to=pic_save)

    param_dict = make_param_dict(stoich_r, stoich_p, r0, p0, p_end, cat_add_rate, k_est, r_ord, cat_ord,
                    t0_est, t_col, TIC_col, r_col, p_col, max_order, scale_avg_num, win, inc, fit_asp)

    # write_fit_data(xlsx_save, df, param_dict, t, r, p, fit_p, fit_r, res_val, res_err, ss_res, r_squared, cat_pois)
    file, _ = write_fit_data_temp(df, param_dict, t, r, p, fit_p, fit_r, res_val, res_err, ss_res, r_squared, cat_pois,
                                  cat_pois_err)
    file.seek(0)
    with open(xlsx_save, "wb") as f:
        f.write(file.getbuffer())
