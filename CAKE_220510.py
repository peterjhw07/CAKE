# CAKE Fitting Programme
# Known parameters and smoothing
stoich_r = 1  # insert stoichiometry of reactant, r
stoich_p = 1  # insert stoichiometry of product, p
r0 = 0.1  # enter value of r0 (moles unit per volume unit) or "" if data are given in appropriate units
p0 = 0  # enter value of p0 (moles unit per volume unit) or "" if data are given in appropriate units
p_end = r0  # enter end value of product (moles unit per volume unit), r0 if equal to start r0 value, or "" if data are given in appropriate units
# cat_add_rate = 0.1  # enter catalyst addition rate (moles unit per volume unit per time unit)
react_vol_init = 5E-3  # enter initial reaction volume (volume unit)
cat_sol_conc = 100  # enter concentration of catalyst solution being injected (moles unit per volume unit)
inject_rate = 5E-6  # enter rate of catalyst solution injection (volume unit per time unit)
win = 1  # enter smoothing window (1 if smoothing not required)

# Parameter fitting
# Enter "" for any order, [exact value] for fixed variable or variable with bounds [estimate, factor difference] or [estimate, lower, upper]
inc = 0  # enter additional data points between adjacent points for improved simulation, i.e., 0 for using raw time points
k_est = ""  # enter rate constant in (M dm^-3)^? time_unit^-1
r_ord = [1, 0, 3]  # enter r order
cat_ord = [1, 0, 3]  # enter cat order
t0_est = [2, 2, 10]  # enter time at which injection began in time_unit^-1
max_order = 3  # enter maximum possible order for species

# Experimental data location
file_name = r'C:\Users\Peter\Documents\Postdoctorate\Work\CAKE\Case studies\AA\PJHW_22042202_260-270.xlsx'  # enter filename as r'file_name'
sheet_name = 'Sheet1'  # enter sheet name as 'sheet_name'
t_col = 2  # enter time column
TIC_col = ""  # enter TIC column or "" if no TIC
r_col = 6  # enter [r] column or "" if no r
p_col = ""  # enter [p] column or "" if no p
scale_avg_num = 300  # enter number of data points from which to calculate r0 and p_end

fit_asp = 'r'  # enter aspect you want to fit to: 'r' for reactant, 'p' for product or 'rp' for both

pic_save = r'C:\Users\Peter\Documents\Postdoctorate\Work\CAKE\Figures\0UV3.1.png'

import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize

# import addcopyfighandler

# general kinetic simulator
def eq_sim_gen(stoich_r, stoich_p, r0, p0, cat_add_rate, t_fit, k, x, y, t0):
    r_calc = np.zeros(len(t_fit))
    p_calc = np.zeros(len(t_fit))
    rate_it = np.zeros(len(t_fit))
    r_calc[0] = r0
    p_calc[0] = p0
    rate_it[0] = k * (r_calc[0] ** x) * ((max(0, t_fit[0] - t0) * cat_add_rate) ** y)
    vol_calc = react_vol_init
    for it in range(1, len(t_fit)):
        time_span = t_fit[it] - t_fit[it - 1]
        conc_adj = vol_calc / (vol_calc + (inject_rate * time_span))
        r_calc[it] = max(0, r_calc[it - 1] - (time_span * rate_it[it - 1] * stoich_r)) * conc_adj
        p_calc[it] = max(0, p_calc[it - 1] + (time_span * rate_it[it - 1] * stoich_p)) * conc_adj
        rate_it[it] = k * (r_calc[it] ** x) * ((max(0, t_fit[it] - t0) * cat_add_rate) ** y)
        vol_calc = vol_calc + (inject_rate * time_span)
    return [r_calc, p_calc, rate_it]

# simulate kinetics and extract appropriate outputs (x4)
def eq_sim_r(stoich_r, stoich_p, r0, p0, cat_add_rate, t, k, x, y, t0):
    t_fit = t
    [r_calc, p_calc, rate_it] = eq_sim_gen(stoich_r, stoich_p, r0, p0, cat_add_rate, t_fit, k, x, y, t0)
    return r_calc

def eq_sim_p(stoich_r, stoich_p, r0, p0, cat_add_rate, t, k, x, y, t0):
    t_fit = t
    [r_calc, p_calc, rate_it] = eq_sim_gen(stoich_r, stoich_p, r0, p0, cat_add_rate, t_fit, k, x, y, t0)
    return p_calc

def eq_sim_rate(stoich_r, stoich_p, r0, p0, cat_add_rate, t, k, x, y, t0):
    t_fit = t
    [r_calc, p_calc, rate_it] = eq_sim_gen(stoich_r, stoich_p, r0, p0, cat_add_rate, t_fit, k, x, y, t0)
    return rate_it

def eq_sim_multi(stoich_r, stoich_p, r0, p0, cat_add_rate, t, k, x, y, t0):
    t_fit = t[:int(len(t) / 2)]
    [r_calc, p_calc, rate_it] = eq_sim_gen(stoich_r, stoich_p, r0, p0, cat_add_rate, t_fit, k, x, y, t0)
    total_calc = np.append(r_calc, p_calc)
    return total_calc

# define additional t values for data sets with few data points
def add_sim_t(t, inc_adj):
    t_fit = np.zeros(((len(t) - 1) * (inc_adj - 1)) + 1)
    for it in range(0, len(t) - 1):
        new_t_it = np.linspace(t[it], t[it + 1], inc_adj)[0:-1]
        t_fit[it * len(new_t_it):(it * len(new_t_it)) + len(new_t_it)] = new_t_it
    t_fit[-1] = t[-1]
    return t_fit

# find the location of original t values
def find_exp_t(t, t_fit, out):
    out_calc_inc = np.zeros(len(t))
    for it in range(0, len(t)):
        index = np.where(t_fit == t[it])
        out_calc_inc[it] = out[index]
    return out_calc_inc

# equivalent to previous simulation functions but with more increments (x4)
def eq_sim_r_inc(stoich_r, stoich_p, r0, p0, cat_add_rate, inc_adj, t, k, x, y, t0):
    t_fit = add_sim_t(t, inc_adj)
    [r_calc, p_calc, rate_it] = eq_sim_gen(stoich_r, stoich_p, r0, p0, cat_add_rate, t_fit, k, x, y, t0)
    r_calc_inc = find_exp_t(t, t_fit, r_calc)
    return r_calc_inc

def eq_sim_p_inc(stoich_r, stoich_p, r0, p0, cat_add_rate, inc_adj, t, k, x, y, t0):
    t_fit = add_sim_t(t, inc_adj)
    [r_calc, p_calc, rate_it] = eq_sim_gen(stoich_r, stoich_p, r0, p0, cat_add_rate, t_fit, k, x, y, t0)
    p_calc_inc = find_exp_t(t, t_fit, p_calc)
    return p_calc_inc

def eq_sim_rate_inc(stoich_r, stoich_p, r0, p0, cat_add_rate, inc_adj, t, k, x, y, t0):
    t_fit = add_sim_t(t, inc_adj)
    [r_calc, p_calc, rate_it] = eq_sim_gen(stoich_r, stoich_p, r0, p0, cat_add_rate, t_fit, k, x, y, t0)
    rate_it_inc = find_exp_t(t, t_fit, rate_it)
    return rate_it_inc

def eq_sim_multi_inc(stoich_r, stoich_p, r0, p0, cat_add_rate, inc_adj, t, k, x, y, t0):
    t_fit = add_sim_t(t[:int(len(t) / 2)], inc_adj)
    [r_calc, p_calc, rate_it] = eq_sim_gen(stoich_r, stoich_p, r0, p0, cat_add_rate, t_fit, k, x, y, t0)
    r_calc_inc = find_exp_t(t[:int(len(t) / 2)], t_fit, r_calc)
    p_calc_inc = find_exp_t(t[:int(len(t) / 2)], t_fit, p_calc)
    total_calc_inc = np.append(r_calc_inc, p_calc_inc)
    return total_calc_inc

# to estimate a k value
def half_life_calc(x_data, y_data):
    half_y = 0.5 * r0
    index = np.where(abs(y_data - half_y) == min(abs(y_data - half_y)))
    half_life = float(x_data[index])
    return half_life

def est_k_order(x, y, t0, r0, cat_add_rate, half_life):
    print([x, y, t0, cat_add_rate, half_life])
    k_guess = (((2 ** (x - 1)) - 1) * (r0 ** (1 - x)) * (y + 1)) / (((half_life - t0) ** (y + 1)) * (cat_add_rate ** y) * (x - 1))
    return k_guess

def est_k_first_order(y, t0, cat_add_rate, half_life):
    k_guess = ((y + 1) * math.log(2)) / \
            (((half_life - t0) ** (y + 1)) * (cat_add_rate ** y))
    return k_guess

# smooth data (if required)
def data_manip(d_col, df):
    d_raw = df.iloc[:, d_col].values
    if win > 1:
        d_ra = df.iloc[:, d_col].rolling(win).mean().values
        d_manip = d_ra[np.logical_not(np.isnan(d_ra))]
    else:
        d_manip = d_raw
    return d_manip

# manipulate to TIC values (for MS only)
def TIC_manip(data, TIC):
    if TIC_col != "":
        data = data / TIC
    else:
        data = data
    return data

def residuals(y_data, fit):
    residuals = y_data - fit
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return [ss_res, r_squared]

def CAKE_overall(stoich_r, stoich_p, r0, p0, p_end, react_vol_init, cat_sol_conc, inject_rate, win,
                    inc, k_est, r_ord, cat_ord, t0_est, max_order,
                    file_name, sheet_name, t_col, TIC_col, r_col, p_col, scale_avg_num,
                    fit_asp, pic_save):
    print(r_ord)
    inc_adj = inc + 2
    cat_add_rate = (cat_sol_conc * inject_rate) / react_vol_init

    df = pd.read_excel(file_name, sheet_name=sheet_name)

    t = data_manip(t_col, df)
    if TIC_col != "":
        print(TIC_col)
        TIC = data_manip(TIC_col, df)
    else:
        TIC = 1
    if r_col != "":
        r = data_manip(r_col, df)
        r = TIC_manip(r, TIC)
        if r0 == "":
            r0 = np.mean(r[0:scale_avg_num])
        else:
            r_scale = np.mean(r[0:scale_avg_num]) / r0
            r = r / r_scale
    if p_col != "":
        p = data_manip(p_col, df)
        p = TIC_manip(p, TIC)
        if p_end == "":
            p_end = np.mean(p[-scale_avg_num:-1])
        else:
            p_scale = np.mean(p[-scale_avg_num:-1]) / p_end
            p = p / p_scale

    if fit_asp == 'r':
        x_data = t
        y_data = r
        half_life = half_life_calc(x_data, y_data)
    elif fit_asp == 'p':
        x_data = t
        y_data = p
        half_life = half_life_calc(x_data, y_data)
    elif 'r' in fit_asp and 'p' in fit_asp:
        x_data = np.append(t, t)
        y_data = np.append(r, p)
        half_life = (half_life_calc(t, r) + half_life_calc(t, p)) / 2

    # define inital values and lower and upper limits for parameters to fit: k, x, y (orders wrt. r and p) and t0
    if r_ord == "":
        r_val = 1
        r_min = 0
        r_max = max_order
    elif len(r_ord) == 1:
        r_val = r_ord[0]
        r_min = r_ord[0] - 0.001
        r_max = r_ord[0] + 0.001
    elif len(r_ord) > 1:
        r_val = r_ord[0]
        r_min = r_ord[1]
        r_max = r_ord[2]
    if cat_ord == "":
        cat_val = 1
        cat_min = 0
        cat_max = max_order
    elif len(cat_ord) == 1:
        cat_val = cat_ord[0]
        cat_min = cat_ord[0] - 0.001
        cat_max = cat_ord[0] + 0.001
    elif len(cat_ord) > 1:
        cat_val = cat_ord[0]
        cat_min = cat_ord[1]
        cat_max = cat_ord[2]
    if len(t0_est) == 1:
        t0_val = t0_est[0]
        t0_min = t0_val - 0.001
        t0_max = t0_val + 0.001
    elif len(t0_est) > 1:
        t0_val = t0_est[0]
        t0_min = t0_est[1]
        t0_max = t0_est[2]
    if k_est == "" or k_est == 0:
        k_guess = np.zeros([16, 4])
        k_guess[:] = [[x, y, 0, 0] for x in [0, 1, 2, 3] for y in [0, 1, 2, 3]]
        for it in range (0, len(k_guess)):
            if k_guess[it, 0] != 1:
                k_guess[it, 2] = est_k_order(k_guess[it, 0], k_guess[it, 1], t0_val, r0, cat_add_rate, half_life)
            else:
                k_guess[it, 2] = est_k_first_order(k_guess[it, 1], t0_val, cat_add_rate, half_life)
            fit_guess = eq_sim_r(r0, p0, cat_add_rate, x_data, k_guess[it, 2], k_guess[it, 0], k_guess[it, 1], t0_val)
            [blank, k_guess[it, 3]] = residuals(y_data, fit_guess)
        index = np.where(k_guess == max(k_guess[:, 3]))
        index_first = index[0]
        k_val = float(k_guess[index_first, 2])
        k_min = k_val / 1E3
        k_max = k_val * 1E3
    elif len(k_est) == 1:
        k_val = k_est[0]
        k_min = k_val - (0.001 * k_val)
        k_max = k_val + (0.001 * k_val)
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
    if fit_asp == 'r':
        if inc == 0:
            res = optimize.curve_fit(lambda t, k, x, y, t0: eq_sim_r(stoich_r, stoich_p, r0, p0, cat_add_rate, t, k, x, y, t0),
                                     x_data, y_data, init_param, maxfev=10000,
                                     bounds=((k_min, r_min, cat_min, t0_min),
                                             (k_max, r_max, cat_max, t0_max)))
        else:
            res = optimize.curve_fit(lambda t, k, x, y, t0: eq_sim_r_inc(stoich_r, stoich_p, r0, p0, cat_add_rate, inc_adj, t, k, x, y, t0),
                                     x_data, y_data, init_param, maxfev=10000,
                                     bounds=((k_min, r_min, cat_min, t0_min),
                                             (k_max, r_max, cat_max, t0_max)))
        kf, xf, yf, t0f = res[0]
        if p_col == '""':
            fit = eq_sim_r(stoich_r, stoich_p, r0, p0, cat_add_rate, x_data, kf, xf, yf, t0f)
            fit_r = fit
        else:
            x_data = np.append(t, t)
            y_data = np.append(r, p)
            if inc == 0:
                fit = eq_sim_multi(stoich_r, stoich_p, r0, p0, cat_add_rate, x_data, kf, xf, yf, t0f)
            else:
                fit = eq_sim_multi_inc(stoich_r, stoich_p, r0, p0, cat_add_rate, inc_adj, x_data, kf, xf, yf, t0f)
            fit_r = fit[:int(len(x_data) / 2)]
            fit_p = fit[int(len(x_data) / 2):]
    elif fit_asp == 'p':
        if inc == 0:
            res = optimize.curve_fit(lambda t, k, x, y, t0: eq_sim_p(stoich_r, stoich_p, r0, p0, cat_add_rate, t, k, x, y, t0),
                                     x_data, y_data, init_param, maxfev=10000,
                                     bounds=((k_min, r_min, cat_min, t0_min),
                                             (k_max, r_max, cat_max, t0_max)))
        else:
            res = optimize.curve_fit(lambda t, k, x, y, t0: eq_sim_p_inc(stoich_r, stoich_p, r0, p0, cat_add_rate, inc_adj, t, k, x, y, t0),
                                     x_data, y_data, init_param, maxfev=10000,
                                     bounds=((k_min, r_min, cat_min, t0_min),
                                             (k_max, r_max, cat_max, t0_max)))
        kf, xf, yf, t0f = res[0]
        if r_col == "":
            if inc == 0:
                fit = eq_sim_p(stoich_r, stoich_p, r0, p0, cat_add_rate, x_data, kf, xf, yf, t0f)
            else:
                fit = eq_sim_p_inc(stoich_r, stoich_p, r0, p0, cat_add_rate, inc_adj, x_data, kf, xf, yf, t0f)
            fit_p = fit
        else:
            x_data = np.append(t, t)
            y_data = np.append(r, p)
            if inc == 0:
                fit = eq_sim_multi(stoich_r, stoich_p, r0, p0, cat_add_rate, x_data, kf, xf, yf, t0f)
            else:
                fit = eq_sim_multi_inc(stoich_r, stoich_p, r0, p0, cat_add_rate, inc_adj, x_data, kf, xf, yf, t0f)
            fit_r = fit[:int(len(x_data) / 2)]
            fit_p = fit[int(len(x_data) / 2):]
    elif 'r' in fit_asp and 'p' in fit_asp:
        if inc == 0:
            res = optimize.curve_fit(lambda t, k, x, y, t0: eq_sim_multi(stoich_r, stoich_p, r0, p0, cat_add_rate, t, k, x, y, t0),
                                     x_data, y_data, init_param, maxfev=10000,
                                     bounds=((k_min, r_min, cat_min, t0_min),
                                             (k_max, r_max, cat_max, t0_max)))
            kf, xf, yf, t0f = res[0]
            fit = eq_sim_multi(stoich_r, stoich_p, r0, p0, cat_add_rate, x_data, kf, xf, yf, t0f)
        else:
            res = optimize.curve_fit(lambda t, k, x, y, t0: eq_sim_multi_inc(stoich_r, stoich_p, r0, p0, cat_add_rate, inc_adj, t, k, x, y, t0),
                                     x_data, y_data, init_param, maxfev=10000,
                                     bounds=((k_min, r_min, cat_min, t0_min),
                                             (k_max, r_max, cat_max, t0_max)))
            kf, xf, yf, t0f = res[0]
            fit = eq_sim_multi_inc(stoich_r, stoich_p, r0, p0, cat_add_rate, inc_adj, x_data, kf, xf, yf, t0f)
        fit_r = fit[:int(len(x_data) / 2)]
        fit_p = fit[int(len(x_data) / 2):]
    if inc == 0:
        fit_rate = eq_sim_rate(stoich_r, stoich_p, r0, p0, cat_add_rate, x_data, kf, xf, yf, t0f)
    else:
        fit_rate = eq_sim_rate_inc(stoich_r, stoich_p, r0, p0, cat_add_rate, inc_adj, x_data, kf, xf, yf, t0f)

    # calculate residuals and errors
    res_val = res[0]
    res_err = np.sqrt(np.diag(res[1]))
    [ss_res, r_squared] = residuals(y_data, fit)

    # print optimal parameters, their associated errors and confidence values
    print('Starting k_value')
    print(k_val)
    print("Optimal values: rate constant k, reactant order x_data, catalyst order y_data and time zero t0")
    print(res_val)
    print("Optimal value errors: rate constant k, reactant order x_data, catalyst order y_data and time zero t0")
    print(res_err)
    print("Residual sum of squares")
    print(ss_res)
    print("R^2")
    print(r_squared)

    # calculate catalyst poisoning, if any
    if len(t0_est) > 1:
        cat_pois = max(0, (t0f - t0_val) * cat_add_rate)
        print("Catalyst poisoning")
        print(cat_pois)

    # graph results
    x_ax_scale = 1
    y_ax_scale = 1
    edge_adj = 0.02
    x_label_text = "Time / min"
    y_label_text = ""
    if r_col != "" and p_col != "":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    if r_col != "":
        if p_col == "":
            plt.figure(figsize=(6, 6))
            plt.rcParams.update({'font.size': 15})
            if len(t) <= 50:
                plt.scatter(t * x_ax_scale, r * y_ax_scale, color='k')  # plt.plot(t, r * 1E6, color='k')
            else:
                plt.plot(t * x_ax_scale, r * y_ax_scale, color='k')  # plt.plot(t, r * 1E6, color='k')
            plt.plot(t * x_ax_scale, fit * y_ax_scale, color='r')
            # plt.title("Raw")
            plt.xlim([min(t * x_ax_scale) - (edge_adj * max(t * x_ax_scale)), max(t * x_ax_scale) * (1 + edge_adj)])
            plt.ylim([min(t * y_ax_scale) - (edge_adj * max(r * y_ax_scale)), max(r * y_ax_scale) * (1 + edge_adj)])
            plt.xlabel(x_label_text)
            plt.ylabel("[R] / mM")
            plt.savefig(pic_save)
            plt.show()
        else:
            if len(t) <= 50:
                ax1.scatter(t * x_ax_scale, r * y_ax_scale, color='k')
            else:
                ax1.plot(t * x_ax_scale, r * y_ax_scale, color='k')
            ax1.plot(t * x_ax_scale, fit_r * y_ax_scale, color='r')
            # plt.title("Raw")
            ax1.set_xlim([min(t * x_ax_scale) - (edge_adj * max(t * x_ax_scale)), max(t * x_ax_scale) * (1 + edge_adj)])
            ax1.set_ylim([min(r * x_ax_scale) - (edge_adj * max(r * x_ax_scale)), max(r * x_ax_scale) * (1 + edge_adj)])
            ax1.set_xlabel(x_label_text)
            ax1.set_ylabel("[R] / $\mathregular{10^{-6}}$ M")
            # ax1.savefig(pic_save)
            # ax1.show()
            # plt.rcParams['svg']
    if p_col != "":
        if r_col == "":
            plt.figure(figsize=(6, 6))
            plt.rcParams.update({'font.size': 15})
            if len(t) <= 50:
                plt.scatter(t * x_ax_scale, p * y_ax_scale, color='k')
            else:
                plt.plot(t * x_ax_scale, p * y_ax_scale, color='k')
            plt.plot(t * x_ax_scale, fit * y_ax_scale, color='r')
            # plt.title("Raw")
            plt.xlim([min(t * x_ax_scale) - (edge_adj * max(t * x_ax_scale)), max(t * x_ax_scale) * (1 + edge_adj)])
            plt.ylim([min(t * y_ax_scale) - (edge_adj * max(p * y_ax_scale)), max(p * y_ax_scale) * (1 + edge_adj)])
            plt.xlabel(x_label_text)
            plt.ylabel("[P] / mM")  # $\mathregular{10^{-6}}$ M
            plt.savefig(pic_save)
            plt.show()
        else:
            if len(t) <= 50:
                ax2.scatter(t * x_ax_scale, p * y_ax_scale, color='k')
            else:
                ax2.plot(t * x_ax_scale, p * y_ax_scale, color='k')
            ax2.plot(t * x_ax_scale, fit_p * y_ax_scale, color='r')
            # plt.title("Raw")
            ax2.set_xlim([min(t * x_ax_scale) - (edge_adj * max(t * x_ax_scale)), max(t * x_ax_scale) * (1 + edge_adj)])
            ax2.set_ylim([min(p * x_ax_scale) - (edge_adj * max(p * x_ax_scale)), max(p * x_ax_scale) * (1 + edge_adj)])
            ax2.set_xlabel(x_label_text)
            ax2.set_ylabel("[P] / $\mathregular{10^{-6}}$ M")
            fig.savefig(pic_save)
            #ax2.savefig(pic_save)
            #ax2.show()
    plt.show()
    # plt.plot(t * x_ax_scale, r * y_ax_scale, color='k')
    # plt.plot(t * x_ax_scale, eq_sim_r(x_data, 1.47, 1, 2, t0_val) * y_ax_scale, color='r')
    # plt.show()

CAKE_overall(stoich_r, stoich_p, r0, p0, p_end, react_vol_init, cat_sol_conc, inject_rate, win,
                    inc, k_est, r_ord, cat_ord, t0_est, max_order,
                    file_name, sheet_name, t_col, TIC_col, r_col, p_col, scale_avg_num,
                    fit_asp, pic_save)