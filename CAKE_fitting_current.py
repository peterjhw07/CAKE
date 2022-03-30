# CAKE Fitting Programme
# Known parameters and smoothing
stoich_r = 1  # insert stoichiometry of r
stoich_p = 1  # insert stoichiometry of p
r0 = 3.2  # enter value of r0 in M dm^-3 or "" if data are given in M dm^-3
p0 = 0  # enter value of p0 in M dm^-3 or "" if data are given in M dm^-3
p_end = r0  # enter end value of product in M dm^-3, r0 if equal to start r0 value, or "" if data are given in M dm^-3
cat_add_rate = 1.57  # enter catalyst addition rate in M time_unit^-1
win = 1  # enter smoothing window (1 if smoothing not required)

# Parameter fitting
# Enter "" for any order, [exact value] for fixed variable or variable with bounds [estimate, factor difference] or [estimate, lower, upper]
inc = 1  # enter increments between adjacent points for improved simulation, "" or 1 for using raw time points
k_est = [1E-2, 1E3]  # enter rate constant in (M dm^-3)^? time_unit^-1
r_ord = [1, 0, 3]  # enter r order
cat_ord = [1, 0, 3]  # enter cat order
t0_est = [0]  # enter time at which injection began in time_unit^-1
max_order = 3  # enter maximum possible order for species

# Experimental data location
file_name = r'C:\Users\Peter\Documents\Postdoctorate\Work\CAKE\Case studies\Visible Chemiluminescence\March 11 Light Meter Test.xlsx'  # enter filename as r'file_name'
sheet_name = 'Sheet2'  # enter sheet name as 'sheet_name'
t_col = 0  # enter time column
TIC_col = ""  # enter TIC column or "" if no TIC
r_col = 3  # enter [r] column or "" if no r
p_col = ""  # enter [p] column or "" if no p
scale_avg_num = 1  # enter number of data points from which to calculate r0 and p_end

fit_asp= 'r'  # enter aspect you want to fit to: 'r' for reactant, 'p' for product or 'rp' for both

pic_save = r'C:\Users\Peter\Documents\Postdoctorate\Work\CAKE\Figures\7.png'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
# import addcopyfighandler

# general kinetic simulator
def eq_sim_gen(t_fit, k, x, y, t0):
    r_calc = np.zeros(len(t_fit))
    p_calc = np.zeros(len(t_fit))
    rate_it = np.zeros(len(t_fit))
    r_calc[0] = r0
    p_calc[0] = p0
    rate_it[0] = k * (r_calc[0] ** x) * ((max(0, t_fit[0] - t0) * cat_add_rate) ** y)
    for it in range(1, len(t_fit)):
        r_calc[it] = max(0, r_calc[it - 1] - ((t_fit[it] - t_fit[it - 1]) * rate_it[it - 1] * stoich_r))
        p_calc[it] = max(0, p_calc[it - 1] + ((t_fit[it] - t_fit[it - 1]) * rate_it[it - 1] * stoich_p))
        rate_it[it] = k * (r_calc[it] ** x) * ((max(0, t_fit[it] - t0) * cat_add_rate) ** y)
    return [r_calc, p_calc, rate_it]

def eq_sim_r(t, k, x, y, t0):
    t_fit = t
    [r_calc, p_calc, rate_it] = eq_sim_gen(t_fit, k, x, y, t0)
    return r_calc

def eq_sim_p(t, k, x, y, t0):
    t_fit = t
    [r_calc, p_calc, rate_it] = eq_sim_gen(t_fit, k, x, y, t0)
    return p_calc

def eq_sim_rate(t, k, x, y, t0):
    t_fit = t
    [r_calc, p_calc, rate_it] = eq_sim_gen(t_fit, k, x, y, t0)
    return rate_it

def eq_sim_multi(t, k, x, y, t0):
    t_fit = t[:int(len(t) / 2)]
    [r_calc, p_calc, rate_it] = eq_sim_gen(t_fit, k, x, y, t0)
    total_calc = np.append(r_calc, p_calc)
    return total_calc

def add_sim_t(t):
    t_fit = np.zeros(((len(t) - 1) * (inc - 1)) + 1)
    for it in range(0, len(t) - 1):
        new_t_it = np.linspace(t[it], t[it + 1], inc)[0:-1]
        t_fit[it * len(new_t_it):(it * len(new_t_it)) + len(new_t_it)] = new_t_it
    t_fit[-1] = t[-1]
    return t_fit

def find_exp_t(t, t_fit, out):
    out_calc_inc = np.zeros(len(t))
    for it in range(0, len(t)):
        index = np.where(t_fit == t[it])
        out_calc_inc[it] = out[index]
    return out_calc_inc

def eq_sim_r_inc(t, k, x, y, t0):
    t_fit = add_sim_t(t)
    [r_calc, p_calc, rate_it] = eq_sim_gen(t_fit, k, x, y, t0)
    r_calc_inc = find_exp_t(t, t_fit, r_calc)
    return r_calc_inc

def eq_sim_p_inc(t, k, x, y, t0):
    t_fit = add_sim_t(t)
    [r_calc, p_calc, rate_it] = eq_sim_gen(t_fit, k, x, y, t0)
    p_calc_inc = find_exp_t(t, t_fit, p_calc)
    return p_calc_inc

def eq_sim_rate_inc(t, k, x, y, t0):
    t_fit = add_sim_t(t)
    [r_calc, p_calc, rate_it] = eq_sim_gen(t_fit, k, x, y, t0)
    rate_it_inc = find_exp_t(t, t_fit, rate_it)
    return rate_it_inc

def eq_sim_multi_inc(t, k, x, y, t0):
    t_fit = add_sim_t(t[:int(len(t) / 2)])
    [r_calc, p_calc, rate_it] = eq_sim_gen(t_fit, k, x, y, t0)
    r_calc_inc = find_exp_t(t[:int(len(t) / 2)], t_fit, r_calc)
    p_calc_inc = find_exp_t(t[:int(len(t) / 2)], t_fit, p_calc)
    total_calc_inc = np.append(r_calc_inc, p_calc_inc)
    return total_calc_inc

def data_manip(d_col):
    d_raw = df.iloc[:, d_col].values
    if win > 1:
        d_ra = df.iloc[:, d_col].rolling(win).mean().values
        d_manip = d_ra[np.logical_not(np.isnan(d_ra))]
    else:
        d_manip = d_raw
    return d_manip

def TIC_manip(data):
    if TIC_col != "":
        data = data / TIC
    else:
        data = data
    return data

df = pd.read_excel (file_name, sheet_name=sheet_name)

inc = inc + 1
t = data_manip(t_col)
if TIC_col != "":
    TIC = data_manip(TIC_col)
if r_col != "":
    r = data_manip(r_col)
    r = TIC_manip(r)
    if r0 == "":
        r0 = np.mean(r[0:scale_avg_num])
    else:
        r_scale = np.mean(r[0:scale_avg_num]) / r0
        r = r / r_scale
if p_col != "":
    p = data_manip(p_col)
    p = TIC_manip(p)
    if p_end == "":
        p_end = np.mean(p[-scale_avg_num:-1])
    else:
        p_scale = np.mean(p[-scale_avg_num:-1]) / p_end
        p = p / p_scale

if k_est == "" or k_est == 0:
    print("ERROR: an estimate of k must be entered")
    exit()
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

init_param = [k_val, r_val, cat_val, t0_val]

if fit_asp == 'r':
    x_data = t
    y_data = r
    if inc == "" or inc == 1:
        res = optimize.curve_fit(eq_sim_r, x_data, y_data, init_param,
                                 bounds=((k_min, r_min, cat_min, t0_min), (k_max, r_max, cat_max, t0_max)))  # maxfev=5000
    else:
        res = optimize.curve_fit(eq_sim_r_inc, x_data, y_data, init_param,
                                 bounds=((k_min, r_min, cat_min, t0_min), (k_max, r_max, cat_max, t0_max)))  # maxfev=5000
    kf, xf, yf, t0f = res[0]
    if p_col == "":
        fit = eq_sim_r(x_data, kf, xf, yf, t0f)
        fit_r = fit
    else:
        x_data = np.append(t, t)
        y_data = np.append(r, p)
        if inc == "" or inc == 1:
            fit = eq_sim_multi(x_data, kf, xf, yf, t0f)
        else:
            fit = eq_sim_multi_inc(x_data, kf, xf, yf, t0f)
        fit_r = fit[:int(len(x_data) / 2)]
        fit_p = fit[int(len(x_data) / 2):]
elif fit_asp == 'p':
    x_data = t
    y_data = p
    if inc == "" or inc == 1:
        res = optimize.curve_fit(eq_sim_p, x_data, y_data, init_param,
                                 bounds=((k_min, r_min, cat_min, t0_min), (k_max, r_max, cat_max, t0_max)))  # maxfev=5000
    else:
        res = optimize.curve_fit(eq_sim_p_inc, x_data, y_data, init_param,
                                 bounds=((k_min, r_min, cat_min, t0_min), (k_max, r_max, cat_max, t0_max)))  # maxfev=5000
    kf, xf, yf, t0f = res[0]
    if r_col == "":
        if inc == "" or inc == 1:
            fit = eq_sim_p(x_data, kf, xf, yf, t0f)
        else:
            fit = eq_sim_p_inc(x_data, kf, xf, yf, t0f)
        fit_p = fit
    else:
        x_data = np.append(t, t)
        y_data = np.append(r, p)
        if inc == "" or inc == 1:
            fit = eq_sim_multi(x_data, kf, xf, yf, t0f)
        else:
            fit = eq_sim_multi_inc(x_data, kf, xf, yf, t0f)
        fit_r = fit[:int(len(x_data) / 2)]
        fit_p = fit[int(len(x_data) / 2):]
elif 'r' in fit_asp and 'p' in fit_asp:
    x_data = np.append(t, t)
    y_data = np.append(r, p)
    if inc == "" or inc == 1:
        res = optimize.curve_fit(eq_sim_multi, x_data, y_data, init_param,
                                 bounds=((k_min, r_min, cat_min, t0_min), (k_max, r_max, cat_max, t0_max)))  # maxfev=5000
        kf, xf, yf, t0f = res[0]
        fit = eq_sim_multi(x_data, kf, xf, yf, t0f)
    else:
        res = optimize.curve_fit(eq_sim_multi_inc, x_data, y_data, init_param,
                                 bounds=((k_min, r_min, cat_min, t0_min), (k_max, r_max, cat_max, t0_max)))  # maxfev=5000
        kf, xf, yf, t0f = res[0]
        fit = eq_sim_multi_inc(x_data, kf, xf, yf, t0f)
    fit_r = fit[:int(len(x_data) / 2)]
    fit_p = fit[int(len(x_data) / 2):]

res_val = res[0]
res_err = np.sqrt(np.diag(res[1]))
residuals = y_data - fit
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

#if inc == "" or inc == 1:
#    fit_rate = eq_sim_rate(x_data, kf, xf, yf, t0f)
#else:
#    fit_rate = eq_sim_rate_inc(x_data, kf, xf, yf, t0f)

print("Optimal values: rate constant k, reactant order x_data, catalyst order y_data and time zero t0")
print(res_val)
print("Optimal value errors: rate constant k, reactant order x_data, catalyst order y_data and time zero t0")
print(res_err)
print("Residual sum of squares")
print(ss_res)
print("R^2")
print(r_squared)

if len(t0_est) > 1:
    cat_pois = max(0, (t0f - t0_val) * cat_add_rate)
    print("Catalyst poisoning")
    print(cat_pois)

ax_scale = 1
edge_adj = 0.02
# pic_save = r'C:\Users\Peter\Documents\Postdoctorate\Work\CAKE\Figures\1.png'
if r_col != "" and p_col != "":
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
if r_col != "":
    if p_col == "":
        plt.figure(figsize=(6, 6))
        plt.rcParams.update({'font.size': 15})
        if len(t) <= 50:
            plt.scatter(t, r * ax_scale, color='k')  # plt.plot(t, r * 1E6, color='k')
        else:
            plt.plot(t, r * ax_scale, color='k')  # plt.plot(t, r * 1E6, color='k')
        plt.plot(t, fit * ax_scale, color='r')
        # plt.title("Raw")
        plt.xlim([- (edge_adj * max(t)), max(t) + (edge_adj * max(t))])
        plt.ylim([- (edge_adj * max(r) * ax_scale), (max(r) * ax_scale) + (edge_adj * max(r) * ax_scale)])
        plt.xlabel("Time / min")
        plt.ylabel("[R] / mM")
        plt.savefig(pic_save)
        plt.show()
    else:
        if len(t) <= 50:
            ax1.scatter(t, r * ax_scale, color='k')
        else:
            ax1.plot(t, r * ax_scale, color='k')
        ax1.plot(t, fit_r * ax_scale, color='r')
        # plt.title("Raw")
        ax1.set_xlim([0, max(t)])
        ax1.set_ylim([0, max(r) * ax_scale])
        ax1.set_xlabel("Time / min")
        ax1.set_ylabel("[R] / $\mathregular{10^{-3}}$ M")
        # ax1.savefig(pic_save)
        # ax1.show()
        # plt.rcParams['svg']

if p_col != "":
    if r_col == "":
        plt.figure(figsize=(6, 6))
        plt.rcParams.update({'font.size': 15})
        if len(t) <= 50:
            plt.scatter(t, p * ax_scale, color='k')
        else:
            plt.plot(t, p * ax_scale, color='k')
        plt.plot(t, fit * ax_scale, color='r')
        # plt.title("Raw")
        plt.xlim([0, max(t)])
        plt.ylim([0, max(p) * ax_scale])
        plt.xlabel("Time / min")
        plt.ylabel("[P] / $\mathregular{10^{-6}}$ M")
        # plt.savefig(pic_save)
        plt.show()
    else:
        if len(t) <= 50:
            ax2.scatter(t, p * ax_scale, color='k')
        else:
            ax2.plot(t, p * ax_scale, color='k')
        ax2.plot(t, fit_p * ax_scale, color='r')
        # plt.title("Raw")
        ax2.set_xlim([0, max(t)])
        ax2.set_ylim([0, max(p) * ax_scale])
        ax2.set_xlabel("Time / min")
        ax2.set_ylabel("[P] / $\mathregular{10^{-6}}$ M")
        #ax2.savefig(pic_save)
        #ax2.show()

plt.show()