# CAKE Fitting Programme
import numpy as np
import pandas as pd
from scripts import cake_fitting_multi
# from CAKE_current import CAKE_overall
import timeit
from datetime import date
import re

def error_bounds_sort(substrate):
    if substrate is None:
        substrate_adj = substrate
    elif ', ' not in str(substrate):
        substrate_adj = [float(substrate)]
    else:
        split = re.split(r',\s*(?![^()]*\))', substrate)
        substrate_adj = [float(x) for x in split]
    return substrate_adj

def make_char_tup(substrate):
    if ', ' in str(substrate):
        substrate_adj = re.split(r',\s*(?![^()]*\))', str(substrate))
        # substrate_adj = tuple(substrate.split(', '))
    return substrate_adj

def make_char_tup_and_float(substrate):
    split = make_char_tup(substrate)
    substrate_adj = [None if 'None' in x else float(x) for x in split]
    return substrate_adj

def make_char_tup_and_int(substrate):
    split = make_char_tup(substrate)
    substrate_adj = [None if 'None' in x else int(x) for x in split]
    return substrate_adj

def make_char_tup_and_sort(substrate):
    split = make_char_tup(substrate)
    substrate_adj = [eval(x) for x in split]
    return substrate_adj

# create dataframe
df = pd.read_excel(r'C:\Users\Peter\Documents\Postdoctorate\Work\CAKE\Programmes\Test_Spectra.xlsx', sheet_name='Overhaul')
df.replace('""', None, inplace=True)

total = np.empty([len(df), 14], object)
for it in range(0, len(df)):
    starttime = timeit.default_timer()
    [number, react_type] = df.iloc[it, :2]
    print(react_type)
    [spec_type, stoich, conc0, conc_end, react_vol_init, cat_sol_conc, inject_rate, t_inj] = df.iloc[it, 2:10]
    [win, inc, k_lim, ord_lim, t_del_lim] = df.iloc[it, 10:15]
    [file_name, sheet_name, t_col, TIC_col, col, scale_avg_num, fit_asp] = df.iloc[it, 15:22]
    [pic_save] = df.iloc[it, 22:]
    spec_type = make_char_tup(spec_type)
    stoich = make_char_tup_and_sort(stoich)
    conc0 = make_char_tup_and_sort(conc0)
    conc_end = make_char_tup_and_sort(conc_end)
    cat_sol_conc = make_char_tup_and_sort(cat_sol_conc)
    inject_rate = make_char_tup_and_sort(inject_rate)
    ord_lim = make_char_tup_and_sort(ord_lim)
    col = make_char_tup_and_sort(col)
    fit_asp = make_char_tup(fit_asp)
    k_lim = error_bounds_sort(k_lim)
    t_del_lim = error_bounds_sort(t_del_lim)
    data = cake_fitting_multi.read_data(file_name, sheet_name)
    cat_add_rate = cake_fitting_multi.get_cat_add_rate(cat_sol_conc, inject_rate, react_vol_init)
    output = cake_fitting_multi.fit_cake(data, spec_type, stoich, conc0, conc_end, cat_add_rate, t_inj, k_lim, ord_lim,
                           t_del_lim, t_col, TIC_col, col, scale_avg_num, win, inc, fit_asp)
    x_data, y_data, fit, k_val, res_val, res_err, ss_res, r_squared, cat_pois, cat_pois_err = output
    time_taken = timeit.default_timer() - starttime

    # plot_output = cake_fitting_multi.plot_cake_results(t, r, p, fit, fit_p, fit_r, r_col, p_col, f_format='png', return_image=False, save_disk=True,
                          # save_to=pic_save, return_fig=False)

    imp_headers = list(data.columns)
    exp_headers = [imp_headers[t_col], *[imp_headers[i] for i in col if i is not None], *[imp_headers[i] for i in col if i is not None], 'a']  # this line will break
    all_data = np.concatenate((x_data, y_data, fit), axis=1)
    exportdf = pd.DataFrame(all_data, columns=exp_headers)
    with pd.ExcelWriter(r'C:\Users\Peter\Documents\Postdoctorate\Work\CAKE\Programmes\Test_Spectra_Results_Fit.xlsx',
                        mode='a', if_sheet_exists='replace') as writer:
        exportdf.to_excel(writer, sheet_name=str(number), index=False)

    total[it] = [number, react_type, k_val, res_val[0], res_err[0], res_val[1], res_err[1], [res_val[i] for i in range(2, len(res_val))],
                 [res_err[i] for i in range(2, len(res_val))], cat_pois, cat_pois_err, ss_res, r_squared, time_taken]

exportdf = pd.DataFrame({"Number":total[:, 0],"Type": total[:, 1],"k_val_est":total[:, 2],
                         "k_fit":total[:, 3], "k_fit_err":total[:, 4], "ord_fit":total[:, 7],
                         "ord_fit_err":total[:, 8],
                         "t_del_fit":total[:, 5], "t_del_fit_err":total[:, 6], "cat_pois":total[:, 9],
                         "cat_pois_err":total[:, 10], "res_sum_squares":total[:, 11], "r_squared":total[:, 12],
                         "script_runtime":total[:, 13]})
date = date.today().strftime("%y%m%d")
with pd.ExcelWriter(r'C:\Users\Peter\Documents\Postdoctorate\Work\CAKE\Programmes\Test_Spectra_Results.xlsx',
                    mode='a', if_sheet_exists='new') as writer:
    exportdf.to_excel(writer, sheet_name=date, index=False)  #usually use 'new'
