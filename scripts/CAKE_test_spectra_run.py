# CAKE Fitting Programme
import numpy as np
import pandas as pd
import cake
# from CAKE_current import CAKE_overall
import timeit
from datetime import date

def error_bounds_sort(substrate):
    if substrate is None:
        substrate_adj = substrate
    elif ', ' not in str(substrate):
        substrate_adj = [float(substrate)]
    else:
        split = substrate.split(', ')
        substrate_adj = [float(x) for x in split]
    return substrate_adj

# create dataframe
df = pd.read_excel(r'C:\Users\Peter\Documents\Postdoctorate\Work\CAKE\Programmes\CAKE_2.0\Test_Spectra_CAKE_2.0.xlsx', sheet_name='Parameters')
df.replace('""', None, inplace=True)

total = np.empty([len(df), 16], object)
for it in range(0, len(df)):
    starttime = timeit.default_timer()
    [number, react_type] = df.iloc[it, :2]
    print(react_type)
    [stoich_r, stoich_p, r0, p0, p_end, react_vol_init, cat_sol_conc, inject_rate, t_inj, win] = df.iloc[it, 2:12]
    [inc, k_lim, r_ord_lim, cat_ord_lim, t_del_lim] = df.iloc[it, 12:17]
    [file_name, sheet_name, t_col, TIC_col, r_col, p_col, scale_avg_num] = df.iloc[it, 17:24]
    [fit_asp, pic_save] = df.iloc[it, 24:]
    k_lim = error_bounds_sort(k_lim)
    r_ord_lim = error_bounds_sort(r_ord_lim)
    cat_ord_lim = error_bounds_sort(cat_ord_lim)
    t_del_lim = error_bounds_sort(t_del_lim)
    data = cake.read_data(file_name, sheet_name)
    cat_add_rate = cake.get_cat_add_rate(cat_sol_conc, inject_rate, react_vol_init)
    output = cake.fit_cake(data, stoich_r, stoich_p, r0, p0, p_end, cat_add_rate, t_inj, k_lim, r_ord_lim, cat_ord_lim,
                           t_del_lim, t_col, TIC_col, r_col, p_col, scale_avg_num, win, inc, fit_asp)
    t, r, p, fit, fit_p, fit_r, k_val, res_val, res_err, ss_res, r_squared, cat_pois, cat_pois_err = output
    time_taken = timeit.default_timer() - starttime

    plot_output = cake.plot_cake_results(t, r, p, fit, fit_p, fit_r, r_col, p_col, f_format='png', return_image=False, save_disk=True,
                          save_to=pic_save, return_fig=False)

    exportdf = pd.DataFrame({"Time": t, "Concentration R": r, "Concentration P": p,
                             "Fit R": fit_r, "Fit P": fit_p})
    with pd.ExcelWriter(r'C:\Users\Peter\Documents\Postdoctorate\Work\CAKE\Programmes\CAKE_2.0\Test_Spectra_Results_Fit_CAKE_2.0.xlsx',
                        mode='a', if_sheet_exists='replace') as writer:
        exportdf.to_excel(writer, sheet_name=str(number), index=False)

    total[it] = [number, react_type, k_val, res_val[0], res_err[0],
                 res_val[1], res_err[1], res_val[2], res_err[2], res_val[3], res_err[3],
                 cat_pois, cat_pois_err, ss_res, r_squared, time_taken]

exportdf = pd.DataFrame({"Number":total[:, 0],"Type": total[:, 1],"k_val_est":total[:, 2],
                         "k_fit":total[:, 3], "k_fit_err":total[:, 4], "r_ord_fit":total[:, 5],
                         "r_ord_fit_err":total[:, 6], "cat_ord_fit":total[:, 7], "cat_ord_fit_err":total[:, 8],
                         "t_del_fit":total[:, 9], "t_del_fit_err":total[:, 10], "cat_pois":total[:, 11],
                         "cat_pois_err":total[:, 12], "res_sum_squares":total[:, 13], "r_squared":total[:, 14],
                         "script_runtime":total[:, 15]})
date = date.today().strftime("%y%m%d")
with pd.ExcelWriter(r'C:\Users\Peter\Documents\Postdoctorate\Work\CAKE\Programmes\CAKE_2.0\Test_Spectra_Results_CAKE_2.0.xlsx',
                    mode='a', if_sheet_exists='new') as writer:
    exportdf.to_excel(writer, sheet_name=date, index=False)  #usually use 'new'
