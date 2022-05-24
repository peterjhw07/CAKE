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
        substrate_adj = [substrate]
    else:
        split = substrate.split(', ')
        substrate_adj = [float(x) for x in split]
    return substrate_adj

# create dataframe
df = pd.read_excel(r'C:\Users\Peter\Documents\Postdoctorate\Work\CAKE\Programmes\Test_Spectra.xlsx', sheet_name='Parameters')
df.replace('""', None, inplace=True)

total = np.empty([len(df), 16], object)
for it in range(0, len(df)):
    starttime = timeit.default_timer()
    [number, type] = df.iloc[it, :2]
    print(type)
    [stoich_r, stoich_p, r0, p0, p_end, react_vol_init, cat_sol_conc, inject_rate, win] = df.iloc[it, 2:11]
    [inc, k_est, r_ord, cat_ord, t0_est, max_order] = df.iloc[it, 11:17]
    [file_name, sheet_name, t_col, TIC_col, r_col, p_col, scale_avg_num] = df.iloc[it, 17:24]
    [fit_asp, pic_save] = df.iloc[it, 24:]
    k_est = error_bounds_sort(k_est)
    r_ord = error_bounds_sort(r_ord)
    cat_ord = error_bounds_sort(cat_ord)
    t0_est = error_bounds_sort(t0_est)
    data = cake.read_data(file_name, sheet_name)
    cat_add_rate = cake.get_cat_add_rate(cat_sol_conc, inject_rate, react_vol_init)
    output = cake.fit_cake(data, stoich_r, stoich_p, r0, p0, p_end, cat_add_rate, k_est, r_ord, cat_ord, t0_est, t_col,
                           TIC_col, r_col, p_col, max_order, scale_avg_num, win, inc, fit_asp)
    t, r, p, fit, fit_p, fit_r, k_val, res_val, res_err, ss_res, r_squared, cat_pois, cat_pois_err = output
    time_taken = timeit.default_timer() - starttime

    exportdf = pd.DataFrame({"Time": t, "Concentration R": r, "Concentration P": p,
                             "Fit R": fit_r, "Fit P": fit_p})
    with pd.ExcelWriter(r'C:\Users\Peter\Documents\Postdoctorate\Work\CAKE\Programmes\Test_Spectra_Results_Fit.xlsx',
                        mode='a', if_sheet_exists='replace') as writer:
        exportdf.to_excel(writer, sheet_name=str(number), index=False)

    total[it] = [number, type, k_val, res_val[0], res_err[0],
                 res_val[1], res_err[1], res_val[2], res_err[2], res_val[3], res_err[3],
                 cat_pois, cat_pois_err, ss_res, r_squared, time_taken]

exportdf = pd.DataFrame({"Number":total[:, 0],"Type": total[:, 1],"k_val_est":total[:, 2],
                         "k_fit":total[:, 3], "k_fit_err":total[:, 4], "r_ord_fit":total[:, 5],
                         "r_ord_fit_err":total[:, 6], "cat_ord_fit":total[:, 7], "cat_ord_fit_err":total[:, 8],
                         "t0_fit":total[:, 9], "t0_fit_err":total[:, 10], "cat_pois":total[:, 11],
                         "cat_pois_err":total[:, 12], "res_sum_squares":total[:, 13], "r_squared":total[:, 14],
                         "script_runtime":total[:, 15]})
date = date.today().strftime("%y%m%d")
with pd.ExcelWriter(r'C:\Users\Peter\Documents\Postdoctorate\Work\CAKE\Programmes\Test_Spectra_Results.xlsx',
                    mode='a', if_sheet_exists='new') as writer:
    exportdf.to_excel(writer, sheet_name=date, index=False)
