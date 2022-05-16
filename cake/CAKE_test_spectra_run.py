# CAKE Fitting Programme
import numpy as np
import pandas as pd
from CAKE_current import CAKE_overall
import timeit
from datetime import date

def error_bounds_sort(substrate):
    if substrate == '""':
        substrate_adj = substrate
    elif len(str(substrate)) == 1:
        substrate_adj = [substrate]
    else:
        split = substrate.split(', ')
        substrate_adj = [float(x) for x in split]
    return substrate_adj

# create dataframe
df = pd.read_excel(r'C:\Users\Peter\Documents\Postdoctorate\Work\CAKE\Programmes\Test_Spectra.xlsx', sheet_name='Parameters')

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
    output = CAKE_overall(stoich_r, stoich_p, r0, p0, p_end, react_vol_init, cat_sol_conc, inject_rate, win,
                          inc, k_est, r_ord, cat_ord, t0_est, max_order,
                          file_name, sheet_name, t_col, TIC_col, r_col, p_col, scale_avg_num,
                          fit_asp, pic_save)
    time_taken = timeit.default_timer() - starttime
    total[it] = [number, type, *output, time_taken]

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