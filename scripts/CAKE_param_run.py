# CAKE Fitting Programme
import numpy as np
import pandas as pd
import cake
# from CAKE_current import CAKE_overall
import timeit
from datetime import date

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
    file, _ = cake.write_fit_data_temp(df, param_dict, t, r, p, fit_p, fit_r, res_val, res_err, ss_res, r_squared, cat_pois,
                                  cat_pois_err)
    file.seek(0)
    with open(xlsx_save, "wb") as f:
        f.write(file.getbuffer())


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
