# CAKE Fitting Programme
import numpy as np
import pandas as pd
from scripts import cake_fitting_multi
# from CAKE_current import CAKE_overall
import timeit
from datetime import date
import re


def make_char_tup(substrate):
    if ', ' in str(substrate):
        substrate_adj = re.split(r',\s*(?![^()]*\))', str(substrate))
    else:
        substrate_adj = substrate
    return substrate_adj


def make_char_tup_and_sort(s):
    split = make_char_tup(s)
    if isinstance(split, (int, float)):
        s_adj = split
    elif "None" in split and isinstance(split, str):
        s_adj = None
    elif not isinstance(split, int):
        s_adj = [eval(str(x)) for x in split]
    else:
        s_adj = split
    return s_adj


# create dataframe
df = pd.read_excel(r'C:\Users\Peter\Documents\Postdoctorate\Work\CAKE\Programmes\Test_Spectra.xlsx', sheet_name='Testing')
df.replace('""', None, inplace=True)

total = np.empty([len(df), 12], object)
for i in range(0, len(df)):
    [number, react_type] = df.iloc[i, :2]
    [spec_type, react_vol_init, stoich, mol0, mol_end, add_sol_conc, add_cont_rate, t_cont] = df.iloc[i, 2:10]
    [add_one_shot, t_one_shot, add_col, sub_cont_rate, sub_aliq, t_aliq, sub_col] = df.iloc[i, 10:17]
    [t_col, col, k_lim, ord_lim, pois_lim, fit_asp] = df.iloc[i, 17:23]
    [TIC_col, scale_avg_num, win, inc, file_name, sheet_name, pic_save] = df.iloc[i, 23:]
    print(number, react_type)
    spec_type = make_char_tup(spec_type)
    stoich = make_char_tup_and_sort(stoich)
    mol0 = make_char_tup_and_sort(mol0)
    mol_end = make_char_tup_and_sort(mol_end)
    add_sol_conc = make_char_tup_and_sort(add_sol_conc)
    add_cont_rate = make_char_tup_and_sort(add_cont_rate)
    t_cont = make_char_tup_and_sort(t_cont)
    add_one_shot = make_char_tup_and_sort(add_one_shot)
    t_one_shot = make_char_tup_and_sort(t_one_shot)
    add_col = make_char_tup_and_sort(add_col)
    sub_cont_rate = make_char_tup_and_sort(sub_cont_rate)
    sub_aliq = make_char_tup_and_sort(sub_aliq)
    t_aliq = make_char_tup_and_sort(t_aliq)
    sub_col = make_char_tup_and_sort(sub_col)
    col = make_char_tup_and_sort(col)
    k_lim = make_char_tup_and_sort(k_lim)
    ord_lim = make_char_tup_and_sort(ord_lim)
    pois_lim = make_char_tup_and_sort(pois_lim)
    fit_asp = make_char_tup(fit_asp)
    sheet_name = str(sheet_name)
    # print(spec_type, react_vol_init, stoich, mol0, mol_end, add_sol_conc, add_cont_rate, t_cont, add_one_shot, t_one_shot, add_col, t_col, col, k_lim, ord_lim, pois_lim, fit_asp, TIC_col, scale_avg_num, win, inc, file_name, sheet_name, pic_save)
    data = cake_fitting_multi.read_data(file_name, sheet_name, t_col, col, add_col, sub_col)
    starttime = timeit.default_timer()
    output = cake_fitting_multi.fit_cake(data, spec_type, react_vol_init, stoich=stoich, mol0=mol0, mol_end=mol_end,
                    add_sol_conc=add_sol_conc, add_cont_rate=add_cont_rate, t_cont=t_cont, add_one_shot=add_one_shot,
                    t_one_shot=t_one_shot, add_col=add_col, sub_cont_rate=sub_cont_rate,
                    sub_aliq=sub_aliq, t_aliq=t_aliq, sub_col=sub_col, t_col=t_col, col=col, k_lim=k_lim, ord_lim=ord_lim,
                    pois_lim=pois_lim, fit_asp=fit_asp, scale_avg_num=scale_avg_num, win=win, inc=inc)
    x_data, y_data, fit, fit_rate, k_val_est, k_fit, k_fit_err, ord_fit, ord_fit_err, pois_fit, pois_fit_err, ss_res, r_squared, col = output
    time_taken = timeit.default_timer() - starttime
    print(k_fit, k_fit_err, ord_fit, ord_fit_err, pois_fit, pois_fit_err)

    imp_headers = list(data.columns)
    fit_headers=[]
    if not isinstance(col, (tuple, list)): col = [col]
    for j in range(len(col)):
        if col[j] is not None:
            fit_headers = [*fit_headers, 'Fit '+imp_headers[col[j]]]
        else:
            fit_headers = [*fit_headers, 'Fit species ' + str(j + 1)]
    exp_headers = [imp_headers[t_col], *['Exp '+imp_headers[i] for i in col if i is not None], *fit_headers, 'Fit rate']

    plot_output = cake_fitting_multi.plot_cake_results(x_data, y_data, fit, col, exp_headers,
                                                       f_format='png', save_disk=True, save_to=pic_save)

    export_fit = "n"
    if "y" in export_fit:
        all_data = np.concatenate((x_data, y_data, fit, fit_rate), axis=1)
        exportdf = pd.DataFrame(all_data, columns=exp_headers)
        with pd.ExcelWriter(r'C:\Users\Peter\Documents\Postdoctorate\Work\CAKE\Programmes\Test_Spectra_Results_Fit.xlsx',
                            mode='a', if_sheet_exists='replace') as writer:
            exportdf.to_excel(writer, sheet_name=str(number), index=False)

    total[i] = [number, react_type, k_val_est, k_fit, k_fit_err, ord_fit, ord_fit_err, pois_fit, pois_fit_err, ss_res, r_squared, time_taken]

exportdf = pd.DataFrame({"Number":total[:, 0],"Type": total[:, 1],"k_val_est":total[:, 2],
                         "k_fit":total[:, 3], "k_fit_err":total[:, 4], "ord_fit":total[:, 5],
                         "ord_fit_err":total[:, 6], "pois_fit":total[:, 7], "pois_fit_err":total[:, 8],
                         "res_sum_squares":total[:, 9], "r_squared":total[:, 10],
                         "script_runtime":total[:, 11]})
date = date.today().strftime("%y%m%d")
with pd.ExcelWriter(r'C:\Users\Peter\Documents\Postdoctorate\Work\CAKE\Programmes\Test_Spectra_Results.xlsx',
                    mode='a', if_sheet_exists='new') as writer:
    exportdf.to_excel(writer, sheet_name=date, index=False)  #usually use 'new'
