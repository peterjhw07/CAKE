# CAKE Fitting Programme
import numpy as np
import pandas as pd
from scripts import cake_fitting_multi
# from CAKE_current import CAKE_overall
import timeit
from datetime import date
import re


def input_sort(s):
    if ', ' in str(s):
        split = re.split(r',\s*(?![^()]*\))', str(s))
    else:
        split = s
    if isinstance(split, (int, float)):
        s_adj = split
    elif "None" in split and isinstance(split, str):
        s_adj = None
    elif not isinstance(split, (int, str)):
        s_adj = [eval(str(x)) for x in split]
    else:
        s_adj = split
    return s_adj


if __name__ == "__main__":
    sim_or_fit = "fit"
    excel_source = "y"
    export_fit = "n"

    if "sim" in sim_or_fit and "n" in excel_source:
        spec_name = ["r1", "r2", "p1", "c1"]
        spec_type = ["r", "r", "p", "c"]
        react_vol_init = 4E-3
        stoich = [1, 1, 1, None]  # insert stoichiometry of reactant, r
        mol0 = [10, 0, 0, 0]
        mol_end = [0, None, 10, 0]
        add_sol_conc = [None, 1E7, None, 0.709]
        add_cont_rate = [None, 10E-6, None, None]
        t_cont = [None, 1, None, None]
        add_one_shot = [None, None, None, 100E-6]
        t_one_shot = [None, None, None, 1]
        k_lim = [3.5]
        ord_lim = [1, 0.0000001, 0, 1]
        pois_lim = [0, 0, 0, 0]
        t_param = (0, 100, 0.1)
        fit_asp = ["y", "n", "y", "y"]
        pic_save = 'cake.svg'

        x_exp_df, y_fit_df, fit_rate_df = cake_fitting_multi.sim_cake(t_param, spec_type, react_vol_init, spec_name=spec_name,
                                                                  stoich=stoich, mol0=mol0, mol_end=mol_end, add_sol_conc=add_sol_conc,
                                                                  add_cont_rate=add_cont_rate, t_cont=t_cont, add_one_shot=add_one_shot,
                                                                  t_one_shot=t_one_shot, k_lim=k_lim, ord_lim=ord_lim, pois_lim=pois_lim)

        plot_output = cake_fitting_multi.plot_sim_results(x_exp_df, y_fit_df, fit_asp,
                                                      f_format='png', save_disk=True, save_to=pic_save)

    elif "sim" in sim_or_fit and "y" in excel_source:
        df = pd.read_excel(r'C:\Users\Peter\Documents\Postdoctorate\Work\CAKE\Programmes\Test_Spectra.xlsx',
                       sheet_name='Sim_testing')
        df.replace('""', None, inplace=True)

        total = np.empty([len(df), 12], object)
        for i in range(0, len(df)):
            [number, react_type] = df.iloc[i, :2]
            [spec_name, spec_type, react_vol_init, stoich, mol0, mol_end, add_sol_conc] = df.iloc[i, 2:9]
            [add_cont_rate, t_cont, add_one_shot, t_one_shot, k_lim, ord_lim, pois_lim, t_param] = df.iloc[i, 9:17]
            [fit_asp, pic_save] = df.iloc[i, 17:]
            print(number, react_type)
            spec_name, spec_type, stoich, mol0, mol_end, add_sol_conc, add_cont_rate, t_cont, add_one_shot,\
            t_one_shot, k_lim, ord_lim, pois_lim, fit_asp, t_param = map(input_sort, [spec_name, spec_type, stoich,
                                        mol0, mol_end, add_sol_conc, add_cont_rate, t_cont, add_one_shot,
                                        t_one_shot, k_lim, ord_lim, pois_lim, fit_asp, t_param])
            t_param = tuple(t_param)
            x_exp_df, y_fit_conc_df, y_fit_rate_df = cake_fitting_multi.sim_cake(t_param, spec_type, react_vol_init,
                            spec_name=spec_name, stoich=stoich, mol0=mol0, mol_end=mol_end, add_sol_conc=add_sol_conc,
                            add_cont_rate=add_cont_rate, t_cont=t_cont, add_one_shot=add_one_shot,
                            t_one_shot=t_one_shot, k_lim=k_lim, ord_lim=ord_lim, pois_lim=pois_lim)

            plot_output = cake_fitting_multi.plot_sim_results(x_exp_df, y_fit_conc_df, fit_asp,
                                                          f_format='png', save_disk=True, save_to=pic_save)

    elif "fit" in sim_or_fit and "y" in excel_source:
        # create dataframe
        df = pd.read_excel(r'C:\Users\Peter\Documents\Postdoctorate\Work\CAKE\Programmes\Test_Spectra.xlsx', sheet_name='Testing')
        df.replace('""', None, inplace=True)

        total = np.empty([len(df), 12], object)
        for i in range(0, len(df)):
            [number, react_type] = df.iloc[i, :2]
            [spec_name, spec_type, react_vol_init, stoich, mol0, mol_end, add_sol_conc] = df.iloc[i, 2:9]
            [add_cont_rate, t_cont, add_one_shot, t_one_shot, add_col, sub_cont_rate, sub_aliq] = df.iloc[i, 9:16]
            [t_aliq, sub_col, t_col, col, k_lim, ord_lim, pois_lim, fit_asp] = df.iloc[i, 16:24]
            [TIC_col, scale_avg_num, win, inc, file_name, sheet_name, pic_save] = df.iloc[i, 24:]
            print(number, react_type)

            #spec_type, fit_asp = map(make_char_tup, [spec_type, fit_asp])
            spec_name, spec_type, stoich, mol0, mol_end, add_sol_conc, add_cont_rate, t_cont, add_one_shot,\
            t_one_shot, add_col, sub_cont_rate, sub_aliq, t_aliq, sub_col, col, \
            k_lim, ord_lim, pois_lim, fit_asp = map(input_sort, [spec_name, spec_type, stoich,
                    mol0, mol_end, add_sol_conc, add_cont_rate, t_cont, add_one_shot, t_one_shot, add_col,
                    sub_cont_rate, sub_aliq, t_aliq, sub_col, col, k_lim, ord_lim, pois_lim, fit_asp])
            sheet_name = str(sheet_name)

            # print(spec_type, react_vol_init, stoich, mol0, mol_end, add_sol_conc, add_cont_rate, t_cont, add_one_shot, t_one_shot, add_col, t_col, col, k_lim, ord_lim, pois_lim, fit_asp, TIC_col, scale_avg_num, win, inc, file_name, sheet_name, pic_save)
            data = cake_fitting_multi.read_data(file_name, sheet_name, t_col, col, add_col, sub_col)
            starttime = timeit.default_timer()
            output = cake_fitting_multi.fit_cake(data, spec_type, react_vol_init, spec_name=spec_name, stoich=stoich,
                        mol0=mol0, mol_end=mol_end, add_sol_conc=add_sol_conc, add_cont_rate=add_cont_rate,
                        t_cont=t_cont, add_one_shot=add_one_shot,t_one_shot=t_one_shot, add_col=add_col,
                        sub_cont_rate=sub_cont_rate, sub_aliq=sub_aliq, t_aliq=t_aliq, sub_col=sub_col, t_col=t_col,
                        col=col, k_lim=k_lim, ord_lim=ord_lim, pois_lim=pois_lim, fit_asp=fit_asp,
                        scale_avg_num=scale_avg_num, win=win, inc=inc)
            x_exp_df, y_exp_df, y_fit_conc_df, y_fit_rate_df, \
            k_val_est, k_fit, k_fit_err, ord_fit, ord_fit_err, pois_fit, pois_fit_err, ss_res, r_squared, col = output
            time_taken = timeit.default_timer() - starttime
            print(k_fit, k_fit_err, ord_fit, ord_fit_err, pois_fit, pois_fit_err)

            plot_output = cake_fitting_multi.plot_fit_results(x_exp_df, y_exp_df, y_fit_conc_df, col,
                                                      f_format='png', save_disk=True, save_to=pic_save)

            if "y" in export_fit:
                all_data = np.concatenate((x_exp_df, y_exp_df, y_fit_conc_df, y_fit_rate_df), axis=1)
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