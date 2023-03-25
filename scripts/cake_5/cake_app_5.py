"""CAKE App Functions"""
# if __name__ == '__main__':
# import matplotlib
# matplotlib.use('Agg')
import math
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import pandas as pd
import io
import itertools
import base64
from scipy import optimize
import logging
import timeit  # to be removed
import warnings

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def write_sim_data(filename, df, param_dict, x_data_df, y_fit_conc_df, y_fit_rate_df):
    param_dict = {key: np.array([value]) for (key, value) in param_dict.items()}
    df_params = pd.DataFrame.from_dict(param_dict)
    df[list(x_data_df.columns)] = x_data_df
    df[list(y_fit_conc_df.columns)] = y_fit_conc_df
    df[list(y_fit_rate_df.columns)] = y_fit_rate_df

    writer = pd.ExcelWriter(filename, engine='openpyxl')
    df.to_excel(writer, sheet_name='FitData')
    df_params.to_excel(writer, sheet_name='InputParams')
    writer.save()


def write_sim_data_temp(df, param_dict, x_data_df, y_fit_conc_df, y_fit_rate_df):
    tmp_file = io.BytesIO()
    write_fit_data(tmp_file, df, param_dict, x_data_df, y_fit_conc_df, y_fit_rate_df)

    return tmp_file, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'


def write_fit_data(filename, df, param_dict, x_data_df, y_exp_df, y_fit_conc_df, y_fit_rate_df,
                   k_fit, k_fit_err, ord_fit, ord_fit_err, pois_fit, pois_fit_err, ss_res, r_squared):
    param_dict = {key: np.array([value]) for (key, value) in param_dict.items()}
    out_dict = {"Rate Constant": [k_fit],
                  "Reaction Orders": [ord_fit],
                  "Species Poisoning": [pois_fit],
                  "Rate Constant Error": [k_fit_err],
                  "Reaction Order Errors": [ord_fit_err],
                  "Species Poisoning Errors": [pois_fit_err],
                  "RSS": [ss_res],
                  "R2": [r_squared]}
    df_outputs = pd.DataFrame.from_dict(out_dict)
    df_params = pd.DataFrame.from_dict(param_dict)
    df[list(x_data_df.columns)] = x_data_df
    df[list(y_exp_df.columns)] = y_exp_df
    df[list(y_fit_conc_df.columns)] = y_fit_conc_df
    df[list(y_fit_rate_df.columns)] = y_fit_rate_df

    writer = pd.ExcelWriter(filename, engine='openpyxl')
    df.to_excel(writer, sheet_name='FitData')
    df_outputs.to_excel(writer, sheet_name='Outputs')
    df_params.to_excel(writer, sheet_name='InputParams')
    writer.save()


def write_fit_data_temp(df, param_dict, x_data_df, y_exp_df, y_fit_conc_df, y_fit_rate_df,
                   k_fit, k_fit_err, ord_fit, ord_fit_err, pois_fit, pois_fit_err, ss_res, r_squared):
    tmp_file = io.BytesIO()
    write_fit_data(tmp_file, df, param_dict, x_data_df, y_exp_df, y_fit_conc_df, y_fit_rate_df,
                   k_fit, k_fit_err, ord_fit, ord_fit_err, pois_fit, pois_fit_err, ss_res, r_squared)

    return tmp_file, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'


def make_param_dict(spec_type, react_vol_init, stoich=1, mol0=None, mol_end=None, add_sol_conc=None, add_cont_rate=None,
                    t_cont=None, add_one_shot=None, t_one_shot=None, add_col=None,
                    sub_cont_rate=None, sub_aliq=None, t_aliq=None, sub_col=None, t_col=0, col=1, k_lim=None,
                    ord_lim=None, pois_lim=None, fit_asp="y", TIC_col=None, scale_avg_num=0, win=1, inc=1):
    param_dict = {'Species types': spec_type,
                  'Initial reaction solution volume': react_vol_init,
                  'Stoichiometries': stoich,
                  'Initial moles': mol0,
                  'Final moles': mol_end,
                  'Addition solution concentrations': add_sol_conc,
                  'Continuous addition rates': add_cont_rate,
                  'Continuous addition start times': t_cont,
                  'One shot additions': add_one_shot,
                  'One shot addition start times': t_one_shot,
                  'Addition columns': add_col,
                  'Continuous subtraction rate': sub_cont_rate,
                  'Subtracted aliquot volumes': sub_aliq,
                  'Subtracted aliquot start times': t_aliq,
                  'Subtraction columns': sub_col,
                  'Time column': t_col,
                  'Species columns': col,
                  'Rate constant limits': k_lim,
                  'Reaction order limits': ord_lim,
                  'Poisoning limits': pois_lim,
                  'Species to fit': fit_asp,
                  'Total ion count column': TIC_col,
                  'Concentration calibration points': scale_avg_num,
                  'Smoothing window': win,
                  'Interpolation multiplier': inc
     }
    if len(k_lim) == 1:
        param_dict['Rate constant starting estimate'] = k_lim[0]
        param_dict['Rate constant minimum'] = k_lim[0] - (1E3 * k_lim[0])
        param_dict['Rate constant maximum'] = k_lim[0] + (1E3 * k_lim[0])
    else:
        param_dict['Rate constant starting estimate'], param_dict['Rate constant minimum'], \
        param_dict['Rate constant maximum'] = k_lim

    if len(ord_lim) == 1:
        param_dict['Reaction order starting estimates'] = ord_lim[0]
        param_dict['Reaction order minima'] = r_ord_lim[0] - 1E6
        param_dict['Reaction order maxima'] = r_ord_lim[0] + 1E6
    else:
        param_dict['Reaction order starting estimates'], param_dict['Reaction order minima'], \
        param_dict['Reaction order maxima'] = ord_lim

    if len(pois_lim) == 1:
        param_dict['Poisoning starting estimates'] = pois_lim[0]
        param_dict['Poisoning starting minima'] = pois_lim[0] - 1E6
        param_dict['Poisoning starting maxima'] = pois_lim[0] + 1E6
    else:
        param_dict['Poisoning starting estimates'], param_dict['Poisoning starting minima'], \
        param_dict['Poisoning starting maxima'] = pois_lim

    return param_dict


# print CAKE results (webapp only)
def pprint_cake(k_fit, k_fit_err, ord_fit, ord_fit_err, pois_fit, pois_fit_err, ss_res, r_squared):
    result = f"""|               | Rate Constant (k) | Reaction Orders |
    |---------------|-------------------|----------------|
    |  Opt. Values  | {k_fit: 17.6E} | {ord_fit[1]: 14.6f} |
    | Est. Error +- | {k_fit_err: 17.6E} | {ord_fit_err: 14.6f} |

    |               | Species Poisoning
    |---------------|----------------|
    |  Opt. Values  | {pois_fit: 14.6f} |
    | Est. Error +- | {pois_fit_err: 14.6f} |

    Residual Sum of Squares for Optimization: {ss_res: 8.6f}.

    R^2 Value of Fit: {r_squared: 8.6f}.
    """

    return result