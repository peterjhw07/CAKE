"""CAKE Concentration vs. Time Plotting"""

import base64
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cake.plot.plot_func import units_adjust, calc_x_lim, calc_y_lim, plot_process


# Plots concentration vs. time
def plot_conc_vs_time(t_df, exp_conc_df=None, fit_conc_df=None, temp_df=None, show_asp='all',
                      time_unit='time_unit', conc_unit='moles_unit volume_unit$^{-1}$',
                      conc_axis_label='Concentration', method='lone',
                      f_format='svg', save_to='', return_fig=False, return_img=True, transparent=False):
    """
    Function to plot CAKE concentration vs. time data.

    Parameters
    ----------
    All data must use consistent units throughout, i.e., all parameters must be inputted with identical units.

    DATA
    t_df : pandas.DataFrame
        Time data in time_unit.
    exp_conc_df : pandas.DataFrame, optional
        Experimental concentration data in moles_unit volume_unit^-1.
        Default is None (experimental data not plotted).
    fit_conc_df : pandas.DataFrame, optional
        Fitted concentration data in moles_unit volume_unit^-1.
        Default is None (fitted data not plotted).
    temp_df : pandas.DataFrame, optional
        Temperature data in K.
        Default is None (temperature data not plotted).
    show_asp : str, or list of str, optional
        Species to fit. Options are 'all', 'exp' (experimentally observed species only), a list of name(s) of species to
        fit as per exp_conc_df/fit_conc_df, or 'y' or 'n' to fit or not fit species respectively (in fit_conc_df order).
        Default is 'all' (all species shown).

    UNITS
    time_unit : str, optional
        Time units. Optional but must be consistent between all parameters.
        Default is 'time_unit'.
    conc_unit : str, optional
        Concentration units. Optional but must be consistent between all parameters.
        Default is 'moles_unit volume_unit$^{-1}$'
    conc_axis_label : str, optional
        Concentration axis label.
        Default is 'Concentration'.

    FIGURE FORMAT
    method : str, optional
        Style of plot. Options are 'lone', 'sep', and 'comp'.
        'lone' plots all selected species on single plot; 'comp' plots the selected species on a single plot
        and all species on another plot; and 'sep' plots all the selected species on separate plots.
        Default is 'lone' (all species plotted on single axis).
    f_format : str, optional
        Image file format. Accepted file formats are 'eps', 'jpg', 'pdf', 'png', and 'svg'.
        Default is 'svg'.
    save_to : str, optional
        Where to save image to.
        Default is '' (does not save).
    return_fig : bool, optional
        Whether to return figure or not, e.g. for subsequent editing.
        Default is False.
    return_img : bool, optional
        Whether to return image or not.
        Default is False.
    transparent : bool, optional
        Set image transparency.
        Default is False.

    Returns
    -------
    if return_fig:
        fig : figure object
    else:
        img : image object
        mimetype : file type
    """

    time_unit, conc_unit = units_adjust([time_unit, conc_unit])
    # Convert data type and get header and species names
    t = pd.DataFrame.to_numpy(t_df)
    if exp_conc_df is not None:
        exp_conc_headers = [i.split(' conc.', 1)[0] for i in list(exp_conc_df.columns)]
        exp_conc_species = [i.split(' exp. conc.', 1)[0] for i in list(exp_conc_df.columns)]
        exp_conc = pd.DataFrame.to_numpy(exp_conc_df)
    else:
        exp_conc_headers = []
        exp_conc = pd.DataFrame.to_numpy(fit_conc_df)
    if fit_conc_df is not None:
        fit_conc_headers = [i.split(' conc.', 1)[0] for i in list(fit_conc_df.columns)]
        fit_conc_species = [i.split(' fit conc.', 1)[0] for i in list(fit_conc_df.columns)]
        fit_conc = pd.DataFrame.to_numpy(fit_conc_df)
    else:
        fit_conc_headers = []
        fit_conc = exp_conc
    if temp_df is not None:
        temp = pd.DataFrame.to_numpy(temp_df)

    # Set aspects to show
    if show_asp == 'all' and fit_conc_df is not None:
        show_asp = ['y'] * len(fit_conc_headers)
    elif show_asp == 'exp' and exp_conc_df is not None:
        show_asp = ['y' if species in exp_conc_species else 'n' for species in fit_conc_species]
    elif isinstance(show_asp, str): show_asp = [show_asp]
    if show_asp[0] != 'y' and show_asp[0] != 'n': show_asp = ['y' if i in show_asp else 'n' for i in fit_conc_species]
    fit_cols = [i for i in range(len(show_asp)) if 'y' in show_asp[i]]
    non_fit_cols = [i for i in range(len(show_asp)) if 'n' in show_asp[i]]

    if exp_conc_df is None: exp_conc = exp_conc[:, fit_cols]

    if 'comp' in method and (len(non_fit_cols) == 0 or fit_conc_df is None): method = 'lone'

    # Set edge limits, colours, and axis labels
    edge_adj = 0.02
    std_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    del std_colours[3]
    std_colours = std_colours * 100
    x_label_text = 'Time' + time_unit
    y_label_text = conc_axis_label + conc_unit

    cur_exp = 0
    cur_clr = 0

    # Plot concentration vs. time
    if 'lone' in method:  # plots a single figure containing all show_asp
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))
        ax1.set_xlabel(x_label_text)
        ax1.set_ylabel(y_label_text)
        # Plot experimental data
        for exp_col in range(len(exp_conc_headers)):
            if len(t) <= 50:
                ax1.scatter(t, exp_conc[:, exp_col], label=exp_conc_headers[exp_col])
            else:
                ax1.plot(t, exp_conc[:, exp_col], label=exp_conc_headers[exp_col])
        # Plot fitted data
        for fit_col in fit_cols:
            ax1.plot(t, fit_conc[:, fit_col], label=fit_conc_headers[fit_col])
        # Plot temperature data as second axis
        if temp_df is not None and not np.all(temp == temp[0]):
            ax2 = ax1.twinx()
            ax2.plot(t, temp, label='Temperature', color='red')
            ax2.set_ylabel('Temperature / K', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.set_xlim(calc_x_lim(t, edge_adj))
            ax2.set_ylim(calc_y_lim(temp, temp, edge_adj))
        if len(fit_cols) == 0: fit_cols = range(len(exp_conc_headers))
        ax1.set_xlim(calc_x_lim(t, edge_adj))
        try:
            ax1.set_ylim(calc_y_lim(exp_conc, fit_conc[:, fit_cols], edge_adj))
        except:
            pass
        ax1.legend(prop={'size': 10}, frameon=False)

    elif 'comp' in method:  # plots two figures, containing show_asp and all fits respectively
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        # Plot experimental data
        for exp_col in range(len(exp_conc_headers)):
            if len(t) <= 50:
                ax1.scatter(t, exp_conc[:, exp_col], color=std_colours[cur_clr],
                            label=exp_conc_headers[exp_col])
                ax2.scatter(t, exp_conc[:, exp_col], color=std_colours[cur_clr],
                            label=exp_conc_headers[exp_col])
                cur_clr += 1
            else:
                ax1.plot(t, exp_conc[:, exp_col], color=std_colours[cur_clr],
                         label=exp_conc_headers[exp_col])
                ax2.plot(t, exp_conc[:, exp_col], color=std_colours[cur_clr],
                         label=exp_conc_headers[exp_col])
                cur_clr += 1
        # Plot fitted data
        for fit_col in fit_cols:
            ax1.plot(t, fit_conc[:, fit_col], color=std_colours[cur_clr],
                     label=fit_conc_headers[fit_col])
            ax2.plot(t, fit_conc[:, fit_col], color=std_colours[cur_clr],
                     label=fit_conc_headers[fit_col])
            cur_clr += 1
        for fit_col in non_fit_cols:
            ax2.plot(t, fit_conc[:, fit_col], color=std_colours[cur_clr],
                     label=fit_conc_headers[fit_col])
            cur_clr += 1
        # Plot temperature data as second axis
        if temp_df is not None and not np.all(temp == temp[0]):
            ax3 = ax2.twinx()
            ax3.plot(t, temp, label='Temperature', color='red')
            ax3.set_ylabel('Temperature / K', color='red')
            ax3.tick_params(axis='y', labelcolor='red')
            ax3.set_xlim(calc_x_lim(t, edge_adj))
            ax3.set_ylim(calc_y_lim(temp, temp, edge_adj))

        ax1.set_xlim(calc_x_lim(t, edge_adj))
        try:
            ax1.set_ylim(calc_y_lim(exp_conc, fit_conc[:, fit_cols], edge_adj))
        except:
            pass
        ax2.set_xlim(calc_x_lim(t, edge_adj))
        try:
            ax2.set_ylim(calc_y_lim(exp_conc, fit_conc, edge_adj))
        except:
            pass

        ax1.set_xlabel(x_label_text)
        ax1.set_ylabel(y_label_text)
        ax2.set_xlabel(x_label_text)
        ax2.set_ylabel(y_label_text)
        ax1.legend(prop={'size': 10}, frameon=False)
        ax2.legend(prop={'size': 10}, frameon=False)

    elif 'sep' in method:  # plots all show_asp separately
        if temp_df is None or np.all(temp == temp[0]):
            grid_shape = (int(round(np.sqrt(len(fit_cols)))), int(math.ceil(np.sqrt(len(fit_cols)))))
        else:
            grid_shape = (int(round(np.sqrt(len(fit_cols) + 1))), int(math.ceil(np.sqrt(len(fit_cols) + 1))))
        fig = plt.figure(figsize=(grid_shape[1] * 6, grid_shape[0] * 5))
        for count, fit_col in enumerate(fit_cols):
            ax = plt.subplot(grid_shape[0], grid_shape[1], count + 1)
            # Plot experimental data
            if exp_conc_df is not None and fit_conc_headers[fit_col].split(' fit', 1)[0] in exp_conc_species:
                if len(t) <= 50:
                    ax.scatter(t, exp_conc[:, cur_exp], color=std_colours[cur_clr],
                               label=exp_conc_headers[cur_exp])
                else:
                    ax.plot(t, exp_conc[:, cur_exp], color=std_colours[cur_clr],
                            label=exp_conc_headers[cur_exp])
                try:
                    ax.set_ylim(calc_y_lim(exp_conc[:, cur_exp], fit_conc[:, fit_col], edge_adj))
                except:
                    pass
                cur_exp += 1
                cur_clr += 1
            else:
                try:
                    ax.set_ylim(calc_y_lim(fit_conc[:, fit_col], fit_conc[:, fit_col], edge_adj))
                except:
                    pass
            # Plot fitted data
            if fit_conc_df is not None:
                ax.plot(t, fit_conc[:, fit_col], color=std_colours[cur_clr], label=fit_conc_headers[fit_col])
            cur_clr += 1

            ax.set_xlim(calc_x_lim(t, edge_adj))
            ax.set_xlabel(x_label_text)
            ax.set_ylabel(y_label_text)
            plt.legend(prop={'size': 10}, frameon=False)
        # Plot temperature data
        if temp_df is not None and not np.all(temp == temp[0]):
            ax = plt.subplot(grid_shape[0], grid_shape[1], count + 2)
            ax.plot(t, temp, color='red', label='Temperature')

            ax.set_xlim(calc_x_lim(t, edge_adj))
            ax.set_ylim(calc_y_lim(temp, temp, edge_adj))
            ax.set_xlabel(x_label_text)
            ax.set_ylabel('Temperature / K', color='red')
            ax.tick_params(axis='y', labelcolor='red')
            plt.legend(prop={'size': 10}, frameon=False)
    else:
        print('Invalid method inputted. Please enter appropriate method or remove method argument.')
        return

    return plot_process(f_format, save_to, transparent, fig, return_fig, return_img)
