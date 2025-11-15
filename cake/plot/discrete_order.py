"""CAKE Conc vs. Rate Plotting"""

import base64
import math
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate
from cake.plot.plot_func import units_adjust, calc_x_lim, calc_y_lim, plot_process


# Plots discrete fits in 2d
def plot_discrete_order_2d(t_df, exp_conc_df=None, fit_conc_df=None, temp_df=None, discrete_order_df=None,
                           show_asp='all', method='lone', metric=None, cutoff=None,
                           time_unit='time_unit', conc_unit='moles_unit volume_unit$^{-1}$',
                           f_format='svg', save_to='', return_fig=False, return_img=False, transparent=False):
    """
    Function to plot CAKE concentration data.

    Parameters
    ----------
    All data must use consistent units throughout, exp_col.e., all parameters must be inputted with identical units.

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
        Species to fit to. Options are 'all', 'exp' (experimentally observed species only), or a list of
        name(s) of species to fit to, or 'y' to fit to species or 'n' not to fit to species (in fit_conc_df ord).
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
    return_img : bool, optional
        Whether to return image or not.
        Default is False.
    return_fig : bool, optional
        Whether to return figure or not.
        Default is False.
    transparent : bool, optional
        Set image transparency.
        Default is False.

    Returns
    -------
    if return_fig:
        fig :
        fig.get_axes() :
    else:
        img :
        mimetype :
    """

    time_unit, conc_unit = units_adjust([time_unit, conc_unit])
    # Convert data type and get header and species names
    t, exp_conc, do_summary = map(pd.DataFrame.to_numpy, [t_df, exp_conc_df, discrete_order_df])
    if exp_conc_df is not None:
        exp_conc_headers = [i.split(' conc.', 1)[0] for i in list(exp_conc_df.columns)]
        exp_conc_species = [i.split(' exp. conc.', 1)[0] for i in list(exp_conc_df.columns)]
        exp_conc = pd.DataFrame.to_numpy(exp_conc_df)
    else:
        exp_conc_headers = []
        exp_conc = pd.DataFrame.to_numpy(fit_conc_df)
    fit_conc_headers = [i.split(' conc.', 1)[0] for i in list(fit_conc_df[0].columns)]
    fit_conc_species = [i.split(' fit conc.', 1)[0] for i in list(fit_conc_df[0].columns)]
    fit_conc = [pd.DataFrame.to_numpy(df) for df in fit_conc_df]
    if temp_df is not None:
        temp = pd.DataFrame.to_numpy(temp_df)

    # Cutoff data below a certain threshold
    if metric == 'best' and cutoff is not None:
        rows_cut = range(min(cutoff, do_summary.shape[0]))
    elif metric in discrete_order_df.columns and cutoff is not None:
        if 'R2' in metric:
            rows_cut = [row for row, x in enumerate(discrete_order_df[metric]) if x >= cutoff]
        else:
            rows_cut = [row for row, x in enumerate(discrete_order_df[metric]) if x <= cutoff]
    else:
        rows_cut = range(do_summary.shape[0])

    # Set aspects to show
    if (show_asp == 'all' or show_asp == 'exp') and fit_conc_df is not None:
        show_asp = ['y' if species in exp_conc_species else 'n' for species in fit_conc_species]
    elif isinstance(show_asp, str): show_asp = [show_asp]
    if show_asp[0] != 'y' and show_asp[0] != 'n': show_asp = ['y' if i in show_asp else 'n' for i in fit_conc_species]
    fit_cols = [i for i in range(len(show_asp)) if 'y' in show_asp[i]]

    if exp_conc_df is None: exp_conc = exp_conc[:, fit_cols]

    # Set edge limits, colours, and axis labels
    edge_adj = 0.02
    std_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    del std_colours[3]
    std_colours = std_colours * 100
    x_label_text = 'Time' + time_unit
    y_label_text = 'Concentration' + conc_unit

    # Plot concentration vs. time
    if 'lone' in method:  # plots a single figure containing all show_asp
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))
        ax1.set_xlabel(x_label_text)
        ax1.set_ylabel(y_label_text)
        # Plot experimental data
        for exp_col in range(len(exp_conc_headers)):
            if len(t) <= 50:
                ax1.scatter(t, exp_conc[:, exp_col], 'k', label=exp_conc_headers[exp_col])
            else:
                ax1.plot(t, exp_conc[:, exp_col], 'k', label=exp_conc_headers[exp_col])
        # Plot fitted data
        for row in rows_cut:
            for fit_col in fit_cols:
                ax1.plot(t, fit_conc[row][:, fit_col],
                         label=str(do_summary[row, 0]) + ' ' + fit_conc_headers[fit_col], alpha=0.2)
        # Plot temperature data as second axis
        if temp_df is not None and not np.all(temp == temp[0]):
            ax2 = ax1.twinx()
            ax2.plot(t, temp, label='Temperature', color='red')
            ax2.set_ylabel('Temperature / K', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.set_xlim(calc_x_lim(t, edge_adj))
            ax2.set_ylim(calc_y_lim(temp, temp, edge_adj))
        ax1.set_xlim(calc_x_lim(t, edge_adj))
        # ax1.legend(prop={'size': 10}, frameon=False)

    elif 'sep' in method:  # plots all show_asp separately
        if temp_df is None or np.all(temp == temp[0]):
            grid_shape = (int(round(np.sqrt(max(rows_cut) + 2))), int(math.ceil(np.sqrt(max(rows_cut) + 2))))
        else:
            grid_shape = (int(round(np.sqrt(max(rows_cut) + 1))), int(math.ceil(np.sqrt(max(rows_cut) + 1))))
        fig = plt.figure(figsize=(grid_shape[1] * 6, grid_shape[0] * 5))
        cur_clr = len(exp_conc_headers) + 1
        for row in rows_cut:
            ax = plt.subplot(grid_shape[0], grid_shape[1], row + 1)
            ax.set_xlabel(x_label_text)
            ax.set_ylabel(y_label_text)
            # Plot experimental data
            for exp_col in range(len(exp_conc_headers)):
                if len(t) <= 50:
                    ax.scatter(t, exp_conc[:, exp_col], color=std_colours[exp_col],
                               label=exp_conc_headers[exp_col])
                else:
                    ax.plot(t, exp_conc[:, exp_col], color=std_colours[exp_col],
                            label=exp_conc_headers[exp_col])
            # Plot fitted data
            for fit_col in fit_cols:
                ax.plot(t, fit_conc[row][:, fit_col], color=std_colours[cur_clr],
                        label=str(do_summary[row, 0]) + ' ' + fit_conc_headers[fit_col])
                cur_clr += 1
        # Plot temperature data
        if temp_df is not None and not np.all(temp == temp[0]):
            ax = plt.subplot(grid_shape[0], grid_shape[1], row + 2)
            ax.plot(t, temp, color='red', label='Temperature')

            ax.set_xlim(calc_x_lim(t, edge_adj))
            ax.set_ylim(calc_y_lim(temp, temp, edge_adj))
            ax.set_xlabel(x_label_text)
            ax.set_ylabel('Temperature / K', color='red')
            ax.tick_params(axis='y', labelcolor='red')
            plt.legend(prop={'size': 10}, frameon=False)

    return plot_process(f_format, save_to, transparent, fig, return_fig, return_img)


# Plots discrete ord fits in 3d (contour and surface)
def plot_discrete_order_3d(discrete_order_df, method='contour', metric=None, cutoff=None, levels=None,
                           f_format='svg', save_to='', return_fig=False, return_img=False, transparent=False):

    do_summary = pd.DataFrame.to_numpy(discrete_order_df)
    labels = ['Order 1', 'Order 2']

    # Cutoff data below a certain threshold
    if cutoff:
        if 'R2' in metric:
            rows_cut = [row for row, x in enumerate(discrete_order_df[metric]) if x >= cutoff]
        else:
            rows_cut = [row for row, x in enumerate(discrete_order_df[metric]) if x <= cutoff]
    else:
        rows_cut = range(do_summary.shape[0])
    do_summary_cut = do_summary[rows_cut]
    if 'R2' in metric:
        cmap = 'coolwarm'
    else:
        cmap = 'coolwarm_r'

    # Setup 3d coordinates
    cont_x_org = [do_summary_cut[i, 0][0] for i in range(len(do_summary_cut))]
    cont_y_org = [do_summary_cut[i, 0][1] for i in range(len(do_summary_cut))]
    cont_z_org = discrete_order_df[metric].iloc[:len(do_summary_cut)].values.astype(float)
    cont_x_add, cont_y_add = np.linspace(min(cont_x_org), max(cont_x_org), 1000), \
                             np.linspace(min(cont_y_org), max(cont_y_org), 1000)
    cont_x_plot, cont_y_plot = np.meshgrid(cont_x_add, cont_y_add)
    cont_z_plot = interpolate.griddata((cont_x_org, cont_y_org), cont_z_org, (cont_x_plot, cont_y_plot),
                                       method='linear')

    # Plot 3d plot
    if 'cont' in method.lower() and 'ex' not in method.lower():  # plots a continuous contour
        fig = plt.imshow(cont_z_plot, vmin=cont_z_org.min(), vmax=cont_z_org.max(), origin='lower', cmap=cmap,
                         extent=[min(cont_x_org), max(cont_x_org), min(cont_y_org), max(cont_y_org)], aspect='auto')
        fig = plt.imshow(cont_z_plot, origin='lower', cmap=cmap,
                          norm=LogNorm(), extent=[min(cont_x_org), max(cont_x_org), min(cont_y_org), max(cont_y_org)], aspect='auto')
        # plt.scatter(cont_x_org, cont_y_org, c=cont_z_org, cmap=cmap)
        plt.xlabel(labels[0]), plt.ylabel(labels[1])
        plt.colorbar(label=metric)

    elif 'cont' in method.lower() and 'ex' in method.lower():  # plots a discrete contour
        # std_colours = plt.rcParams['axes.prop_cycle'].by_key()['color'] * 100
        std_colours = [plt.get_cmap(cmap)(i) for i in np.linspace(0, 1, len(levels) - 1)]
        fig = plt.contourf(cont_x_plot, cont_y_plot, cont_z_plot, levels=levels, origin='lower', colors=std_colours,
                           extent=[min(cont_x_org), max(cont_x_org), min(cont_y_org), max(cont_y_org)])
        # plt.scatter(cont_x_org, cont_y_org, c=cont_z_org, cmap=cmap)
        plt.xlabel(labels[0]), plt.ylabel(labels[1])
        plt.colorbar(label=metric)

    elif '3d' in method.lower():  # plots a continuous surface
        fig = plt.axes(projection='3d')
        fig.plot_surface(cont_x_plot, cont_y_plot, -np.log(cont_z_plot), cmap=cmap)
        fig.set_xlabel(labels[0]), fig.set_ylabel(labels[1]), fig.set_zlabel(metric)

    return plot_process(f_format, save_to, transparent, fig, return_fig, return_img)
