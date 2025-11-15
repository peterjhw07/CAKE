"""CAKE Concentration vs. Rate Plotting"""

import base64
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cake.plot.plot_func import units_adjust, calc_x_lim, calc_y_lim, plot_process
import cake.rate_eqs as rate_eqs


# Plots rate vs. concentration
def plot_rate_vs_conc(t_df, fit_conc_df, fit_rate_df, k, ord, ord_loc, rate_eq_type='standard', exp_conc_df=None,
                      temp0=293.15, temp_df=None, show_asp='all',
                      rate_unit='moles_unit volume_unit$^{-1}$ time_unit$^{-1}$', f_format='svg',
                      save_to='', return_fig=False, return_img=False, transparent=False):
    """
    Function to plot CAKE rate vs. concentration data.

    Parameters
    ----------
    All data must use consistent units throughout, i.e., all parameters must be inputted with identical units.

    DATA
    t_df : pandas.DataFrame, required
        Time data in time_unit.
    fit_conc_df : pandas.DataFrame, required
        Fitted concentration data in moles_unit volume_unit^-1.
    fit_rate_df : pandas.DataFrame, required
        Fitted rate data in moles_unit volume_unit^-1 time_unit^-1.
    k : float, or list of float, or list of list of float, required
        Estimated constant(s) (in rxns order) in appropriate units.
        Multiple constants for a rate equation must be input as a list, e.g. [constant_1, constant_2, ...].
        Multiple reactions must be input as a list, e.g. [reaction_1_constants, reaction_2_constants, ...].
    ord : list of float, required
        Species reaction order (in spec_name/type order).
    ord_loc : list of tuple of int, required
        Indices of stoichiometry of each species (in fit_conc_df order).
        Multiple indexes for a rate equation must be input as a tuple, e.g., (index_1, index_2, ...).
        Multiple reactions must be input as a list, e.g., [reaction_1_indices, reaction_2_indices, ...].
    rate_eq_type : str, optional
        Rate equation type used for fitting. Available options are 'standard', 'Arrhenius', 'Eyring',
        'Michaelis-Menten', or 'custom'. A 'custom' rate equation can be formatted in the rate_eqs.py file.
        Default is 'standard'.
    exp_conc_df : pandas.DataFrame, optional
        Experimental concentration data in moles_unit volume_unit^-1. Only required if show_asp is 'exp'.
        Default is None.
    temp0 : float, optional
        Initial temperature in K.
        Default is 293.15 K (room temperature).
    temp_df : pandas.DataFrame, optional
        Temperature data in K.
        Default is None (temperature data not plotted).
    show_asp : str, or list of str, optional
        Species to fit. Options are 'all', 'exp' (experimentally observed species only), a list of name(s) of species to
        fit as per exp_conc_df/fit_conc_df, or 'y' or 'n' to fit or not fit species respectively (in fit_conc_df order).
        Default is 'all' (all species shown).

    UNITS
    rate_unit : str, optional
        Rate units. Optional but must be consistent between all parameters.
        Default is 'moles_unit volume_unit$^{-1}$ time_unit$^{-1}$'.

    FIGURE FORMAT
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

    rate_unit = units_adjust(rate_unit)[0]
    # Convert data type and get header and species names
    t, fit_conc, fit_rate = map(pd.DataFrame.to_numpy, [t_df, fit_conc_df, fit_rate_df])
    if isinstance(ord, list) and not isinstance(ord[0], list):
        ord = [ord]
    if exp_conc_df is not None:
        exp_conc_species = [i.split(' exp. conc.', 1)[0] for i in list(exp_conc_df.columns)]
    if fit_conc_df is not None:
        fit_conc_headers = list(fit_conc_df.columns)
        fit_conc_species = [i.split(' fit conc.', 1)[0] for i in list(fit_conc_df.columns)]
        fit_conc = pd.DataFrame.to_numpy(fit_conc_df)
    if temp_df is not None:
        temp = pd.DataFrame.to_numpy(temp_df).reshape(-1)
        temp_headers = list(temp_df.columns)[0]
    else:
        temp = np.empty(len(t))
        temp[0] = float(temp0)

    rate_eq = rate_eqs.rate_eq_map.get(rate_eq_type.lower())

    # Set aspects to show
    if show_asp == 'all' and fit_conc_df is not None:
        show_asp = ['y'] * len(fit_conc_headers)
    elif show_asp == 'exp' and fit_conc_df is not None:
        show_asp = ['y' if species in exp_conc_species else 'n' for species in fit_conc_species]
    elif isinstance(show_asp, str): show_asp = [show_asp]
    fit_cols = [i for i in range(len(show_asp)) if 'y' in show_asp[i]]

    # Calculate rate vs. concentration and temperature
    ord_fit_rate_adj = np.empty((len(t), fit_conc.shape[1]))
    for i in range(fit_conc.shape[1]):
        ord_loc_adj = [tuple(x for x in rxn if x == i) for rxn in ord_loc]
        for j in range(len(t)):
            ord_fit_rate_adj[j, i] = rate_eqs.rate_calc(k, fit_conc[j], ord, ord_loc_adj, temp[j], rate_eq)[0]
    if not np.all(temp == temp[0]):
        temp_fit_rate_adj = np.empty((len(t), 1))
        for j in range(len(t)):
            temp_fit_rate_adj[j, 0] = rate_eqs.rate_calc(k, fit_conc[j], ord, [[]], temp[j], rate_eq)[0]
    else:
        temp_fit_rate_adj = []

    # Set edge limits, colours, and axis labels
    edge_adj = 0.02
    std_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    del std_colours[3]
    std_colours = std_colours * 100
    y_label_text = 'Rate' + rate_unit

    # Plot rate vs. concentration for each rxn
    for i in ord:
        if temp_df is None or np.all(temp == temp[0]):
            grid_shape = (int(round(np.sqrt(len(fit_cols)))), int(math.ceil(np.sqrt(len(fit_cols)))))
        else:
            grid_shape = (int(round(np.sqrt(len(fit_cols) + 1))), int(math.ceil(np.sqrt(len(fit_cols) + 1))))
        fig = plt.figure(figsize=(grid_shape[1] * 6, grid_shape[0] * 5))
        # plt.subplots_adjust(hspace=0.4, wspace=0.4)
        # Plot rate vs. concentration
        for count, fit_col in enumerate(fit_cols):
            ax = plt.subplot(grid_shape[0], grid_shape[1], count + 1)
            ax.plot(fit_conc[:, fit_col], ord_fit_rate_adj[:, fit_col], color=std_colours[count])
            ax.set_xlim(calc_x_lim(fit_conc[:, fit_col], edge_adj))
            try:
                ax.set_ylim(calc_y_lim(ord_fit_rate_adj[:, fit_col], ord_fit_rate_adj[:, fit_col], edge_adj))
            except:
                pass
            ax.set_xlabel(fit_conc_headers[fit_col])
            ax.set_ylabel(y_label_text)

        if not np.all(temp == temp[0]):
            ax = plt.subplot(grid_shape[0], grid_shape[1], count + 2)
            ax.set_xlabel(temp_headers)
            ax.set_ylabel(y_label_text)
            ax.plot(temp, temp_fit_rate_adj, color='red')
            ax.set_xlim([float(min(temp) - (edge_adj * max(temp))), float(max(temp) * (1 + edge_adj))])

        return plot_process(f_format, save_to, transparent, fig, return_fig, return_img)
