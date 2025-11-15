"""CAKE Rate vs. Temperature Plotting"""

import base64
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cake.plot.plot_func import units_adjust, calc_x_lim, calc_y_lim, plot_process
from cake import rate_eqs


# Plots rates vs. temp
def plot_rate_vs_temp(t_df, fit_conc_df, fit_rate_df, temp_df, stoich, stoich_loc, k, rate_eq_type='standard',
                      exp_conc_df=None, show_asp='all', rate_constant_unit='rate_constant_unit',
                      f_format='svg', save_to='', return_fig=False, return_img=False, transparent=False):
    """
    Function to plot CAKE rate versus temperature.

    Parameters
    ----------
    All data must use consistent units throughout, i.e., all parameters must be inputted with identical units.

    DATA
    t_df : pandas.DataFrame, required
        Time data in time_unit.
    fit_conc_df : pandas.DataFrame, required
        Fitted concentration data in moles_unit volume_unit^-1.
        Default is None (fitted data not plotted).
    fit_rate_df : pandas.DataFrame, required
        Fitted rate data in moles_unit volume_unit^-1 time_unit^-1.
    temp_df : pandas.DataFrame, required
        Temperature data in K.
        Default is None (temperature data not plotted).
    stoich : list of int, or None, required
        Stoichiometry of species, use 'None' for catalyst.
    stoich_loc : list of tuple, required
        Indexes of rxns associated with each species (in fit_conc_df order).
        Multiple indices for different rxns must be input as a tuple for each species, e.g., (index_1, index_2, ...).
        Multiple species must be input as a list, e.g., [species_1_indices, reaction_2_indices, ...].
    k : float, or list of float, or list of list of float, required
        Estimated constant(s) (in rxns ord) in appropriate units.
        Multiple constants for a rate equation must be input as a list, e.g. [constant_1, constant_2, ...].
        Multiple reactions must be input as a list, e.g. [reaction_1_constants, reaction_2_constants, ...].
    rate_eq_type : str, optional
        Rate equation type used for fitting. Available options are 'standard', 'Arrhenius', 'Eyring',
        'Michaelis-Menten', or 'custom'. A 'custom' rate equation can be formatted in the rate_eqs.py file.
        Default is 'standard'.
    exp_conc_df : pandas.DataFrame, optional
        Experimental concentration data in moles_unit volume_unit^-1. Only required if show_asp is 'exp'.
        Default is None.
    show_asp : str, or list of str, optional
        Species to fit. Options are 'all', 'exp' (experimentally observed species only), a list of name(s) of species to
        fit as per exp_conc_df/fit_conc_df, or 'y' or 'n' to fit or not fit species respectively (in fit_conc_df order).
        Default is 'all' (all species shown).

    UNITS
    rate_constant_unit : str, optional
        Rate constant units. Optional but must be consistent between all parameters.
        Default is 'rate_constant_unit'.

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

    rate_constant_unit = units_adjust(rate_constant_unit)[0]
    # Convert data type and get header and species names
    t, fit_conc, fit_rate = map(pd.DataFrame.to_numpy, [t_df, fit_conc_df, fit_rate_df])
    if exp_conc_df is not None:
        exp_conc_species = [i.split(' exp. conc.', 1)[0] for i in list(exp_conc_df.columns)]
    fit_conc_species = [i.split(' fit conc.', 1)[0] for i in list(fit_conc_df.columns)]
    fit_rate_headers_adj = [species + ' partial fit rate' + rate_constant_unit for species in fit_conc_species]
    temp = pd.DataFrame.to_numpy(temp_df).reshape(-1)
    temp_headers = list(temp_df.columns)[0]

    rate_eq = rate_eqs.rate_eq_map.get(rate_eq_type.lower())

    # Set aspects to show
    if show_asp == 'all' and fit_conc_df is not None:
        show_asp = ['y'] * len(fit_conc_species)
    elif show_asp == 'exp' and fit_conc_df is not None:
        show_asp = ['y' if species in exp_conc_species else 'n' for species in fit_conc_species]
    elif isinstance(show_asp, str): show_asp = [show_asp]
    fit_cols = [i for i in range(len(show_asp)) if 'y' in show_asp[i]]

    # Calculate rate vs. temperature
    stoich_fit_rates_adj = np.empty((len(t), fit_conc.shape[1]))
    for p in range(len(t)):
        rates = rate_eqs.rate_calc(k, fit_conc[p], [[]] * fit_conc.shape[1], [[]] * fit_conc.shape[1], temp[p], rate_eq)
        stoich_fit_rates_adj[p, :] = [sum(([rates[m[i]] * stoich[m[i], j]
                                            for i in range(len(m))])) for j, m in enumerate(stoich_loc)]

    # Set edge limits and colours
    edge_adj = 0.02
    std_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    del std_colours[3]
    std_colours = std_colours * 100

    # Plot rate vs. temperature for each rxn
    for i in stoich:
        grid_shape = (int(round(np.sqrt(len(fit_cols)))), int(math.ceil(np.sqrt(len(fit_cols)))))
        fig = plt.figure(figsize=(grid_shape[1] * 6, grid_shape[0] * 5))
        # Plot rate vs. temperature
        for count, fit_col in enumerate(fit_cols):
            ax = plt.subplot(grid_shape[0], grid_shape[1], count + 1)
            ax.set_xlabel(temp_headers)
            ax.set_ylabel(fit_rate_headers_adj[fit_col])
            ax.plot(temp, stoich_fit_rates_adj[:, fit_col], color=std_colours[count])
            ax.set_xlim(calc_x_lim(temp, edge_adj))
            try:
                ax.set_ylim(calc_y_lim(stoich_fit_rates_adj[:, fit_col], stoich_fit_rates_adj[:, fit_col], edge_adj))
            except:
                pass

        return plot_process(f_format, save_to, transparent, fig, return_fig, return_img)
