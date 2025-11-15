"""CAKE Fitting Programme"""

# Imports
import copy
import numpy as np
import pandas as pd
import itertools
from scipy import optimize
import logging
from cake import rate_eqs
from cake.fitting import ode_solver
from cake.fitting.residuals import residuals
from cake.prep import get_prep, get_events, import_data, store_objs


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

np.seterr(divide='ignore', invalid='ignore')


# Fits CAKE experiments
def fit(df, spec_name=None, spec_type=None, stoich=None, rxns=None, t_col=0, col=1, fit_asp=None,
        mol0=None, mol_end=None, vol0=None, add_sol_conc=None, cont_add_rate=None, t_cont_add=None,
        disc_add_vol=None, t_disc_add=None, cont_sub_rate=None, disc_sub_vol=None, t_disc_sub=None,
        temp0=293.15, cont_temp_rate=None, t_cont_temp=None, temp_col=None,
        rate_eq_type='standard', rate_method='Radau', rtol=1E-6, atol=1E-9, k_lim=None, ord_lim=None, pois_lim=None,
        scale_avg_num=0, win=1, inc=1, tic_col=None,
        time_unit='time_unit', conc_unit='moles_unit volume_unit$^{-1}$'):
    """
    Function to fit CAKE data.

    Parameters
    ----------
    All data must use consistent units throughout, i.e., all parameters must be inputted with identical units.

    DATA
    df : numpy.array or pandas.DataFrame, required
        The reaction data, including time, concentration or signal intensity from monitored species,
        and temperature (if temperature data required).

    SPECIES INFORMATION
    spec_name : str, list of str, or None, optional
        Name of each species.
        Default is None (species are given default names).
    spec_type : str, or list of str, or None, optional
        Type of each species: 'r' for reactant, 'p' for product, 'c' for catalyst.
        Default is None (not used if rxns).
    stoich : list of int, or None, optional
        Stoichiometry of species, use 'None' for catalyst.
        Default is None (not used if rxns).
    rxns : list of tuple of tuple of str, or None, optional
        Reaction mechanisms for fitting.
        Takes the form [reaction_1, reaction_2, ...] where each reaction is of the form
        (('reactant_1', 'reactant_2', ...), ('product_1', 'product_2', ...)).
        Default is None (not required if using spec_type, stoich, etc.).

    DATA LOCATION
    t_col : int, or str, optional
        Index or name of time column.
        Default is 0 (first column).
    col : int, str, or list of int and str, optional
        Index(es) or name(s) of species column(s) (in spec_name/type order).
        Default is 1 (second column).
    fit_asp : str, or list of str, or None, optional
        Name(s) of species to fit to, or 'y' to fit to species, 'n' not to fit to species (in spec_name/type order).
        Default is None which fits all species with data.

    REACTION CONDITIONS
    mol0 : float, or list of float, required
        Initial moles of species (in spec_name/type order) in moles_unit.
    mol_end : float, list of float, or None, optional
        Final moles of species (in spec_name/type order) in moles_unit or None if data do not need scaling.
        Default is None (in spec_name/type order) in moles_unit.
    vol0 : float, required
        Initial reaction solution volume in volume_unit.

    SOLUTION ADDITIONS AND SUBTRACTIONS
    add_sol_conc : list of float, or None, optional
        Concentration of solution being added for each species (in spec_name/type order) in moles_unit volume_unit^-1.
        Default is None (no addition solution for any species).
    cont_add_rate : list of float, or list of tuple of float, or None, optional
        Continuous addition rates of species (in spec_name/type order) in volume_unit time_unit^-1.
        Multiple conditions for each species are input as tuples.
        Is paired with t_cont_add.
        Default is None (no continuous addition for any species).
    t_cont_add : list of tuple of float, or None, optional
        Times at which continuous addition began for each species (in spec_name/type order) in time_unit^-1.
        Multiple conditions for each species are input as tuples.
        Is paired with cont_add_rate.
        Default is None (no continuous addition for any species).
    disc_add_vol : list of tuple of float, or None, optional
        Discrete addition volumes for each species (in spec_name/type order) in volume_unit.
        Multiple conditions for each species are input as tuples.
        Is paired with t_disc_add.
        Default is None (no discrete additions for any species).
    t_disc_add : list of tuple of float, or None, optional
        Times of discrete additions for each species (in spec_name/type order) in time_unit^-1.
        Multiple conditions for each species are input as tuples.
        Is paired with disc_add_vol.
        Default is None (no discrete additions for any species).
    cont_sub_rate : float, or None, optional
        Continuous subtraction rate in volume_unit time_unit^-1.
        Default is None (no continuous subtraction).
    disc_sub_vol : float, or list of float, or None, optional
        Discrete subtraction volumes in volume_unit.
        Is paired with t_disc_sub.
        Default is None (no discrete subtractions).
    t_disc_sub : float, or list of float, or None, optional
        Times of discrete subtractions in time_unit^-1.
        Is paired with disc_sub_vol.
        Default is None (no discrete subtractions).

    TEMPERATURE ALTERATION
    temp0 : float, optional
        Initial temperature in K.
        Default is 293.15 K (room temperature).
    cont_temp_rate : float, or None, optional
        Temperature alteration rates in K time_unit^-1.
        Is paired with t_cont_temp.
        Default is None (no temperature alteration).
    t_cont_temp : float, or None, optional
        Times at which temperature alterations occur in time_unit.
        Is paired with cont_temp_rate.
        Default is None (no temperature alteration).
    temp_col : int, or None, optional
        Index of temperature column (for use of monitored temperature).
        Default is None.

    FITTING PARAMETERS
    rate_eq_type : str, optional
        Rate equation type used for fitting. Available options are 'standard', 'Arrhenius', 'Eyring',
        'Michaelis-Menten', or 'custom'. A 'custom' rate equation can be formatted in the rate_eqs.py file.
        Default is 'standard'.
    rate_method : str, optional
        Rate method used for fitting. Available options are 'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', and 'LSODA'.
        For more information see scipy.integrate.solve_ivp documentation.
        Default is 'Radau'.
    rtol : float, optional
        Relative tolerance (number of correct digits) used for fitting.
        For more information see scipy.integrate.solve_ivp documentation.
        Default is '1E-6'.
    atol : float, optional
        Absolute tolerance (number of correct decimal places) used for fitting.
        For more information see scipy.integrate.solve_ivp documentation.
        Default is '1E-9'.
    k_lim : float, or list of float and tuple of float, or list of list of float and tuple of float, or None, optional
        Estimated constant(s) (in rxns ord) in appropriate units. Constants related to the molar gas constant
        must be in units of J K^-1 mol^-1.
        Each constant can be input as float for a fixed variable
        or as tuple for floating variable with bounds (estimate, factor difference) or (estimate, lower, upper).
        Multiple constants for a rate equation must be input as a list, e.g., [constant_1, constant_2, ...].
        Multiple reactions must be input as a list, e.g., [reaction_1_constants, reaction_2_constants, ...].
        Default is None which sets appropriate bounds for all reactions and constants.
    ord_lim : list of float and tuple, or None, optional
        Species reaction ord (in spec_name/type order).
        Specified as exact value for fixed unfitted variable or variable with bounds (estimate, lower, upper).
        Default is None which assumes (1, 0, 2) for 'r' and 'c' species and 0 for 'p' species (not used if rxns).
    pois_lim : list of float, str, and tuple of float and str, or None, optional
        Moles of species poisoned (in spec_name/type order) in moles_unit.
        Specified as float for fixed unfitted variable, 'max' as bounds (0, 0, max species concentration),
        or tuple as bounds (estimate, lower, upper).
        Default is None (no poisoning occurs for any species).

    DATA MANIPULATION
    scale_avg_num : int, optional
        Number of data points from which to calculate mol0 and mol_end.
        Default is 0 (no scaling).
    win : int, optional
        Smoothing window.
        Default is 1 (no smoothing).
    inc : int, or float, optional
        Increments between adjacent points for improved simulation (>1)
        or fraction of points removed, e.g. if unecessarily large amount of data (<1).
        Default is 1 (no addition or subtraction of points).
    tic_col : int, or None, optional
        Index of TIC column or None if no TIC.
        Default is None.

    UNITS
    time_unit : str, optional
        Time units. Optional but must be consistent between all parameters - only used in output and plotting.
        Default is 'time_unit'.
    conc_unit : str, optional
        Concentration units. Optional but must be consistent between all parameters - only used in output and plotting.
        Default is 'moles_unit volume_unit$^{-1}$'.

    Returns
    -------
    res : obj, with the following fields defined:
        DATA
        t_df : pandas.DataFrame
            Time data in time_unit.
        exp_conc_df : pandas.DataFrame
            Experimental concentration data in moles_unit volume_unit^-1.
        exp_rate_df : pandas.DataFrame
            Experimental rate data in moles_unit volume_unit^-1 time_unit^-1.
        fit_conc_df : pandas.DataFrame
            Fitted concentration data in moles_unit volume_unit^-1.
        fit_rate_df : pandas.DataFrame
            Fitted rate data in moles_unit volume_unit^-1 time_unit^-1.
        temp_df : pandas.DataFrame
            Temperature data in K.
        all_df : pandas.DataFrame
            The above combined.

        FITTED PARAMETERS
        k_val : np.ndarray of float
            Initial constant estimate(s) (in rxns and rate equation ord).
        k_fit : np.ndarray of float
            Fitted constant(s) (in rxns and rate equation ord) in appropriate units.
        k_fit_err : np.ndarray of float
            Fitted constant error(s) (in rxns and rate equation ord) in appropriate units
        ord_fit : np.ndarray of float
            Fitted ord(s) (in spec_name/type order).
        ord_fit_err : np.ndarray of float
            Fitted ord error(s) (in spec_name/type order).
        pois_fit : np.ndarray of float
            Fitted poisoning(s) (in spec_name/type order) in moles.
        pois_fit_err : np.ndarray of float
            Fitted poisoning error(s) (in spec_name/type order) in moles.

        GOODNESS OF FIT
        rss : float
            Residual sum of squares.
        r2 : float
            R squared.
        rmse : float
            Root mean square average.
        mae : float
            Mean average error.
        aic : float
            Akaike information criterion (AIC).
        bic : float
            Bayesian information criterion (BIC).

        FUNCTIONS
        plot_conc_vs_time(self, show_asp='all', method='lone', f_format='svg', save_to='cake_conc_vs_time.svg',
                     return_img=False, return_fig=False, transparent=False) : function
             Function to plot CAKE concentration data. See cake plot_conc_vs_time documentation for more details.
        plot_rate_vs_conc(self, show_asp='all', f_format='svg', save_to='cake_conc_vs_time.svg',
                     return_img=False, return_fig=False, transparent=False) : function
             Function to plot CAKE rate data. See cake plot_rate_vs_conc documentation for more details.
        plot_rate_vs_temp(self, show_asp='all', f_format='svg', save_to='cake_conc_vs_time.svg',
                     return_img=False, return_fig=False, transparent=False) : function
             Function to plot CAKE rate versus temperature. See cake plot_rate_vs_temp documentation for more details.
    """
    data = pre_fit(df, spec_name, spec_type, stoich, rxns, t_col, col, fit_asp, mol0, mol_end, vol0, add_sol_conc,
                   cont_add_rate, t_cont_add, disc_add_vol, t_disc_add, cont_sub_rate, disc_sub_vol, t_disc_sub, temp0,
                   cont_temp_rate, t_cont_temp, temp_col, rate_eq_type, rate_method, rtol, atol,
                   k_lim, ord_lim, pois_lim, tic_col, scale_avg_num, win, inc, time_unit, conc_unit)
    data = fitting_func(data)
    return data


# Prepares parameters for CAKE fitting
def pre_fit(df, spec_name, spec_type, stoich, rxns, t_col, col, fit_asp, mol0, mol_end, vol0,
            add_sol_conc, cont_add_rate, t_cont_add, disc_add_vol, t_disc_add, cont_sub_rate, disc_sub_vol,
            t_disc_sub, temp0, cont_temp_rate, t_cont_temp, temp_col, rate_eq_type, rate_method, rtol, atol,
            k_lim, ord_lim, pois_lim, tic_col, scale_avg_num, win, inc, time_unit, conc_unit):

    if fit_asp is None:
        if col and isinstance(col, (list, tuple)):
            fit_asp = ['y' if i else 'n' for i in col]
        else:
            fit_asp = ['y']

    spec_name, num_spec, stoich, t_col, col, fit_asp, mol0, mol_end, add_sol_conc, cont_add_rate, t_cont_add, \
    disc_add_vol, t_disc_add, disc_sub_vol, t_disc_sub, cont_temp_rate, t_cont_temp, stoich_loc, ord_loc, \
    k_lim, ord_lim, pois_lim, fix_pois_locs, var_locs, fit_asp_locs, fit_param_locs, \
    inc = get_prep.param_prep(spec_name, spec_type, stoich, rxns, t_col, col, fit_asp, mol0, mol_end, add_sol_conc,
                              cont_add_rate, t_cont_add, disc_add_vol, t_disc_add, disc_sub_vol, t_disc_sub,
                              cont_temp_rate, t_cont_temp, temp_col, rate_eq_type, k_lim, ord_lim, pois_lim, inc)

    if isinstance(df, pd.DataFrame):
        data_org = df.to_numpy()

    # Convert df header names into indices
    if t_col and isinstance(t_col, str): t_col = df.columns.get_loc(t_col)
    if col and isinstance(col, str):
        col = df.columns.get_loc(col)
    elif col and isinstance(col, (list, tuple)):
        if isinstance(col[0], str):
            col = [df.columns.get_loc(i) for i in col]
    if temp_col and isinstance(temp_col, str): temp_col = df.columns.get_loc(temp_col)
    if tic_col and isinstance(tic_col, str): tic_col = df.columns.get_loc(tic_col)

    # Get t
    t = get_prep.data_smooth(data_org, t_col, win, inc)
    t = np.reshape(t, len(t))
    t_add = get_prep.add_sim(t, inc)

    # Get tic
    tic = get_prep.data_smooth(data_org, tic_col, win, inc) if tic_col is not None else None

    # Calculate iterative species additions, volumes, temperatures and the appropriate rate function
    t_split, disc_event, cont_event = get_events.get_conc_events(t_add, num_spec, vol0, add_sol_conc, cont_add_rate,
                                                                 t_cont_add, disc_add_vol, t_disc_add, cont_sub_rate,
                                                                 disc_sub_vol, t_disc_sub, temp0, cont_temp_rate, t_cont_temp)
    temp, cont_event = get_prep.get_temp_data(t, data_org, cont_event, temp_col, win=win, inc=inc)
    rate_eq = rate_eqs.rate_eq_map.get(rate_eq_type.lower())
    ode_func = ode_solver.temp_map.get('temp_norm') if not temp_col else ode_solver.temp_map.get('temp_col')

    # Determine mol0, mol_end and instrument_scale data as required
    data_mod = np.empty((len(t), num_spec))
    col_ext = []
    for i in range(num_spec):
        if col[i] is not None:
            col_ext.append(i)
            data_i = get_prep.data_smooth(data_org, col[i], win, inc)
            data_i = get_prep.tic_norm(data_i, tic)

            vol = get_prep.get_vol(t_add, cont_event)
            if mol0[i] is None and scale_avg_num == 0:
                mol0[i] = data_i[0] * vol[0]
            elif mol0[i] is None and scale_avg_num > 0:
                mol0[i] = np.mean([data_i[j] * vol[j] for j in range(scale_avg_num)])
            elif mol0[i] is not None and mol0[i] != 0 and scale_avg_num > 0 and (
                        mol_end[i] is None or mol0[i] >= mol_end[i]):
                data_scale = np.mean([data_i[j] / (mol0[i] / vol[j]) for j in range(scale_avg_num)])
                data_i = data_i / data_scale

            if mol_end[i] is None and scale_avg_num == 0:
                mol_end[i] = data_i[-1] * vol[-1]
            elif mol_end[i] is None and scale_avg_num > 0:
                mol_end[i] = np.mean([data_i[j] * vol[j] for j in range(-scale_avg_num, 0)])
            elif mol_end[i] is not None and mol_end[i] != 0 and scale_avg_num > 0 and (
                        mol0[i] is None or mol_end[i] >= mol0[i]):
                data_scale = np.mean([data_i[j] / (mol_end[i] / vol[j]) for j in range(-scale_avg_num, 0)])
                data_i = data_i / data_scale
            data_mod[:, i] = data_i
        if col[i] is None and mol0[i] is None:
            mol0[i] = 0  # May cause issues
        if col[i] is None and mol_end[i] is None:
            mol_end[i] = 0  # May cause issues

    exp_t_rows = list(range(0, len(t_add), int(max(1, inc))))
    exp_conc = data_mod[:, col_ext]
    exp_rates = np.zeros((len(exp_t_rows), len(col_ext)))
    #for i, j in enumerate(col_ext):
    #    exp_rates[:, i] = np.gradient(data_mod[:, j], t_add[exp_t_rows])

    # Manipulate data for fitting
    t_add_to_fit = np.empty(0)
    y_data_to_fit = np.empty(0)

    for i in range(len(fit_asp_locs)):
        t_add_to_fit = np.append(t_add_to_fit, t_add)
        y_data_to_fit = np.append(y_data_to_fit, data_mod[:, fit_asp_locs[i]], axis=0)

    data = store_objs.FitData(num_spec, spec_name, spec_type, fit_asp, mol0, mol_end, stoich, stoich_loc, temp,
                              disc_event, cont_event, k_lim, ord_lim, pois_lim, ord_loc, col, inc, rate_eq_type,
                              rate_eq, ode_func, rate_method, rtol, atol, fix_pois_locs, var_locs, fit_asp_locs,
                              fit_param_locs, t, t_add, t_add_to_fit, t_split, y_data_to_fit, data_mod, col_ext,
                              exp_conc, exp_rates, time_unit, conc_unit)

    return data


# Fits CAKE experiment data
def fitting_func(data):
    exp_t_rows = list(range(0, len(data.t_add), int(max(1, data.inc))))

    # Dictionary for defining lower and upper constant limits
    param_comb_dict = {
        'standard': [(-10, 10, 10, 'g')],
        'michaelis-menten': [(-13, 13, 10, 'g'), (-10, 0, 10, 'g')],
        'arrhenius': [(-15, 15, 10, 'g'), (0, 1E6, 51, 'a')],
        'eyring': [(-1E3, 1E3, 51, 'a'), (-1E6, 1E6, 51, 'a')]
    }

    # Function to sort data for fitting
    def eq_sim_fit(_, *fit_param):
        conc = ode_solver.eq_sim_gen(data.t_split, exp_t_rows, data.mol0, data.stoich, data.stoich_loc,
                                     data.disc_event, data.cont_event, data.k_lim, data.ord_lim, data.ord_loc,
                                     data.var_locs, fit_param, data.fit_param_locs, data.rate_eq, data.ode_func,
                                     data.rate_method, data.rtol, data.atol)
        conc_fit_reshape = np.empty(0)
        for i in data.fit_asp_locs:
            conc_fit_reshape = np.append(conc_fit_reshape, np.reshape(conc[:, i], len(conc[:, i])))
        return conc_fit_reshape

    # Function for guessing initial constant values, based on given allowed parameter combinations and spacing type
    def guess_sim(t, y_data, ord_val, pois_val, param_comb):
        guess_list = []
        for i in range(len(param_comb)):
            if len(param_comb[i]) < 4:
                guess_list.append([param_comb[i][0]])
            elif 'a' in param_comb[i][3]:
                guess_list.append(np.linspace(*param_comb[i][:-1], param_comb[i][2] + 1).tolist())
            elif 'g' in param_comb[i][3]:
                guess_list.append([param_comb[i][2] ** j for j in range(param_comb[i][0], param_comb[i][1])])
        pre_k_guess_arr = np.array(list(itertools.product(*guess_list)))
        k_guess_arr = np.column_stack((pre_k_guess_arr, np.zeros(len(pre_k_guess_arr))))
        for i in range(len(k_guess_arr)):
            try:
                fit_guess = eq_sim_fit(t, *k_guess_arr[i, :-1], *ord_val, *pois_val)
                k_guess_arr[i, -1] = residuals(y_data, fit_guess, [])[0]
            except:
                k_guess_arr[i, -1] = -1
        k_guess_arr[k_guess_arr[:, -1] == -1, -1] = max(k_guess_arr[:, -1])
        sort_indices = np.argsort(k_guess_arr[:, -1])
        k_guess_sort = k_guess_arr[sort_indices]
        return k_guess_sort[0, :-1].tolist()

    # Define initial values and lower and upper limits for parameters to fit: ord and pois
    ord_val, ord_min, ord_max, pois_val, pois_min, pois_max = [], [], [], [], [], []
    for i, j in data.var_locs[1]:
        ord_val.append(data.ord_lim[i][j][0])
        ord_min.append(data.ord_lim[i][j][1])
        ord_max.append(data.ord_lim[i][j][2])
    for i, j in enumerate(data.var_locs[2]):
        unpack_pois_lim = data.pois_lim[j]
        if 'max' in unpack_pois_lim:
            pois_val.append(0)
            pois_min.append(0)
            pois_max.append(max(data.mol0[i], data.mol_end[i]))
        else:
            pois_val.append(unpack_pois_lim[0])
            pois_min.append(unpack_pois_lim[1])
            pois_max.append(unpack_pois_lim[2])
    for i in data.fix_pois_locs:
        data.mol0[i] -= data.pois_lim[i]
        data.mol_end[i] -= data.pois_lim[i]
    # Define initial constant (k) values to fit
    # starttime = timeit.default_timer()
    k_val, k_min, k_max, param_comb = [], [], [], []
    for i, j in data.var_locs[0]:
        if isinstance(data.k_lim[i][j], (tuple, list)) and data.k_lim[i][j][0] is None:
            param_comb.append(param_comb_dict[data.rate_eq_type.lower()][j])  # swapped from i to j - may need repair
        else:
            param_comb.append(data.k_lim[i][j])
    k_val = guess_sim(data.t_add_to_fit, data.y_data_to_fit, ord_val, pois_val, param_comb)

    # Define lower and upper limits for constant(s) (k) values to fit
    bound_adj = 1E-6
    for i, (j, m) in enumerate(data.var_locs[0]):
        if isinstance(data.k_lim[j][m], (tuple, list)) and (len(data.k_lim[j][m]) > 1 and data.k_lim[j][m][1] is None)\
                or (len(data.k_lim[j][m]) > 2 and data.k_lim[j][m][2] is None):
            if ('standard' or 'michaelis-menten') in data.rate_eq_type.lower() or ('arrhenius' in data.rate_eq_type.lower() and m == 0):
                k_min.append(k_val[i] * bound_adj)
                k_max.append(k_val[i] / bound_adj)
            if 'eyring' in data.rate_eq_type.lower() or ('arrhenius' in data.rate_eq_type.lower() and m == 1):
                k_min.append(param_comb_dict[data.rate_eq_type.lower()][m][0])
                k_max.append(param_comb_dict[data.rate_eq_type.lower()][m][1])
        elif len(data.k_lim[j][m]) == 2:
            k_min.append(k_val[i] * data.k_lim[j][m][1])
            k_max.append(k_val[i] / data.k_lim[j][m][1])
        elif len(data.k_lim[j][m]) == 3:
            k_min.append(data.k_lim[j][m][1])
            k_max.append(data.k_lim[j][m][2])

    init_param = [*k_val, *ord_val, *pois_val]
    low_bounds = [*k_min, *ord_min, *pois_min]
    up_bounds = [*k_max, *ord_max, *pois_max]

    # Path if no parameters were set to fit that avoids applying fitting
    if init_param and low_bounds and up_bounds:
        # Fit, determine optimal parameters and determine resulting fits
        fit_popt, fit_param_pcov = optimize.curve_fit(eq_sim_fit, data.t_add_to_fit, data.y_data_to_fit,
                                                    init_param, maxfev=800, bounds=(low_bounds, up_bounds), method='trf')

        k_fit = fit_popt[data.fit_param_locs[0]]
        ord_fit = fit_popt[data.fit_param_locs[1]]
        pois_fit = fit_popt[data.fit_param_locs[2]]

        fit_conc = ode_solver.eq_sim_gen(data.t_split, exp_t_rows, data.mol0, data.stoich, data.stoich_loc,
                                         data.disc_event, data.cont_event, data.k_lim, data.ord_lim, data.ord_loc, data.var_locs,
                                         fit_popt, data.fit_param_locs, data.rate_eq, data.ode_func, data.rate_method,
                                         data.rtol, data.atol)
        fit_rates = np.empty((len(data.t), len(data.ord_loc)))

        k_adj = copy.deepcopy(data.k_lim)  # added to prevent editing of variables
        ord_adj = copy.deepcopy(data.ord_lim)  # added to prevent editing of variables
        for i, (j, m) in enumerate(data.var_locs[0]):
            k_adj[j] = [k_fit[i] if idx == m else val for idx, val in enumerate(k_adj[j])]
        for i, (j, m) in enumerate(data.var_locs[1]):
            ord_adj[j] = [ord_fit[i] if idx == m else val for idx, val in enumerate(ord_adj[j])]
        for i in range(len(data.t)):
            fit_rates[i] = rate_eqs.rate_calc(k_adj, fit_conc[i], ord_adj, data.ord_loc, data.temp[i], data.rate_eq)

        # Calculate residuals and errors
        fit_param_err = np.sqrt(np.diag(fit_param_pcov))
        k_fit_err = fit_param_err[data.fit_param_locs[0]]
        ord_fit_err = fit_param_err[data.fit_param_locs[1]]
        pois_fit_err = fit_param_err[data.fit_param_locs[2]]
        rss, r2, r2_adj, rmse, mae, aic, bic = residuals(data.y_data_to_fit,
                                                         eq_sim_fit(data.t_add_to_fit, *k_fit, *ord_fit, *pois_fit),
                                                         init_param)

    else:
        print('No parameters set to fit - no fitting applied.')
        fit_conc = ode_solver.eq_sim_gen(data.t_split, exp_t_rows, data.mol0, data.stoich, data.stoich_loc,
                                     data.disc_event, data.cont_event, data.k_lim, data.ord_lim, data.ord_loc,
                                     data.var_locs, [], data.fit_param_locs, data.rate_eq, data.ode_func,
                                     data.rate_method, data.rtol, data.atol)
        fit_rates = np.empty((len(data.t), len(data.ord_loc)))
        for i in range(len(data.t)):
            fit_rates[i] = rate_eqs.rate_calc(data.k_lim, fit_conc[i], data.ord_lim, data.ord_loc, data.temp[i], data.rate_eq)
        rss, r2, r2_adj, rmse, mae, aic, bic = residuals(data.y_data_to_fit, eq_sim_fit(data.t_add_to_fit), init_param)

    # Prepare data for output
    t_df = pd.DataFrame(data.t, columns=['Time / ' + data.time_unit])
    exp_conc_headers = [data.spec_name[i] + ' exp. conc. / ' + data.conc_unit
                          for i in range(data.num_spec) if data.col[i] is not None]
    exp_rate_headers = [data.spec_name[i] + ' exp. rate / ' + data.rate_unit
                          for i in range(data.num_spec) if data.col[i] is not None]
    fit_conc_headers = [i + ' fit conc. / ' + data.conc_unit for i in data.spec_name]
    fit_rate_headers = ['Reaction ' + str(i + 1) + ' fit rate / ' + data.rate_unit for i in range(len(data.k_lim))]
    exp_conc_df = pd.DataFrame(data.exp_conc, columns=exp_conc_headers)
    exp_rate_df = pd.DataFrame(data.exp_rates, columns=exp_rate_headers)
    fit_conc_df = pd.DataFrame(fit_conc, columns=fit_conc_headers)
    fit_rate_df = pd.DataFrame(fit_rates, columns=fit_rate_headers)
    temp_df = pd.DataFrame(data.temp, columns=['Temperature / K'])
    all_df = pd.concat([t_df, exp_conc_df, exp_rate_df, fit_conc_df, fit_rate_df, temp_df], axis=1)

    if not data.var_locs[0]:
        k_val, k_fit, k_fit_err = 'N/A', 'N/A', 'N/A'
    if not data.var_locs[1]:
        ord_fit, ord_fit_err = 'N/A', 'N/A'
    if not data.var_locs[2]:
        pois_fit, pois_fit_err = 'N/A', 'N/A'
        t_del_fit, t_del_fit_err = 'N/A', 'N/A'
    else:
        pois_fit, pois_fit_err = pois_fit / data.cont_event.vol[0], pois_fit_err / data.cont_event.vol[0]
        t_del_fit, t_del_fit_err = pois_fit * 1, pois_fit_err * 1  # need to make t_del work somehow

    data.add_fit(t_df, exp_conc_df, exp_rate_df, fit_conc_df, fit_rate_df, temp_df, all_df,
                 k_val, k_fit, k_fit_err, ord_fit, ord_fit_err, pois_fit, pois_fit_err,
                 rss, r2, r2_adj, rmse, mae, aic, bic)

    return data


if __name__ == "__main__":
    spec_name = ['r1', 'r2', 'p1', 'c1']
    spec_type = ['r', 'r', 'p', 'c']
    vol0 = 0.1
    stoich = [1, 1, 1, None]  # insert stoichiometry of reactant, r
    mol0 = [0.1, 0.2, 0, 0]
    mol_end = [0, 0.1, 0.1, None]
    add_sol_conc = [None, None, None, 10]
    cont_add_rate = [None, None, None, 0.001]
    t_cont_add = [None, None, None, 1]
    t_col = 0
    col = [1, 2, 3, None]
    ord_lim = [(1, 0, 2), 1, 0, (1, 0, 2)]
    fit_asp = ['y', 'n', 'y', 'n']
    file_name = r'C:\Users\Peter\Documents\Postdoctorate\Work\CAKE\Case studies\CAKE preliminary trials.xlsx'
    sheet_name = r'Test_data'
    pic_save = r'/Users/bhenders/Desktop/CAKE/cake_app_test.png'
    xlsx_save = r'/Users/bhenders/Desktop/CAKE/fit_data.xlsx'

    df = import_data(file_name, sheet_name, t_col, col)
    output = fit(df, spec_name=spec_name, spec_type=spec_type, stoich=stoich, t_col=t_col, col=col, fit_asp=fit_asp,
                 mol0=mol0, mol_end=mol_end, vol0=vol0, add_sol_conc=add_sol_conc, cont_add_rate=cont_add_rate,
                 t_cont_add=t_cont_add, ord_lim=ord_lim)
    t_df, exp_df, fit_conc_df, fit_rate_df, k_val_est, k_fit, k_fit_err, \
    ord_fit, ord_fit_err, pois_fit, pois_fit_err, ss_res, r2, col = output

    if not isinstance(col, (tuple, list)): col = [col]

    html = plot_conc_vs_time(t_df, exp_df, fit_conc_df, col,
                             f_format='svg', return_img=False, save_disk=True, save_to=pic_save)

    param_dict = app.make_param_dict(spec_type, vol0, stoich=stoich, fit_asp=fit_asp, mol0=mol0, mol_end=mol_end,
                                     add_sol_conc=add_sol_conc, t_cont_add=t_cont_add, t_col=t_col, col=col,
                                     ord_lim=None)

    # write_fit_data(xlsx_save, df, param_dict, t, r, p, fit_p, fit_r, res_val, res_err, res_rss, res_r2, cat_pois)
    file, _ = app.write_fit_data_temp(df, param_dict, t_df, exp_df, fit_conc_df, fit_rate_df,
                                      k_fit, k_fit_err, ord_fit, ord_fit_err, pois_fit, pois_fit_err, ss_res, r2)
    file.seek(0)
    with open(xlsx_save, 'wb') as f:
        f.write(file.getbuffer())
