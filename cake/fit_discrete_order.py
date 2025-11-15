"""CAKE Fitting Programme for Discrete Orders"""

import multiprocessing
import numpy as np
import os
import pandas as pd
import pickle
import itertools
from cake.prep import get_prep
from cake import fit
from cake.fit import fitting_func


def run_with_timeout(func, args=(), kwargs={}, timeout=10):
    def wrapper(queue, *args, **kwargs):
        try:
            result = func(*args, **kwargs)
            queue.put((True, result))
        except Exception as e:
            queue.put((False, e))

    queue = mp.Queue()
    proc = mp.Process(target=wrapper, args=(queue, *args), kwargs=kwargs)
    proc.start()
    proc.join(timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        raise TimeoutError("Function call timed out")

    success, value = queue.get()
    if success:
        return value
    else:
        raise value


def fit_discrete_order(df, spec_name=None, spec_type=None, stoich=None, rxns=None, t_col=0, col=1, fit_asp=None,
                       mol0=None, mol_end=None, vol0=None, add_sol_conc=None, cont_add_rate=None, t_cont_add=None,
                       disc_add_vol=None, t_disc_add=None, cont_sub_rate=None, disc_sub_vol=None, t_disc_sub=None,
                       temp0=293.15, cont_temp_rate=None, t_cont_temp=None, temp_col=None,
                       rate_eq_type='standard', rate_method='Radau', rtol=1E-3, atol=1E-6,
                       k_lim=None, ord_num=None, ord_inc=5, pois_lim=None, scale_avg_num=0, win=1, inc=1, tic_col=None,
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
        Index(es) or name(s) of species column(s) (in spec_name/type ord).
        Default is 1 (second column).
    fit_asp : str, or list of str, or None, optional
        Name(s) of species to fit to, or 'y' to fit to species, 'n' not to fit to species (in spec_name/type ord).
        Default is None which fits all species with data.

    REACTION CONDITIONS
    mol0 : float, or list of float, required
        Initial moles of species (in spec_name/type ord) in moles_unit.
    mol_end : float, list of float, or None, optional
        Final moles of species (in spec_name/type ord) in moles_unit or None if data do not need scaling.
        Default is None (data must be provided in moles).
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
        Default is '1E-3'.
    atol : float, optional
        Absolute tolerance (number of correct decimal places) used for fitting.
        For more information see scipy.integrate.solve_ivp documentation.
        Default is '1E-6'.
    k_lim : float, or list of float and tuple of float, or list of list of float and tuple of float, or None, optional
        Estimated constant(s) (in rxns ord) in appropriate units. Constants related to the molar gas constant
        must be in units of J K^-1 mol^-1.
        Each constant can be input as float for a fixed variable
        or as tuple for floating variable with bounds (estimate, factor difference) or (estimate, lower, upper).
        Multiple constants for a rate equation must be input as a list, e.g. [constant_1, constant_2, ...].
        Multiple reactions must be input as a list, e.g. [reaction_1_constants, reaction_2_constants, ...].
        Default is None which sets appropriate bounds for all reactions and constants.
    ord_num : list of float and tuple, or None, optional
        Species reaction ord (in spec_name/type ord).
        Specified as exact value for fixed unfitted variable or variable with bounds (estimate, lower, upper).
        Default is None which assumes (1, 0, 2) for 'r' and 'c' species and 0 for 'p' species (not used if rxns).
    ord_num : int or float, optional
        Number of orders to test, split uniformly between the lower and upper ord_lim for each species.
        Default is 5.
    pois_lim : list of float, str, and tuple of float and str, or None, optional
        Moles of species poisoned (in spec_name/type ord) in moles_unit.
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
        Default is 'moles_unit volume_unit$^{-1}$'

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
        self.discrete_order_fit_df : pandas.DataFrame
            Fitted parameters and goodness of fit metrics as below, in ord of minimum rss.
        discrete_order_fit_conc_df : pandas.DataFrame
            Fitted concentration data for each ord combination in moles_unit volume_unit^-1, in ord of minimum rss.
        self.discrete_order_fit_rate_df : pandas.DataFrame
            Fitted rate data for each ord combination in moles_unit volume_unit^-1, in ord of minimum rss.

        FITTED PARAMETERS
        k_val : list of list of float
            Initial constant estimate(s) (in rxns and rate equation ord).
        k_fit :
            Fitted constant(s) (in rxns and rate equation ord) in appropriate units.
        k_fit_err :
            Fitted constant error(s) (in rxns and rate equation ord) in appropriate units
        ord_fit :
            Fitted ord(s) (in spec_name/type ord).
        ord_fit_err :
            Fitted ord error(s) (in spec_name/type ord).
        pois_fit :
            Fitted poisoning(s) (in spec_name/type ord) in moles.
        pois_fit_err :
            Fitted poisoning error(s) (in spec_name/type ord) in moles.

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
    """

    # Run normal fitting
    data = fit(df, spec_name=spec_name, spec_type=spec_type, stoich=stoich, rxns=rxns, t_col=t_col, col=col,
               fit_asp=fit_asp, mol0=mol0, mol_end=mol_end, vol0=vol0, add_sol_conc=add_sol_conc,
               cont_add_rate=cont_add_rate, t_cont_add=t_cont_add, disc_add_vol=disc_add_vol, t_disc_add=t_disc_add,
               cont_sub_rate=cont_sub_rate, disc_sub_vol=disc_sub_vol, t_disc_sub=t_disc_sub, temp0=temp0,
               cont_temp_rate=cont_temp_rate, t_cont_temp=t_cont_temp, temp_col=temp_col, rate_eq_type=rate_eq_type,
               rate_method=rate_method, rtol=rtol, atol=atol, k_lim=k_lim, ord_lim=ord_num, pois_lim=pois_lim,
               scale_avg_num=scale_avg_num, win=win, inc=inc, time_unit=time_unit, conc_unit=conc_unit)

    best_df = data.df_data
    best_k_est = data.k_est
    best_res = data.res

    # Get allowed orders for each species and the ord combinations of all species
    ord_var_list = [np.round(np.linspace(data.ord_lim[0][i][1], data.ord_lim[0][i][2], ord_inc), 2).tolist()
                    for i in range(len(data.ord_lim[0])) if isinstance(data.ord_lim[0][i], tuple)]
    ord_var_combi = list(itertools.product(*ord_var_list))
    ord_all_list = [np.linspace(data.ord_lim[0][i][1], data.ord_lim[0][i][2], ord_inc).tolist()
                        if isinstance(data.ord_lim[0][i], tuple) else [data.ord_lim[0][i]]
                    for i in range(len(data.ord_lim[0]))]
    ord_all_combi = list(itertools.product(*ord_all_list))

    # Redefine locations for fitting
    data.fix_pois_locs, data.var_locs, data.fit_asp_locs, data.fit_param_locs = \
        get_prep.get_var_locs(data.spec_type, data.num_spec, data.k_lim, [list(ord_all_combi[0])],
                              data.pois_lim, data.fit_asp)

    # Setup total arrays
    do_fit = np.empty([len(ord_all_combi) + 1, 9], dtype=object)
    do_fit_conc = np.empty([len(ord_all_combi) + 1], dtype=object)
    do_fit_rate = np.empty([len(ord_all_combi) + 1], dtype=object)

    # Add the best (non-discrete) fit
    def run_with_timeout(data, timeout=10.0):
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=fitting_func_queue, args=(data, result_queue))
        process.start()
        process.join(timeout=timeout)

        if process.is_alive():
            process.terminate()
            process.join()
            print(f"Process took longer than " + str(timeout) + " s. Skipped.")
            return False
        else:
            return True

    # Add discrete fits
    for i in range(len(ord_all_combi)):
        data.ord_lim = [list(ord_all_combi[i])]
        if run_with_timeout(data, timeout=20):
            with open('temp.pkl', 'rb') as f:
                data = pickle.load(f)
        else:
            continue
        do_fit[i + 1] = [[*ord_var_combi[i]], data.k_fit, data.k_fit_err, data.pois_fit, data.pois_fit_err,
                         data.rss, data.r2, data.rmse, data.mae]
        do_fit_conc[i + 1] = data.fit_conc_df
        do_fit_rate[i + 1] = data.fit_rate_df
    try:
        os.remove('temp.pkl')
    except:
        pass

    data.ord_lim = ord_all_combi

    # Sort fits by RSS
    mask = do_fit[:, -4] != None
    sort_indices = do_fit[mask, -4].argsort()
    do_fit_sort = do_fit[mask][sort_indices]
    do_fit_conc_sort = do_fit_conc[mask][sort_indices]
    do_fit_rate_sort = do_fit_rate[mask][sort_indices]

    headers = ['Order', 'k', 'k error', 'Poisoning', 'Poisoning fit', 'RSS', 'R2', 'RMSE', 'MAE']
    do_fit_sort_df = pd.DataFrame(do_fit_sort, columns=headers)
    identical_columns = [col for col in do_fit_sort_df.columns if do_fit_sort_df.astype(str)[col].nunique() == 1]
    do_fit_sort_df.drop(columns=identical_columns, inplace=True)

    data.add_discrete_order(do_fit_sort_df, do_fit_conc_sort, do_fit_rate_sort)

    data.add_fit(*best_df, *best_k_est, *best_res)

    return data


def fitting_func_queue(data, result_queue):
    result = fitting_func(data)
    result_queue.put(result.rss)
    with open('temp.pkl', 'wb') as f:
        pickle.dump(result, f)
