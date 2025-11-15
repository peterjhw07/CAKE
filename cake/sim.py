"""CAKE Simulation Programme"""

import numpy as np
import pandas as pd
from cake import rate_eqs
from cake.fitting import ode_solver
from cake.prep import get_events, get_prep, store_objs


np.seterr(divide='ignore', invalid='ignore')


# Simulates CAKE experiments
def sim(t, spec_name=None, spec_type=None, stoich=None, rxns=None, mol0=None, vol0=None,
        add_sol_conc=None, cont_add_rate=None, t_cont_add=None, disc_add_vol=None, t_disc_add=None,
        cont_sub_rate=None, disc_sub_vol=None, t_disc_sub=None, temp0=293.15, cont_temp_rate=None, t_cont_temp=None,
        rate_eq_type='standard', rate_method='Radau', rtol=1E-6, atol=1E-9, k=None, ord=None, pois=None, inc=1,
        rand_fac=None, scale=1, time_unit='time_unit', conc_unit='moles_unit volume_unit$^{-1}$'):
    """
    Function to simulate CAKE data.

    Parameters
    ----------
    All data must use consistent units throughout, i.e., all parameters must be inputted with identical units.

    DATA
    t : tuple, list, or numpy.array, required
        Time values to perform simulation with.
        Tuple of the form (start, end, step size) will make time values using these parameters,
        numpy.array or list will use the exact values.

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
        Reaction mechanisms for simulation.
        Takes the form [reaction_1, reaction_2, ...]) where each reaction is of the form
        (('reactant_1', 'reactant_2', ...), ('product_1', 'product_2', ...)).
        Default is None (not required if using spec_type, stoich, etc.).

    REACTION CONDITIONS
    mol0 : list of float, required
        Initial moles of species (in spec_name/type order) in moles_unit.
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

    SIMULATION PARAMETERS
    rate_eq_type : str, optional
        Rate equation type used for simulation. Available options are 'standard', 'Arrhenius', 'Eyring',
        'Michaelis-Menten', or 'custom'. A 'custom' rate equation can be formatted in the rate_eqs.py file.
        Default is 'standard'.
    rate_method : str, optional
        Rate method used for simulation. Available options are 'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', and 'LSODA'.
        For more information see scipy.integrate.solve_ivp documentation.
        Default is 'Radau'.
    rtol : float, optional
        Relative tolerance (number of correct digits) used for simulation.
        For more information see scipy.integrate.solve_ivp documentation.
        Default is '1E-6'.
    atol : float, optional
        Absolute tolerance (number of correct decimal places) used for simulation.
        For more information see scipy.integrate.solve_ivp documentation.
        Default is '1E-9'.
    k : float, or list of float, or list of list of float, required
        Estimated constant(s) (in rxns ord) in appropriate units. Constants related to the molar gas constant
        must be in units of J K^-1 mol^-1.
        Multiple constants for a rate equation must be input as a list, e.g., [constant_1, constant_2, ...].
        Multiple reactions must be input as a list, e.g., [reaction_1_constants, reaction_2_constants, ...].
    ord : list of float, or None, optional
        Species reaction ord (in spec_name/type order).
        Default is None which sets ord as 1 for 'r' and 'c' species and 0 for 'p' species (not used if rxns).
    pois : list of float and str, or None, optional
        Moles of species poisoned (in spec_name/type order) in moles_unit.
        Default is None (no poisoning occurs for any species).

    DATA MANIPULATION
    inc : int or float, optional
        Increments between adjacent points for improved simulation (>1).
        or fraction of points removed, e.g. if unecessarily large amount of data (<1).
        Default is 1 (no addition or subtraction of points).
    rand_fac : float, optional
        Noise addition.
        Default is None (no noise).
    scale : float, optional
        Scaling to emulate an arbitary relationship between concentration and instrument intensity.
        Default is None (no scaling).

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
        fit_conc_df : pandas.DataFrame
            Simulated concentration data in moles_unit volume_unit^-1.
        fit_rate_df : pandas.DataFrame
            Simulated rate data in moles_unit volume_unit^-1 time_unit^-1.
        temp_df : pandas.DataFrame
            Temperature data in K.
        all_df : pandas.DataFrame
            The above combined.

        FUNCTIONS
        plot_conc_vs_time(self, show_asp='all', conc_unit='', conc_axis_label='Concentration',
                     method='lone', f_format='svg', save_to='cake_conc_vs_time.svg',
                     return_img=False, return_fig=False, transparent=False) : function
             Function to plot CAKE concentration data. conc_unit defaults to self.conc if not specified.
             See cake plot_conc_vs_time documentation for more details.
        plot_rate_vs_conc(self, show_asp='all', f_format='svg', save_to='cake_conc_vs_time.svg',
                     return_img=False, return_fig=False, transparent=False) : function
             Function to plot CAKE rate data. See cake plot_rate_vs_conc documentation for more details.
        plot_rate_vs_temp(self, show_asp='all', f_format='svg', save_to='cake_conc_vs_time.svg',
                     return_img=False, return_fig=False, transparent=False) : function
             Function to plot CAKE rate versus temperature. See cake plot_rate_vs_temp documentation for more details.
        """

    # Prepare parameters
    if type(t) is tuple:
        t = np.linspace(t[0], t[1], int((t[1] - t[0]) / t[2]) + 1)
    spec_name, num_spec, stoich, t_col, col, fit_asp, mol0, _, add_sol_conc, cont_add_rate, t_cont_add, \
    disc_add_vol, t_disc_add, disc_sub_vol, t_disc_sub, cont_temp_rate, t_cont_temp, stoich_loc, ord_loc, k, ord, pois, \
    fix_pois_locs, var_locs, fit_asp_locs, fit_param_locs, \
    inc = get_prep.param_prep(spec_name, spec_type, stoich, rxns, None, None, None, mol0, None,
                              add_sol_conc, cont_add_rate, t_cont_add, disc_add_vol, t_disc_add, disc_sub_vol,
                              t_disc_sub, cont_temp_rate, t_cont_temp, None, rate_eq_type, k, ord, pois, inc)
    exp_t_rows = list(range(0, len(t), int(max(1, inc))))

    # Calculate iterative species additions, volumes, temperature and define rate equation
    t_split, disc_event, cont_event = get_events.get_conc_events(t, num_spec, vol0, add_sol_conc, cont_add_rate,
                                                                 t_cont_add, disc_add_vol, t_disc_add, cont_sub_rate,
                                                                 disc_sub_vol, t_disc_sub, temp0, cont_temp_rate,
                                                                 t_cont_temp)
    temp, cont_event = get_prep.get_temp_data(t, [], cont_event, None, win=1, inc=1)
    rate_eq = rate_eqs.rate_eq_map.get(rate_eq_type.lower())
    ode_func = ode_solver.temp_map.get('temp_norm')

    mol0 = [mol0[i] if mol0[i] else 0 for i in range(num_spec)]

    data = store_objs.SimData(num_spec, spec_name, spec_type, mol0, stoich, stoich_loc, temp, disc_event, cont_event,
                              k, ord, pois, ord_loc, col, inc, rate_eq_type, rate_eq, ode_func, rate_method,
                              rtol, atol, t, time_unit, conc_unit)

    for i in fix_pois_locs:
        mol0[i] -= pois[i]

    # Run simulation
    fit_conc = ode_solver.eq_sim_gen(t_split, exp_t_rows, mol0, stoich, stoich_loc, disc_event,
                                     cont_event, k, ord, ord_loc, [[], [], []],
                                     k, [[], [], []], rate_eq, ode_func, rate_method, rtol, atol)
    fit_rates = np.empty((len(t), len(ord_loc)))
    for i in range(len(t)):
        fit_rates[i] = rate_eqs.rate_calc(k, fit_conc[i], ord, ord_loc, temp[i], rate_eq)

    # Add noise component
    rand_fac = [i if i else 0 for i in get_prep.type_to_list(rand_fac)]
    if len(rand_fac) == 1: rand_fac = rand_fac * num_spec
    fit_conc += (mol0[0] / vol0) * ((np.random.rand(*fit_conc.shape) - 0.5) * 2) * rand_fac
    fit_conc *= scale

    # Make numpy arrays into DataFrames for improved presentation
    t_df = pd.DataFrame(t, columns=['Time / ' + data.time_unit])
    fit_conc_headers = [i + ' fit conc. / ' + data.conc_unit for i in spec_name]
    fit_rate_headers = ['Reaction ' + str(i + 1) + ' fit rate / ' + data.rate_unit for i in range(len(k))]
    fit_conc_df = pd.DataFrame(fit_conc, columns=fit_conc_headers)
    fit_rate_df = pd.DataFrame(fit_rates, columns=fit_rate_headers)
    temp_df = pd.DataFrame(temp, columns=['Temperature / K'])
    all_df = pd.concat([t_df, fit_conc_df, fit_rate_df, temp_df], axis=1)

    data.add_fit(t_df, fit_conc_df, fit_rate_df, temp_df, all_df)
    return data
