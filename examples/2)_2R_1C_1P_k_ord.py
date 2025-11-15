"""
CAKE Example 2
2 reactants, 1 catalyst, 1 product: Reactant 1 + Reactant 2 -> Product, rate constant (k) = 0.003 mM / min,
first, zeroth, and first orders in Reactant 1, Reactant 2, and Catalyst respectively.
CAKE reaction performed with continuous addition of catalyst.
The reaction is first simulated.
The simulation is repeated but with noise added, only Reactant 1 assumed to be detectable,
and instrument intensities converted to concentration via calibration.
These concentrations are then fitted, assuming rate constant and Reactant 1 and Catalyst orders are unknown.
"""

from cake import sim, fit

# Reaction conditions
t = (0, 100, 1 / 12)  # one reading every 5 s between 0-100 min
spec_name = ['Reactant 1', 'Reactant 2', 'Catalyst', 'Product']
spec_type = ['r', 'r', 'c', 'p']  # species are reactants ('r'), catalysts ('c'), and products ('p')
stoich = [1, 1, None, 1]  # one Reactant 1 and one Reactant 2 produce one Product
mol0 = [10, 20, 0, 0]  # initially 10 mmol Reactant 1, 20 mmol Reactant 2, and 0 mmol Catalyst and Product
vol0 = 0.05  # initially 0.05 L reaction volume
add_sol_conc = [None, None, 1000, None]  # catalyst addition solution of concentration 1000 mM
cont_add_rate = [None, None, 0.00005, None]  # catalyst addition solution added at rate 0.00005 L / min
t_cont_add = [None, None, 5, None]  # catalyst addition solution started at 5 min
k = 0.003  # 0.003 mM / min
ord = [1, 0, 1, 0]  # first order in Reactant 1 and Catalyst, and zero order in Reactant 2 and Product

time_unit = 'min'
conc_unit = 'mM'

# Simulation of species concentrations versus time
sim_data = sim(t, spec_name=spec_name, spec_type=spec_type, stoich=stoich, mol0=mol0, vol0=vol0,
               add_sol_conc=add_sol_conc, cont_add_rate=cont_add_rate, t_cont_add=t_cont_add, k=k, ord=ord,
               time_unit=time_unit, conc_unit=conc_unit)
sim_data.plot_conc_vs_time(f_format='png', save_to='figures/2.1)_sim_conc_vs_time.png')

# Run instrument simulation
noise = 0.02  # 2% noise of maximum intensity

sim_data = sim(t, spec_name=spec_name, spec_type=spec_type, stoich=stoich, mol0=mol0, vol0=vol0,
               add_sol_conc=add_sol_conc, cont_add_rate=cont_add_rate, t_cont_add=t_cont_add, k=k, ord=ord,
               rand_fac=noise, time_unit=time_unit, conc_unit=conc_unit)
sim_data.plot_conc_vs_time(show_asp='Reactant 1', f_format='png',
                           save_to='figures/2.2)_sim_recorded_conc_vs_time.png')

# Run fitting
t_col = 0  # Time output as column 0
col = [1, None, None, None]  # Reactant intensity output as column 1 and assumed no data for other species
ord_lim = [None, 0, None, 0]  # Setting allowed order limits for Reactant 2 and Product as 0

fit_data = fit(sim_data.all_df, spec_name=spec_name, spec_type=spec_type, stoich=stoich, t_col=t_col, col=col,
               mol0=mol0, vol0=vol0, add_sol_conc=add_sol_conc, cont_add_rate=cont_add_rate, t_cont_add=t_cont_add,
               ord_lim=ord_lim, time_unit=time_unit, conc_unit=conc_unit)
fig = fit_data.plot_conc_vs_time(show_asp='exp', f_format='png', save_to='figures/2.3)_fit_conc_vs_time.png')
fit_data.plot_rate_vs_conc(show_asp='exp', f_format='png', save_to='figures/2.4)_fit_rate_vs_conc.png')
print('Fitted k: ' + str(fit_data.k_fit))
print('Fitted Reactant 1 order: ' + str(fit_data.ord_fit[0]))
print('Fitted Catalyst order: ' + str(fit_data.ord_fit[1]))
