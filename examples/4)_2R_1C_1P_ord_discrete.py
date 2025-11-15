"""
CAKE Example 4
2 reactants, 1 catalyst, 1 product: Reactant 1 + Reactant 2 -> Product, rate constant (k) = 0.1 mM / min,
first, zeroth, and first orders in Reactant 1, Reactant 2, and Catalyst respectively.
CAKE reaction performed with continuous addition of catalyst.
The reaction is first simulated.
The simulation is repeated but with noise added, concentrations multiplied by a random factor to emulate an uncalibrated
but linear intensity detected by machine, and only Reactant 1 and Product intensities assumed to be detectable.
These intensities are then fitted, assuming rate constant and Reactant 1 and Catalyst orders are unknown.
Additionally, Reactant 1 and Catalyst orders are screened at discrete intervals.
"""

import random
from cake import sim, fit_discrete_order

# Reaction conditions
t = (0, 50, 1 / 12)  # one reading every 5 s between 0-100 min
spec_name = ['Reactant 1', 'Reactant 2', 'Catalyst', 'Product']
spec_type = ['r', 'r', 'c', 'p']  # species are reactants ('r'), catalysts ('c'), and products ('p')
stoich = [1, 1, None, 1]  # one Reactant 1 and one Reactant 2 produce one Product
mol0 = [5, 10, 0, 0]  # initially 5 mmol Reactant 1, 10 mmol Reactant 2, and 0 mmol Catalyst and Product
mol_end = [None, None, None, 5]  # final Product molarity expected to be 5 mmol
vol0 = 0.025  # initially 0.025 L reaction volume
add_sol_conc = [None, None, 100, None]  # catalyst addition solution of concentration 100 mM
cont_add_rate = [None, None, 0.00005, None]  # catalyst addition solution added at rate 0.00005 L / min
t_cont_add = [None, None, 5, None]  # catalyst addition solution started at 5 min
k = 0.1  # 0.1 mM / min
ord = [1, 0, 1, 0]  # first order in Reactant 1 and Catalyst, zero order in Reactant 2 and Product

time_unit = 'min'
conc_unit = 'mM'

# Simulation of species concentrations versus time
sim_data = sim(t, spec_name=spec_name, spec_type=spec_type, stoich=stoich, mol0=mol0, vol0=vol0,
               add_sol_conc=add_sol_conc, cont_add_rate=cont_add_rate, t_cont_add=t_cont_add, k=k, ord=ord,
               time_unit=time_unit, conc_unit=conc_unit)
sim_data.plot_conc_vs_time(f_format='png', save_to='figures/4.1)_sim_conc_vs_time.png')

# Run instrument simulation
noise = 0.02  # 2% noise of maximum intensity
scale = random.uniform(0.01, 100)  # random scale between 0.01-100 to emulate arbitrary instrument intensity

sim_data = sim(t, spec_name=spec_name, spec_type=spec_type, stoich=stoich, mol0=mol0, vol0=vol0,
               add_sol_conc=add_sol_conc, cont_add_rate=cont_add_rate, t_cont_add=t_cont_add, k=k, ord=ord,
               rand_fac=noise, time_unit=time_unit, conc_unit=conc_unit)
sim_data.plot_conc_vs_time(show_asp=['Reactant 1', 'Product'], conc_axis_label='Intensity', conc_unit='AU',
                           f_format='png', save_to='figures/4.2)_sim_intensity_vs_time.png')

# Run fitting
t_col = 0  # Time output as column 0
col = [1, None, None, 4]  # # Reactant 1 and Product concentrations output as columns 1 and 4
scale_avg_num = 10  # Estimate maximum intensities based on last 10 data points
ord_lim = [None, 0, None, 0]  # Setting allowed ord limits for Reactant 2 and Product as 0
ord_inc = 11  # Number of orders to test

fit_data = fit_discrete_order(sim_data.all_df, spec_name=spec_name, spec_type=spec_type, stoich=stoich, t_col=t_col,
                              col=col, mol0=mol0, mol_end=mol_end, vol0=vol0, add_sol_conc=add_sol_conc,
                              cont_add_rate=cont_add_rate, t_cont_add=t_cont_add, ord_num=ord_lim, ord_inc=ord_inc,
                              scale_avg_num=scale_avg_num, time_unit=time_unit, conc_unit=conc_unit)
fit_data.plot_conc_vs_time(show_asp='exp', f_format='png', save_to='figures/4.3)_best_fit_conc_vs_time.png')
fit_data.plot_discrete_order_2d(show_asp='exp', metric='RSS', cutoff=25000, f_format='png',
                                save_to='figures/4.4)_discrete_order_fits_conc_vs_time.png')
fit_data.plot_discrete_order_3d(method='exact_contour', metric='RSS',
                                levels=[0, 7000, 10000, 25000, 50000, 100000, 1000000, 10000000],
                                f_format='png', save_to='figures/4.5)_discrete_order_fits_contour.png')
fit_data.plot_discrete_order_3d(method='3d', metric='RSS', cutoff=25000, f_format='png',
                                save_to='figures/4.6)_discrete_order_fits_3d.png')
print('Best fit k: ' + str(fit_data.k_fit))
print('Best Reactant 1 order: ' + str(fit_data.ord_fit[0]))
print('Best Catalyst order: ' + str(fit_data.ord_fit[1]))
print(fit_data.discrete_order_fit_df[['Order', 'k', 'RSS']])
