"""
CAKE Example 3
2 reactants, 1 catalyst, 1 product: Reactant 1 + Reactant 2 -> Product, rate constant (k) = 0.5 M / min,
zeroth, first, and first orders in Reactant 1, Reactant 2, and Catalyst respectively.
CAKE reaction performed with continuous addition of catalyst.
The reaction is first simulated, with the sample solution lost to an instrument at 10 uL / min.
The simulation is repeated but with noise added, only Reactant 1 and Product assumed to be detectable,
and instrument intensities converted to concentration via calibration.
These concentrations are then fitted, assuming rate constant and Reactant 1 and Catalyst orders are unknown.
"""

from cake import sim, fit

# Reaction conditions
t = (0, 100, 1 / 12)  # one reading every 5 s between 0-100 min
spec_name = ['Reactant 1', 'Reactant 2', 'Catalyst', 'Product']
spec_type = ['r', 'r', 'c', 'p']  # species are reactants ('r'), catalysts ('c'), and products ('p')
stoich = [1, 1, None, 1]  # one Reactant 1 and one Reactant 2 produce 1 Product
mol0 = [3, 6, 0, 0]  # initially 3 mmol Reactant 1, 6 mmol Reactant 2, and 0 mmol Catalyst and Product
vol0 = 100  # initial reaction volume 100 mL
add_sol_conc = [None, None, 1, None]  # catalyst addition solution of concentration 1 M
cont_add_rate = [None, None, 0.05, None]  # catalyst addition solution added at rate 0.05 mL / min
cont_sub_rate = 0.01  # sample solution lost to instrument at 0.01 mL / min
t_cont_add = [None, None, 5, None]  # catalyst addition solution started at 5 min
k = 0.5  # 0.5 mol M / min
ord = [0.001, 1, 1, 0]  # 'zero' order in Reactant 1 and Product, and first order in Reactant 2 and Catalyst

time_unit = 'min'
conc_unit = 'M'

# Simulation of species concentrations versus time
sim_data = sim(t, spec_name=spec_name, spec_type=spec_type, stoich=stoich, mol0=mol0, vol0=vol0,
               add_sol_conc=add_sol_conc, cont_add_rate=cont_add_rate, t_cont_add=t_cont_add,
               cont_sub_rate=cont_sub_rate, k=k, ord=ord, time_unit=time_unit, conc_unit=conc_unit)
sim_data.plot_conc_vs_time(f_format='png', save_to='figures/3.1)_sim_conc_vs_time.png')

# Run instrument simulation
noise = 0.05  # 5% noise of maximum intensity

sim_data = sim(t, spec_name=spec_name, spec_type=spec_type, stoich=stoich, mol0=mol0, vol0=vol0,
               add_sol_conc=add_sol_conc, cont_add_rate=cont_add_rate, t_cont_add=t_cont_add,
               cont_sub_rate=cont_sub_rate, k=k, ord=ord, rand_fac=noise, time_unit=time_unit, conc_unit=conc_unit)
sim_data.plot_conc_vs_time(show_asp=['Reactant 1', 'Product'],
                           f_format='png', save_to='figures/3.2)_sim_recorded_conc_vs_time.png')

# Run fitting
t_col = 0  # Time output as column 0
col = [1, None, None, 4]  # Reactant 1 and Product concentrations output as columns 1 and 4
ord_lim = [None, 1, None, 0]  # Setting allowed ord limits for Reactant 2 as 1 and Product as 0

fit_data = fit(sim_data.all_df, spec_name=spec_name, spec_type=spec_type, stoich=stoich, t_col=t_col, col=col,
               mol0=mol0, vol0=vol0, add_sol_conc=add_sol_conc, cont_add_rate=cont_add_rate, t_cont_add=t_cont_add,
               cont_sub_rate=cont_sub_rate, ord_lim=ord_lim, time_unit=time_unit, conc_unit=conc_unit)
fit_data.plot_conc_vs_time(show_asp='exp', f_format='png', save_to='figures/3.3)_fit_conc_vs_time.png')
print('Fitted k: ' + str(fit_data.k_fit))
print('Fitted Reactant 1 order: ' + str(fit_data.ord_fit[0]))
print('Fitted Catalyst order: ' + str(fit_data.ord_fit[1]))
