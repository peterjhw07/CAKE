"""
CAKE Example 6
Multistep reaction system:
R1 + C -> I, rate constant (k) = 0.01 mM / min;
I + R2 -> P + C, rate constant (k) = 0.10 mM / min.
CAKE reaction performed with continuous addition of C.
The reaction is first simulated.
The simulation is repeated but with noise added, only R1 concentration assumed to be detectable,
and instrument intensities converted to concentration via calibration.
These concentrations are then fitted, assuming rate constants and R1 and C orders are unknown.
"""

from cake import sim, fit

# Reaction conditions
t = (0, 100, 1 / 6)  # one reading every 10 s between 0-100 min
spec_name = ['R1', 'R2', 'C']  # species with conditions needing input
spec_type = ['r', 'r', 'c']  # species are reactants ('r'), catalysts ('c'), and products ('p')
stoich = [1, 1, None]  # one Reactant 1 and one Reactant 2 produce one Product (not described)
rxns = ['R1 + C  -> I', 'I + R2 -> P + C']  # reaction mechanism
mol0 = [10, 15, 0]  # initially 10 mmol R1, 15 mmol R2, and 0 mmol C
vol0 = 0.1  # initially 0.1 L reaction volume
add_sol_conc = [None, None, 1000]  # catalyst addition solution of concentration 1000 mM
cont_add_rate = [None, None, 0.00005]  # catalyst addition solution added at rate 0.00005 L / min
t_cont_add = [None, None, 5]  # catalyst addition solution started at 1 min
k = [[0.01], [0.10]]  # 0.01 and 0.10 mM / min for each reaction respectively

time_unit = 'min'
conc_unit = 'mM'

# Simulation of species concentrations versus time
sim_data = sim(t, spec_name=spec_name, rxns=rxns, mol0=mol0, vol0=vol0, add_sol_conc=add_sol_conc,
               cont_add_rate=cont_add_rate, t_cont_add=t_cont_add, k=k, ord=ord, time_unit=time_unit,
               conc_unit=conc_unit)
sim_data.plot_conc_vs_time(f_format='png', save_to='figures/6.1)_sim_conc_vs_time.png')

# Run instrument simulation
noise = 0.05  # 5% noise of maximum intensity

sim_data = sim(t, spec_name=spec_name, rxns=rxns, mol0=mol0, vol0=vol0, add_sol_conc=add_sol_conc,
               cont_add_rate=cont_add_rate, t_cont_add=t_cont_add, k=k, ord=ord, rand_fac=noise, time_unit=time_unit,
               conc_unit=conc_unit)
sim_data.plot_conc_vs_time(show_asp='R1', f_format='png', save_to='figures/6.2)_sim_recorded_conc_vs_time.png')

# Run fitting
t_col = 0  # Time output as column 0
col = [1, None, None]  # Reactant intensity output as column 1 and assumed no data for R2 or C
ord_lim = [None, 0, None]  # Setting allowed ord limits for R2 as 0, fitting orders for R1 and C

fit_data = fit(sim_data.all_df, spec_name=spec_name, spec_type=spec_type, stoich=stoich, t_col=t_col, col=col,
               mol0=mol0, vol0=vol0, add_sol_conc=add_sol_conc, cont_add_rate=cont_add_rate, t_cont_add=t_cont_add,
               ord_lim=ord_lim, time_unit=time_unit, conc_unit=conc_unit)
fit_data.plot_conc_vs_time(show_asp='exp', f_format='png',
                           save_to='figures/6.3)_fit_conc_vs_time.png')
print('Fitted k: ' + str(fit_data.k_fit))
print('Fitted R1 order: ' + str(fit_data.ord_fit[0]))
print('Fitted C order: ' + str(fit_data.ord_fit[1]))
