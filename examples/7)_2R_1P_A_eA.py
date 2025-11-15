"""
CAKE Example 7
2 reactants, 1 product: Reactant 1 + Reactant 2 -> 2 Product,
pre-exponential factor = 150000 M / h, activation energy = 28000 J / mol,
first orders in Reactant 1 and Reactant 2.
CAKE reaction performed with continuous alteration of temperature.
The reaction is first simulated.
The simulation is repeated but with noise added, all concentrations assumed to be detectable,
and instrument intensities converted to concentration via calibration.
These concentrations are then fitted, and pre-exponential factor and activation energy are unknown.
"""

from cake import sim, fit

# Reaction conditions
t = (0, 10, 1 / 60)  # one reading every 1 min between 0-10 h
spec_name = ['Reactant 1', 'Reactant 2', 'Product']
spec_type = ['r', 'r', 'p']  # species are reactants ('r') and products ('p')
stoich = [1, 1, 2]  # one Reactant 1 and one Reactant 2 produce two Product
mol0 = [0.1, 0.12, 0]  # initially 0.1 mol Reactant 1, 0.12 mol Reactant 2, and 0 mol Product
vol0 = 0.2  # initially 0.2 L reaction volume
temp0 = 293.15  # initial temperature 293.15 K
cont_temp_rate = 10  # temperature increase at 10 K / h
t_cont_temp = 5 / 60  # temperature alteration started at 5 min
A = 150000  # 150000 M / h pre-exponential factor
eA = 28000  # 28000 J / mol activation energy
constants = [A, eA]
ord = [1, 1, 0]  # first order in reactant, zero order in product

time_unit = 'h'
conc_unit = 'M'

# Simulation of species concentrations versus time
sim_data = sim(t, spec_name=spec_name, spec_type=spec_type, stoich=stoich, mol0=mol0, vol0=vol0, temp0=temp0,
               cont_temp_rate=cont_temp_rate, t_cont_temp=t_cont_temp, rate_eq_type='Arrhenius', k=constants, ord=ord,
               time_unit=time_unit, conc_unit=conc_unit)
sim_data.plot_conc_vs_time(f_format='png', save_to='figures/7.1)_sim_conc_vs_time.png')

# Run instrument simulation
noise = 0.02  # 2% noise of maximum intensity

sim_data = sim(t, spec_name=spec_name, spec_type=spec_type, stoich=stoich, mol0=mol0, vol0=vol0, temp0=temp0,
               cont_temp_rate=cont_temp_rate, t_cont_temp=t_cont_temp, rate_eq_type='Arrhenius', k=constants, ord=ord,
               rand_fac=noise, time_unit=time_unit, conc_unit=conc_unit)
sim_data.plot_conc_vs_time(f_format='png', save_to='figures/7.2)_sim_recorded_conc_vs_time.png')

# Run fitting
t_col = 0  # Time output as column 0
col = [1, 2, 3]  # Reactant intensity output as column 1 and assumed no data for Product
ord_lim = [1, 1, 0]  # Fixing ord values for Reactant 1, Reactant 2 and Product

fit_data = fit(sim_data.all_df, spec_name=spec_name, spec_type=spec_type, stoich=stoich, t_col=t_col, col=col,
               mol0=mol0, vol0=vol0, temp0=temp0, cont_temp_rate=cont_temp_rate, t_cont_temp=t_cont_temp,
               rate_eq_type='Arrhenius', ord_lim=ord_lim, time_unit=time_unit, conc_unit=conc_unit)
fit_data.plot_conc_vs_time(f_format='png', save_to='figures/7.3)_fit_conc_vs_time.png')
fit_data.plot_rate_vs_temp(rate_constant_unit='M$^{-1}$ h$^{-1}$',
                           f_format='png', save_to='figures/7.4)_fit_rate_vs_temp.png')
print('Fitted A: ' + str(fit_data.k_fit[0]))
print('Fitted eA: ' + str(fit_data.k_fit[1]))
