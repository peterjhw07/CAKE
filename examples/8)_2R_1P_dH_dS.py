"""
CAKE Example 8
2 reactants, 1 catalyst, 1 product: Reactant 1 + Reactant 2 -> Product 1 + Product 2,
dH = -3000 J / mol, dS = -200 J / K / mol,
first orders in Reactant 1 and Catalyst, zero ord in Reactant 2.
CAKE reaction performed with continuous alteration of temperature.
The reaction is first simulated.
The simulation is repeated but with noise added, only Reactant 1 assumed to be detectable,
and instrument intensities converted to concentration via calibration.
These concentrations are then fitted, assuming temperature was recorded, and dH and dS are unknown.
"""

from cake import sim, fit

# Reaction conditions
t = (0, 240, 1 / 30)  # one reading every 2 s between 0-240 min
spec_name = ['Reactant 1', 'Reactant 2', 'Catalyst', 'Product 1', 'Product 2']
spec_type = ['r', 'r', 'c', 'p', 'p']  # species are reactants ('r'), catalysts ('c') and products ('p')
stoich = [1, 1, None, 1, 1]  # one Reactant 1 and one Reactant 2 produce one Product 1 and one Product 2
mol0 = [0.02, 0.04, 0, 0, 0]  # initially 0.02 mol Reactant 1, 0.04 mol Reactant 2, and 0 mol Catalyst, Products 1 and 2
vol0 = 0.4  # initially 0.4 L reaction volume
add_sol_conc = [None, None, 1, None, None]  # 1 M catalyst addition solution
disc_add_vol = [None, None, 0.00002, None, None]  # 0.00002 L discrete addition of catalyst solution
t_disc_add = [None, None, 5, None, None]  # discrete addition of catalyst solution at 5 min
temp0 = 293.15  # initial temperature 293.15 K
cont_temp_rate = 0.2  # temperature increase at 0.2 K / min
t_cont_temp = 5  # temperature alteration started at 5 min
dS = -200  # -200 J / K / mol
dH = -3000  # -3000 J / mol
constants = [dS, dH]
ord = [1, 0, 1, 0, 0]  # first order in reactant, zero order in product

time_unit = 'min'
conc_unit = 'M'

# Simulation of species concentrations versus time
sim_data = sim(t, spec_name=spec_name, spec_type=spec_type, stoich=stoich, mol0=mol0, vol0=vol0,
               add_sol_conc=add_sol_conc, disc_add_vol=disc_add_vol, t_disc_add=t_disc_add, temp0=temp0,
               cont_temp_rate=cont_temp_rate, t_cont_temp=t_cont_temp, rate_eq_type='Eyring', k=constants, ord=ord,
               time_unit=time_unit, conc_unit=conc_unit)
sim_data.plot_conc_vs_time(f_format='png', save_to='figures/8.1)_sim_conc_vs_time.png')

# Run instrument simulation
noise = 0.01  # 1% noise of maximum intensity

sim_data = sim(t, spec_name=spec_name, spec_type=spec_type, stoich=stoich, mol0=mol0, vol0=vol0,
               add_sol_conc=add_sol_conc, disc_add_vol=disc_add_vol, t_disc_add=t_disc_add, temp0=temp0,
               cont_temp_rate=cont_temp_rate, t_cont_temp=t_cont_temp, rate_eq_type='Eyring', k=constants, ord=ord,
               rand_fac=noise, time_unit=time_unit, conc_unit=conc_unit)
sim_data.plot_conc_vs_time(show_asp=['Reactant 1'], f_format='png',
                           save_to='figures/8.2)_sim_recorded_conc_vs_time.png')

# Run fitting
t_col = 0  # Time output as column 0
col = [1, None, None, None, None]  # Reactant 1 intensity output as column 1 and assumed no data for other species
ord_lim = [1, 0, 1, 0, 0]  # Fixing ord values for species

fit_data = fit(sim_data.all_df, spec_name=spec_name, spec_type=spec_type, stoich=stoich, t_col=t_col, col=col,
               mol0=mol0, vol0=vol0, add_sol_conc=add_sol_conc, disc_add_vol=disc_add_vol, t_disc_add=t_disc_add,
               temp0=temp0, cont_temp_rate=cont_temp_rate, t_cont_temp=t_cont_temp, rate_eq_type='Eyring', ord_lim=ord_lim,
               time_unit=time_unit, conc_unit=conc_unit)
fit_data.plot_conc_vs_time(show_asp='exp', f_format='png', save_to='figures/8.3)_fit_conc_vs_time.png')
print('Fitted dS: ' + str(fit_data.k_fit[0]))
print('Fitted dH: ' + str(fit_data.k_fit[1]))
