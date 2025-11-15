"""
CAKE Example 1
1 reactant, 1 product: 2 Reactant -> Product, rate constant (k) = 0.75 mM / h, first order in reactant.
The reaction is first simulated.
The simulation is repeated but with noise added.
These intensities are then fitted assuming only Reactant 1 was detectable, instrument intensities
were converted to concentration via calibration, and rate constant and reactant order are unknown.
"""

from cake import sim, fit

# Reaction conditions
t = (0, 5, 1 / 60)  # one reading every 1 min between 0-5 h
spec_name = ['Reactant', 'Product']
spec_type = ['r', 'p']  # species are reactants ('r') and products ('p')
stoich = [2, 1]  # 2 Reactant produce 1 Product
mol0 = [50, 0]  # initially 50 mmol Reactant, 0 mmol Product
vol0 = 0.1  # initially 0.1 L reaction volume
k = 0.75  # 0.75 mM / h
ord = [1, 0]  # first order in Reactant, zero order in Product

time_unit = 'h'
conc_unit = 'mM'

# Simulation of species concentrations versus time
sim_data = sim(t, spec_name=spec_name, spec_type=spec_type, stoich=stoich, mol0=mol0, vol0=vol0, k=k, ord=ord,
               time_unit=time_unit, conc_unit=conc_unit)
sim_data.plot_conc_vs_time(f_format='png', save_to='figures/1.1)_sim_conc_vs_time.png')

# Run fitting
t_col = 0  # Time output as column 0
col = [1, None]  # Reactant concentration output as column 1 and assumed no data for Product

fit_data = fit(sim_data.all_df, spec_name=spec_name, spec_type=spec_type, stoich=stoich, t_col=t_col, col=col,
               mol0=mol0, vol0=vol0, time_unit=time_unit, conc_unit=conc_unit)
fit_data.plot_conc_vs_time(show_asp='exp', f_format='png', save_to='figures/1.2)_fit_conc_vs_time.png')
print('Fitted k: ' + str(*fit_data.k_fit))
print('Fitted Reactant order: ' + str(*fit_data.ord_fit))
