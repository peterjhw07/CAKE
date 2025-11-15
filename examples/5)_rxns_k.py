"""
CAKE Example 5
Multistep reaction system:
R1 + C -> I, rate constant (k) = 0.01 mM / min;
I + R2 -> P + C, rate constant (k) = 0.10 mM / min.
CAKE reaction performed with continuous addition of C.
The reaction is first simulated.
The simulation is repeated but with noise added, reactant concentrations multiplied by a random factor to emulate an
uncalibrated but linear intensity detected by machine, and only the reactant concentration assumed to be detectable.
These intensities are then fitted, assuming rate constants are unknown.
"""

import random
from cake import sim, fit

# Reaction conditions
t = (0, 100, 1 / 30)  # one reading every 2 s between 0-100 min
spec_name = ['R1', 'R2', 'C']  # species with conditions needing input
rxns = ['R1 + C -> I', 'I + R2 -> P + C']  # reaction mechanism
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
               cont_add_rate=cont_add_rate, t_cont_add=t_cont_add, k=k, time_unit=time_unit, conc_unit=conc_unit)
sim_data.plot_conc_vs_time(f_format='png', save_to='figures/5.1)_sim_conc_vs_time.png')

# Run instrument simulation
noise = 0.02  # 2% noise of maximum intensity
scale = random.uniform(0.01, 100)  # random scale between 0.01-100 to emulate arbitrary instrument intensity

sim_data = sim(t, spec_name=spec_name, rxns=rxns, mol0=mol0, vol0=vol0, add_sol_conc=add_sol_conc,
               cont_add_rate=cont_add_rate, t_cont_add=t_cont_add, k=k, rand_fac=noise, scale=scale,
               time_unit=time_unit, conc_unit=conc_unit)
sim_data.plot_conc_vs_time(show_asp='R1', conc_axis_label='Intensity', conc_unit='AU', f_format='png',
                           save_to='figures/5.2)_sim_intensity_vs_time.png')

# Run fitting
t_col = 0  # Time output as column 0
col = [1, None, None]  # Reactant intensity output as column 1 and assumed no data for R2 or C
scale_avg_num = 30  # Estimate maximum intensities based on first 30 data points

fit_data = fit(sim_data.all_df, spec_name=spec_name, rxns=rxns, t_col=t_col, col=col, mol0=mol0, vol0=vol0,
               add_sol_conc=add_sol_conc, cont_add_rate=cont_add_rate, t_cont_add=t_cont_add,
               scale_avg_num=scale_avg_num, time_unit=time_unit, conc_unit=conc_unit)
fit_data.plot_conc_vs_time(show_asp='exp', f_format='png',
                           save_to='figures/5.3)_fit_conc_vs_time.png')
print('First fitted rate constant: ' + str(fit_data.k_fit[0]))
print('Second fitted rate constant: ' + str(fit_data.k_fit[1]))
