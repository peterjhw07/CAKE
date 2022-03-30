import math
import numpy as np
import matplotlib.pyplot as plt

init_a = 200  # insert initial concentration of reactant a
init_b = 0  # insert initial concentration of reactant b
init_cat = 0  # insert initial concentration of catalyst
k = 0.00066  # insert rate constant in (M)^x s^-1

a_order = 1  # insert order of reaction wrt a
b_order = 0  # insert order of reaction wrt b
cat_order = 1  # insert order of reaction wrt cat

stoich_a = 1  # insert stoichiometry of a
stoich_b = 0  # insert stoichiometry of b
stoich_prod = 0.5  # insert stoichiometry of prod

alt_init_time = 2  # insert time at which species injection begins
time_unit = 0.1  # inset period of time in unit^-1
run_time = 120  # insert run time for simulation

add_conc_rate_a = 0  # insert addition rate of reactant a in unit^-1
add_conc_rate_b = 0  # insert addition rate of reactant b in unit^-1
add_conc_rate_cat = 2  # insert addition rate of catalyst in unit^-1

time_it = 0
a_conc_it = init_a
b_conc_it = init_b
cat_conc_it = init_cat
prod_conc_it = 0

it_num = run_time / time_unit
np.linspace(0, int(run_time), num=int(it_num) + 1)
pre_inj = np.repeat(np.array([[0, 0, 0]]), int(alt_init_time / time_unit), 0)
post_inj = np.repeat(np.array([[add_conc_rate_a, add_conc_rate_b, add_conc_rate_cat]]), int((run_time - alt_init_time) / time_unit) + 1, 0)
add_amounts = (np.row_stack((pre_inj, post_inj))) * time_unit
answer = np.zeros((int(it_num) + 1, 6))
for it in range(0, int(it_num) + 1):
    rate_inst = k * (a_conc_it ** a_order) * (b_conc_it ** b_order) * (cat_conc_it ** cat_order)
    answer[it, :] = [time_it, a_conc_it, b_conc_it, cat_conc_it, rate_inst, prod_conc_it]
    a_conc_it = max([a_conc_it - (rate_inst * time_unit) + (stoich_a * int(add_amounts[it, 0])), 0])
    b_conc_it = max([b_conc_it - (rate_inst * time_unit) + (stoich_b * int(add_amounts[it, 1])), 0])
    cat_conc_it = cat_conc_it + add_amounts[it, 2]
    prod_conc_it = prod_conc_it + (stoich_prod * (rate_inst * time_unit))
    time_it = time_it + time_unit
    ...
data = np.row_stack([["time", "[a]", "[b]", "[cat]", "rate", "[p]"], answer])
start_max = max(init_a, init_b, init_cat)
# print(data)
plt.plot(answer[:, 0], answer[:, 1]*100/start_max)
if init_b > 0 or add_conc_rate_b > 0:
    plt.plot(answer[:, 0], answer[:, 2] * 100 / start_max)
    ...
plt.plot(answer[:, 0], answer[:, 3]*100/start_max)
plt.plot(answer[:, 0], answer[:, 5]*100/start_max)
plt.xlim([0, run_time])
# plt.ylim([0, max(answer[:, [1, 2, 3, 5]])])
plt.xlabel('Time / min')
plt.ylabel('Relative amount / %')
if init_b > 0 or add_conc_rate_b > 0:
    plt.legend(["Reactant A", "Reactant B", "Catalyst", "Product"])
else:
    plt.legend(["Reactant", "Catalyst", "Product"])
    ...
plt.show()
