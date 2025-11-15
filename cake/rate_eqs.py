"""CAKE Rate Equations"""

import math
import numpy as np


# Custom rate equation
def rate_eq_custom(k, conc, ord_loc, ord, temp):
    # print('Manipulate equation using k[i], conc[i], ord_loc[i], ord[i] and temp as required')
    return (k[0] * conc[0]) / (k[1] + conc[0])


# Standard rate equation
def rate_eq_standard(k, conc, ord_loc, ord, temp):
    return k[0] * np.prod([conc[i] ** ord[i] for i in ord_loc])


# Arrhenius rate equation (k is proportional to A and exp(-Ea / RT))
def rate_eq_Arrhenius(k, conc, ord_loc, ord, temp):
    return k[0] * math.exp(-k[1] / (temp * 8.314462)) * np.prod([conc[i] ** ord[i] for i in ord_loc])


# Eyring rate equation (k is proportional to exp(dS / R) and exp(-dH / RT))
def rate_eq_Eyring(k, conc, ord_loc, ord, temp):
    return ((1 * 1.380649E-23 * temp) / 6.626070E-34) * math.exp(k[0] / 8.314462) * math.exp(-k[1] / (temp * 8.314462)) * np.prod([conc[i] ** ord[i] for i in ord_loc])


# Michaelis-Menten rate equation
def rate_eq_MM(k, conc, ord_loc, ord, temp):
    return k[0] * (conc[0] ** ord[0]) * (conc[3] ** ord[3]) * (conc[1] / (k[1] + conc[1]))


# Rate calculator
def rate_calc(k, conc, ord, ord_loc, temp, rate_eq):
    return [rate_eq(j, conc, ord_loc[i], ord[i], temp) for i, j in enumerate(k)]


# Map to get required rate equation
rate_eq_map = {
    'standard': rate_eq_standard,
    'arrhenius': rate_eq_Arrhenius,
    'eyring': rate_eq_Eyring,
    'michaelis-menten': rate_eq_MM,
    'custom': rate_eq_custom,
}
