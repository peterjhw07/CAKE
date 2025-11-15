"""
CAKE
Continuous Addition Kinetic Elucidation is a method for analyzing the kinetics of reactions
performed using continuous addition of a species.
"""

from cake.prep.import_data import import_data
from cake.sim import sim
from cake.fit import fit
from cake.fit_discrete_order import fit_discrete_order
from cake.plot import plot_conc_vs_time, plot_rate_vs_conc, plot_rate_vs_temp
from cake.plot import plot_discrete_order_2d, plot_discrete_order_3d
