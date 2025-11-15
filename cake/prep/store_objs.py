"""CAKE Object Storage"""

import copy
from cake.plot.conc_vs_time import plot_conc_vs_time
from cake.plot.rate_vs_conc import plot_rate_vs_conc
from cake.plot.rate_vs_temp import plot_rate_vs_temp
from cake.plot.discrete_order import plot_discrete_order_2d, plot_discrete_order_3d


# Sim data object
class SimData:
    def __init__(self, num_spec, spec_name, spec_type, mol0, stoich, stoich_loc, temp, disc_event, cont_event,
                 k_lim, ord_lim, pois_lim, ord_loc, col, inc, rate_eq_type, rate_eq, ode_func, rate_method, rtol, atol,
                 t, time_unit, conc_unit):
        self.num_spec = num_spec
        self.spec_name = spec_name
        self.spec_type = spec_type
        self.mol0 = mol0
        self.stoich = stoich
        self.stoich_loc = stoich_loc
        self.temp = temp
        self.disc_event = disc_event
        self.cont_event = cont_event
        self.k_lim = k_lim
        self.ord_lim = ord_lim
        self.pois_lim = pois_lim
        self.ord_loc = ord_loc
        self.col = col
        self.inc = inc
        self.rate_eq_type = rate_eq_type
        self.rate_eq = rate_eq
        self.ode_func = ode_func
        self.rate_method = rate_method
        self.rtol = rtol
        self.atol = atol
        self.t = t
        self.time_unit = time_unit
        self.conc_unit = conc_unit
        self.rate_unit = conc_unit + ' ' + time_unit + '$^{-1}$'

    # Add fitted data
    def add_fit(self, t_df, fit_conc_df, fit_rate_df, temp_df, all_df):
        self.t_df = t_df
        self.fit_conc_df = fit_conc_df
        self.fit_rate_df = fit_rate_df
        self.temp_df = temp_df
        self.all_df = all_df

    # Object link for simple plotting
    def plot_conc_vs_time(self, conc_unit='', conc_axis_label='Concentration',
                          show_asp='all', method='lone', f_format='svg', save_to='cake_conc_vs_time.svg',
                          return_fig=False, return_img=False, transparent=False):
        if not conc_unit:
            conc_unit = self.conc_unit
        img, mimetype = plot_conc_vs_time(self.t_df, fit_conc_df=self.fit_conc_df, temp_df=self.temp_df,
                        show_asp=show_asp, time_unit=self.time_unit, conc_unit=conc_unit,
                        conc_axis_label=conc_axis_label, method=method, f_format=f_format, save_to=save_to,
                        return_fig=return_fig, return_img=return_img, transparent=transparent)
        return img, mimetype

    def plot_rate_vs_conc(self, show_asp='all', f_format='svg', save_to='cake_conc_vs_rate.svg',
                          return_fig=False, return_img=False, transparent=False):
        img, mimetype = plot_rate_vs_conc(self.t_df, self.fit_conc_df, self.fit_rate_df, self.k_lim, self.ord_lim,
                        self.ord_loc, self.rate_eq_type, temp0=self.temp[0], temp_df=self.temp_df, show_asp=show_asp,
                          rate_unit=self.rate_unit, f_format=f_format, save_to=save_to, return_fig=return_fig,
                          return_img=return_img, transparent=transparent)
        return img, mimetype

    def plot_rate_vs_temp(self, show_asp='all', rate_constant_unit='rate_constant_unit',
                          f_format='svg', save_to='cake_conc_vs_rate.svg',
                          return_fig=False, return_img=False, transparent=False):
        img, mimetype = plot_rate_vs_temp(self.t_df, self.fit_conc_df, self.fit_rate_df, self.temp_df, self.stoich,
                        self.stoich_loc, self.k_lim, self.rate_eq_type, show_asp=show_asp,
                        rate_constant_unit=rate_constant_unit, f_format=f_format, save_to=save_to,
                        return_fig=return_fig, return_img=return_img, transparent=transparent)
        return img, mimetype


# Fit data object
class FitData:
    def __init__(self, num_spec, spec_name, spec_type, fit_asp, mol0, mol_end, stoich, stoich_loc, temp, disc_event,
                 cont_event, k_lim, ord_lim, pois_lim, ord_loc, col, inc, rate_eq_type, rate_eq, ode_func, rate_method,
                 rtol, atol, fix_pois_locs, var_locs, fit_asp_locs, fit_param_locs, t, t_add, t_add_to_fit, t_split,
                 y_data_to_fit, data_mod, col_ext, exp_conc, exp_rates, time_unit, conc_unit):
        self.num_spec = num_spec
        self.spec_name = spec_name
        self.spec_type = spec_type
        self.fit_asp = fit_asp
        self.mol0 = mol0
        self.mol_end = mol_end
        self.stoich = stoich
        self.stoich_loc = stoich_loc
        self.temp = temp
        self.disc_event = disc_event
        self.cont_event = cont_event
        self.k_lim = k_lim
        self.ord_lim = ord_lim
        self.pois_lim = pois_lim
        self.ord_loc = ord_loc
        self.col = col
        self.inc = inc
        self.rate_eq_type = rate_eq_type
        self.rate_eq = rate_eq
        self.ode_func = ode_func
        self.rate_method = rate_method
        self.rtol = rtol
        self.atol = atol
        self.fix_pois_locs = fix_pois_locs
        self.var_locs = var_locs
        self.fit_asp_locs = fit_asp_locs
        self.fit_param_locs = fit_param_locs
        self.t = t
        self.t_add = t_add
        self.t_add_to_fit = t_add_to_fit
        self.t_split = t_split
        self.y_data_to_fit = y_data_to_fit
        self.data_mod = data_mod
        self.col_ext = col_ext
        self.exp_conc = exp_conc
        self.exp_rates = exp_rates
        self.time_unit = time_unit
        self.conc_unit = conc_unit
        self.rate_unit = conc_unit + ' ' + time_unit + '$^{-1}$'

    # Add fitted data
    def add_fit(self, t_df, exp_conc_df, exp_rate_df, fit_conc_df, fit_rate_df,
                temp_df, all_df, k_est, k_fit, k_fit_err, ord_fit, ord_fit_err, pois_fit, pois_fit_err,
                rss, r2, r2_adj, rmse, mae, aic, bic):
        self.t_df = t_df
        self.exp_conc_df = exp_conc_df
        self.exp_rate_df = exp_rate_df
        self.fit_conc_df = fit_conc_df
        self.fit_rate_df = fit_rate_df
        self.temp_df = temp_df
        self.all_df = all_df
        self.k_est = k_est
        self.k_fit = k_fit
        self.k_fit_err = k_fit_err
        self.ord_fit = ord_fit
        self.ord_fit_err = ord_fit_err
        self.pois_fit = pois_fit
        self.pois_fit_err = pois_fit_err
        self.rss = rss
        self.r2 = r2
        self.r2_adj = r2_adj
        self.rmse = rmse
        self.mae = mae
        self.aic = aic
        self.bic = bic
        self.df_data = [t_df, exp_conc_df, exp_rate_df, fit_conc_df, fit_rate_df, temp_df, all_df]
        self.res = [k_fit, k_fit_err, ord_fit, ord_fit_err, pois_fit, pois_fit_err,
                    rss, r2, r2_adj, rmse, mae, aic, bic]

        self.k_fit_all = copy.deepcopy(self.k_lim)  # added to prevent editing of variables
        self.ord_fit_all = copy.deepcopy(self.ord_lim)  # added to prevent editing of variables
        for l, (j, m) in enumerate(self.var_locs[0]):
            self.k_fit_all[j] = [self.k_fit[l] if idx == m else val for idx, val in enumerate(self.k_fit_all[j])]
        for l, (j, m) in enumerate(self.var_locs[1]):
            self.ord_fit_all[j] = [self.ord_fit[l] if idx == m else val for idx, val in enumerate(self.ord_fit_all[j])]

    # Add fitted dicrete order data
    def add_discrete_order(self, discrete_order_fit_df, discrete_order_fit_conc_df, discrete_order_fit_rate_df):
        self.discrete_order_fit_df = discrete_order_fit_df
        self.discrete_order_fit_conc_df = discrete_order_fit_conc_df
        self.discrete_order_fit_rate_df = discrete_order_fit_rate_df

    # Object link for simple plotting
    def plot_conc_vs_time(self, show_asp='all', method='lone', f_format='svg', save_to='cake_conc_vs_time.svg',
                          return_fig=False, return_img=False, transparent=False):
        img, mimetype = plot_conc_vs_time(self.t_df, exp_conc_df=self.exp_conc_df, fit_conc_df=self.fit_conc_df,
                        temp_df=self.temp_df, show_asp=show_asp, time_unit=self.time_unit, conc_unit=self.conc_unit,
                        method=method, f_format=f_format, save_to=save_to, return_fig=return_fig, return_img=return_img,
                        transparent=transparent)
        return img, mimetype

    def plot_rate_vs_conc(self, show_asp='all', f_format='svg', save_to='cake_conc_vs_rate.svg',
                          return_fig=False, return_img=False, transparent=False):
        img, mimetype = plot_rate_vs_conc(self.t_df, self.fit_conc_df, self.fit_rate_df, self.k_fit_all,
                        self.ord_fit_all, self.ord_loc, self.rate_eq_type, exp_conc_df=self.exp_conc_df,
                        temp_df=self.temp_df, show_asp=show_asp, rate_unit=self.rate_unit, f_format=f_format,
                        save_to=save_to, return_fig=return_fig, return_img=return_img, transparent=transparent)
        return img, mimetype

    def plot_rate_vs_temp(self, show_asp='all', rate_constant_unit='rate_constant_unit',
                          f_format='svg', save_to='cake_conc_vs_rate.svg',
                          return_fig=False, return_img=False, transparent=False):
        img, mimetype = plot_rate_vs_temp(self.t_df, self.fit_conc_df, self.fit_rate_df, self.temp_df, self.stoich,
                        self.stoich_loc, self.k_fit_all, self.rate_eq_type, exp_conc_df=self.exp_conc_df,
                        show_asp=show_asp, rate_constant_unit=rate_constant_unit, f_format=f_format, save_to=save_to,
                        return_fig=return_fig, return_img=return_img, transparent=transparent)
        return img, mimetype

    def plot_discrete_order_2d(self, method='lone', show_asp='all', metric=None, cutoff=None,
                               f_format='svg', save_to='cake_discrete_fit_2d.svg',
                               return_fig=False, return_img=False, transparent=False):
        img, mimetype = plot_discrete_order_2d(self.t_df, self.exp_conc_df, self.discrete_order_fit_conc_df,
                        self.temp_df, self.discrete_order_fit_df, show_asp=show_asp, method=method, metric=metric,
                        cutoff=cutoff, time_unit=self.time_unit, conc_unit=self.conc_unit, f_format=f_format,
                        save_to=save_to, return_fig=return_fig, return_img=return_img, transparent=transparent)
        return img, mimetype

    def plot_discrete_order_3d(self, method='contour', metric='RSS', cutoff=None, levels=None,
                               f_format='svg', save_to='cake_discrete_fit_3d.svg',
                               return_fig=False, return_img=False, transparent=False):
        img, mimetype = plot_discrete_order_3d(self.discrete_order_fit_df, method=method, metric=metric, cutoff=cutoff,
                        levels=levels, f_format=f_format, save_to=save_to, return_fig=return_fig, return_img=return_img,
                        transparent=transparent)
        return img, mimetype


# Fit summary object
class FitSummary:
    def __init__(self, t, exp_conc, exp_rate, fit_conc, fit_rate, k_val, k_fit, k_fit_err, ord_fit, ord_fit_err,
                 pois_fit, pois_fit_err, rss, r2, r2_adj, rmse, mae, aic, bic, temp, col, ord_lim):
        self.t = t
        self.exp_conc = exp_conc
        self.exp_rate = exp_rate
        self.fit_conc = fit_conc
        self.fit_rate = fit_rate
        self.k_est = k_val
        self.k_fit = k_fit
        self.k_fit_err = k_fit_err
        self.ord_fit = ord_fit
        self.ord_fit_err = ord_fit_err
        self.pois_fit = pois_fit
        self.pois_fit_err = pois_fit_err
        self.rss = rss
        self.r2 = r2
        self.r2_adj = r2_adj
        self.rmse = rmse
        self.mae = mae
        self.aic = aic
        self.bic = bic
        self.temp = temp
        self.col = col
        self.ord_lim = ord_lim

    def __repr__(self):
        return (
            f"Optimization Result:\n"
            f"  t: {self.t, self.exp_conc}\n"
            f"  Experimental concentration(s): {self.exp_conc}\n"
            f"  Experimental rate: {self.exp_rate}\n"
            f"  Fitted concentration(s): {self.fit_conc}\n"
            f"  Fitted rate: {self.fit_rate}\n"
            f"  Initial constant estimate(s): {self.k_est}\n"
            f"  Fitted constant(s): {self.k_fit}\n"
            f"  Fitted constant error(s): {self.k_fit_err}\n"
            f"  Fitted order(s): {self.ord_fit}\n"
            f"  Fitted order errors(s): {self.ord_fit_err}\n"
            f"  Fitted poisoning(s): {self.pois_fit}\n"
            f"  Fitted poisoning error(s): {self.pois_fit_err}\n"
            f"  Residual sum of squares: {self.rss}\n"
            f"  R squared (not recommended): {self.r2}\n"
            f"  R squared adjusted (not recommended): {self.r2_adj}\n"
            f"  Root mean square error: {self.rmse}\n"
            f"  Mean average error: {self.mae}\n"
            f"  Akaike information criterion: {self.aic}\n"
            f"  Bayesian information criterion: {self.bic}\n"
            f"  Initial rate constant estimates: {self.k_est}\n"
        )
