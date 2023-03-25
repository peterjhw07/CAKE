"""CAKE Plotting Functions"""
# if __name__ == '__main__':
# import matplotlib
# matplotlib.use('Agg')
import math
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import pandas as pd
import io
import itertools
import base64
from scipy import optimize
import logging
import timeit  # to be removed
import warnings

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


# calculate x limits from x data
def calc_x_lim(x_data, edge_adj):
    return [float(min(x_data) - (edge_adj * max(x_data))), float(max(x_data) * (1 + edge_adj))]


# calculate y limits from y data
def calc_y_lim(y_exp, y_fit, edge_adj):
    return [float(min(np.min(y_exp), np.min(y_fit)) - edge_adj * max(np.max(y_exp), np.max(y_fit))),
            float(max(np.max(y_exp), np.max(y_fit)) * (1 + edge_adj))]


# processes plotted data
def plot_process(return_fig, fig, f_format, save_disk, save_to, transparent):
    if return_fig:
        return fig, fig.get_axes()

    # correct mimetype based on filetype (for displaying in browser)
    if f_format == 'svg':
        mimetype = 'image/svg+xml'
    elif f_format == 'png':
        mimetype = 'image/png'
    elif f_format == 'jpg':
        mimetype = 'image/jpg'
    elif f_format == 'pdf':
        mimetype = 'application/pdf'
    elif f_format == 'eps':
        mimetype = 'application/postscript'
    else:
        raise ValueError('Image format {} not supported.'.format(format))

    # save to disk if desired
    if save_disk:
        plt.savefig(save_to, transparent=transparent)

    # save the figure to the temporary file-like object
    img = io.BytesIO()  # file-like object to hold image
    plt.savefig(img, format=f_format, transparent=transparent)
    plt.close()
    img.seek(0)
    return img, mimetype


# plot time vs conc
def plot_conc_vs_time(x_data_df, y_exp_conc_df=None, y_fit_conc_df=None, col=None, show_asp=None, method="lone", f_format='svg', return_image=False, save_disk=False,
                      save_to='cake_fit.svg', return_fig=False, transparent=False):
    # methods
    x_data = pd.DataFrame.to_numpy(x_data_df)

    if y_exp_conc_df is not None:
        y_exp_conc_headers = [i.replace(' conc. / moles_unit volume_unit$^{-1}$', '') for i in list(y_exp_conc_df.columns)]
        y_exp_conc = pd.DataFrame.to_numpy(y_exp_conc_df)
    else:
        y_exp_conc_headers = []
        y_exp_conc = pd.DataFrame.to_numpy(y_fit_conc_df)
    if y_fit_conc_df is not None:
        y_fit_conc_headers = [i.replace(' conc. / moles_unit volume_unit$^{-1}$', '') for i in list(y_fit_conc_df.columns)]
        y_fit_conc = pd.DataFrame.to_numpy(y_fit_conc_df)
    else:
        y_fit_conc_headers = []
        y_fit_conc = y_exp_conc
        y_fit_col = []

    if col is not None and show_asp is None:
        y_fit_col = [i for i in range(len(col)) if col[i] is not None]
        non_y_fit_col = [i for i in range(len(col)) if col[i] is None]
    if "lone all" in method and show_asp is None and y_fit_conc_df is not None:
        show_asp = ["y"] * len(y_fit_conc_headers)
    if show_asp is not None:
        y_fit_col = [i for i in range(len(show_asp)) if 'y' in show_asp[i]]
        non_y_fit_col = [i for i in range(len(show_asp)) if 'n' in show_asp[i]]
    if "comp" in method and (len(non_y_fit_col) == 0 or y_fit_conc_df is None):
        method = "lone"
    if show_asp is not None and 'y' not in show_asp:
        print("If used, show_asp must contain at least one 'y'. Plot time_vs_conc has been skipped.")
        return

    # graph results
    x_ax_scale = 1
    y_ax_scale = 1
    edge_adj = 0.02
    std_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

    x_data_adj = x_data * x_ax_scale
    y_exp_conc_adj = y_exp_conc * y_ax_scale
    y_fit_conc_adj = y_fit_conc * y_ax_scale

    x_label_text = list(x_data_df.columns)[0]
    y_label_text = "Concentration / moles_unit volume_unit$^{-1}$"

    cur_exp = 0
    cur_clr = 0
    if "lone" in method:  # lone plots a single figure containing all exps and fits as specified
        fig = plt.figure(figsize=(5, 5))
        #plt.rcParams.update({'font.size': 15})
        plt.xlabel(x_label_text)
        plt.ylabel(y_label_text)
        for i in range(len(y_exp_conc_headers)):
            if len(x_data_adj) <= 50:
                plt.scatter(x_data_adj, y_exp_conc_adj[:, i], label=y_exp_conc_headers[i])
            else:
                plt.plot(x_data_adj, y_exp_conc_adj[:, i], label=y_exp_conc_headers[i])
        for i in y_fit_col:
            plt.plot(x_data_adj, y_fit_conc_adj[:, i], label=y_fit_conc_headers[i])
        if len(y_fit_col) == 0: y_fit_col = range(len(y_exp_conc_headers))
        plt.xlim(calc_x_lim(x_data_adj, edge_adj))
        plt.ylim(calc_y_lim(y_exp_conc_adj, y_fit_conc_adj[:, y_fit_col], edge_adj))
        plt.legend(prop={'size': 10}, frameon=False)
    elif "comp" in method:  # plots two figures, with the first containing show_asp (or col if show_asp not specified) and the second containing all fits
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        #plt.rcParams.update({'font.size': 15})
        for i in range(len(y_exp_conc_headers)):
            if len(x_data_adj) <= 50:
                ax1.scatter(x_data_adj, y_exp_conc_adj[:, i], color=std_colours[cur_clr],
                            label=y_exp_conc_headers[i])
                ax2.scatter(x_data_adj, y_exp_conc_adj[:, i], color=std_colours[cur_clr],
                            label=y_exp_conc_headers[i])
                cur_clr += 1
            else:
                ax1.plot(x_data_adj, y_exp_conc_adj[:, i], color=std_colours[cur_clr],
                            label=y_exp_conc_headers[i])
                ax2.plot(x_data_adj, y_exp_conc_adj[:, i], color=std_colours[cur_clr],
                            label=y_exp_conc_headers[i])
                cur_clr += 1
        for i in y_fit_col:
            ax1.plot(x_data_adj, y_fit_conc_adj[:, i], color=std_colours[cur_clr],
                            label=y_fit_conc_headers[i])
            ax2.plot(x_data_adj, y_fit_conc_adj[:, i], color=std_colours[cur_clr],
                            label=y_fit_conc_headers[i])
            cur_clr += 1
        for i in non_y_fit_col:
            ax2.plot(x_data_adj, y_fit_conc_adj[:, i], color=std_colours[cur_clr],
                            label=y_fit_conc_headers[i])
            cur_clr += 1

        ax1.set_xlim(calc_x_lim(x_data_adj, edge_adj))
        ax1.set_ylim(calc_y_lim(y_exp_conc_adj, y_fit_conc_adj[:, y_fit_col], edge_adj))
        ax2.set_xlim(calc_x_lim(x_data_adj, edge_adj))
        ax2.set_ylim(calc_y_lim(y_exp_conc_adj, y_fit_conc_adj, edge_adj))

        ax1.set_xlabel(x_label_text)
        ax1.set_ylabel(y_label_text)
        ax2.set_xlabel(x_label_text)
        ax2.set_ylabel(y_label_text)
        ax1.legend(prop={'size': 10}, frameon=False)
        ax2.legend(prop={'size': 10}, frameon=False)

    elif "sep" in method:
        num_spec = max([len(y_exp_conc_headers), len(y_fit_conc_headers)])
        grid_shape = (int(round(np.sqrt(num_spec))), int(math.ceil(np.sqrt(num_spec))))
        fig = plt.figure(figsize=(grid_shape[0] * 6, grid_shape[1] * 5))
        # plt.subplots_adjust(hspace=0.4, wspace=0.4)
        for i in range(num_spec):
            ax = plt.subplot(grid_shape[0], grid_shape[1], i + 1)
            if col is not None and col[i] is not None and y_exp_conc_df is not None:
                if len(x_data_adj) <= 50:
                    ax.scatter(x_data_adj, y_exp_conc_adj[:, cur_exp], color=std_colours[cur_clr], label=y_exp_conc_headers[cur_exp])
                else:
                    ax.plot(x_data_adj, y_exp_conc_adj[:, cur_exp], color=std_colours[cur_clr], label=y_exp_conc_headers[cur_exp])
                ax.set_ylim(calc_y_lim(y_exp_conc_adj[:, cur_exp], y_fit_conc_adj[:, i], edge_adj))
                cur_exp += 1
                cur_clr += 1
            else:
                ax.set_ylim(calc_y_lim(y_fit_conc_adj[:, i], y_fit_conc_adj[:, i], edge_adj))
            ax.plot(x_data_adj, y_fit_conc_adj[:, i], color=std_colours[cur_clr], label=y_fit_conc_headers[i])
            cur_clr += 1

            ax.set_xlim(calc_x_lim(x_data_adj, edge_adj))
            ax.set_xlabel(x_label_text)
            ax.set_ylabel(y_label_text)
            plt.legend(prop={'size': 10}, frameon=False)

    else:
        print("Invalid method inputted. Please enter appropriate method or remove method argument.")
        return

    # plt.show()
    img, mimetype = plot_process(return_fig, fig, f_format, save_disk, save_to, transparent)
    if not return_image:
        graph_url = base64.b64encode(img.getvalue()).decode()
        return 'data:{};base64,{}'.format(mimetype, graph_url)
    else:
        return img, mimetype


# plot rate vs conc
def plot_rate_vs_conc(x_data_df, y_fit_conc_df, y_fit_rate_df, orders, y_exp_conc_df=None, y_exp_rate_df=None,
                      f_format='svg', return_image=False, save_disk=False,
                     save_to='cake_conc_vs_rate.svg', return_fig=False, transparent=False):
    x_data, y_fit_conc, y_fit_rate = map(pd.DataFrame.to_numpy, [x_data_df, y_fit_conc_df, y_fit_rate_df])
    if y_exp_conc_df is not None:
        pd.DataFrame.to_numpy(y_exp_conc_df)
    if y_exp_rate_df is not None:
        pd.DataFrame.to_numpy(y_exp_rate_df)

    num_spec = len(orders)
    # y_exp_conc_headers = list(y_exp_conc_df.columns)
    y_fit_conc_headers = list(y_fit_conc_df.columns)
    # y_exp_rate_adj_headers = [i.replace('fit conc. / moles_unit volume_unit$^{-1}$', 'exp.') for i in list(y_fit_conc_df.columns)]
    y_fit_rate_adj_headers = [i.replace(' conc. / moles_unit volume_unit$^{-1}$', '') for i in list(y_fit_conc_df.columns)]

    # y_exp_rate_adj = np.empty((len(y_exp_rate), num_spec))
    y_fit_rate_adj = np.empty((len(y_fit_rate), num_spec))
    for i in range(num_spec):
        # y_exp_rate_adj[:, i] = np.divide(y_exp_rate.reshape(len(y_exp_rate)), np.product([y_fit_conc[:, j] ** orders[j] for j in range(num_spec) if i != j], axis=0))
        y_fit_rate_adj[:, i] = np.divide(y_fit_rate.reshape(len(y_fit_rate)), np.product([y_fit_conc[:, j] ** orders[j] for j in range(num_spec) if i != j], axis=0))
    # y_exp_rate_adj_df = pd.DataFrame(y_exp_rate_adj, columns=y_exp_rate_adj_headers)
    y_fit_rate_adj_df = pd.DataFrame(y_fit_rate_adj, columns=y_fit_rate_adj_headers)

    grid_shape = (int(round(np.sqrt(num_spec))), int(math.ceil(np.sqrt(num_spec))))

    x_ax_scale = 1
    y_ax_scale = 1
    edge_adj = 0.02
    fig = plt.figure(figsize=(grid_shape[0] * 5, grid_shape[1] * 5))
    # plt.subplots_adjust(hspace=0.5)
    std_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    y_label_text = "Rate / moles_unit volume_unit$^{-1}$ time_unit$^{-1}$"
    for i in range(num_spec):
        x_label_text = y_fit_conc_headers[i]
        ax = plt.subplot(grid_shape[0], grid_shape[1], i + 1)
        ax.set_xlabel(x_label_text)
        ax.set_ylabel(y_label_text)
        # ax.scatter(y_fit_conc[:, i] * x_ax_scale, y_exp_rate_adj[:, i] * y_ax_scale, color=std_colours[i])
        ax.plot(y_fit_conc[:, i] * x_ax_scale, y_fit_rate_adj[:, i] * y_ax_scale, color=std_colours[i])
        ax.set_xlim([float(min(y_fit_conc[:, i] * x_ax_scale) - (edge_adj * max(y_fit_conc[:, i] * x_ax_scale))),
                float(max(y_fit_conc[:, i] * x_ax_scale) * (1 + edge_adj))])
        # ax.set_ylim([float(np.nanmin(y_fit_rate_adj[:, i]) - edge_adj * np.nanmax(y_fit_rate_adj[:, i]) * y_ax_scale),
        #        float(np.nanmax(y_fit_rate_adj[:, i])) * y_ax_scale * (1 + edge_adj)])
    # plt.show()
    save_to_replace = save_to.replace('.png', '_rates.png')
    img, mimetype = plot_process(return_fig, fig, f_format, save_disk, save_to_replace, transparent)
    if not return_image:
        graph_url = base64.b64encode(img.getvalue()).decode()
        return 'data:{};base64,{}'.format(mimetype, graph_url)
    else:
        return img, mimetype


# plot other fits in 2D
def plot_other_fits_2D(x_data_df, y_exp_conc_df, y_fit_conc_df_arr, real_err_df, col, cutoff=1, f_format='svg', return_image=False,
                       save_disk=False, save_to='cake_other_fits.svg', return_fig=False, transparent=False):
    num_spec = len(col)
    x_data, y_exp_conc, real_err = map(pd.DataFrame.to_numpy, [x_data_df, y_exp_conc_df, real_err_df])
    # np.savetxt(r"C:\Users\Peter\Desktop\real_err.csv", real_err, delimiter="\t", fmt='%s')
    cut_thresh = cutoff * real_err[0, -1]
    rows_cut = [i for i, x in enumerate(real_err[:, -1] > cut_thresh) if x]
    cur_clr = 0

    col_ext = [i for i in range(len(col)) if col[i] is not None]
    grid_shape = (int(round(np.sqrt(len(col_ext)))), int(math.ceil(np.sqrt(len(col_ext)))))

    x_ax_scale = 1
    y_ax_scale = 1
    edge_adj = 0.02

    x_data_adj = x_data * x_ax_scale
    fig = plt.figure(figsize=(grid_shape[0] * 5, grid_shape[1] * 5))
    # plt.subplots_adjust(hspace=0.5)
    std_colours = plt.rcParams['axes.prop_cycle'].by_key()['color'] * 100

    x_label_text = "Time / time_unit"
    y_label_text = "Concentration / moles_unit volume_unit$^{-1}$"
    for i in range(len(col_ext)):
        ax = plt.subplot(grid_shape[0], grid_shape[1], i + 1)
        ax.plot(x_data * x_ax_scale, y_exp_conc[:, i] * y_ax_scale, '-o', color="black", label="Exp")
        for j in rows_cut:
            y_fit_conc_df = y_fit_conc_df_arr[j][0]
            y_fit_conc = y_fit_conc_df.to_numpy()
            ax.plot(x_data_adj, y_fit_conc[:, col_ext[i]] * y_ax_scale, label=real_err[j, 0])
            #color=std_colours[j]
        ax.set_xlim(calc_x_lim(x_data_adj, edge_adj))
        #ax.set_ylim([float(np.nanmin(y_fit_rate_adj[:, i]) - edge_adj * np.nanmax(y_fit_rate_adj[:, i]) * y_ax_scale),
        #    float(np.nanmax(y_fit_rate_adj[:, i])) * y_ax_scale * (1 + edge_adj)])

        ax.set_xlabel(x_label_text)
        ax.set_ylabel(y_label_text)
        ax.legend(prop={'size': 10}, frameon=False)

    save_to_replace = save_to.replace('.png', '_other_fits_2D.png')
    img, mimetype = plot_process(return_fig, fig, f_format, save_disk, save_to_replace, transparent)
    # plt.show()

    for i in range(len(col_ext)):
        grid_shape = (int(round(np.sqrt(len(rows_cut)))), int(math.ceil(np.sqrt(len(rows_cut)))))
        fig = plt.figure(figsize=(grid_shape[0] * 5, grid_shape[1] * 5))
        plt.subplots_adjust(hspace=0.2, wspace=0.08)
        for j in rows_cut:
            ax = plt.subplot(grid_shape[0], grid_shape[1], j + 1)
            ax.plot(x_data * x_ax_scale, y_exp_conc[:, i] * y_ax_scale, '-o', color="black", label="Exp")
            y_fit_conc_df = y_fit_conc_df_arr[j][0]
            y_fit_conc = y_fit_conc_df.to_numpy()
            ax.plot(x_data * x_ax_scale, y_fit_conc[:, col_ext[i]] * y_ax_scale, color=std_colours[j], label=real_err[j, 0])
            # color=std_colours[j]
            ax.set_xlim(calc_x_lim(x_data_adj, edge_adj))
            # ax.set_ylim([float(np.nanmin(y_fit_rate_adj[:, i]) - edge_adj * np.nanmax(y_fit_rate_adj[:, i]) * y_ax_scale),
            #    float(np.nanmax(y_fit_rate_adj[:, i])) * y_ax_scale * (1 + edge_adj)])

            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.tick_params(length=0, width=0)
            # ax.set_xlabel(x_label_text)
            # ax.set_ylabel(y_label_text)

    plt.show()

    if not return_image:
        graph_url = base64.b64encode(img.getvalue()).decode()
        return 'data:{};base64,{}'.format(mimetype, graph_url)
    else:
        return img, mimetype


# plot other fits in 3D (contour map and 3D projection)
def plot_other_fits_3D(real_err_df, cutoff=1, f_format='svg', return_image=False, save_disk=False,
                     save_to='cake_other_fits.svg', return_fig=False, transparent=False):
    real_err_arr = pd.DataFrame.to_numpy(real_err_df)
    real_err_arr_cut = real_err_arr[real_err_arr[:, -1] > cutoff, :]
    cont_x_org = [real_err_arr_cut[i, 0][0] for i in range(len(real_err_arr_cut))]
    cont_y_org = [real_err_arr_cut[i, 0][1] for i in range(len(real_err_arr_cut))]
    cont_z_org = real_err_arr_cut[:, -1]
    cont_x_add, cont_y_add = np.linspace(min(cont_x_org), max(cont_x_org), 1000), \
                             np.linspace(min(cont_y_org), max(cont_y_org), 1000)
    cont_x_plot, cont_y_plot = np.meshgrid(cont_x_add, cont_y_add)
    cont_z_plot = interpolate.griddata((cont_x_org, cont_y_org), cont_z_org, (cont_x_plot, cont_y_plot), method='linear')
    # rbf = scipy.interpolate.Rbf(cont_x_org, cont_y_org, cont_z_org, function='linear')
    # cont_z_plot = rbf(cont_x_plot, cont_y_plot)

    cont_fig = plt.imshow(cont_z_plot, vmin=cont_z_org.min(), vmax=cont_z_org.max(), origin='lower', cmap='coolwarm',
               extent=[min(cont_x_org), max(cont_x_org), min(cont_y_org), max(cont_y_org)], aspect='auto')
    # plt.scatter(cont_x_org, cont_y_org, c=cont_z_org, cmap='coolwarm')
    plt.xlabel('Order 1'), plt.ylabel('Order 2')
    plt.colorbar()
    img, mimetype = plot_process(return_fig, cont_fig, f_format, save_disk, save_to.replace('.png', '_other_fits_contour.png'), transparent)

    fig_3D = plt.axes(projection='3d')
    fig_3D.plot_surface(cont_x_plot, cont_y_plot, cont_z_plot, cmap='coolwarm')  # rstride=1, cstride=1
    fig_3D.set_xlabel('Order 1'), fig_3D.set_ylabel('Order 2'), fig_3D.set_zlabel('r^2')
    img, mimetype = plot_process(return_fig, fig_3D, f_format, save_disk, save_to.replace('.png', '_other_fits_3D.png'), transparent)
    if not return_image:
        graph_url = base64.b64encode(img.getvalue()).decode()
        return 'data:{};base64,{}'.format(mimetype, graph_url)
    else:
        return img, mimetype
