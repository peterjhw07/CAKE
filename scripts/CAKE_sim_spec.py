"""CAKE Fitting Programme"""
# if __name__ == '__main__':
# import matplotlib
# matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
import itertools
import base64
from scipy import optimize
import logging
import re
import cake_fitting_multi


def sim_cake(t, spec_type, react_vol_init, stoich=1, mol0=None, mol_end=None, add_sol_conc=None, add_cont_rate=None,
             t_cont=None, add_one_shot=None, t_one_shot=None, add_col=None, sub_cont_rate=None,
             sub_aliq=None, t_aliq=None, sub_col=None, k_lim=None, ord_lim=None,
             pois_lim=None, win=1, inc=1):

    spec_type = cake_fitting_multi.type_to_list(spec_type)
    num_spec = len(spec_type)
    r_locs = [i for i in range(num_spec) if 'r' in spec_type[i]]
    p_locs = [i for i in range(num_spec) if 'p' in spec_type[i]]
    c_locs = [i for i in range(num_spec) if 'c' in spec_type[i]]

    if stoich is None: stoich = [1] * num_spec
    mol0, mol_end, add_sol_conc, add_cont_rate, t_cont, add_one_shot, t_one_shot, add_col = \
        map(cake_fitting_multi.return_all_nones, [mol0, mol_end, add_sol_conc, add_cont_rate, t_cont, add_one_shot, t_one_shot, add_col], [num_spec] * 8)
    if ord_lim is None:
        ord_lim = []
        for i in spec_type:
            if 'r' in i or 'c' in i:
                ord_lim.append((1, 0, 2))
            elif 'p' in i:
                ord_lim.append(0)
    elif p_locs:
        for i in p_locs:
            if ord_lim[i] is None: ord_lim[i] = 0
    if pois_lim is None: pois_lim = [0] * num_spec

    stoich, mol0, mol_end, add_sol_conc, add_cont_rate, t_cont, add_one_shot, t_one_shot, add_col, sub_aliq, t_aliq, \
    ord_lim, pois_lim = map(cake_fitting_multi.type_to_list, [stoich, mol0, mol_end, add_sol_conc,
    add_cont_rate, t_cont, add_one_shot, t_one_shot, add_col, sub_aliq, t_aliq, ord_lim, pois_lim])
    add_cont_rate, t_cont, add_one_shot, t_one_shot = map(cake_fitting_multi.tuple_of_lists_from_tuple_of_int_float,
                                            [add_cont_rate, t_cont, add_one_shot, t_one_shot])
    #print(spec_type, react_vol_init, stoich, mol0, mol_end, add_sol_conc, add_cont_rate, t_cont, add_one_shot,
          #t_one_shot, add_col, t_col, col, k_lim, ord_lim, pois_lim)

    fix_ord_locs = [i for i in range(num_spec) if (isinstance(ord_lim[i], (int, float))
                    or (isinstance(ord_lim[i], (tuple, list)) and len(ord_lim[i]) == 1))]
    fix_pois_locs = [i for i in range(num_spec) if (isinstance(pois_lim[i], (int, float))
                    or (isinstance(pois_lim[i], (tuple, list)) and len(pois_lim[i]) == 1))]
    inc += 1

    # Calculate iterative species additions and volumes
    add_pops, vol_data, vol_loss_rat = cake_fitting_multi.get_add_pops_vol(t, t, t, num_spec, react_vol_init,
                            add_sol_conc, add_cont_rate, t_cont, add_one_shot, t_one_shot, add_col,
                            sub_cont_rate, sub_aliq, t_aliq, sub_col, win=win)

    add_pops_new = np.zeros((len(t), num_spec))
    for i in range(1, len(add_pops)):
        add_pops_new[i] = add_pops[i] - add_pops[i - 1]
    add_pops = add_pops_new

    # Define initial values and lower and upper limits for parameters to fit: ord and pois
    for i in range(num_spec):
        if mol0[i] is None:
            mol0[i] = 0  # May cause issues
        if mol_end[i] is None:
            mol_end[i] = 0  # May cause issues
    for i in fix_pois_locs:
        mol0[i] -= pois_lim[i]
        mol_end[i] -= pois_lim[i]

    fit_pops_all, fit_rate_all = cake_fitting_multi.eq_sim_gen(stoich, mol0, mol_end, add_pops, vol_data, vol_loss_rat,
                                    inc, ord_lim, r_locs, p_locs, c_locs, fix_ord_locs, [], [], t, k_lim, [0, [], []])

    return fit_pops_all, np.reshape(fit_rate_all, (len(fit_rate_all), 1))


def plot_cake_results(x_data, fit, headers, spec_to_graph, f_format='svg', return_image=False, save_disk=False,
                      save_to='cake.svg', return_fig=False, transparent=False):

    spec_to_graph_locs = [i for i in range(len(spec_to_graph)) if "y" in spec_to_graph[i]]
    # graph results
    x_ax_scale = 1
    y_ax_scale = 1
    edge_adj = 0.02
    x_label_text = "Time / time_unit"
    y_label_text = "Concentration / moles_unit volume_unit$^{-1}$"
    fig = plt.figure(figsize=(5, 5))
    #plt.rcParams.update({'font.size': 15})
    plt.xlabel(x_label_text)
    plt.ylabel(y_label_text)
    for i in spec_to_graph_locs:
        plt.plot(x_data, fit[:, i] * y_ax_scale, label=headers[i])
    fit = fit[:, spec_to_graph_locs]
    plt.xlim([float(min(x_data * x_ax_scale) - (edge_adj * max(x_data * x_ax_scale))), float(max(x_data * x_ax_scale) * (1 + edge_adj))])
    plt.ylim([float(np.min(fit) - edge_adj * np.max(fit) * x_ax_scale), float(np.max(fit)) * x_ax_scale * (1 + edge_adj)])
    plt.legend(prop={'size': 10}, frameon=False)

    #plt.show()

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
    if not return_image:
        graph_url = base64.b64encode(img.getvalue()).decode()
        return 'data:{};base64,{}'.format(mimetype, graph_url)
    else:
        return img, mimetype


def make_char_tup(substrate):
    if ', ' in str(substrate):
        substrate_adj = re.split(r',\s*(?![^()]*\))', str(substrate))
    else:
        substrate_adj = substrate
    return substrate_adj


def make_char_tup_and_sort(s):
    split = make_char_tup(s)
    if isinstance(split, (int, float)):
        s_adj = split
    elif "None" in split and isinstance(split, str):
        s_adj = None
    elif not isinstance(split, int):
        s_adj = [eval(str(x)) for x in split]
    else:
        s_adj = split
    return s_adj


if __name__ == "__main__":
    excel_source = "n"
    if "n" in excel_source:
        spec_type = ["r", "r", "p", "c"]
        react_vol_init = 4E-3
        stoich = [1, 1, 1, None]  # insert stoichiometry of reactant, r
        mol0 = [10, 0, 0, 0]
        mol_end = [0, None, 10, 0]
        add_sol_conc = [None, 1E7, None, 0.709]
        add_cont_rate = [None, 10E-6, None, None]
        t_cont = [None, 1, None, None]
        add_one_shot = [None, None, None, 100E-6]
        t_one_shot = [None, None, None, 1]
        k_lim = [3.5]
        ord_lim = [1, 0.0000001, 0, 1]
        pois_lim = [0, 0, 0, 0]
        t_end = 100
        t_inc = 0.1
        spec_to_graph = ["y", "n", "y", "y"]
        pic_save = 'cake.svg'

        t = np.linspace(0, t_end, int(t_end / t_inc) + 1)
        fit, fit_rate = sim_cake(t, spec_type, react_vol_init, stoich=stoich, mol0=mol0, mol_end=mol_end,
                                 add_sol_conc=add_sol_conc, add_cont_rate=add_cont_rate, t_cont=t_cont,
                                 add_one_shot=add_one_shot, t_one_shot=t_one_shot,
                                 k_lim=k_lim, ord_lim=ord_lim, pois_lim=pois_lim)

        headers = []
        for i in range(len(spec_type)):
            headers = [*headers, 'Species ' + str(i + 1)]

        plot_output = plot_cake_results(np.reshape(t, (len(t), 1)), fit, headers, spec_to_graph,
                                        f_format='png', save_disk=True, save_to=pic_save)
    else:
        df = pd.read_excel(r'C:\Users\Peter\Documents\Postdoctorate\Work\CAKE\Programmes\Test_Spectra.xlsx',
                           sheet_name='Sim_testing')
        df.replace('""', None, inplace=True)

        total = np.empty([len(df), 12], object)
        for i in range(0, len(df)):
            [number, react_type] = df.iloc[i, :2]
            [spec_type, react_vol_init, stoich, mol0, mol_end, add_sol_conc, add_cont_rate, t_cont] = df.iloc[i, 2:10]
            [add_one_shot, t_one_shot, k_lim, ord_lim, pois_lim, t_end, t_inc, spec_to_graph] = df.iloc[i, 10:18]
            [pic_save] = df.iloc[i, 18:]
            print(number, react_type)
            spec_type = make_char_tup(spec_type)
            stoich = make_char_tup_and_sort(stoich)
            mol0 = make_char_tup_and_sort(mol0)
            mol_end = make_char_tup_and_sort(mol_end)
            add_sol_conc = make_char_tup_and_sort(add_sol_conc)
            add_cont_rate = make_char_tup_and_sort(add_cont_rate)
            t_cont = make_char_tup_and_sort(t_cont)
            add_one_shot = make_char_tup_and_sort(add_one_shot)
            t_one_shot = make_char_tup_and_sort(t_one_shot)
            k_lim = [make_char_tup_and_sort(k_lim)]
            ord_lim = make_char_tup_and_sort(ord_lim)
            pois_lim = make_char_tup_and_sort(pois_lim)
            spec_to_graph = make_char_tup(spec_to_graph)

            t = np.linspace(0, t_end, int(t_end / t_inc) + 1)
            fit, fit_rate = sim_cake(t, spec_type, react_vol_init, stoich=stoich, mol0=mol0, mol_end=mol_end,
                         add_sol_conc=add_sol_conc, add_cont_rate=add_cont_rate, t_cont=t_cont,
                         add_one_shot=add_one_shot, t_one_shot=t_one_shot,
                         k_lim=k_lim, ord_lim=ord_lim, pois_lim=pois_lim)

            headers = []
            for i in range(len(spec_type)):
                headers = [*headers, 'Species ' + str(i + 1)]

            plot_output = plot_cake_results(np.reshape(t, (len(t), 1)), fit, headers, spec_to_graph,
                                                f_format='png', save_disk=True, save_to=pic_save)