"""General Plotting Functions"""

import base64
import numpy as np
import matplotlib.pyplot as plt
import io
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

np.seterr(divide='ignore', invalid='ignore')


# Adjusts DataFrame and axis titles
def units_adjust(units):
    if not isinstance(units, list):
        units = [units]
    for i in range(len(units)):
        if units[i]:
            units[i] = ' / ' + units[i]
        else:
            units[i] = ''
    return units


# Calculates limits from data
def calc_lim(lower_lim, upper_lim, edge_adj):
    return [float(lower_lim - (edge_adj * (upper_lim - lower_lim))),
            float(upper_lim + (edge_adj * (upper_lim - lower_lim)))]


# Calculates x limits from x data
def calc_x_lim(t, edge_adj):
    return calc_lim(min(t), max(t), edge_adj)


# Calculates y limits from y data
def calc_y_lim(exp, fit, edge_adj):
    lower_lim = min(np.nanmin(exp), np.nanmin(fit))
    upper_lim = max(np.nanmax(exp), np.nanmax(fit))
    if lower_lim != upper_lim:
        return calc_lim(lower_lim, upper_lim, edge_adj)
    else:
        return [float(lower_lim * (1 - edge_adj)), float(lower_lim * (1 + edge_adj))]


# Processes plotted data
def plot_process(f_format, save_to, transparent, fig, return_fig=False, return_img=False):

    # Correct mimetype based on filetype (for displaying in browser)
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

    # Save to disk if desired
    if save_to:
        plt.savefig(save_to, format=f_format, transparent=transparent)

    # Save the figure to the temporary file-like object
    img = io.BytesIO()  # file-like object to hold image
    plt.savefig(img, format=f_format, transparent=transparent)

    if return_fig:
        return fig, fig.get_axes()
    elif return_img:
        plt.close()
        img.seek(0)
        return img, mimetype
    else:
        plt.close()
        graph_url = base64.b64encode(img.getvalue()).decode()
        return 'data:{};base64,{}'.format(mimetype, graph_url), mimetype
