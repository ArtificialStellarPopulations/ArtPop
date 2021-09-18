# Third-party
import numpy as np
import matplotlib.pyplot as plt


__all__ = ['mpl_style', 'jpg_style', 'show_image']


mpl_style = {
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': True,
    'axes.spines.right': True,
    'axes.edgecolor': 'black',
    'axes.linewidth': 2.0,
    'axes.titlesize': 'xx-large',
    'axes.labelsize': 'xx-large',

    'xtick.major.size': 8,
    'xtick.minor.size': 4,
    'xtick.major.width': 2.0,
    'xtick.minor.width': 2.0,
    'xtick.direction': 'in',
    'xtick.labelsize': 'x-large',

    'ytick.major.size': 8,
    'ytick.minor.size': 4,
    'ytick.major.width': 2.0,
    'ytick.minor.width': 2.0,
    'ytick.direction': 'in',
    'ytick.labelsize': 'x-large',

    'xtick.top': True,
    'ytick.right': True,

    'legend.numpoints': 1,
    'legend.fontsize': 'x-large',
    'legend.handletextpad': 0.3,
    'legend.frameon': False,
    'legend.scatterpoints': 1,
    'savefig.bbox': 'tight'
}


jpg_style = {
    'font.family': 'serif',
    'font.serif': 'Times New Roman',

    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': True,
    'axes.spines.right' : True,
    'axes.edgecolor': 'black',
    'axes.linewidth': 2.0,
    'axes.titlesize': 18,
    'axes.labelsize': 18,

    'xtick.major.size': 10,
    'xtick.minor.size': 5,
    'xtick.major.width': 2.0,
    'xtick.minor.width': 2.0,
    'xtick.labelsize': 15,
    'xtick.direction': 'in',
    'ytick.major.size': 10,
    'ytick.minor.size': 5,
    'ytick.major.width': 2.0,
    'ytick.minor.width': 2.0,
    'ytick.labelsize': 15,
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,

    'legend.numpoints': 1,
    'legend.handletextpad': 0.3,
    'legend.frameon': False,
    'legend.scatterpoints' : 1,

    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',

    'text.usetex': True
}


def show_image(image, percentile=[0.1, 99.9], subplots=None, cmap='gray_r',
               figsize=(10, 10), **kwargs):
    """
    Display image using matplotlib.

    Parameters
    ----------
    image : `~numpy.ndarray`
        The image pixels.
    percentile : list-like or None, optional
        Set the min and max pixel values to the given low and high percentile
        values: [low, high]. If None, use all pixel values.
    subplots : tuple or None, optional
        The ``matplotlib`` figure and axis objects (`fig`, `ax`). If None, a
        new figure will be created.
    cmap : str, optional
        ``matplotlib`` color map.
    figsize : tuple, optional
         Figure size. Only used if subplots is None.
    **kwargs
        Keyword arguments for `~matplotlib.pyplot.imshow`.

    Returns
    -------
    fig : `~matplotlib.figure.Figure`
        The Figure object.
    ax : `~matplotlib.axes._subplots.AxesSubplot`
        The axis.
    """
    if subplots is None:
        fig, ax = plt.subplots(figsize=figsize,
                               subplot_kw=dict(xticks=[], yticks=[]))

    else:
        fig, ax = subplots
    if percentile is not None:
        vmin, vmax = np.nanpercentile(image, percentile)
    else:
        vmin, vmax = None, None

    interp = kwargs.pop('interpolation', None)
    ax.imshow(image, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
              interpolation=interp, **kwargs)

    return fig, ax
