# Third-party
import numpy as np
import matplotlib.pyplot as plt


__all__ = ['mpl_style', 'show_image']

mpl_style = {
    'font.family': 'serif',
    'font.serif': 'Times New Roman',

    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': True,
    'axes.spines.right': True,
    'axes.edgecolor': 'black',

    'axes.linewidth': 2.0,
    'axes.titlesize': 'large',
    'axes.labelsize': 'large',

    'xtick.major.size': 10,
    'xtick.minor.size': 5,
    'xtick.major.width': 2.0,
    'xtick.minor.width': 2.0,
    'xtick.direction': 'in',
    'xtick.labelsize': 'medium',

    'ytick.major.size': 10,
    'ytick.minor.size': 5,
    'ytick.major.width': 2.0,
    'ytick.minor.width': 2.0,
    'ytick.direction': 'in',
    'ytick.labelsize': 'medium',

    'xtick.top': True,
    'ytick.right': True,

    'legend.numpoints': 1,
    'legend.handletextpad': 0.3,
    'legend.frameon': False,
    'legend.scatterpoints': 1,

    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
}


def show_image(image, percentile=[1, 99], subplots=None, cmap='gray_r', 
               rasterized=False, **kwargs):
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
    rasterized : bool, optional
        If True, set `rasterized=True` in `~matplotlib.pyplot.imshow`.
    **kwargs
        Keyword arguments for `~matplotlib.pyplot.subplots`.

    Returns
    -------
    fig : `~matplotlib.figure.Figure`
        The Figure object.
    ax : `~matplotlib.axes._subplots.AxesSubplot`
        The axis. 
    """
    if subplots is None:
        figsize = kwargs.pop('figsize', (10, 10))
        fig, ax = plt.subplots(figsize=figsize,
                               subplot_kw=dict(xticks=[], yticks=[]),
                               **kwargs)
    else:
        fig, ax = subplots
    if percentile is not None:
        vmin, vmax = np.nanpercentile(image, percentile)
    else:
        vmin, vmax = None, None
    ax.imshow(image, origin='lower', cmap=cmap, rasterized=rasterized,
              vmin=vmin, vmax=vmax)

    return fig, ax
