# Third-party
import numpy as np
import matplotlib.pyplot as plt


__all__ = ['mpl_style', 'show_image']


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


def show_image(image, percentile=[0.1, 99.9], subplots=None, cmap='gray_r',
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
