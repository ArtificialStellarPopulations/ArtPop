import numpy as np
import matplotlib.pyplot as plt


__all__ = ['show_image']


def show_image(image, percentile=[1, 99], subplots=None,
               cmap='gray_r', rasterized=False, **kwargs):
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
