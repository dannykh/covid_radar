import matplotlib.pyplot as plt

from filter_params import dist_range_bottom, dist_range_top, time_filter_bottom, time_filter_top, slow_sample_rate
import numpy as np


def plot_named_list(named_map, x_range, xlable, ylable, scatter=False):
    drawer = plt.scatter if scatter else plt.plot
    fig, ax = plt.subplots(1, 1)
    for name, phase_plot in named_map.items():
        drawer(x_range, phase_plot)

    plt.legend(named_map.keys())
    plt.xlabel(xlable)
    plt.ylabel(ylable)

    fig.show()
    plt.waitforbuttonpress()
    plt.close(fig)


def forceAspect(ax, aspect):
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)


def plot_range_maps(named_range_maps, xlable, ylable, title):
    plt.xlabel(xlable)
    plt.ylabel(ylable)

    for name, plot in named_range_maps.items():
        fig, ax = plt.subplots(1, 1)
        # plot = plot / np.max(plot)
        im = ax.imshow(plot, extent=[dist_range_bottom, dist_range_top, 0, 50])
        forceAspect(ax, aspect=1)
        fig.colorbar(im, ax=ax, orientation='horizontal', fraction=.1)

        plt.title(f"{name}-{title}")
        fig.show()
        plt.waitforbuttonpress()
        plt.close(fig)


def fixed_aspect_ratio(ratio, ax):
    '''
    Set a fixed aspect ratio on matplotlib plots
    regardless of axis units
    '''
    xvals, yvals = ax.axes.get_xlim(), ax.axes.get_ylim()

    xrange = xvals[1] - xvals[0]
    yrange = yvals[1] - yvals[0]
    ax.set_aspect(ratio * (xrange / yrange), adjustable='box')
