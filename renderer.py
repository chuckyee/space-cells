#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.mplot3d import Axes3D

from geometry import has_hole_3x3x3


def plot_voxels_solid(voxels, alpha=1.0):
    # set color of voxels
    colors = np.zeros(voxels.shape + (4,))
    colors[...,1] = 0.5
    colors[voxels,3] = alpha

    # set color of voxels
    # colors = np.empty(voxels.shape, dtype=object)
    # colors[voxels] = 'green'

    # and plot everything
    fig = plt.figure(figsize=(5,5))
    ax = fig.gca(projection='3d')
    ax.voxels(voxels, facecolors=colors, edgecolor='k')

    # turn off all ticks / grid lines on panes
    ax.w_xaxis.line.set_lw(0.)
    ax.set_xticks([])
    ax.w_yaxis.line.set_lw(0.)
    ax.set_yticks([])
    ax.w_zaxis.line.set_lw(0.)
    ax.set_zticks([])

    # turn off back panes, keeping bottom pane
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

# Code from terbium.io

def explode(data):
    shape_arr = np.array(data.shape)
    size = shape_arr[:3]*2 - 1
    exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
    exploded[::2, ::2, ::2] = data
    return exploded

def expand_coordinates(indices):
    x, y, z = indices
    x[1::2, :, :] += 1
    y[:, 1::2, :] += 1
    z[:, :, 1::2] += 1
    return x, y, z

def plot_voxels(voxels, height=256, width=256, dpi=100, to_buffer=False):
    # renderer puts lots of white space around voxels, so render to bigger canvas then crop
    factor = 13/8 / dpi
    fig = plt.figure(figsize=(width * factor, height * factor), dpi=dpi)
    ax = fig.gca(projection='3d')

    # turn off all ticks / grid lines / panes
    plt.axis('off')

    # explode to create transparency
    colors = np.array([[['#00800080']*3]*3]*3)
    colors = explode(colors)
    filled = explode(voxels)
    x, y, z = expand_coordinates(np.indices(np.array(filled.shape) + 1))

    # plot
    ax.voxels(x, y, z, filled, facecolors=colors, edgecolors='gray')

    if to_buffer:
        # convert to numpy array
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        image = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        image = image.reshape((h, w, -1))
        image = np.roll(image, 3, axis=2) # convert ARGB => RGBA
        
        return image[72:72+256,72:72+256,:]
    else:
        return ax

def main():
    n = 3
    threshold = 0.5

    # prepare coordinates
    x, y, z = np.indices((n, n, n))

    # randomly fill in voxels
    voxels = np.random.rand(n, n, n) < threshold

    # render image
    to_buffer = False
    
    image = plot_voxels(voxels, to_buffer=to_buffer)

    if to_buffer:
        # write image to file
        dpi = 100
        fig = plt.figure(dpi=dpi)
        w, h, _ = image.shape
        figsize = (w/dpi, h/dpi)
        fig.set_size_inches(*figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(image, interpolation='none')
        plt.axis('off')
        plt.savefig('test.png')
    else:
        plt.show()


if __name__ == '__main__':
    main()
