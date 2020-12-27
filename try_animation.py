import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ArtistAnimation


def animate_process(frames, fig, plot_func, *args, **kwargs):
    frames = [plot_func(frame) for frame in frames]

    ani = ArtistAnimation(fig, frames, interval=50, blit=True, repeat=False)
    plt.show()
