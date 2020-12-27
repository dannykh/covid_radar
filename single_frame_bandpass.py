import numpy as np
import matplotlib.pyplot as plt
from mmwave import dsp
# Radar specific parameters
from mmwave.dsp import Window

from data_loader import load_file
from etc.config_1 import adc_samples, NUM_CHIRPS

fig = plt.figure()

# Discard Rx axis
all_data, range_res, velocity_res = load_file("data/mom_1_Raw_0.bin")


# Apply the range resolution factor to the range indices
ranges = np.arange(adc_samples) * range_res

partial_stacked = np.vstack(all_data[500:600])

range_plot = dsp.range_processing(partial_stacked, window_type_1d=Window.HAMMING)
# for rbin in range(20, 50):
#     # Select middle frame, at ~1.6m range bin
#     range_plot_target = range_plot.T[rbin]
#     phases = np.arctan2(range_plot_target.imag, range_plot_target.real)
#
#     plt.plot(np.arange(NUM_CHIRPS*100), phases)
#     plt.title(rbin * range_res)
#     plt.show()

best_bin_idx = 37
range_plot_target = range_plot.T[best_bin_idx]
phases = np.arctan2(range_plot_target.imag, range_plot_target.real)

# y = butter_bandpass_filter(phases, 0.1, 0.6, 4000 * 1000, order=6)
# plt.plot(np.arange(NUM_CHIRPS * 100), y)
# plt.title(best_bin_idx * range_res)
# plt.show()

freqs = np.fft.fft(phases)
freqs = np.fft.fftshift(freqs)
freq_range = np.arange(NUM_CHIRPS * 100)
plt.plot( freqs)
plt.title(best_bin_idx * range_res)
plt.show()
