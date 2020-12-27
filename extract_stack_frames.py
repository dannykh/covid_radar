import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mmwave import dsp
# Radar specific parameters
from mmwave.dsp import Window

from data_loader import load_file
from dsp_utils import unwrap, extract_phases
from etc.config_1 import adc_samples
from filter_params import dist_range_bottom, dist_range_top, freq_range_bottom, freq_range_top
from plot_utils import forceAspect, plot_frequencies

fig = plt.figure()

empty_1, _, _ = load_file("data/old/no_chair_Raw_0.bin")
empty_2, _, _ = load_file("data/old/no_chair_2_Raw_0.bin")

empty_avg = np.average(np.stack([empty_1, empty_1]), axis=0)

chair_1, _, _ = load_file("data/old/empty_1_Raw_0.bin")
chair_2, _, _ = load_file("data/old/empty_2_Raw_0.bin")
chair_3, _, _ = load_file("data/old/empty_3_Raw_0.bin")

chair_avg = np.average(np.stack([chair_1, chair_2, chair_3]), axis=0)

tot_empty = np.average(np.stack([chair_avg, empty_avg]), axis=0)

all_data, range_res, velocity_res = load_file("data/data_0312/moved_Raw_0.bin")
all_data_2, range_res, velocity_res = load_file("data/old/blanket.bin")
# range_res = RANGE_RESOLUTION

# all_data = all_data_2
# all_data_2 = all_data_2 - chair_avg
# all_data = np.maximum(chair_avg - empty_avg,0)

# Apply the range resolution factor to the range indices
ranges = np.arange(adc_samples) * range_res

chair_range_plot = dsp.range_processing(chair_avg, window_type_1d=Window.HAMMING)
range_plot = dsp.range_processing(all_data, window_type_1d=Window.HAMMING)
range_plot_2 = dsp.range_processing(all_data_2, window_type_1d=Window.HAMMING)


# range_plot -= empty_avg

# range_plot = range_plot - np.mean(range_plot,axis=0)


powers = abs(range_plot)
vmin = powers.min()
vmax = powers.max()
norm = colors.Normalize(vmin=vmin, vmax=vmax)

ax = fig.add_subplot(111)

# im = ax.imshow(powers, extent=[0, 200 * range_res, 0, 50])
# # im.set_norm(norm)
# forceAspect(ax, aspect=1)
# fig.colorbar(im, ax=ax, orientation='horizontal', fraction=.1)
# plt.show()

range_mask = (dist_range_bottom < ranges) & (ranges < dist_range_top)
ranges_filtered = ranges[range_mask]
range_plot_filtered = range_plot[:, range_mask]
range_plot_filtered_2 = range_plot_2[:, range_mask]

powers = np.sqrt(range_plot_filtered.imag ** 2 + range_plot_filtered.real ** 2)
print(np.max(powers))
powers = powers / np.max(powers)
# powers -= np.average(powers, axis=0)


im = ax.imshow(powers, extent=[dist_range_bottom, dist_range_top, 0, 50])
forceAspect(ax, aspect=1)
fig.colorbar(im, ax=ax, orientation='horizontal', fraction=.1)
plt.xlabel("Range(m)")
plt.ylabel("Time(s)")
plt.title("Range-Slow time matrix")
plt.show()

avg_power = np.average(powers, axis=0)
plt.plot(ranges_filtered, avg_power)
plt.xlabel("Range(m)")
plt.ylabel("AVG. power")
plt.title("Average power per range")
plt.show()


unwrapped_phases = extract_phases(range_plot_filtered)
unwrapped_phases_2 = extract_phases(range_plot_filtered_2)



sel_range = 200
max_pwr_idx = np.argmax(avg_power)
print(f"max power bin selected : {max_pwr_idx * range_res + dist_range_bottom}")
phase_plot = unwrapped_phases.T[max_pwr_idx][:sel_range]
phase_plot_2 = unwrapped_phases_2.T[max_pwr_idx][:sel_range]

plt.plot(np.arange(0, sel_range) * 0.05, phase_plot)
plt.xlabel("Time(s)")
plt.ylabel("Unwrapped Phase(Rad)")
plt.title("Phase - Time")
plt.show()

freq_res = 20 / sel_range
phase_fft_1 = np.maximum(np.fft.fft(phase_plot), 0)[:int(sel_range / 2)]
phase_fft_2 = np.maximum(np.fft.fft(phase_plot_2), 0)[:int(sel_range / 2)]
# phase_fft = np.average([phase_fft_1, phase_fft_2], axis=0)

phase_fft = phase_fft_1
# phase_fft = np.fft.fftshift(phase_fft)
phases = np.arange(0, sel_range / 2) * freq_res
min_freq = 0.05
max_freq = 2
phase_mask = (phases > min_freq) & (phases < max_freq)
phases_filtered = phases[phase_mask]
fft_plot_filtered = phase_fft[phase_mask]
plt.plot(phases_filtered * 60, fft_plot_filtered)
plt.title("Spectrum")
plt.xlabel("Frequency(Hz)")
plt.show()

phases = np.arange(0, 500) * freq_res
phase_mask = (phases > freq_range_bottom) & (phases < freq_range_top)
phases_filtered = phases[phase_mask]

# plot_frequencies()