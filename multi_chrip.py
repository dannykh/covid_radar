import matplotlib.pyplot as plt
import numpy as np
from mmwave import dsp
# Radar specific parameters
from mmwave.dsp import Window

from radar_utils.data_loader import load_file
from dsp_utils import extract_phases
from etc.config_1 import adc_samples, NUM_FRAMES
from filter_params import dist_range_bottom, dist_range_top, all_data, freq_range_top, freq_range_bottom, \
    slow_sample_rate, time_filter_bottom, time_filter_top
from plot_utils import plot_named_list, plot_range_maps

data, range_res, vel_res = load_file("data/010121/adc_data_Raw_0.bin")

data_avgd = np.average(data, axis=1)
data_first = data[:, 0, :]
range_map = dsp.range_processing(data_first, Window.HANNING)

ranges = np.arange(adc_samples) * range_res

range_mask = (dist_range_bottom < ranges) & (ranges < dist_range_top)
ranges_filtered = ranges[range_mask]
range_plot_filtered = range_map[:, range_mask]

powers = np.sqrt(range_plot_filtered.imag ** 2 + range_plot_filtered.real ** 2)
print(np.max(powers))

plot_range_maps({"multi": powers})

avg_power = np.average(powers, axis=0)
plt.plot(ranges_filtered, avg_power)
plt.show()
