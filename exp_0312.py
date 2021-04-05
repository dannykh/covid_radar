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
from signal_generation import table

fig = plt.figure()
range_res, _ = dsp.range_resolution(200, 4000, 70)
freq_res = 20 / 1000  # 20 Hz / 1000 samples
ranges = np.arange(adc_samples) * range_res
range_mask = (dist_range_bottom < ranges) & (ranges < dist_range_top)
ranges_filtered = ranges[range_mask]
frequencies = np.arange(0, NUM_FRAMES / 2) * freq_res
frequency_mask = (frequencies > freq_range_bottom) & (frequencies < freq_range_top)
frequencies_filtered = frequencies[frequency_mask]
times = np.arange(0, NUM_FRAMES) * slow_sample_rate
time_mask = (time_filter_bottom < times) & (times < time_filter_top)
times_filtered = times[time_mask]

keyword = "s"

data_set = [
    # "nimrod_10_1",
    # "empty_10_1",
    # "txrx1",
    # "nimrod_10db_12hz"
]

working_data = {
    # name: load_file(all_data[name])[0] for name in data_set
    "stationary": table
}

# working_data = {
#     "nimrod_minus_empty_-10db": raw_data["nimrod_10_1"] - raw_data["empty_10_1"]
# }

# data, range_res, vel_res = load_file("data/010121/adc_data_Raw_0.bin")

# data_avgd = np.average(data, axis=1)

# working_data = {"avg" : data_avgd}

range_maps = {
    name: dsp.range_processing(raw, window_type_1d=Window.HANNING) for name, raw in working_data.items()
}

# range_maps = working_data

range_maps_filtered = {
    name: rtm[:, range_mask] for name, rtm in range_maps.items()
}

raw_power_maps = {
    name: np.sqrt(rtm.imag ** 2 + rtm.real ** 2) for name, rtm in working_data.items()
}

range_power_maps = {
    name: np.sqrt(rtm.imag ** 2 + rtm.real ** 2) for name, rtm in range_maps_filtered.items()
}

i_maps = {
    name: rtm.real for name, rtm in working_data.items()
}

q_maps = {
    name: rtm.imag for name, rtm in working_data.items()
}

avg_power_maps = {
    name: np.average(powers, axis=0) for name, powers in range_power_maps.items()
}

max_power_maps = {
    name: np.std(powers, axis=0) for name, powers in range_power_maps.items()
}

working_set = range_power_maps

for name, powers in working_set.items():
    print(f"{name} : {np.max(powers):,}")

# Extract phases
phase_maps = {
    name: extract_phases(rtm) for name, rtm in range_maps_filtered.items()
}

max_power_idx = np.argmax(avg_power_maps[[x for x in working_data if keyword in x][0]]) + 7
max_power_range = ranges_filtered[0] + max_power_idx * range_res

print(f"Selected range : {max_power_range}")

# Extract bin with max power
max_power_idx_map = {
    name: np.argmax(avg_power) for name, avg_power in avg_power_maps.items()
}

best_phase_maps = {
    name: phase_maps[name].T[max_power_idx] for name, max_idx in max_power_idx_map.items()
}

best_phase_map_partial = {
    name: phases[time_mask] for name, phases in best_phase_maps.items()
}

phase_diffs_maps = {
    name: phases[1:] - phases[:-1] for name, phases in best_phase_map_partial.items()
}

frequency_map = {
    name: np.maximum(0, np.fft.fft(phases))[:int(NUM_FRAMES / 2)] for name, phases in best_phase_maps.items()
}

partial_frequency_maps = {
    name: freqs[frequency_mask] for name, freqs in frequency_map.items()
}

# plot preprocessing
# best_phase_map_partial["nimrod_10_1"] -= 10

# Plots
# while True:
plot_range_maps(raw_power_maps, "Sample", "Power", "Raw")
plot_range_maps(i_maps, "Range(m)", "FFT amp", "In-phase")
plot_range_maps(q_maps, "Range(m)", "FFT amp", "Quadrature")
plot_range_maps(working_set, "Range(m)", "time(s)", "Range-Time")
plot_range_maps(phase_maps, "phase(rad)", "time(m)", "Range-Phase-Time")
plot_named_list(avg_power_maps, ranges_filtered, "range(m)", "Power", scatter=True)
plot_named_list(best_phase_map_partial, times_filtered, "times(s)", "Phase(rad)")
plot_named_list(partial_frequency_maps, frequencies_filtered * 60, "Frequency(BPM)", "Amplitude")
