import numpy as np
# Radar specific parameters
from mmwave.dataloader import DCA1000
from mmwave.dsp import range_resolution, doppler_resolution

from etc.config_1 import NUM_RX, NUM_CHIRPS, NUM_ADC_SAMPLES, NUM_FRAMES, sample_rate, freq_slope, adc_samples, \
    start_freq, idle_time, ramp_end_time, RANGE_RESOLUTION

# Data specific parameters

# DSP processing parameters

# Data sampling configuration

# Range and Velocity resolution
range_res, band_width = range_resolution(NUM_ADC_SAMPLES, sample_rate, freq_slope)
# range_res = RANGE_RESOLUTION
print(f'Range Resolution: {range_res} [meters]')
velocity_res = doppler_resolution(band_width, start_freq, ramp_end_time, idle_time, NUM_CHIRPS, 1)
print(f'Velocity Resolution: {velocity_res} [meters/second]')

# Apply the range resolution factor to the range indices
ranges = np.arange(adc_samples) * range_res


def load_file(file_path, radar_config=None, discard_rx=True):
    adc_data = np.fromfile(file_path, dtype=np.uint16)[::2]
    adc_data = adc_data.reshape(NUM_FRAMES, -1)
    all_data = np.apply_along_axis(DCA1000.organize, 1, adc_data, num_chirps=NUM_CHIRPS, num_rx=NUM_RX,
                                   num_samples=NUM_ADC_SAMPLES)
    if discard_rx:
        # Discard Rx axis
        all_data = all_data.reshape(all_data.shape[0],  all_data.shape[3])

    return all_data, range_res, velocity_res
