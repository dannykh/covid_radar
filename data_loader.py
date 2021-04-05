import numpy as np
# Radar specific parameters
from mmwave.dsp import range_resolution, doppler_resolution

from etc.config_1 import NUM_CHIRPS, NUM_ADC_SAMPLES, sample_rate, freq_slope, adc_samples, \
    start_freq, idle_time, ramp_end_time

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


