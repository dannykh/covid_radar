import numpy as np

from etc.config_1 import start_freq, freq_slope, NUM_FRAMES, RANGE_RESOLUTION
from matplotlib import pyplot as plt

T_c = 50 * 10e-6  # 50 micro sec chirp time
f_c = 77 * 10e9  # 77 GHz initial frequency
B = 4 * 10e9  # 4 GHz bandwidth
num_frames = NUM_FRAMES
c = 3e8  # m/s
sample_rate = 4 * 10e6  # Hz
K = B / T_c  # B/T_c
PI = np.pi

amp_t = 1  # db
amp_r = 1  # db
# object :
d_0 = 1  # meter
t_delay = 2 * d_0 / c

num_samples = 200

time_line = np.linspace(0, T_c, num_samples)
print(time_line)


def _transmit_func(t):
    return amp_t * np.cos(2 * PI * f_c * t + PI * (B / T_c) * t ** 2)


single_chirp_t = _transmit_func(time_line)
single_chirp_r = _transmit_func(time_line - t_delay)


def _mixing_func(t, t_d):
    return amp_t * amp_r * np.exp(1j * (2 * PI * (B * t_d / T_c) * t +
                                        2 * PI * f_c * t_d +
                                        PI * (B / T_c) * t_d ** 2))


mixed_sig = _mixing_func(time_line, t_delay)


# plt.xlim(0, T_c)
# plt.scatter(time_line, mixed_sig)
# plt.show()

def _freq_to_dist(freq):
    return (T_c * c / (2 * B)) * freq


freq_res = sample_rate / num_samples

plt.plot(_freq_to_dist(np.arange(0, num_samples) * freq_res), np.abs(np.fft.fft(mixed_sig)))
plt.show()

# x_t = np.cos()
