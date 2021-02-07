import numpy as np
from matplotlib import pyplot as plt

from data_loader import load_file
from dsp_utils import extract_phases
from filter_params import all_data, dist_range_bottom, dist_range_top
from plot_utils import plot_range_maps, forceAspect, fixed_aspect_ratio
from scipy.fftpack import fft

c = 3e8
PI = np.pi

# chirp sequence frequency
f_chirp = 20  # Hz

# ramp frequency
T_r = 50 * 1e-6  # duration of one cycle
f_r = 4 * 1e9  # Hz
m_w = f_r / T_r

n_r = 1000  # number of chirps
T_M = 50  # sec

# sample settings
f_s = 4e6  # 4 MHz
n_s = int(T_r * f_s) + 1

f_0 = 77.7 * 1e9

# some helpful constants
w_0 = 2 * np.pi * f_0
lambda_0 = c / f_0


def f_transmitted(t):
    return f_0 + m_w * (t % T_r)


def chirp(t):
    return np.cos(2 * np.pi * (f_transmitted(t)) * t)


t = np.linspace(0, T_r, n_s)

# plt.figure(figsize=(15, 5))
# plt.plot(t, f_transmitted(t))
# plt.xlabel("t [s]")
# plt.ylabel("frequency [Hz]")
# plt.title("Chirp sequence Modulation, transmitted signal $f_t(t)$")
# plt.show()

amp_t = 1  # db
amp_r = 1  # db

targets = [
    # (dist, [( Amplitude (m) , Frequency (Hz) , Phase (Rad))...])
    (1, [(0, 0.1, 0), ]),

    (2, [(0.05, 2, 0), ]),

    (3, [(0.05, 2, 0),
         (0.1, 0.1, 0), ]),

]


def get_range(dist, harmonics, t):
    return dist + np.sum([amp * np.cos(2 * PI * freq * t + phase) for (amp, freq, phase) in harmonics], axis=0)


def get_all_ranges(t):
    return np.array([get_range(dist, harmonics, t) for dist, harmonics in targets])


def itr(t):
    rs = get_all_ranges(t)

    def _w_itr(r):
        v_veh = 0  # legacy, remove when ready
        return 2 * f_0 * v_veh / c + 2 * m_w * r / c

    # we do t%T_r because the eq. above only valid within the ramp
    v = np.sum([np.cos(2 * np.pi * _w_itr(r) * (t % T_r) + 2 * r * 2 * np.pi * f_0 / c) for r in rs], axis=0)
    return v


def get_delays(t):
    return 2 * get_all_ranges(t) / c


def _mixing_func(t):
    t_d = get_delays(t % T_r)
    return amp_t * amp_r * np.exp(1j * (2 * PI * (m_w * t_d) * (t % T_r) +
                                        2 * PI * f_0 * t_d +
                                        PI * m_w * t_d ** 2))


def _freq_to_dist(freq):
    return (T_r * c / (2 * f_r)) * freq


# time_line = np.linspace(0, T_r, n_s)
# print(time_line)
# mixed_sig = _mixing_func(time_line)

freq_res = f_s / n_s

# plt.plot(_freq_to_dist(np.arange(0, n_s) * freq_res), np.abs(np.fft.fft(mixed_sig)))
# plt.show()

# t_sample = np.linspace(0, T_M, int(T_M * f_s))
# t_frames = t_sample[::]
# print(len(t_sample))
# v_sample = _mixing_func(t_sample)

table = np.zeros((n_r, n_s))

for chirp_nr in range(n_r):
    t_start = int(chirp_nr * (1 / f_chirp))
    t_frame = np.linspace(t_start, t_start + T_r, n_s)
    v_sample = itr(t_frame)
    table[chirp_nr, :] = v_sample

table = load_file(all_data["nimrod_10_1"])[0]

chirp0_samples = table[0, :]
chirp0_magnitude = np.abs(fft(chirp0_samples))
frequencies = np.arange(0, n_s // 2) * f_s / n_s


def freq_to_range(f):
    return f * c / (2 * m_w)


ranges = freq_to_range(frequencies)
range_filter = (ranges > dist_range_bottom) & (ranges < dist_range_top)
ranges = ranges[range_filter]

plt.figure(figsize=(10, 5))
plt.plot(ranges, 2.0 / n_s * chirp0_magnitude[0:n_s // 2][range_filter])
# plt.plot(ranges, 2.0 / n_s * np.abs(chirp0_magnitude[0:n_s // 2]), "k+")
plt.xlabel("range $r$ [m]")
plt.title("derived range (FFT of chirp_0)")
print(freq_to_range(frequencies)[np.argmax(2.0 / n_s * np.abs(chirp0_magnitude[0:n_s // 2]))])
# plt.show()

range_table = np.zeros((n_r, n_s // 2), dtype=np.csingle)

for chirp_nr in range(n_r):
    chirp_ad_values = table[chirp_nr, :]
    chirp_fft = fft(chirp_ad_values)  # FFT
    range_table[chirp_nr, :] = 2.0 / n_s * chirp_fft[:n_s // 2]

range_table = range_table[:,range_filter]

aspect = 0.1

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
abs_axes = ax[0, 0]
phi_axes = ax[0, 1]
real_axes = ax[1, 0]
imag_axes = ax[1, 1]

im_asb = abs_axes.imshow(np.abs(range_table), cmap=plt.get_cmap('RdYlBu'))
abs_axes.set_xticks(range(ranges.size)[::10])
abs_axes.set_xticklabels(ranges[::10], rotation=90)
fig.colorbar(im_asb, ax=abs_axes)
abs_axes.set_xlabel("range [m]")
abs_axes.set_ylabel("chirp number")
abs_axes.set_title("$|A(j\omega)|$")
abs_axes.set_aspect(aspect, adjustable='box')

im_phi = phi_axes.imshow(extract_phases(range_table), cmap=plt.get_cmap('RdYlBu'))
fig.colorbar(im_phi, ax=phi_axes)
phi_axes.set_xlabel("range [m]")
phi_axes.set_ylabel("chirp number")
phi_axes.set_title("$∠ A(j\omega)$")
phi_axes.set_xticks(range(ranges.size)[::10])
phi_axes.set_xticklabels(ranges[::10], rotation=90)
phi_axes.set_aspect(aspect, adjustable='box')

im_real = real_axes.imshow(np.real(range_table), cmap=plt.get_cmap('RdYlBu'))
fig.colorbar(im_real, ax=real_axes)
real_axes.set_xlabel("range [m]")
real_axes.set_ylabel("chirp number")
real_axes.set_title("Real{$A(j\omega)$}")
real_axes.set_xticks(range(ranges.size)[::10])
real_axes.set_xticklabels(ranges[::10], rotation=90)
real_axes.set_aspect(aspect, adjustable='box')

im_imag = imag_axes.imshow(np.imag(range_table), cmap=plt.get_cmap('RdYlBu'))
fig.colorbar(im_imag, ax=imag_axes)
imag_axes.set_xlabel("range [m]")
imag_axes.set_ylabel("chirp number")
imag_axes.set_title("Imag{$A(j\omega)$}")
imag_axes.set_xticks(range(ranges.size)[::10])
imag_axes.set_xticklabels(ranges[::10], rotation=90)
imag_axes.set_aspect(aspect, adjustable='box')

fig.suptitle("Range FFT table visualized.")

plt.show()
