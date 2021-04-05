import numpy as np
import scipy
from matplotlib import pyplot as plt

from radar_utils.data_loader import load_file
from dsp_utils import extract_phases
from filter_params import all_data, dist_range_bottom, dist_range_top, time_filter_bottom, time_filter_top, \
    freq_range_bottom, freq_range_top
from plot_utils import plot_range_maps, forceAspect, fixed_aspect_ratio
from scipy.fftpack import fft, ifft
from scipy.signal import hanning

c = 3e8
PI = np.pi

# chirp sequence frequency
f_chirp = 20  # Hz

# ramp frequency
T_r = 50 * 1e-6  # duration of one cycle
f_r = 3.5e9  # Hz
m_w = 70e12

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
    # (1, [(0.1, 0.1, 0), ]),

    (2, [(0.006, 0.2, 0), ]),

    # (2.5, [(0.05, 2, 0),
    #      (0.1, 0.1, 0), ]),

]
belt_dist = 2

mat = scipy.io.loadmat(f"chest_measurements/Free_T{4}.mat")

belt_data = mat["mp36_s"][1] / 1e3  # sampled at 100 Hz, in mm
belt_data_20hz = belt_data[::5][:n_r]


def get_belt_at_chirp(chirp_idx: int):
    return np.zeros((1, n_s)) + belt_dist + belt_data_20hz[chirp_idx]


def generate_harmonic_motion(dist, harmonics, t):
    return dist + np.sum([amp * np.cos(2 * PI * freq * t + phase) for (amp, freq, phase) in harmonics], axis=0)


def get_belt_at_time(t):
    return get_belt_at_chirp(int(t[0] * f_chirp))


def get_all_ranges(t):
    return np.array([generate_harmonic_motion(dist, motion, t) for dist, motion in targets])


def get_all_ranges_belt(t):
    return np.array([get_belt_at_time(t)])


# timeline_2 = np.linspace(0,5,1000)
# ranges_2 = get_all_ranges(timeline_2)[0]
# plt.plot(timeline_2, ranges_2)
# timeline = np.linspace(0, 10, 300)
# ranges = get_all_ranges(timeline)[0]
# plt.plot(timeline, ranges)
# plt.show()
# warped = np.arctan2(np.sin(ranges), np.cos(ranges))
# plt.plot(timeline, warped)
# plt.show()
# unwrapped = np.unwrap(warped)
# plt.plot(timeline, unwrapped)
# plt.show()


def itr(t):
    rs = get_all_ranges(t)

    def _w_itr(r):
        v_veh = 0  # legacy, remove when ready
        return 2 * f_0 * v_veh / c + 2 * m_w * r / c

    # we do t%T_r because the eq. above only valid within the ramp
    v = np.sum([np.cos(2 * np.pi * _w_itr(r) * (t % T_r) + 2 * r * 2 * np.pi * f_0 / c) for r in rs], axis=0)
    return v


def get_delays(ranges_t):
    return 2 * ranges_t / c


def _mixing_func_td(t):
    rngs = get_all_ranges_belt(t)
    t_ds = get_delays(rngs)

    def _mix(t_d):
        return amp_t * amp_r * np.exp(1j * (2 * PI * (m_w * t_d) * (t % T_r) +
                                            2 * PI * f_0 * t_d))

    return np.sum([_mix(t_d) for t_d in t_ds], axis=0)


def _freq_to_dist(freq):
    return (T_r * c / (2 * f_r)) * freq


freq_res = f_s / n_s

# plt.plot(_freq_to_dist(np.arange(0, n_s) * freq_res), np.abs(np.fft.fft(mixed_sig)))
# plt.show()

# t_sample = np.linspace(0, T_M, int(T_M * f_s))
# t_frames = t_sample[::]
# print(len(t_sample))
# v_sample = _mixing_func(t_sample)

table = np.zeros((n_r, n_s), dtype=np.complex)

for chirp_nr in range(n_r):
    t_start = chirp_nr * (1 / f_chirp)
    t_frame = np.linspace(t_start, t_start + T_r, n_s)
    v_sample = _mixing_func_td(t_frame)
    table[chirp_nr, :] = v_sample

# table = load_file(all_data["65cm"])[0]
table -= np.average(table)

chirp0_samples = table[0, :]
chirp0_magnitude = np.abs(fft(chirp0_samples, axis=0))
frequencies = np.arange(0, n_s // 2) * f_s / n_s


def freq_to_range(f):
    return f * c / (2 * m_w)


ranges = np.around(freq_to_range(frequencies), 3)
range_filter = (ranges >= dist_range_bottom) & (ranges < dist_range_top)
ranges = ranges[range_filter]

range_table = np.zeros((n_r, n_s // 2), dtype=np.csingle)

for chirp_nr in range(n_r):
    chirp_ad_values = table[chirp_nr, :]
    chirp_fft = fft(chirp_ad_values)  # FFT
    range_table[chirp_nr, :] = 2.0 / n_s * chirp_fft[:n_s // 2]

range_table = range_table[:, range_filter]
phase_range_map = extract_phases(range_table)

avg_abs_power = np.average(np.abs(range_table), axis=0)
max_power_bin_idx = np.argmax(avg_abs_power)
# max_power_bin_idx = 46
max_power_bin = ranges[max_power_bin_idx]
print(f"Max power at index {max_power_bin_idx} is {max_power_bin}")

max_bin = range_table[:, max_power_bin_idx]
phases = phase_range_map[:, max_power_bin_idx]

displacements = phases * lambda_0 / (4 * PI)

slow_timeline = np.arange(0, n_r) * (1 / f_chirp)
slow_time_filter = (slow_timeline > time_filter_bottom) & (slow_timeline < time_filter_top)
slow_time_filtered = slow_timeline[slow_time_filter]

h = hanning(n_r)
phase_fft = fft(phases * (h))
phase_freqs_full = np.arange(0, n_r) * f_chirp / n_r
phase_frequencies = phase_freqs_full[:len(phase_freqs_full) // 2]

phase_freq_filter = (phase_frequencies > freq_range_bottom) & (phase_frequencies < freq_range_top)
phase_freq_filtered = phase_frequencies[phase_freq_filter]

phase_filter = (np.abs(phase_fft) < 150) | ((phase_freqs_full < 0.1) | (phase_freqs_full > 2))
phases_fft_filtered = phase_fft.copy()
phases_fft_filtered[phase_filter] = 0

reconstructed_phases = ifft(phases_fft_filtered)

aspect = 0.1

fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10, 8), constrained_layout=True)
abs_axes = ax[0, 0]
phi_axes = ax[0, 1]
best_phase_axes = ax[1, 0]
fft_axes = ax[1, 1]
reconstructed_phases_axes = ax[2, 0]
filtered_fft_axes = ax[2, 1]

im_asb = abs_axes.imshow(np.abs(range_table), cmap=plt.get_cmap('RdYlBu'))
abs_axes.set_xticks(range(ranges.size)[::10])
abs_axes.set_xticklabels(ranges[::10], rotation=90)
fig.colorbar(im_asb, ax=abs_axes)
abs_axes.set_xlabel("range [m]")
abs_axes.set_ylabel("chirp number")
abs_axes.set_title("$|A(j\omega)|$")
abs_axes.set_aspect(aspect, adjustable='box')

im_phi = phi_axes.imshow(phase_range_map, cmap=plt.get_cmap('RdYlBu'))
fig.colorbar(im_phi, ax=phi_axes)
phi_axes.set_xlabel("range [m]")
phi_axes.set_ylabel("chirp number")
phi_axes.set_title("$∠ A(j\omega)$")
phi_axes.set_xticks(range(ranges.size)[::10])
phi_axes.set_xticklabels(ranges[::10], rotation=90)
phi_axes.set_aspect(aspect, adjustable='box')

best_phase_axes.plot(slow_time_filtered, belt_data_20hz[slow_time_filter])
best_phase_axes.plot(slow_time_filtered, displacements[slow_time_filter])
best_phase_axes.set_xlabel("Time [s]")
best_phase_axes.set_ylabel("Phase [Rad]")
best_phase_axes.set_title(f"$∠A(j\omega)$ at dist {max_power_bin}")

fft_axes.plot(phase_freq_filtered, np.abs(phase_fft[:n_r // 2][phase_freq_filter]))
fft_axes.set_xlabel("Frequency [Hz]")
fft_axes.set_ylabel("FFT power [pure]")
fft_axes.set_title(f"$∠A(j\omega)$ at dist {max_power_bin} FFT")

filtered_fft_axes.plot(phase_freq_filtered, np.abs(phases_fft_filtered[:n_r // 2][phase_freq_filter]))
filtered_fft_axes.set_xlabel("Frequency [Hz]")
filtered_fft_axes.set_ylabel("FFT power [pure]")
filtered_fft_axes.set_title(f"$∠A(j\omega)$ FFT filtered")

reconstructed_phases_axes.plot(slow_time_filtered, reconstructed_phases[slow_time_filter])
reconstructed_phases_axes.set_xlabel("time [s]")
reconstructed_phases_axes.set_ylabel("Phase [Rad]")
reconstructed_phases_axes.set_title(f"$∠A(j\omega)$ reconstructed")

fig.suptitle("Range FFT table visualized.")

plt.show()
