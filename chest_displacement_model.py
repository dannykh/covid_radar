import scipy.io
import matplotlib.pyplot as plt
import numpy as np


def do_person(idx):
    mat = scipy.io.loadmat(f"chest_measurements/Free_T{idx}.mat")

    belt_data = mat["mp36_s"][1]
    vic32_data = (mat["vicon_s"][29])
    heart_data = mat["mp36_s"][2]

    belt_data -= np.average(belt_data)
    vic32_data -= np.average(vic32_data)
    heart_data -= np.average(heart_data)

    sample_duration = 180  # sec
    sample_rate = len(belt_data) / sample_duration  # Hz
    sample_period = np.arange(0, len(belt_data)) / sample_rate

    time_window_min, time_window_max = 0, 50
    filtered_period_mask = (time_window_min <= sample_period) & (sample_period <= time_window_max)
    filtered_period = sample_period[filtered_period_mask]

    filtered_sample_belt = belt_data[filtered_period_mask]
    filtered_sample_vic = vic32_data[filtered_period_mask]
    filtered_sample_heart = heart_data[filtered_period_mask]

    belt_fft = np.fft.fft(filtered_sample_belt)[:int(len(filtered_sample_belt) / 2)]

    belt_fft -= np.average(belt_fft)

    freq_min, freq_max = 0, 2  # Hz

    freq_res = sample_rate / len(filtered_sample_belt)
    freqs = np.arange(len(belt_fft)) * freq_res
    freq_mask = (freq_min <= freqs) & (freqs <= freq_max)
    freqs_filtered = freqs[freq_mask]
    belt_fft_filtered = belt_fft[freq_mask]

    plt.plot(filtered_period, filtered_sample_belt)
    # plt.plot(filtered_period, filtered_sample_vic)
    plt.plot(filtered_period, filtered_sample_heart)
    plt.legend(["Full Chest", "ECG"])
    plt.show()

    plt.plot(freqs_filtered * 60, belt_fft_filtered)
    plt.show()


for idx in range(2, 10):
    do_person(idx)
