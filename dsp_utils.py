import numpy as np


def unwrap(phase_map):
    for r in range(1, phase_map.shape[1]):
        phase_diff = phase_map[:, r] - phase_map[:, r - 1]
        add_2pi = np.argwhere(phase_diff < np.pi)
        sub_2pi = np.argwhere(phase_diff > np.pi)
        phase_map[add_2pi, r] += 2 * np.pi
        phase_map[sub_2pi, r] -= 2 * np.pi

    return phase_map


def extract_phases(im_vec):
    phases = np.arctan2(im_vec.imag, im_vec.real)
    unwrapped_phases = unwrap(phases)
    return unwrapped_phases


def get_phase_fft(phase_vec):
    return np.maximum(np.fft.fft(phase_vec), 0)[:int(len(phase_vec) / 2)]
