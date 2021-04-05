import numpy as np
from mmwave.dataloader import DCA1000

from data_loader import range_res, velocity_res
from etc.config_1 import NUM_FRAMES, NUM_CHIRPS, NUM_RX, NUM_ADC_SAMPLES


def organize(raw_frame, num_chirps, num_rx, num_samples):
    """Reorganizes raw ADC data into a full frame

    Args:
        raw_frame (ndarray): Data to format
        num_chirps: Number of chirps included in the frame
        num_rx: Number of receivers used in the frame
        num_samples: Number of ADC samples included in each chirp

    Returns:
        ndarray: Reformatted frame of raw data of shape (num_chirps, num_rx, num_samples)

    """
    ret = np.zeros(len(raw_frame) // 2, dtype=complex)

    # Separate IQ data
    ret[0::2] = raw_frame[0::4] + 1j * raw_frame[2::4]
    ret[1::2] = raw_frame[1::4] + 1j * raw_frame[3::4]
    return ret.reshape((num_chirps, num_rx, num_samples))


def load_file(file_path, radar_config=None, discard_rx=True):
    adc_data = np.fromfile(file_path, dtype=np.uint16)[::2]
    adc_data = adc_data.reshape(NUM_FRAMES, -1)
    all_data = np.apply_along_axis(organize, 1, adc_data, num_chirps=NUM_CHIRPS, num_rx=NUM_RX,
                                   num_samples=NUM_ADC_SAMPLES)
    if discard_rx:
        # Discard Rx axis
        if NUM_CHIRPS > 1:
            all_data = all_data.reshape(all_data.shape[0], all_data.shape[1], all_data.shape[3])
        else:
            all_data = all_data.reshape(all_data.shape[0], all_data.shape[3])

    return all_data, range_res, velocity_res
