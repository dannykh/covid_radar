import xml.etree.ElementTree as ET
from collections.abc import Callable


class RadarConfig:
    def __init__(self, num_chirps: int, num_rx: int, num_tx: int, num_adc_samples: int,
                 num_frames: int, chirp_sample_rate: int,
                 chirp_rate: int, freq_slope: int, start_freq: float,
                 ramp_start_time: int, ramp_end_time: int,
                 idle_time: int, frame_rate: int, rx_gain: int):
        self.num_chirps = num_chirps
        self.num_rx = num_rx
        self.num_tx = num_tx
        self.num_adc_samples = num_adc_samples,
        self.num_frames = num_frames
        self.chirp_sample_rate = chirp_sample_rate
        self.chirp_rate = chirp_rate
        self.freq_slope = freq_slope
        self.start_freq = start_freq
        self.ramp_start_time = ramp_start_time
        self.ramp_end_time = ramp_end_time
        self.idle_time = idle_time
        self.rx_gain = rx_gain
        self.ramp_duration = self.ramp_end_time - self.ramp_start_time
        self.chirp_bandwidth = self.freq_slope * self.ramp_duration
        self.frame_rate = frame_rate


def get_param_val(root, catagory_name: str, param_name: str, type_converter: Callable = float):
    """
    Get value of a parameter (param_name) in a configuration catagory (catagory_name)
    """
    vals = root.findall(f"./{catagory_name}/param/[@name='{param_name}']")
    return type_converter(vals[0].attrib["value"])


def config_file_parser(fpath: str) -> RadarConfig:
    tree = ET.parse(fpath)
    root = tree.getroot()

    num_tx = sum([get_param_val(root, "apiname_channel_cfg", f"tx{idx}En", int) for idx in (0, 1, 2)])
    num_rx = sum([get_param_val(root, "apiname_channel_cfg", f"rx{idx}En", int) for idx in (0, 1, 2, 3)])
    start_freq = get_param_val(root, "apiname_calmonfreqtxpowlimit_cfg", "freqLimitLowTx1")
    num_adc_samples = get_param_val(root, "apiname_profile_cfg", "numAdcSamples", int)
    freq_slope = get_param_val(root, "apiname_profile_cfg", "freqSlopeConst") * 1e12
    chirp_sample_rate = get_param_val(root, "apiname_profile_cfg", "digOutSampleRate", int) * 1e3  # ksps to hz
    ramp_start_time = get_param_val(root, "apiname_profile_cfg", "txStartTime") * 1e-6
    ramp_end_time = get_param_val(root, "apiname_profile_cfg", "rampEndTime") * 1e-6
    idle_time = get_param_val(root, "apiname_profile_cfg", "idleTimeConst") * 1e-6
    rx_gain = get_param_val(root, "apiname_profile_cfg", "rxGain")
    num_frames = get_param_val(root, "apiname_frame_cfg", "frameCount", int)
    frame_rate = get_param_val(root, "apiname_frame_cfg", "periodicity", int)
    num_chirps = get_param_val(root, "apiname_frame_cfg", "loopCount", int)

    return RadarConfig(num_chirps, num_rx, num_tx, num_adc_samples, num_frames, chirp_sample_rate,
                       0, freq_slope, start_freq, ramp_start_time, ramp_end_time, idle_time, frame_rate, rx_gain)


print(config_file_parser("../configs/1000_frames_50ms_tm_50_tc_1_lvds2_tx1.xml").__dict__)
