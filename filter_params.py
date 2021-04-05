import glob

dist_range_bottom, dist_range_top = 0, 3
freq_range_bottom, freq_range_top = 0.1, 2

slow_sample_rate = 0.05  # sec

time_filter_bottom, time_filter_top = 0, 50 # sec

override_range_bin = None

data_dir = "data/"
all_paths = glob.glob(f"{data_dir}/*Raw*")

all_data = {
    fname[fname.find("\\") + 1:fname.find("_Raw")]: fname for fname in all_paths
}
