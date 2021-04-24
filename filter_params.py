import glob

dist_range_bottom, dist_range_top = 0.1, 3
freq_range_bottom, freq_range_top = 0, 10

slow_sample_rate = 0.05  # sec

time_filter_bottom, time_filter_top = 20, 30  # sec

override_range_bin = None

data_dir = "data/"
all_paths = glob.glob(f"{data_dir}/12_04_21/*Raw*.bin*")

all_data = {
    fname[fname.find("\\") + 1:fname.find("_Raw")]: fname for fname in all_paths
}
