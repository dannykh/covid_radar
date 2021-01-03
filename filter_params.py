import glob

dist_range_bottom, dist_range_top = 1.8, 2.4
freq_range_bottom, freq_range_top = 0, 2

slow_sample_rate = 0.05  # sec

time_filter_bottom, time_filter_top = 0, 10  # sec

override_range_bin = None

data_dir = "data/231221_weiz/"
all_paths = glob.glob(f"{data_dir}/*Raw*")

all_data = {
    fname[fname.find("\\") + 1:fname.find("_Raw")]: fname for fname in all_paths
}
