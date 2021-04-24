import csv
import numpy as np
import pandas as pd
from data_loader import load_file
from scipy.io import savemat
import glob


for idx in range(15,16):
    fname = glob.glob(f"data/19_04_2021/*_{idx}.bin*")[0]
    table, range_res, vel_res = load_file(fname)
    table = np.average(table,1)
    table = table.reshape((table.shape[0], table.shape[1] ))

    # pd.DataFrame(table).to_csv(fname.replace(".bin", ".csv"))
    savemat(fname.replace(".bin", ".mat"), {"radar": table})
