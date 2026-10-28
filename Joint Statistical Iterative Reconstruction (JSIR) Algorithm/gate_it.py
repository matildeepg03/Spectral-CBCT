import uproot
import numpy as np

nz, ny = 88, 128
n_proj = 360

z_min, z_max = -44, 44
y_min, y_max = -64, 64

E_thresh = 50.0

def build_energy_matrices(file_name):

    f = uproot.open(file_name)
    tree = f["psa;1"]

    z = tree["PrePosition_X"].array(library="np")
    y = tree["PrePosition_Y"].array(library="np")
    energy = tree["KineticEnergy"].array(library="np") * 1000

    iz = ((z - z_min) / ((z_max - z_min) / nz)).astype(int)
    iy = ((y - y_min) / ((y_max - y_min) / ny)).astype(int)

    iz = np.clip(iz, 0, nz - 1)
    iy = np.clip(iy, 0, ny - 1)

    matrix_low = np.zeros((nz, ny), dtype=np.float32)
    matrix_high = np.zeros((nz, ny), dtype=np.float32)

    for zi, yi, Ei in zip(iz, iy, energy):

        if Ei <= E_thresh and Ei > 20:
            matrix_low[zi, yi] += 1
        elif Ei > E_thresh:
            matrix_high[zi, yi] += 1
        else:
            pass

    return matrix_low, matrix_high

open_files = [f"ROOT_correct64/root_{i}.root" for i in range(n_proj)]
print("open_files:", open_files)

open_low = []
open_high = []

for file_name in open_files:
    low, high = build_energy_matrices(file_name)

    open_low.append(low)
    open_high.append(high)

open_low = np.stack(open_low, axis=1)
open_high = np.stack(open_high, axis=1)

np.save("low_64correct.npy", open_low)
np.save("high_64correct.npy", open_high)