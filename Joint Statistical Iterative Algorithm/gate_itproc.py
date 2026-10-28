import uproot
import numpy as np

nz, ny = 44, 64

z_min, z_max = -2.2, 2.2
y_min, y_max = -3.2, 3.2

E_thresh = 50.0

epsilon = 1e-3

min_counts = 5  

window_size = 10


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

    matrix_low = np.zeros((ny, nz), dtype=np.float32)
    matrix_high = np.zeros((ny, nz), dtype=np.float32)

    for zi, yi, Ei in zip(iz, iy, energy):

        if Ei <= E_thresh and Ei >= 20:
            matrix_low[yi, zi] += 1
        elif Ei > E_thresh:
            matrix_high[yi, zi] += 1
        else:
            pass

    return matrix_low, matrix_high

open_files = [f"ROOT/root_{i}.root" for i in range(100)]
print("open_files:", open_files)

open_low_all = []
open_high_all = []

for file_name in open_files:
    low, high = build_energy_matrices(file_name)

    open_low_all.append(low)
    open_high_all.append(high)

open_low_all = np.stack(open_low_all, axis=0)
open_high_all = np.stack(open_high_all, axis=0)

# Phantom files
phantom_files = [f"ROOT_123/root_{i}.root" for i in range(100)]

low_all = []
high_all = []

for i, file_name in enumerate(phantom_files):
    indices = [(i + j) % len(open_files) for j in range(window_size)]

    matrix_I0_low = np.mean(open_low_all[indices], axis=0)
    matrix_I0_high = np.mean(open_high_all[indices], axis=0)

    matrix_low, matrix_high = build_energy_matrices(file_name)

    low = np.zeros((ny, nz), dtype=np.float32)
    high = np.zeros((ny, nz), dtype=np.float32)

    mask_low = (matrix_I0_low > min_counts) & (matrix_low > 0)
    mask_high = (matrix_I0_high > min_counts) & (matrix_high > 0)

    low[mask_low] = -np.log(matrix_low[mask_low] / matrix_I0_low[mask_low])
    high[mask_high] = -np.log(matrix_high[mask_high] / matrix_I0_high[mask_high])

    low = np.clip(low, 0, 5)
    high = np.clip(high, 0, 5)

    low_all.append(low)
    high_all.append(high)

low_stack = np.stack(low_all, axis=0)
high_stack = np.stack(high_all, axis=0)


np.save("log_low.npy", low_stack)
np.save("log_high.npy", high_stack)