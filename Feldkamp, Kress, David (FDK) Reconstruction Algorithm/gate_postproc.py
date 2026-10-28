import uproot
import numpy as np

nx, ny = 256, 256

x_min, x_max = -50.0, 50.0
y_min, y_max = -50.0, 50.0

E_thresh = 50.0

epsilon = 1e-3

min_counts = 5  # threshold to avoid log(0)/low statistics

# Open Beam
open_file = uproot.open("ROOT/phasespace_open.root")
tree = open_file["psa;1"]

x0 = tree["PrePosition_X"].array(library="np")
y0 = tree["PrePosition_Y"].array(library="np")
energy0 = tree["KineticEnergy"].array(library="np") * 1000  # keV

# Pixelization
ix0 = ((x0 - x_min) / ((x_max - x_min) / nx)).astype(int)
iy0 = ((y0 - y_min) / ((y_max - y_min) / ny)).astype(int)

ix0 = np.clip(ix0, 0, nx - 1)
iy0 = np.clip(iy0, 0, ny - 1)

# Count open beam
matrix_I0_low = np.zeros((ny, nx))
matrix_I0_high = np.zeros((ny, nx))

for xi, yi, Ei in zip(ix0, iy0, energy0):
    if Ei > 20.0:
        if Ei <= E_thresh:
            matrix_I0_low[yi, xi] += 1
        else:
            matrix_I0_high[yi, xi] += 1


# Phantom files
phantom_files = [f"ROOT/phasespace_{i}.root" for i in range(360)]

atten_low_all = []
atten_high_all = []

for file_name in phantom_files:

    f = uproot.open(file_name)
    tree = f["psa;1"]

    x = tree["PrePosition_X"].array(library="np")
    y = tree["PrePosition_Y"].array(library="np")
    energy = tree["KineticEnergy"].array(library="np") * 1000

    ix = ((x - x_min) / ((x_max - x_min) / nx)).astype(int)
    iy = ((y - y_min) / ((y_max - y_min) / ny)).astype(int)

    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)

    matrix_low = np.zeros((ny, nx))
    matrix_high = np.zeros((ny, nx))

    for xi, yi, Ei in zip(ix, iy, energy):
        if Ei <= E_thresh:
            matrix_low[yi, xi] += 1
        else:
            matrix_high[yi, xi] += 1

    atten_low = np.zeros((ny, nx), dtype=np.float32)
    atten_high = np.zeros((ny, nx), dtype=np.float32)

    mask_low = (matrix_I0_low > min_counts) & (matrix_low > 0)
    mask_high = (matrix_I0_high > min_counts) & (matrix_high > 0)

    atten_low[mask_low] = -np.log(matrix_low[mask_low] / matrix_I0_low[mask_low])
    atten_high[mask_high] = -np.log(matrix_high[mask_high] / matrix_I0_high[mask_high])

    # Optional safety clip (VERY important for FDK)
    atten_low = np.clip(atten_low, 0, 5)
    atten_high = np.clip(atten_high, 0, 5)

    atten_low_all.append(atten_low)
    atten_high_all.append(atten_high)

atten_low_stack = np.stack(atten_low_all, axis=0)
atten_high_stack = np.stack(atten_high_all, axis=0)

print("Values for low energy bin")
print("min, max sinogram:", atten_low_stack.min(), atten_low_stack.max())
print("sinogram value at [128, 90, 128]:", atten_low_stack[128, 90, 128])
print("sinogram value at [128, 0, 128]:", atten_low_stack[128, 0, 128])
print("Values for high energy bin")
print("min, max sinogram:", atten_high_stack.min(), atten_high_stack.max())
print("sinogram value at [128, 90, 128]:", atten_high_stack[127, 90, 128])
print("sinogram value at [128, 0, 128]:", atten_high_stack[128, 0, 128])

np.save("atten_low2.npy", atten_low_stack)
np.save("atten_high2.npy", atten_high_stack)