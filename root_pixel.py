import uproot
import os
import numpy as np
import matplotlib.pyplot as plt
import opengate as gate

# Simulation Parameters
nx, ny = 128, 128             # detector pixels
x_min, x_max = 0.0, 40.0      # cm
y_min, y_max = 0.0, 40.0      # cm
E_thresh = 60.0               # keV energy threshold

epsilon = 1e-8                # to avoid log(0)

# Root files
open = uproot.open("phasespace_open.root")
phantom_files = ["phasespace0.root"]

tree = open["psa;1"]
x0 = tree["PrePosition_X"].array(library="np")
y0 = tree["PrePosition_Y"].array(library="np")
energy0 = tree["KineticEnergy"].array(library="np") * 1000

# Pixelization for open beam
ix0 = ((x0 - x_min) / ((x_max-x_min)/nx)).astype(int)
iy0 = ((y0 - y_min) / ((y_max-y_min)/ny)).astype(int)
ix0 = np.clip(ix0, 0, nx-1)
iy0 = np.clip(iy0, 0, ny-1)

# Count photons per pixel per energy bin
matrix_I0_low = np.zeros((ny, nx))
matrix_I0_high = np.zeros((ny, nx))

for xi, yi, Ei in zip(ix0, iy0, energy0):
    if Ei <= E_thresh:
        matrix_I0_low[yi, xi] += 1
    else:
        matrix_I0_high[yi, xi] += 1

# Process phantom projections
atten_low_all = []
atten_high_all = []

for file_name in phantom_files:
    f = uproot.open(file_name)
    tree = f["psa;1"]
    
    x = tree["PrePosition_X"].array(library="np")  # cm
    y = tree["PrePosition_Y"].array(library="np")
    energy = tree["KineticEnergy"].array() * 1000  # keV

    # Pixelization
    ix = ((x - x_min) / ((x_max-x_min)/nx)).astype(int)
    iy = ((y - y_min) / ((y_max-y_min)/ny)).astype(int)
    ix = np.clip(ix, 0, nx-1)
    iy = np.clip(iy, 0, ny-1)

    # Count photons per pixel per energy bin
    matrix_low = np.zeros((ny, nx))
    matrix_high = np.zeros((ny, nx))

    for xi_idx, yi_idx, Ei in zip(ix, iy, energy):
        if Ei <= E_thresh:
            matrix_low[yi_idx, xi_idx] += 1
        else:
            matrix_high[yi_idx, xi_idx] += 1

    # Compute attenuation
    atten_low = -np.log((matrix_low + epsilon) / (matrix_I0_low + epsilon))
    atten_high = -np.log((matrix_high + epsilon) / (matrix_I0_high + epsilon))

    atten_low_all.append(atten_low)
    atten_high_all.append(atten_high)

# Step 4: Stack and save for FDK
atten_low_stack = np.stack(atten_low_all, axis=0)    # shape: [angles, ny, nx]
atten_high_stack = np.stack(atten_high_all, axis=0)

np.save("atten_low_stack.npy", atten_low_stack)
np.save("atten_high_stack.npy", atten_high_stack)