import uproot
import numpy as np
import astra

nx, ny = 256, 256

x_min, x_max = -50.0, 50.0
y_min, y_max = -50.0, 50.0

E_thresh = 50.0

epsilon = 1e-3

# I0 definition (ASTRA)
N = 256
N_ANGLES = 360
angles = np.deg2rad(np.arange(0, N_ANGLES, 1))
det_spacing = 100/ N  # mm/pixel
source_origin = 2300   # mm
origin_detector = 900  # mm

# Volume
voxel_size = 40 / N  # mm/voxel
min_x, max_x = -20, 20
min_y, max_y = -20, 20
min_z, max_z = -20, 20
vol_geom = astra.create_vol_geom(N, N, N, min_x, max_x, min_y, max_y, min_z, max_z)

blank_phantom = np.zeros((N, N, N), dtype=np.float32)
volume_id = astra.data3d.create('-vol', vol_geom, blank_phantom)

# Projection geometry
proj_geom = astra.create_proj_geom('cone', det_spacing, det_spacing, N, N, angles, source_origin, origin_detector)

# Forward projection
sinogram_id, sinogram = astra.create_sino3d_gpu(volume_id, proj_geom, vol_geom)
print("the astra forward projection shape is:" , sinogram.shape)


# I definition(360 projections)
phantom_files = [f"ROOT2/root_open{i}.root" for i in range(360)]

atten_low_all = []
atten_high_all = []

for proj_idx, file_name in enumerate(phantom_files):

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

    I0 = sinogram[:, proj_idx, :]

    mask_low = (matrix_low > 0) & (I0 > 5)
    mask_high = (matrix_high > 0) & (I0 > 5)

    atten_low[mask_low] = -np.log(matrix_low[mask_low] / I0[mask_low])
    atten_high[mask_high] = -np.log(matrix_high[mask_high] / I0[mask_high])


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
print("sinogram value at [128, 90, 128]:", atten_high_stack[128, 90, 128])
print("sinogram value at [128, 0, 128]:", atten_high_stack[128, 0, 128])

np.save("sinogram_low.npy", atten_low_stack)
np.save("sinogram_high.npy", atten_high_stack)