import astra
import numpy as np

# Load projection data
projections = np.load("atten_high2.npy")
projections = np.transpose(projections, (1, 0, 2)) # only with gate data

# Parameters
nx, num_angles, ny = projections.shape
angles_deg = np.arange(0, num_angles, 1)
angles = np.deg2rad(angles_deg)
N_ANGLES = 360
det_spacing = 100 / 256  # mm/pixel
src_obj_dist = 2300
obj_det_dist = 900

# Define volume geometry
vol_geom = astra.create_vol_geom(ny, nx, nx, -20, 20, -20, 20, -20, 20)

# Define projection geometry
proj_geom = astra.create_proj_geom('cone', det_spacing, det_spacing, nx, ny, angles, src_obj_dist, obj_det_dist)

# Create ASTRA 3D projection data object
sinogram_id = astra.data3d.create('-proj3d', proj_geom, projections)

# Create reconstruction data object
recon_id = astra.data3d.create('-vol', vol_geom)

# FDK reconstruction
cfg = astra.astra_dict('FDK_CUDA')
cfg['ProjectionDataId'] = sinogram_id
cfg['ReconstructionDataId'] = recon_id
algorithm_id = astra.algorithm.create(cfg)
astra.algorithm.run(algorithm_id)

# Get the reconstructed volume
reconstruction = astra.data3d.get(recon_id)

print("min, max reconstruction:", reconstruction.min(), reconstruction.max())
print("reconstruction value at [128, 128, 128]:", reconstruction[128, 128, 128])
print("reconstruction value at [1, 1, 1]:", reconstruction[1, 1, 1])

np.save("recon_low_stack.npy", reconstruction)
