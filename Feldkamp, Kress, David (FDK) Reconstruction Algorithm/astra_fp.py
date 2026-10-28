import astra
import numpy as np
from xraydb import material_mu

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

# Phanton
radius_mm = 10
radius_pixels = int(radius_mm / voxel_size)

energy_min = 50
energy_max = 100
energy = (energy_min + energy_max) / 2
mu = material_mu('H2O', energy * 1000)
mu = mu/10
#print("mu:", mu)

def cylinder_y(N, radius, value):

    phantom = np.zeros((N, N, N), dtype=np.float32) 
    center = N//2
    xc = yc = center
    z_min = int(center - 15/voxel_size)
    z_max = int(center + 15/voxel_size)

    # XZ circle mask
    X, Y = np.ogrid[:N, :N]
    mask = (X - xc)**2 + (Y - yc)**2 <= radius**2

    for z in range(z_min, z_max):
        phantom[:, :, z] = mask * value
    return phantom
    
phantom = cylinder_y(N, radius_pixels, mu)

volume_id = astra.data3d.create('-vol', vol_geom, phantom)

# Projection geometry
proj_geom = astra.create_proj_geom('cone', det_spacing, det_spacing, N, N, angles, source_origin, origin_detector)

# Forward projection
sinogram_id, sinogram = astra.create_sino3d_gpu(volume_id, proj_geom, vol_geom)
#print("min, max sinogram:", sinogram.min(), sinogram.max())
#print("sinogram value at [128, 90, 128]:", sinogram[128, 90, 128])
#print("sinogram value at [128, 0, 128]:", sinogram[128, 0, 128])

# I0 calculation (particles per pixel)
I_source = 1e8
pixel_area = det_spacing ** 2
SDD = source_origin + origin_detector
half_angle = np.deg2rad(0.7)
term2 = pixel_area / (4* np.pi * SDD**2)
term3 = 1 / (np.sin(half_angle/2) ** 2)
I_total = I_source * term2 * term3
print("I_total:", I_total)
spectrum_weights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0001, 0.0, 0.0003, 0.0003, 0.0007, 0.0014, 0.0024, 0.0036, 0.0051, 0.0067, 0.0085, 0.0102, 0.0119, 0.0136, 0.0151, 0.0164, 0.0177, 0.0187, 0.0196, 0.0204, 0.021, 0.0214, 0.0217, 0.0219, 0.022, 0.022, 0.0219, 0.0218, 0.0216, 0.0213, 0.021, 0.0207, 0.0203, 0.0199, 0.0195, 0.0191, 0.0187, 0.0182, 0.0178, 0.0173, 0.0168, 0.0164, 0.0159, 0.0155, 0.015, 0.0146, 0.0141, 0.0137, 0.0257, 0.0129, 0.0346, 0.012, 0.0116, 0.0112, 0.0109, 0.0105, 0.0101, 0.0124, 0.0147, 0.0091, 0.0101, 0.007, 0.0068, 0.0066, 0.0063, 0.0061, 0.0059, 0.0057, 0.0054, 0.0052, 0.005, 0.0047, 0.0045, 0.0043, 0.004, 0.0038, 0.0036, 0.0034, 0.0031, 0.0029, 0.0027, 0.0024, 0.0022, 0.0019, 0.0017, 0.0014, 0.0012, 0.001, 0.0007, 0.0004, 0.0005]
weight_low = sum(spectrum_weights[20:49])  # sum of weights for energies <= 20 keV
weight_high = sum(spectrum_weights[49:99])  # sum of weights for energies > 20 keV and <= 50 keV
I0 = I_total * weight_high  # for high energy bin
 # to detector count
I = I0 * np.exp(-sinogram)
# add poison noise
I_poisson = np.zeros_like(I)
I_poisson = np.random.poisson(I)
# to density
#val = 1e-20  # to avoid log(0)
sinogram_noise = -np.log(I_poisson / I0)


print("min, max sinogram:", sinogram_noise.min(), sinogram_noise.max())
print("sinogram value at [128, 90, 128]:", sinogram_noise[128, 90, 128])
print("sinogram value at [128, 0, 128]:", sinogram_noise[128, 0, 128])
# Save projections
np.save(f"projections_poisson_high.npy", sinogram_noise)


