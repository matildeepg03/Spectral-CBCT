import astra
import numpy as np
import matplotlib.pyplot as plt
import phantom
from xraydb import material_mu

N = 32
ny = 64
nz = 44
n_proj = 100
angles = np.arange(0, n_proj) * (np.pi / n_proj)

det_spacing = 0.1  # mm/pixel
source_origin = 2300   # mm
origin_detector = 900  # mm

# Volume
voxel_size = 0.1  # mm/voxel
lim = total_width_cm = (N * voxel_size) / 2
vol_geom = astra.create_vol_geom(N, N, N, -lim, lim, -lim, lim, -lim, lim)

# Phanton
radius_mm = 1
radius_pixels = int(radius_mm / voxel_size)
length_mm = radius_mm

energy1 = 35
energy2 = 75
water_mu1 = material_mu('H2O', energy1 * 1000)
water_mu2 = material_mu('H2O', energy2 * 1000)
print("Water mu:", water_mu2)
bone_mu1 = 1.73970371
bone_mu2 = 0.45686043

radius_rod = 0.1
radius_rod_pixels = int(radius_rod / voxel_size)
num_rods = 8

def cylinder_y(N, radius, radius_rod):

    phantom = np.zeros((N, N, N), dtype=np.float32) 
    center = N/2
    xc = yc = center
    z_min = int(center - length_mm / voxel_size)
    z_max = int(center + length_mm / voxel_size)

    # Cylinder creation
    X, Y = np.ogrid[:N, :N]
    mask = (X - xc)**2 + (Y - yc)**2 <= radius**2
    for z in range(z_min, z_max):
        phantom[:, :, z] = mask * water_mu1 # fill the cylinder with water_mu value (this value depends on the energy used)

    # Add rods
    for i in range(num_rods):
        angle = i * 2*np.pi / num_rods
        cx = xc + int((radius/2) * np.cos(angle))
        cy = yc + int((radius/2) * np.sin(angle))

        rod_mask = (X - cx)**2 + (Y - cy)**2 <= radius_rod**2

        for z in range(z_min, z_max):
            phantom[:, :, z][rod_mask] = bone_mu1 # fill the rods with bone_mu value (this value depends on the energy used)

    return phantom

phantom = cylinder_y(N, radius_pixels, radius_rod_pixels)
phantom = phantom.transpose(2, 0, 1)

volume_id = astra.data3d.create('-vol', vol_geom, phantom)

# Projection geometry
proj_geom = astra.create_proj_geom('cone', det_spacing, det_spacing, nz, ny, angles, source_origin, origin_detector)

# Forward projection
sinogram_id, sinogram = astra.create_sino3d_gpu(volume_id, proj_geom, vol_geom)

np.save("E1_bw_sinogram.npy", sinogram)

"""

# Uncomment the following code to add Poisson noise to the sinogram and save it as a new file. 
# This simulates the effect of photon counting noise in X-ray imaging.

# I0 calculation (particles per pixel)
I_source = 1e7
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
I0 = I_total * weight_low  # for low energy bin
 # to detector count
I = I0 * np.exp(-sinogram)
# add poison noise
I_poisson = np.zeros_like(I)
I_poisson = np.random.poisson(I)

sinogram_noise = -np.log(I_poisson / I0)

region = I_poisson[:, 90, :]

plt.figure()
plt.hist(region.ravel(), bins=50)
plt.title("Noisy projection (angle 90°)")
plt.xlabel("Photon counts")
plt.ylabel("Frequency")

#plt.savefig("poisson_high.png", dpi=300, bbox_inches='tight')
plt.close()

print("Mean:", np.mean(region))
print("Variance:", np.var(region))

print("min, max sinogram:", sinogram_noise.min(), sinogram_noise.max())
print("sinogram value at [128, 90, 128]:", sinogram_noise[128, 90, 128])
print("sinogram value at [128, 0, 128]:", sinogram_noise[128, 0, 128])
# Save projections
np.save(f"ASTRA_low.npy", sinogram_noise)
"""

