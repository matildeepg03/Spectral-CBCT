import numpy as np

def extract_cylinder_values(volume, center, radius, voxel_size, height_mm=15):
    N = volume.shape[0]
    xc, yc, zc = center

    z_half = int(height_mm / voxel_size)

    X, Y, Z = np.ogrid[:N, :N, :N]

    mask = (
        ((X - xc)**2 + (Y - yc)**2 <= radius**2) &
        (np.abs(Z - zc) <= z_half)
    )

    return volume[mask]

volume = np.load("recon_high_stack.npy")

N = 256
voxel_size = 40 / N

radius_pixels = int(10 / voxel_size)

values = extract_cylinder_values(
    volume,
    center=(128, 128, 128),
    radius=radius_pixels,
    voxel_size=voxel_size
)

expected_voxels = np.pi * radius_pixels**2 * int(30/voxel_size)

mu = values.mean()
sigma = values.std()
snr = mu/ sigma
print("mu =", mu)
print("std =", sigma)
print("SNR =", snr)