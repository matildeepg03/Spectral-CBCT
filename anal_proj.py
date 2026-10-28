import numpy as np

volume = np.load("G0rot_sinogram_high.npy")
print("volume shape:", volume.shape)

proj_90 = volume[90, :, :].transpose(1,0)     # projection at 90 degrees
proj_0 = volume[0, :, :]       # projection at 0 degrees

# Rectangle bounds
x_min, x_max = 89, 166
y_min, y_max = 102, 153

# Create grid
Y, X = np.ogrid[:proj_0.shape[0], :proj_0.shape[1]]

# Rectangle mask
mask_0 = (X >= x_min) & (X <= x_max) & (Y >= y_min) & (Y <= y_max)

# Extract values
values_0 = proj_0[mask_0]

# Statistics
sinogram = values_0.mean()
sigma = values_0.std()
snr = sinogram / sigma

print("Projection at 0 degrees:")
print("sinogram value =", sinogram)
print("std =", sigma)
print("SNR =", snr)

# Circle center
cx, cy = 128, 128

# Radius
r = (153 - 102) // 2   # 64

# Create coordinate grid
Y, X = np.ogrid[:proj_90.shape[0], :proj_90.shape[1]]

# Circular mask
mask_90 = (X - cx)**2 + (Y - cy)**2 <= r**2

values_90 = proj_90[mask_90]


sinogram = values_90.mean()
sigma = values_90.std()
snr = sinogram / sigma
print("Projection at 90 degrees:")
print("sinogram value =", sinogram)
print("std =", sigma)
print("SNR =", snr)