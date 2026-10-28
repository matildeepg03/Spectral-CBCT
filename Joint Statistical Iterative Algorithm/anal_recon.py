import numpy as np

volume = np.load("ASTRArecon_high.npy")
print("shape:", volume.shape)


x_min, x_max = 32, 224

sub_volume = volume[x_min:x_max, :, :]


cz = sub_volume.shape[1] // 2
cy = sub_volume.shape[2] // 2
r = 64

Z, Y = np.ogrid[:sub_volume.shape[1], :sub_volume.shape[2]]

mask = (Z - cz)**2 + (Y - cy)**2 <= r**2

values = sub_volume[:, mask]


mu = values.mean()
sigma = values.std()
snr = mu / sigma

print("Cylinder (ZY plane):")
print("mean =", mu)
print("std =", sigma)
print("SNR =", snr)