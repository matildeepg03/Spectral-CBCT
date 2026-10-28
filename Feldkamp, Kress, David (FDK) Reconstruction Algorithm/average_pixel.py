import uproot
import numpy as np


nx, ny = 256, 256

x_min, x_max = -50.0, 50.0
y_min, y_max = -50.0, 50.0

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

# Count particles per pixel
matrix_count = np.zeros((ny, nx))

for xi, yi in zip(ix0, iy0):
        matrix_count[yi, xi] += 1

# Total particles counted
total_particles = int(matrix_count.sum())

# Average particles per pixel (mean across all pixels)
avg_per_pixel = float(matrix_count.mean())

print(f"Total particles: {total_particles}")
print(f"Average particles per pixel: {avg_per_pixel}")

# Define pixel centers (coordinates in mm)
dx = (x_max - x_min) / nx
dy = (y_max - y_min) / ny
x_centers = x_min + (np.arange(nx) + 0.5) * dx
y_centers = y_min + (np.arange(ny) + 0.5) * dy
X, Y = np.meshgrid(x_centers, y_centers)

# Circle mask (radius in mm)
radius = 40.0
mask_circle = (X**2 + Y**2) <= (radius ** 2)

count_in_circle = int(matrix_count[mask_circle].sum())
avg_in_circle = float(matrix_count[mask_circle].mean()) if mask_circle.sum() > 0 else 0.0

print(f"Particles inside circle radius {radius} mm: {count_in_circle}")
print(f"Average per pixel inside circle: {avg_in_circle}")

# Print center pixel value (128,128) and save results
center_val = int(matrix_count[128, 128])
print(f"Value at pixel [128,128]: {center_val}")


