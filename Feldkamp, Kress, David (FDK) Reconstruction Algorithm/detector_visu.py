import uproot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


# Detector / image parameters (match your other scripts)
nx, ny = 256, 256
x_min, x_max = -50.0, 50.0
y_min, y_max = -50.0, 50.0

angle_deg = 0

file_path = f"ROOT/phasespace_open.root"

# Read phase space 0
f = uproot.open(file_path)
tree = f["psa;1"]

x = tree["PrePosition_X"].array(library="np")
y = tree["PrePosition_Y"].array(library="np")
energy = tree["KineticEnergy"].array(library="np") * 1000.0  # keV if needed

# Bin into detector pixels using a 2D histogram
H, xedges, yedges = np.histogram2d(x, y, bins=[nx, ny], range=[[x_min, x_max], [y_min, y_max]])
# histogram2d returns shape (nx, ny) with x first; transpose to (ny, nx) for imshow
counts = H.T

# Plot
fig, ax = plt.subplots(figsize=(6, 6))

# Use log scale if there are large count variations
if counts.max() > 0:
    norm = colors.LogNorm(vmin=max(counts.min(), 1e-1), vmax=counts.max())
else:
    norm = None

im = ax.imshow(counts, origin="lower", cmap="viridis", norm=norm,
               extent=[x_min, x_max, y_min, y_max], aspect='equal')
ax.set_title(f"Detector Hit Map (Phantom θ = {angle_deg}°)")
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Counts per pixel")

plt.tight_layout()
# Save preview
out_png = f"detection_map{angle_deg}.png"
fig.savefig(out_png, dpi=200)
print(f"Saved detection map to {out_png}")

# Also show (interactive)
plt.show()

print("Max counts:", counts.max())