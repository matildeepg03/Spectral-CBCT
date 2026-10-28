import uproot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


# Detector / image parameters (match your other scripts)
nx, ny = 88, 128
x_min, x_max = -44, 44
y_min, y_max = -64, 64

angle_deg = 0

file_path = "root_0_p.root"  # Adjust as needed

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

positive_counts = counts[counts > 0]

# Linear colour scale
vmin = positive_counts.min()
vmax = positive_counts.max()

norm = colors.Normalize(vmin=vmin, vmax=vmax)


im = ax.imshow(counts, origin="lower", cmap="viridis", norm=norm,
               extent=[x_min, x_max, y_min, y_max], aspect='equal')
ax.set_xlabel("Detector X (mm)")
ax.set_ylabel("Detector Y (mm)")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Counts per pixel")

plt.tight_layout()
# Save preview
out_png = f"detection_map{angle_deg}.png"
fig.savefig(out_png, dpi=200)
print(f"Saved detection map to {out_png}")

# Also show (interactive)
plt.show()

mean_counts = positive_counts.mean()
total_counts = positive_counts.sum()

positive_counts = counts[counts > 0]

print("Total pixels:", counts.size)
print("Non-zero pixels:", len(positive_counts))
print("Mean all pixels:", counts.mean())
print("Mean non-zero pixels:", positive_counts.mean())
print("Median non-zero pixels:", np.median(positive_counts))
print("Total detected counts:", total_counts)
print("Maximum:", counts.max())