import uproot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# load root file from phase space
file = uproot.open("phasespace.root")
tree = file["psa;1"]

# read branches
energy = tree["KineticEnergy"].array(library="np")   # MeV
x = tree["PrePosition_X"].array(library="np")       # mm
y = tree["PrePosition_Y"].array(library="np")       # mm
z = tree["PrePosition_Z"].array(library="np")       # mm
weight = tree["Weight"].array(library="np")
px = tree["TrackVertexMomentumDirection_X"].array(library="np")
py = tree["TrackVertexMomentumDirection_Y"].array(library="np")
pz = tree["TrackVertexMomentumDirection_Z"].array(library="np")

# convert energy to keV
energy_keV = energy * 1000

# total number of particles
n_total = len(energy_keV)

# direction filter: along +Z
theta_max_deg = 0
# normalize momentum vectors
# compute magnitude of momentum using pytgagorean theorem
p_mag = np.sqrt(px**2 + py**2 + pz**2)
px_norm = px / p_mag
py_norm = py / p_mag
pz_norm = pz / p_mag
# mask for direction
mask = pz_norm < 0 # creates a boolean array that identifies particles with negative pz (the ones that are alligned)

# Apply mask
energy_keV_filtered = energy_keV[mask]
x_filtered = x[mask]
y_filtered = y[mask]
z_filtered = z[mask]
weight_filtered = weight[mask]
px_filtered = px[mask]
py_filtered = py[mask]
pz_filtered = pz[mask]

n_filtered = len(energy_keV_filtered)

print(f"Particles reaching detector: {n_total}")
print(f"Particles along +Z within {theta_max_deg}°: {n_filtered}")

# Define source spectrum (discrete energies)
source_energies = np.array([7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5, 52.5, 57.5, 62.5, 67.5])  # keV
source_weights = np.array([0.4073, 0.0784, 0.0738, 0.0842, 0.0804, 0.0702, 0.0583, 0.047, 0.0368, 0.0277, 0.0196, 0.012, 0.0043]
)

# Build non-overlapping bin edges from `source_energies` centers
centers = source_energies
midpoints = (centers[:-1] + centers[1:]) / 2.0
bin_edges = np.empty(len(centers) + 1)
bin_edges[1:-1] = midpoints
# extend first/last edge by same half-spacing as nearest midpoint
bin_edges[0] = centers[0] - (midpoints[0] - centers[0])
bin_edges[-1] = centers[-1] + (centers[-1] - midpoints[-1])
 
# histogram of detected energies using filtered weights
counts_weighted, _ = np.histogram(energy_keV_filtered, bins=bin_edges, weights=weight_filtered)
 
# Convert to normalized probabilities
total = counts_weighted.sum()
if total > 0:
    probabilities = counts_weighted / total
 
print("\nDetected weighted probabilities:")
for E, P in zip(centers, probabilities):
     print(f"{E} keV : {P:.5f}")
print("Sum of probabilities =", np.sum(probabilities))
 
# Normalize source weights for direct probability comparison
source_norm = source_weights / np.sum(source_weights)

# Plot: Source vs Detected using index-based positions
group_positions = np.arange(len(centers))
bar_width = 0.4
plt.figure(figsize=(10,5))
plt.bar(group_positions - bar_width/2, source_norm, width=bar_width, label="Source", alpha=0.7)
plt.bar(group_positions + bar_width/2, probabilities, width=bar_width, label="Detected", alpha=0.7)
plt.xticks(group_positions, centers)
plt.xlabel("Energy [keV]")
plt.ylabel("Probability")
plt.title("Source vs Detected Energy Distribution")
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# Save filtered CSV
df = pd.DataFrame({
    "Energy_keV": energy_keV_filtered,
    "X_mm": x_filtered,
    "Y_mm": y_filtered,
    "Z_mm": z_filtered,
    "Weight": weight_filtered,
    "Px": px_filtered,
    "Py": py_filtered,
    "Pz": pz_filtered
})
df.to_csv("ct_phasespace_filtered.csv", index=False)
print("Filtered CSV saved as ct_phasespace_filtered.csv")