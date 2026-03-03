import uproot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ------------------------------
# Load ROOT phase space file
# ------------------------------
file = uproot.open("phasespace.root")
tree = file["psa;1"]

# Read branches
energy = tree["KineticEnergy"].array(library="np")   # MeV
x = tree["PrePosition_X"].array(library="np")       # mm
y = tree["PrePosition_Y"].array(library="np")       # mm
z = tree["PrePosition_Z"].array(library="np")       # mm
weight = tree["Weight"].array(library="np")

px = tree["TrackVertexMomentumDirection_X"].array(library="np")
py = tree["TrackVertexMomentumDirection_Y"].array(library="np")
pz = tree["TrackVertexMomentumDirection_Z"].array(library="np")

# Convert energy to keV
energy_keV = energy * 1000

# Total number of particles
n_total = len(energy_keV)

# Direction filter: along +Z
theta_max_deg = 0

# Normalize momentum vectors
p_mag = np.sqrt(px**2 + py**2 + pz**2)
px_norm = px / p_mag
py_norm = py / p_mag
pz_norm = pz / p_mag

# Boolean mask for direction
mask = pz_norm < 0

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

# Weighted average energy
average_energy = np.sum(energy_keV_filtered * weight_filtered) / np.sum(weight_filtered)
print(f"Weighted average photon energy (filtered): {average_energy:.2f} keV")

# Define source spectrum (discrete energies)
source_energies = np.array([7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5, 52.5, 57.5, 62.5, 67.5])  # keV
source_weights = np.array([241093747.19029588, 46387505.50826389, 43705892.59321092, 49858992.22455113, 47612263.51117678, 41533300.05580628, 34508462.16303297, 27800778.030590918, 21776933.99701297, 16423394.86364773, 11605532.26763803, 7078135.240462244, 2554392.4610825721]
)

# Build non-overlapping bin edges from `source_energies` centers
centers = source_energies
if len(centers) < 2:
    raise ValueError("Need at least two source energy centers to build bins")

midpoints = (centers[:-1] + centers[1:]) / 2.0
bin_edges = np.empty(len(centers) + 1)
bin_edges[1:-1] = midpoints
# extend first/last edge by same half-spacing as nearest midpoint
bin_edges[0] = centers[0] - (midpoints[0] - centers[0])
bin_edges[-1] = centers[-1] + (centers[-1] - midpoints[-1])
 
# Weighted histogram of detected energies using filtered weights
counts_weighted, _ = np.histogram(energy_keV_filtered, bins=bin_edges, weights=weight_filtered)
 
# Convert to normalized probabilities (safe against zero total)
total = counts_weighted.sum()
if total > 0:
    probabilities = counts_weighted / total
else:
    probabilities = np.zeros_like(counts_weighted, dtype=float)
 
print("\nDetected weighted probabilities:")
for E, P in zip(centers, probabilities):
     print(f"{E} keV : {P:.4f}")
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