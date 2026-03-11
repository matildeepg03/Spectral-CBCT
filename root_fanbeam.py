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
source_energies = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29.5, 30.5, 31.5, 32.5, 33.5, 34.5, 35.5, 36.5, 37.5, 38.5, 39.5, 40.5, 41.5, 42.5, 43.5, 44.5, 45.5, 46.5, 47.5, 48.5, 49.5, 50.5, 51.5, 52.5, 53.5, 54.5, 55.5, 56.5, 57.5, 58.5, 59.5, 60.5, 61.5, 62.5, 63.5, 64.5, 65.5, 66.5,67.5 ,68.5 ,69.5, 70.5, 71.5, 72.5, 73.5, 74.5, 75.5, 76.5, 77.5, 78.5, 79.5, 80.5, 81.5, 82.5, 83.5, 84.5, 85.5, 86.5, 87.5, 88.5, 89.5, 90.5, 91.5, 92.5, 93.5, 94.5, 95.5, 96.5, 97.5, 98.5, 99.5])
source_weights = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0001, 0.0, 0.0003, 0.0003, 0.0007, 0.0015, 0.0025, 0.0038, 0.0054, 0.0072, 0.009, 0.0109, 0.0125, 0.0141, 0.0155, 0.0169, 0.0181, 0.0191, 0.02, 0.0207, 0.0214, 0.0218, 0.0221, 0.0223, 0.0223, 0.0224, 0.0223, 0.0222, 0.022, 0.0217, 0.0215, 0.0212, 0.0208, 0.0204, 0.02, 0.0195, 0.0191, 0.0186, 0.018, 0.0175, 0.0169, 0.0163, 0.0157, 0.0151, 0.0145, 0.0139, 0.0133, 0.0127, 0.0249, 0.0124, 0.0335, 0.0116, 0.0112, 0.0109, 0.0105, 0.0101, 0.0098, 0.012, 0.0142, 0.0087, 0.0098, 0.0068, 0.0066, 0.0063, 0.0061, 0.0059, 0.0057, 0.0055, 0.0052, 0.005, 0.0048, 0.0046, 0.0044, 0.0041, 0.0039, 0.0037, 0.0035, 0.0032, 0.003, 0.0028, 0.0026, 0.0023, 0.0021, 0.0019, 0.0016, 0.0014, 0.0012, 0.0009, 0.0007, 0.0004, 0.0001])

bin_edges = np.arange(1, 101, 1)
centers = bin_edges[:-1] + 0.5
 
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
group_positions = centers
bar_width = 0.4
plt.figure(figsize=(10,5))
plt.bar(group_positions - bar_width/2, source_norm, width=bar_width, label="Source", alpha=0.7)
plt.bar(group_positions + bar_width/2, probabilities, width=bar_width, label="Detected", alpha=0.7)
plt.xticks(group_positions, centers)
plt.xlabel("Energy [keV]")
plt.ylabel("Probability")
plt.title("Source vs Detected Energy Distribution")

# Only show every 10 keV label
tick_positions = np.arange(10.5, 100, 20)
plt.xticks(tick_positions)
# Optional: rotate labels slightly if needed
#plt.xticks(np.arange(0, 101, 10), rotation=45)

plt.xlim(0, 100)  # restrict x-axis to the relevant range
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