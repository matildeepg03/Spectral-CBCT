import uproot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Open ROOT file
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

# Store total number of particles reaching detector
n_total = len(energy_keV)

# ------------------------------
# Direction filter: along +Z
# ------------------------------
theta_max_deg = 0
cos_theta_max = np.cos(np.deg2rad(theta_max_deg))

# Normalize momentum vectors
p_mag = np.sqrt(px**2 + py**2 + pz**2)
px_norm = px / p_mag
py_norm = py / p_mag
pz_norm = pz / p_mag

# Boolean mask
mask = pz_norm >= cos_theta_max

# Apply mask to get direction-filtered particles
energy_keV_filtered = energy_keV[mask]
x_filtered = x[mask]
y_filtered = y[mask]
z_filtered = z[mask]
weight_filtered = weight[mask]

n_filtered = len(energy_keV_filtered)

# ------------------------------
# Print numbers clearly
# ------------------------------
print(f"Particles reaching detector: {n_total}")
print(f"Particles along +Z within {theta_max_deg}°: {n_filtered}")

# ------------------------------
# Weighted average energy for filtered particles
# ------------------------------
average_energy = np.sum(energy_keV_filtered * weight_filtered) / np.sum(weight_filtered)
print(f"Weighted average photon energy (filtered): {average_energy:.2f} keV")

# ------------------------------
# Plots
# ------------------------------
plt.figure()
plt.hist(energy_keV_filtered, bins=80, weights=weight_filtered)
plt.xlabel("Energy [keV]")
plt.ylabel("Counts")
plt.title("CT X-ray Spectrum (Direction Filtered)")
plt.grid(True)
plt.show()

plt.figure()
plt.hist2d(x_filtered, y_filtered, bins=150, weights=weight_filtered)
plt.xlabel("X [mm]")
plt.ylabel("Y [mm]")
plt.title(f"Detector Fluence Map (θ < {theta_max_deg}°)")
plt.colorbar(label="Counts")
plt.show()

# ------------------------------
# Save filtered CSV
# ------------------------------
df = pd.DataFrame({
    "Energy_keV": energy_keV_filtered,
    "X_mm": x_filtered,
    "Y_mm": y_filtered,
    "Z_mm": z_filtered,
    "Weight": weight_filtered,
    "Px": px[mask],
    "Py": py[mask],
    "Pz": pz[mask]
})

df.to_csv("ct_phasespace_filtered.csv", index=False)
print("Filtered CSV saved as ct_phasespace_filtered.csv")