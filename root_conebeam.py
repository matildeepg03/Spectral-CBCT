import uproot
import os
import numpy as np
import matplotlib.pyplot as plt
import opengate as gate

# define units
cm = gate.g4_units.cm
mm = gate.g4_units.mm

# source and detector positions
source = np.array([0.0, 0.0, 20*cm])     
detector_size = np.array([40*cm, 40*cm])     
detector_z = -18*cm                          

# READ VACUUM ROOT FILE
file = uproot.open("phasespace_vacuum.root")
tree_vacuum = file["psa;1"]
x_vac = tree_vacuum["PrePosition_X"].array(library="np")
y_vac = tree_vacuum["PrePosition_Y"].array(library="np")
weight_vac = tree_vacuum["Weight"].array(library="np")

# READ SIMULATION ROOT FILE
file = uproot.open("phasespace.root")
tree = file["psa;1"]
x = tree["PrePosition_X"].array(library="np") # array of x positions
y = tree["PrePosition_Y"].array(library="np")
weight = tree["Weight"].array(library="np")


# Simulation Parameters
phantom = "water"
rotation = 0  # degrees

# Define detector grid
detector_size = np.array([40, 40])
nx = ny = 128
x_edges = np.linspace(-detector_size[0]/2, detector_size[0]/2, nx+1)
y_edges = np.linspace(-detector_size[1]/2, detector_size[1]/2, ny+1)

# Compute 2D histogram for vacuum and simulation data
# I0: vacuum (incident beam)
I0, _, _ = np.histogram2d(x_vac, y_vac, bins=[x_edges, y_edges], weights=weight_vac)
# I: phantom (after attenuation)
I, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=weight)

# Compute attenuation map
eps = 1e-12  # prevent log(0)
mu_map = -np.log((I + eps) / (I0 + eps))
mu_map = np.clip(mu_map, 0, np.inf)
# Display it
plt.figure(figsize=(6,6))
plt.imshow(mu_map.T, origin='lower',
           extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
           cmap='viridis')
plt.colorbar(label='attenuation')
plt.xlabel('x [mm]')
plt.ylabel('y [mm]')
plt.title("Attenuation Map")
plt.show()

# Compute projection (line integral) along z for each pixel
projection = np.sum(mu_map, axis=1)

# Save prpjection data
out_folder = "projections"
os.makedirs(out_folder, exist_ok=True)
filename = os.path.join(out_folder, f"{phantom}_{np.angle}.npy")
np.save(filename, projection)


""" DIRECTION FILTERING

# expected particle direction
z = tree["PrePosition_Z"].array(library="np")
px = tree["PreDirection_X"].array(library="np")
py = tree["PreDirection_Y"].array(library="np")
pz = tree["PreDirection_Z"].array(library="np")
energy = tree["KineticEnergy"].array(library="np")
preposition = np.vstack((x, y, z)).T
predirection = np.vstack((px, py, pz)).T
pixel_id = tree["TrackVolumeCopyNo"].array(library="np")
r = preposition - source
u_expected = r / np.linalg.norm(r, axis=1, keepdims=True)

# expected particle position at the detector
dz = detector_z - source[2]  # distance along z
t = dz / u_expected[:, 2]    # scale factor to reach detector plane
x_hit = source[0] + u_expected[:, 0] * t
y_hit = source[1] + u_expected[:, 1] * t

# select particles inside the detector bounds
mask_detector = (np.abs(x_hit) <= detector_size[0]/2) & (np.abs(y_hit) <= detector_size[1]/2)
preposition_detected = preposition[mask_detector]
predirection_detected = predirection[mask_detector]
energy_detected = energy[mask_detector] * 1000  # convert to keV
weight_detected = weight[mask_detector]
x_hits = x_hit[mask_detector]
y_hits = y_hit[mask_detector]
u_expected_detected = u_expected[mask_detector]
pixel_id_detected = pixel_id[mask_detector]

# compute angular deviation
u_measured = predirection_detected / np.linalg.norm(predirection_detected, axis=1, keepdims=True)
dot = np.sum(u_expected_detected * u_measured, axis=1)
dot = np.clip(dot, -1.0, 1.0)
angle_deg = np.degrees(np.arccos(dot))

# filter particles based in their deviation
threshold = 0.1  # degrees
no_change = angle_deg <= threshold
changed = angle_deg > threshold

#apply this mask
energy_aligned = energy_detected[no_change]
weight_aligned = weight_detected[no_change]
pixel_id_aligned = pixel_id_detected[no_change]
x_ballistic = x_hits[no_change]
y_ballistic = y_hits[no_change]

total_detected = len(preposition_detected)
scattered_detected = np.sum(changed)
filter = total_detected - scattered_detected
print("Total particles reaching detector:", total_detected)
print("Particles reaching detector that are scattered:", scattered_detected)
print("Fraction scattered:", scattered_detected / total_detected)

 """


projection = np.sum(mu_map, axis=1)

# save projection data
out_folder = "projections"
os.makedirs(out_folder, exist_ok=True)
np.save(os.path.join(out_folder, f"projection_{angle_deg:03d}.npy"), projection)





"""
ENERGY SPECTRUM ANALYSIS

# Define source spectrum (discrete energies)
source_energies = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29.5, 30.5, 31.5, 32.5, 33.5, 34.5, 35.5, 36.5, 37.5, 38.5, 39.5, 40.5, 41.5, 42.5, 43.5, 44.5, 45.5, 46.5, 47.5, 48.5, 49.5, 50.5, 51.5, 52.5, 53.5, 54.5, 55.5, 56.5, 57.5, 58.5, 59.5, 60.5, 61.5, 62.5, 63.5, 64.5, 65.5, 66.5,67.5 ,68.5 ,69.5, 70.5, 71.5, 72.5, 73.5, 74.5, 75.5, 76.5, 77.5, 78.5, 79.5, 80.5, 81.5, 82.5, 83.5, 84.5, 85.5, 86.5, 87.5, 88.5, 89.5, 90.5, 91.5, 92.5, 93.5, 94.5, 95.5, 96.5, 97.5, 98.5, 99.5])
source_weights = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0001, 0.0, 0.0003, 0.0003, 0.0007, 0.0015, 0.0025, 0.0038, 0.0054, 0.0072, 0.009, 0.0109, 0.0125, 0.0141, 0.0155, 0.0169, 0.0181, 0.0191, 0.02, 0.0207, 0.0214, 0.0218, 0.0221, 0.0223, 0.0223, 0.0224, 0.0223, 0.0222, 0.022, 0.0217, 0.0215, 0.0212, 0.0208, 0.0204, 0.02, 0.0195, 0.0191, 0.0186, 0.018, 0.0175, 0.0169, 0.0163, 0.0157, 0.0151, 0.0145, 0.0139, 0.0133, 0.0127, 0.0249, 0.0124, 0.0335, 0.0116, 0.0112, 0.0109, 0.0105, 0.0101, 0.0098, 0.012, 0.0142, 0.0087, 0.0098, 0.0068, 0.0066, 0.0063, 0.0061, 0.0059, 0.0057, 0.0055, 0.0052, 0.005, 0.0048, 0.0046, 0.0044, 0.0041, 0.0039, 0.0037, 0.0035, 0.0032, 0.003, 0.0028, 0.0026, 0.0023, 0.0021, 0.0019, 0.0016, 0.0014, 0.0012, 0.0009, 0.0007, 0.0004, 0.0001])

# Build non-overlapping bin edges from `source_energies` centers
bin_edges = np.arange(1,101,1)
centers = bin_edges[:-1] + 0.5  # centers at 1.5, 2.5, ..., 69.5
 
# histogram of detected energies using filtered weights
counts_weighted, _ = np.histogram(energy_aligned, bins=bin_edges, weights=weight_aligned)
 
# Convert to normalized probabilities
total = counts_weighted.sum()
if total > 0:
    probabilities = counts_weighted / total
 
print("\nAligned weighted probabilities:")
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
tick_positions = np.arange(10.5, 100, 10)
plt.xticks(tick_positions)
# Optional: rotate labels slightly if needed
#plt.xticks(np.arange(0, 101, 10), rotation=45)

plt.xlim(0, 100)  # restrict x-axis to the relevant range
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.show() """



