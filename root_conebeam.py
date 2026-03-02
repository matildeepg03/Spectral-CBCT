import uproot
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

# open root file
file = uproot.open("phasespace.root")
tree = file["psa;1"]

# read branched of root file
x = tree["PrePosition_X"].array(library="np") # array of x positions
y = tree["PrePosition_Y"].array(library="np")
z = tree["PrePosition_Z"].array(library="np")
parent_id = tree["ParentID"].array(library="np")
px = tree["PreDirection_X"].array(library="np")
py = tree["PreDirection_Y"].array(library="np")
pz = tree["PreDirection_Z"].array(library="np")
energy = tree["KineticEnergy"].array(library="np")

# do the transpose of the arrays in order to get the particles coordinates
preposition = np.vstack((x, y, z)).T
predirection = np.vstack((px, py, pz)).T

# expected particle direction
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
energy_detected = energy[mask_detector]
x_hits = x_hit[mask_detector]
y_hits = y_hit[mask_detector]
u_expected_detected = u_expected[mask_detector]

# compute angular deviation
u_measured = predirection_detected / np.linalg.norm(predirection_detected, axis=1, keepdims=True)
dot = np.sum(u_expected_detected * u_measured, axis=1)
dot = np.clip(dot, -1.0, 1.0)
angle_deg = np.degrees(np.arccos(dot))

# filter particles based in their deviation
threshold = 0.1  # degrees
no_change = angle_deg <= threshold
changed = angle_deg > threshold


total_detected = len(preposition_detected)
scattered_detected = np.sum(changed)
print("Total particles reaching detector:", total_detected)
print("Particles reaching detector that are scattered:", scattered_detected)
print("Fraction scattered:", scattered_detected / total_detected)

# Average energies
# Convert energies to keV
energy_detected_keV = energy_detected * 1000
# Average energies in keV
avg_energy_no_change = np.mean(energy_detected_keV[no_change])
avg_energy_changed = np.mean(energy_detected_keV[changed])
print(f"Average energy of ballistic photons: {avg_energy_no_change:.2f} keV")
print(f"Average energy of scattered photons: {avg_energy_changed:.2f} keV")


# plot scatter map
x_ballistic = x_hits[no_change]
y_ballistic = y_hits[no_change]
x_scattered = x_hits[changed]
y_scattered = y_hits[changed]

plt.figure(figsize=(8,8))
plt.scatter(x_scattered, y_scattered, s=1, color="red", label="Scattered", alpha=0.5)
plt.scatter(x_ballistic, y_ballistic, s=1, color="blue", label="Ballistic", alpha=0.5)
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.title("Detector Map")
plt.legend()
plt.xlim(-detector_size[0]/2, detector_size[0]/2)
plt.ylim(-detector_size[1]/2, detector_size[1]/2)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()