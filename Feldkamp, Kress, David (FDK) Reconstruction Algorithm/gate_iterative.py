"""This script defines an iterative OpenGATE simulation with angular projections and geometry setup."""

import opengate as gate
import numpy as np
import sys
# define units
cm = gate.g4_units.cm
keV = gate.g4_units.keV
deg = gate.g4_units.deg
sec = gate.g4_units.s
n_proj = 360
projAngles = np.arange(0, n_proj) * (np.pi / n_proj)

#n = int(sys.argv[1])
n=0
# Create a simulation
sim = gate.Simulation()
sim.visu = True
sim.visu_type = "vrml"
sim.visu_verbose = True
sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option4"
sim.volume_manager.add_material_database("GateMaterials.db")
# World
world = sim.world
world.size = [300*cm, 300*cm, 460*cm]
world.material = "G4_Galactic"

# Phantom
phantom = sim.add_volume("TubsVolume", "phantom")
phantom.mother = "world"
phantom.rmin = 0 * cm
phantom.rmax = 2.56 * cm
phantom.material = "G4_WATER"
phantom.dz = 0.3 * cm # half-z
phantom.translation = [0, 0, 0]
phantom.color = [0.2, 0.6, 1.0, 0.25]
# mini phantoms that mimmic the tissues
mini_radius = 0.256 * cm
rod_count = 8
rod_material = "BONE_COMPACT_192"
rod_color = [0.7, 0.7, 0.7, 1]

for i in range(rod_count):
    s = sim.add_volume("TubsVolume", f"insert_compact{i + 1}")
    s.mother = "phantom"
    s.rmin = 0 * cm
    s.rmax = mini_radius
    s.material = rod_material
    s.color = rod_color
    s.dz = 0.3 * cm

    # arrange in a ring
    angle = i * 2 * np.pi / rod_count
    s.translation = [
        1.28 *  np.cos(angle) * cm,
        1.28 * np.sin(angle) * cm,
        0
    ]
# Set initial 90 degree rotation around Y to make horizontal
rotation_y = np.array([
    [np.cos(90*deg), 0, np.sin(90*deg)],
    [0, 1, 0],
    [-np.sin(90*deg), 0, np.cos(90*deg)]
])
phantom.rotation = rotation_y
rotation_x = np.array([
    [1, 0, 0],
    [0, np.cos(projAngles[n]), -np.sin(projAngles[n])],
    [0, np.sin(projAngles[n]),  np.cos(projAngles[n])]
])
final_rotation = rotation_x @ rotation_y
phantom.rotation = final_rotation
"""
# Source
source = sim.add_source("GenericSource", "xraygun")
source.particle = "gamma"
source.attached_to = "world"
source.n = int(1e7)
source_initial_pos = np.array([0, 0.0, 230.0]) * cm
source.position.translation = source_initial_pos.tolist()
source.start_time = 0 * sec
source.end_time = 1 * sec
source.energy.type = "spectrum_discrete"
source.energy.spectrum_energies = [1.5*keV, 2.5*keV, 3.5*keV, 4.5*keV, 5.5*keV, 6.5*keV, 7.5*keV, 8.5*keV, 9.5*keV, 10.5*keV, 11.5*keV, 12.5*keV, 13.5*keV, 14.5*keV, 15.5*keV, 16.5*keV, 17.5*keV, 18.5*keV, 19.5*keV, 20.5*keV, 21.5*keV, 22.5*keV, 23.5*keV, 24.5*keV, 25.5*keV, 26.5*keV, 27.5*keV, 28.5*keV, 29.5*keV, 30.5*keV, 31.5*keV, 32.5*keV, 33.5*keV, 34.5*keV, 35.5*keV, 36.5*keV,37.5*keV, 38.5*keV, 39.5*keV, 40.5*keV, 41.5*keV, 42.5*keV, 43.5*keV, 44.5*keV, 45.5*keV, 46.5*keV, 47.5*keV, 48.5*keV, 49.5*keV, 50.5*keV, 51.5*keV, 52.5*keV, 53.5*keV, 54.5*keV, 55.5*keV, 56.5*keV, 57.5*keV, 58.5*keV, 59.5*keV, 60.5*keV, 61.5*keV, 62.5*keV, 63.5*keV, 64.5*keV, 65.5*keV, 66.5*keV, 67.5*keV, 68.5*keV, 69.5*keV, 70.5*keV, 71.5*keV,72.5*keV,73.5*keV,74.5*keV,75.5*keV,76.5*keV,77.5*keV,78.5*keV,79.5*keV,80.5*keV,81.5*keV,82.5*keV,83.5*keV,84.5*keV,85.5*keV,86.5*keV,87.5*keV,88.5*keV,89.5*keV,90.5*keV,91.5*keV,92.5*keV,93.5*keV,94.5*keV,95.5*keV,96.5*keV,97.5*keV,98.5*keV,99.5*keV]
source.energy.spectrum_weights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0001, 0.0, 0.0003, 0.0003, 0.0007, 0.0014, 0.0024, 0.0036, 0.0051, 0.0067, 0.0085, 0.0102, 0.0119, 0.0136, 0.0151, 0.0164, 0.0177, 0.0187, 0.0196, 0.0204, 0.021, 0.0214, 0.0217, 0.0219, 0.022, 0.022, 0.0219, 0.0218, 0.0216, 0.0213, 0.021, 0.0207, 0.0203, 0.0199, 0.0195, 0.0191, 0.0187, 0.0182, 0.0178, 0.0173, 0.0168, 0.0164, 0.0159, 0.0155, 0.015, 0.0146, 0.0141, 0.0137, 0.0257, 0.0129, 0.0346, 0.012, 0.0116, 0.0112, 0.0109, 0.0105, 0.0101, 0.0124, 0.0147, 0.0091, 0.0101, 0.007, 0.0068, 0.0066, 0.0063, 0.0061, 0.0059, 0.0057, 0.0054, 0.0052, 0.005, 0.0047, 0.0045, 0.0043, 0.004, 0.0038, 0.0036, 0.0034, 0.0031, 0.0029, 0.0027, 0.0024, 0.0022, 0.0019, 0.0017, 0.0014, 0.0012, 0.001, 0.0007, 0.0004, 0.0005]
source.direction.type = "iso"
source.direction.theta = [0*deg, 1.38*deg]
# Detector
detector = sim.add_volume("Box", "detector")
detector.mother = "world"
detector.size = np.array([8.8, 12.8, 0.5]) * cm
detector_initial_pos = np.array([0.0, 0.0, -90.0]) * cm
detector.translation = detector_initial_pos.tolist()
detector.color = [0.35, 0.5, 0.29, 0.75]
detector.material = "G4_Galactic"
# Phase space actor
psa = sim.add_actor("PhaseSpaceActor", "psa")
psa.attached_to = "detector"
psa.steps_to_store = "first" # comment if you want the scattered particles as well
psa.attributes = ["KineticEnergy", "PrePosition"]
psa.output_filename = f"ROOT0_correct64/root_{n}.root"
"""
sim.run_timing_intervals = [[0, 1*sec]]
sim.run()
print(f"Simulation completed {n}.")