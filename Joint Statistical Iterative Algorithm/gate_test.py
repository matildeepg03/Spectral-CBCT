import opengate as gate
import numpy as np
import sys


# define units
cm = gate.g4_units.cm
keV = gate.g4_units.keV
deg = gate.g4_units.deg
mm = gate.g4_units.mm
sec = gate.g4_units.s

n= 270

# Create a simulation
sim = gate.Simulation()
sim.visu = True
sim.visu_type = "vrml"
sim.visu_verbose = True
sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option4"

# World
world = sim.world
world.size = [320*cm, 320*cm, 500*cm]
world.material = "G4_Galactic"


# principal phantom
phantom = sim.add_volume("TubsVolume", "phantom")
phantom.mother = "world"
phantom.rmin = 0 * cm # inner radius
phantom.rmax = 18 * cm # outer radius
phantom.dz =  5 * cm   # half-height
phantom.material = "G4_WATER"
phantom.color = [0.2, 0.6, 1.0, 0.25]


# mini phantoms that mimmic the tissues
mini_radius = 1 * cm

tissues = [
    ("lung",   "G4_LUNG_ICRP",           [0.7, 0.7, 0.7, 1]), # light gray
    ("adipose","G4_ADIPOSE_TISSUE_ICRP", [1.0, 0.9, 0.8, 1]), # beige
    ("muscle", "G4_MUSCLE_SKELETAL_ICRP",[1.0, 0.5, 0.5, 1]), # light red
    ("bone",   "G4_BONE_COMPACT_ICRU",   [1.0, 1.0, 1.0, 1]), # white
    ("blood",  "G4_BLOOD_ICRP",          [1.0, 0.0, 0.0, 1]), # red
]


for i, (name, mat, color) in enumerate(tissues):
    s = sim.add_volume("TubsVolume", f"insert_{name}")
    s.mother = "phantom"
    s.rmin = 0 * cm
    s.rmax = mini_radius
    s.material = mat
    s.color = color
    s.dz = 3 * cm

    # arrange in a ring
    angle = i * 2 * np.pi / len(tissues)
    s.translation = [
        9 * np.cos(angle) * cm,
        9 * np.sin(angle) * cm,
        0
    ]

# Set initial 90 degree rotation around Y to make horizontal
rotation_y = np.array([
    [np.cos(90*deg), 0, np.sin(90*deg)],
    [0, 1, 0],
    [-np.sin(90*deg), 0, np.cos(90*deg)]
])

rotation_x = np.array([
    [1, 0, 0],
    [0, np.cos(n*deg), -np.sin(n*deg)],
    [0, np.sin(n*deg),  np.cos(n*deg)]
])

final_rotation = rotation_x @ rotation_y
phantom.rotation = final_rotation

sim.run_timing_intervals = [[0, 1*sec]]
sim.run()
print("Simulation completed")