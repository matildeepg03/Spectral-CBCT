import opengate as gate
import numpy as np

# create a simulation
sim = gate.Simulation()

# to visualize
sim.visu = True
sim.visu_type = "vrml"
sim.visu_verbose = True

# define units
cm = gate.g4_units.cm
keV = gate.g4_units.keV
deg = gate.g4_units.deg

# set a air world
world = sim.world
world.size = [40*cm, 40*cm, 40*cm]
world.material = "G4_AIR"

# principal phantom
phantom = sim.add_volume("TubsVolume", "phantom")
phantom.mother = "world"
phantom.rmin = 0 * cm # inner radius
phantom.rmax = 8 * cm # outer radius
phantom.material = "G4_WATER"
phantom.color = [0.2, 0.6, 1.0, 0.25]

# mini phantoms that mimmic the tissues
mini_radius = 0.6 * cm

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

    # arrange in a ring
    angle = i * 2 * np.pi / len(tissues)
    s.translation = [
        4 * np.cos(angle) * cm,
        4 * np.sin(angle) * cm,
        0
    ]


# to run the simulation
sim.run()
