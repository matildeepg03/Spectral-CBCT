import opengate as gate
import numpy as np

# create a simulation
sim = gate.Simulation()

# enable the visualization
sim.visu = True
sim.visu_type = "vrml"
sim.visu_verbose = True

# define units
cm = gate.g4_units.cm
keV = gate.g4_units.keV
deg = gate.g4_units.deg
mm = gate.g4_units.mm
sec = gate.g4_units.s
um = gate.g4_units.um

# set a world
world = sim.world
world.size = [40*cm, 40*cm, 40*cm]
world.material = "G4_Galactic"

"""
# principal phantom
phantom = sim.add_volume("TubsVolume", "phantom")
phantom.mother = "world"
phantom.rmin = 0 * cm # inner radius
phantom.rmax = 18 * cm # outer radius
phantom.dz =  0.5 * cm   # half-height
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
    ]  """

# # SOURCE

source = sim.add_source("GenericSource", "xraygun")
source.particle = "gamma"
source.attached_to = "world"
source.n = int(1000)
source.start_time = 0 * sec
source.end_time = 1 * sec
source.energy.type = "mono"
source.energy.mono = 70*keV
source.position.translation = [0, 0, 20*cm]
# Angular distribution (cone)
source.direction.type = "iso"
source.direction.theta = [0*deg, 12.4*deg]


""" source = sim.add_source("GenericSource", "ct_source")
source.particle = "gamma"
source.attached_to = "world"
source.n = int(1000)
# # time intervals
source.start_time = 0 * sec
source.end_time = 1 * sec
# # position
source.position.type = "disc"
source.position.translation = [0, 0, 20*cm]
source.position.dimension = 0.25 * mm
# # energy
source.energy.type = "mono"
source.energy.mono = 70*keV
# direction
source.direction.type = "iso"
source.direction.theta = 12.4 * deg """


# DETECTOR
# geometry element
detector = sim.add_volume("Box", "detector")
detector.mother = "world"
detector.size = [40*cm, 40*cm, 5*mm]
detector.translation = [0, 0, -18*cm]
detector.color = [0.35, 0.5, 0.29, 0.75]
detector.material = "G4_Galactic"

# phase space actor
psa = sim.add_actor("PhaseSpaceActor", "psa")
psa.attached_to = "detector"
psa.output_filename = "phasespace.root"
# to store only the first interaction of the particle with the volume
psa.steps_to_store = "first"
psa.attributes = [
      "KineticEnergy",
      "PrePosition",
      "EventID",
      "Weight",
      "TrackVertexMomentumDirection"
  ]

# FILTERS
# unscattered particles
f_unsc = sim.add_filter("UnscatteredPrimaryFilter", "f_unsc")
psa.filters.append(f_unsc)
# angle


# to run the simulation
sim.run_timing_intervals = [[0, 1 * sec]]
sim.run()
