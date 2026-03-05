import opengate as gate
import numpy as np

# just to test commit

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

# SOURCE
source = sim.add_source("GenericSource", "xraygun")
source.particle = "gamma"
source.attached_to = "world"
source.n = int(100000)
# time intervals
source.start_time = 0 * sec
source.end_time = 1 * sec
# energy
source.energy.type = "spectrum_discrete"
source.energy.spectrum_energies = [7.5*keV, 12.5*keV, 17.5*keV, 22.5*keV, 27.5*keV, 32.5*keV, 37.5*keV, 42.5*keV, 47.5*keV, 52.5*keV, 57.5*keV, 62.5*keV, 67.5*keV]
source.energy.spectrum_weights = [0.4073, 0.0784, 0.0738, 0.0842, 0.0804, 0.0702, 0.0583, 0.047, 0.0368, 0.0277, 0.0196, 0.012, 0.0043]
#source.energy.mono = 70*keV
# FAN BEAM
# position
source.position.type = "point"
source.position.translation = [0, 0, 20*cm]
source.position.dimension = 1 * um
# direction
source.direction.type = "momentum"
source.direction.momentum = [0, 0, -1]
""" # CONE BEAM
# position
source.position.translation = [0, 0, 20*cm]
# angular distribution
source.direction.type = "iso"
source.direction.theta = [0*deg, 12.4*deg] """

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
psa.steps_to_store = "first" # to store only the first interaction of the particle with the volume
psa.attributes = [
    "KineticEnergy",
    "PrePosition",
    "PreDirection",
    "Weight",
    "TrackID",
    "ParentID",
    "TrackVertexMomentumDirection"
]

# phase space FILTERS
# unscattered particles
f_unsc = sim.add_filter("UnscatteredPrimaryFilter", "f_unsc")
psa.filters.append(f_unsc)

# to run the simulation
sim.run_timing_intervals = [[0, 1 * sec]]
sim.run()
