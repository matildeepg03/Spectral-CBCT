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
