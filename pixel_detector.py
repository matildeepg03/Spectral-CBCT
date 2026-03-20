import opengate as gate
import numpy as np

# create a simulation
sim = gate.Simulation()

# enable the visualization
sim.visu = False
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
sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option4"
world.size = [40*cm, 40*cm, 40*cm]
world.material = "G4_Galactic"

# principal phantom
phantom = sim.add_volume("TubsVolume", "phantom")
phantom.mother = "world"
phantom.rmin = 0 * cm # inner radius
phantom.rmax = 8 * cm # outer radius
phantom.material = "G4_WATER"
phantom.dz = 3 * cm
phantom.color = [0.2, 0.6, 1.0, 0.25]

# SOURCE
source = sim.add_source("GenericSource", "xraygun")
source.particle = "gamma"
source.attached_to = "world"
source.n = int(1e6)
# time intervals
source.start_time = 0 * sec
source.end_time = 1 * sec
# energy
# spectrum
source.energy.type = "spectrum_discrete"
source.energy.spectrum_energies = [1.5*keV, 2.5*keV, 3.5*keV, 4.5*keV, 5.5*keV, 6.5*keV, 7.5*keV, 8.5*keV, 9.5*keV, 10.5*keV, 11.5*keV, 12.5*keV, 13.5*keV, 14.5*keV, 15.5*keV, 16.5*keV, 17.5*keV, 18.5*keV, 19.5*keV, 20.5*keV, 21.5*keV, 22.5*keV, 23.5*keV, 24.5*keV, 25.5*keV, 26.5*keV, 27.5*keV, 28.5*keV, 29.5*keV, 30.5*keV, 31.5*keV, 32.5*keV, 33.5*keV, 34.5*keV, 35.5*keV, 36.5*keV, 37.5*keV, 38.5*keV, 39.5*keV, 40.5*keV, 41.5*keV, 42.5*keV, 43.5*keV, 44.5*keV, 45.5*keV, 46.5*keV, 47.5*keV, 48.5*keV, 49.5*keV, 50.5*keV, 51.5*keV, 52.5*keV, 53.5*keV, 54.5*keV, 55.5*keV, 56.5*keV, 57.5*keV, 58.5*keV, 59.5*keV, 60.5*keV, 61.5*keV, 62.5*keV, 63.5*keV, 64.5*keV, 65.5*keV, 66.5*keV, 67.5*keV, 68.5*keV, 69.5*keV, 70.5*keV, 71.5*keV, 72.5*keV, 73.5*keV, 74.5*keV, 75.5*keV, 76.5*keV, 77.5*keV, 78.5*keV, 79.5*keV, 80.5*keV, 81.5*keV, 82.5*keV, 83.5*keV, 84.5*keV, 85.5*keV, 86.5*keV, 87.5*keV, 88.5*keV, 89.5*keV, 90.5*keV, 91.5*keV, 92.5*keV, 93.5*keV, 94.5*keV, 95.5*keV, 96.5*keV, 97.5*keV, 98.5*keV, 99.5*keV]
source.energy.spectrum_weights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0001, 0.0, 0.0003, 0.0003, 0.0007, 0.0015, 0.0025, 0.0038, 0.0054, 0.0072, 0.009, 0.0109, 0.0125, 0.0141, 0.0155, 0.0169, 0.0181, 0.0191, 0.02, 0.0207, 0.0214, 0.0218, 0.0221, 0.0223, 0.0223, 0.0224, 0.0223, 0.0222, 0.022, 0.0217, 0.0215, 0.0212, 0.0208, 0.0204, 0.02, 0.0195, 0.0191, 0.0186, 0.018, 0.0175, 0.0169, 0.0163, 0.0157, 0.0151, 0.0145, 0.0139, 0.0133, 0.0127, 0.0249, 0.0124, 0.0335, 0.0116, 0.0112, 0.0109, 0.0105, 0.0101, 0.0098, 0.012, 0.0142, 0.0087, 0.0098, 0.0068, 0.0066, 0.0063, 0.0061, 0.0059, 0.0057, 0.0055, 0.0052, 0.005, 0.0048, 0.0046, 0.0044, 0.0041, 0.0039, 0.0037, 0.0035, 0.0032, 0.003, 0.0028, 0.0026, 0.0023, 0.0021, 0.0019, 0.0016, 0.0014, 0.0012, 0.0009, 0.0007, 0.0004, 0.0001]
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

""" # pixelization
pixel_side = 128
pixel_size = 40*cm/pixel_side
dose = sim.add_actor("DoseActor", "pixelization")
dose.attached_to = "detector"
dose.size = [pixel_side, pixel_side, 1]      # number of voxels
dose.spacing = [pixel_size, pixel_size, 5*mm]  # voxel size
dose.edep.active = False
dose.counts.active = True
dose.output_filename = "pixel_counts.mhd"
dose.write_to_disk = True """

""" # pixelization
pixel = sim.add_volume("Box", "pixel")
pixel.mother = "detector"
pixel.size = [pixel_size, pixel_size, 5*mm]
pixel.material = "G4_Galactic"
pixel.color = [0.8, 0.8, 0.8, 0.5]
pixel.translation = gate.geometry.utility.get_grid_repetition(size=[pixel_side, pixel_side, 1], spacing=[pixel_size, pixel_size, 0], start=[-20*cm + pixel_size/2, -20*cm + pixel_size/2, 0])
"""
# phase space actor
psa = sim.add_actor("PhaseSpaceActor", "psa")
psa.attached_to = "detector"
psa.output_filename = "phasespace0.root"
# to store only the first interaction of the particle with the volume
psa.steps_to_store = "first"
psa.attributes = [
      "KineticEnergy",
      "PrePosition",
  ]

# to run the simulation
sim.run_timing_intervals = [[0, 1 * sec]]
sim.run()
