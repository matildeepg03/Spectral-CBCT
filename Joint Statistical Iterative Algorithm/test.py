import numpy as np
import scipy
import xraydb

current_dir = Path(__file__).parent

incident_spectrum_path = current_dir.parent /"Matlab_files" /"incidentSpectrum.mat"
material_attenuation_path = current_dir.parent /"Matlab_files" /"materialAttenuationBoneWater.mat"
detector_response_path = current_dir.parent /"Matlab_files" /"detectorResponse.mat"
# Load MATLAB .mat files containing the required data
incident_spectrum = scipy.io.loadmat(str(incident_spectrum_path))['incidentSpectrum']
material_attenuations = scipy.io.loadmat(str(material_attenuation_path))['materialAttenuations']
detector_response = scipy.io.loadmat(str(detector_response_path))['detectorResponse']

energies = np.arange(1000, 151000, 1000, dtype=float)
print("energies", len(energies))

water_composition = {'O': 0.888102, 'H': 0.111898}
bone_composition = {'O': 0.435, 'C': 0.155, 'H': 0.034, 'N': 0.042, 'Na': 0.001, 'Mg': 0.002, 'P': 0.103, 'S': 0.003, 'Ca': 0.225}

mu_rho_water = np.zeros_like(energies, dtype=float)
mu_rho_bone = np.zeros_like(energies, dtype=float)

for element, fraction in water_composition.items():
    mu_rho_water += fraction * xraydb.mu_elam(element, energies)

for element, fraction in bone_composition.items():
    mu_rho_bone += fraction * xraydb.mu_elam(element, energies)

mu_water = mu_rho_water * 1
mu_bone = mu_rho_bone * 1.92

M = np.column_stack([mu_bone, mu_water])
print("M shape:", M.shape)

s_unbinned = detector_response * incident_spectrum.T
S = np.vstack([
    np.sum(s_unbinned[29:50, :], axis=0),
    np.sum(s_unbinned[50:61, :], axis=0),
    np.sum(s_unbinned[61:71, :], axis=0),
    np.sum(s_unbinned[71:82, :], axis=0),
    np.sum(s_unbinned[82:, :], axis=0),
    ])