import numpy as np
# SpekpyWeb (Spekpy ver. 2.0.8)
# -----------------------
# Physics: casim
# Attenuation data: nist
# Energy bin (keV): 5
# Target material: W
# Anode angle (°): 12
# kV: 70
# x (cm): 0
# y (cm): 0
# z (cm): 38
# mAs: 1
# Bremsstrahlung: true
# Char. x-ray: true

# Filtration: None

# Mid-bin energy (keV)	Fluence (cm⁻² keV⁻¹)
# SpekpyWeb (Spekpy ver. 2.0.8)
# -----------------------
# Physics: casim
# Attenuation data: nist
# Energy bin (keV): 5
# Target material: W
# Anode angle (°): 7
# kV: 70
# x (cm): 0
# y (cm): 0
# z (cm): 38
# mAs: 1
# Bremsstrahlung: true
# Char. x-ray: true

# Filtration: None

# Mid-bin energy (keV)	Fluence (cm⁻² keV⁻¹)
energy_bins = [7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5, 52.5, 57.5, 62.5, 67.5]
fluence = [241093747.19029588, 46387505.50826389, 43705892.59321092, 49858992.22455113, 47612263.51117678, 41533300.05580628, 34508462.16303297, 27800778.030590918, 21776933.99701297, 16423394.86364773, 11605532.26763803, 7078135.240462244, 2554392.4610825721]

# normalize the fluence
weights = []
n = len(energy_bins)
for i in range (0, n):
    weights.append(fluence[i] / sum(fluence))
    i = i + 1

print("Weights: ", weights)
    