import numpy as np
import matplotlib.pyplot as plt
import xraydb

def compute_vmi(x_bone, x_water, E1, E2): 
    """
    Compute the Virtual Monochromatic Image (VMI) using the water and bone images.

    Parameters:
        x_water (ndarray): Water image.
        x_bone (ndarray): Bone image.
        E1 (float): Energy level 1.
        E2 (float): Energy level 2.

    Returns:
        VMI (ndarray): Virtual Monochromatic Image.
    """

    vmi_energies = np.array([E1, E2])

    # Define the compositions of water and bone
    water_composition = {'O': 0.888102, 'H': 0.111898}
    bone_composition = {'O': 0.435, 'C': 0.155, 'H': 0.034, 'N': 0.042, 'Na': 0.001, 'Mg': 0.002, 'P': 0.103,
                            'S': 0.003, 'Ca': 0.225}

    mu_rho_water = np.zeros_like(vmi_energies, dtype=float)
    mu_rho_bone = np.zeros_like(vmi_energies, dtype=float)

    for element, fraction in water_composition.items():
        mu_rho_water += fraction * xraydb.mu_elam(element, vmi_energies)

    for element, fraction in bone_composition.items():
        mu_rho_bone += fraction * xraydb.mu_elam(element, vmi_energies)

    mu_water = (mu_rho_water * 1)
    mu_bone = (mu_rho_bone * 1.92)

    VMI_E1 = x_water * mu_water[0] + x_bone * mu_bone[0]

    VMI_E2 = x_water * mu_water[1] + x_bone * mu_bone[1]


    fig_E1 = plt.figure()
    plt.imshow(VMI_E1, cmap="gray", vmin=VMI_E1.min(), vmax=VMI_E1.max())
    cbar = plt.colorbar()
    cbar.set_label(' μ (cm⁻¹)')
    plt.title("VMI (E = 35 keV)")
    fig_E1.savefig(f'/scratch/matildegoncalv/VMI_35keV.png')
    plt.close(fig_E1)

    np.save("VMI_E1.npy", VMI_E1)
    np.save("VMI_E2.npy", VMI_E2)

    fig_E2 = plt.figure()
    plt.imshow(VMI_E2, cmap="gray", vmin=VMI_E2.min(), vmax=VMI_E2.max())
    cbar2 = plt.colorbar()
    cbar2.set_label(' μ (cm⁻¹)')
    plt.title("VMI (E = 75 keV)")
    fig_E2.savefig(f'/scratch/matildegoncalv/VMI_75keV.png')
    plt.close(fig_E2)
    plt.show()
    
    return VMI_E1, VMI_E2, mu_water, mu_bone


def mask_roi_water(objectsize):
    radius = (objectsize*0.3)/2
    center = objectsize /2
    size = objectsize
    Y, X = np.ogrid[:size, :size]
    water_mask = (X - center)**2 + (Y - center)**2 <= radius**2

    return water_mask

   

def mask_roi_bone(objectsize):

    num_rods = 8
    center = objectsize /2
    size = objectsize
    radius_outer = objectsize * 0.4
    radius_small_circles = objectsize * 0.03
    Y, X = np.ogrid[:size, :size]
    mask = np.zeros((size, size), dtype=bool)

    for i in range(num_rods):
        angle = i * 2*np.pi / num_rods
        cx = center + int((radius_outer/2) * np.cos(angle))
        cy = center + int((radius_outer/2) * np.sin(angle))
        rod_mask = (X - cx)**2 + (Y - cy)**2 <= radius_small_circles**2
        mask |= rod_mask      # Add this rod to the mask

    return mask

def roi_statistics(
    image: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """
    Calculate statistics inside an ROI.
    """
    values = image[mask]

    if values.size < 2:
        raise ValueError("The ROI must contain at least two pixels.")

    return {
        "mean": round(float(np.mean(values)), 4),
        "std": round(float(np.std(values, ddof=1)), 4),
        "variance": round(float(np.var(values, ddof=1)), 4),
        "minimum": round(float(np.min(values)), 4),
        "maximum": round(float(np.max(values)), 4),
    }


def calculate_crc(vmi, vmi_true, bone_mask, water_mask):

    # Reconstructed contrast
    vmi_bone = np.mean(vmi[bone_mask])
    vmi_water = np.mean(vmi[water_mask])
    C_recon = (vmi_bone - vmi_water) / vmi_water

    # True contrast
    true_bone = np.mean(vmi_true[bone_mask])
    true_water = np.mean(vmi_true[water_mask])
    C_true = (true_bone - true_water) / true_water

    crc = C_recon / C_true

    return C_recon, crc

