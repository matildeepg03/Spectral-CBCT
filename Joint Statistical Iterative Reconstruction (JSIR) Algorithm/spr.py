
from dbm import error

import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import scipy.io
import xraydb


def convert_element_name_to_symbol(name):
    name_dict = {}
    
    name_dict['Hydrogen'] = 'H'
    name_dict['Helium'] = 'He'
    name_dict['Lithium'] = 'Li'
    name_dict['Boron'] = 'B'
    name_dict['Beryllium'] = 'Be'
    name_dict['Carbon'] = 'C'
    name_dict['Nitrogen'] = 'N'
    name_dict['Oxygen'] = 'O'
    name_dict['Fluorine'] = 'F'
    name_dict['Neon'] = 'Ne'
    name_dict['Sodium'] = 'Na'
    name_dict['Magnesium'] = 'Mg'
    name_dict['Aluminium'] = 'Al'
    name_dict['Silicon'] = 'Si'
    name_dict['Phosphor'] = 'P'
    name_dict['Sulfur'] = 'S'
    name_dict['Chlorine'] = 'Cl'
    name_dict['Argon'] = 'Ar'
    name_dict['Potassium'] = 'K'
    name_dict['Calcium'] = 'Ca'
    name_dict['Scandium'] = 'Sc'
    name_dict['Titanium'] = 'Ti'
    name_dict['Vandium'] = 'V'
    name_dict['Chromium'] = 'Cr'
    name_dict['Manganese'] = 'Mn'
    name_dict['Iron'] = 'Fe'
    name_dict['Cobalt'] = 'Co'
    name_dict['Nickel'] = 'Ni'
    name_dict['Copper'] = 'Cu'
    name_dict['Zinc'] = 'Zn'
    name_dict['Gallium'] = 'Ga'
    name_dict['Germanium'] = 'Ge'
    name_dict['Yttrium'] = 'Y'
    name_dict['Cadmium'] = 'Cd'
    name_dict['Tellurium'] = 'Te'
    name_dict['Iodine'] = 'I'
    name_dict['Cesium'] = 'Cs'
    name_dict['Gadolinium'] = 'Gd'
    name_dict['Lutetium'] = 'Lu'
    name_dict['Tungsten'] = 'W'
    name_dict['Thallium'] = 'Tl'
    name_dict['Lead'] = 'Pb'
    name_dict['Bismuth'] = 'Bi'
    name_dict['Uranium'] = 'U'

    return name_dict[name]



def convert_element_identifier_to_atomic_number(identifier):
    """
    The identifier could be:
     - element symbol
     - element number, as int or str
     - element name.
     
    This function will return the element number.
    """
    
    
    if type(identifier) == str:
        # The atomic number could be entered
        # as a stirng:
        try:
            atomic_number = int(identifier)
        except ValueError:
            try:
                # Or the element could be entered
                # as the symbol:
                atomic_number = xraydb.atomic_number(identifier)
            except ValueError:
                # Or the element is entered with its
                # full name:
                atomic_number = xraydb.atomic_number(convert_element_name_to_symbol(identifier))



    elif type(identifier) == float:
        atomic_number = int(identifier)

    elif type(identifier) == int:
        atomic_number = int(identifier)

    else:
        raise AttributeError
        
    return atomic_number



def convert_mass_fractions_to_stochiometry(fract_dict):
    """
    
    Input:
        fract_dict: dict
            dictionary with entries "element: weight_fraction".
            For the elements their symbol, their atomic number 
            or their full name can be used.
    """
    
    fraction_sum = 0
    
    for key in fract_dict:
        fraction_sum += fract_dict[key]
        
    assert abs(fraction_sum - 1) < 1E-2, "Weight fractions don't add up to 1."
       
    # For nicer output, find the lowest stochiometric abundance value:
    min_val = 1E6
    for key in fract_dict:
        atomic_number = convert_element_identifier_to_atomic_number(key)
        stoch = fract_dict[key] / xraydb.atomic_mass(atomic_number) 
        
        if (stoch < min_val) and (stoch > 1E-6):
            min_val = stoch

    # Finally, create the output:
    output_string = ""
    for key in fract_dict:
        atomic_number = convert_element_identifier_to_atomic_number(key)
        stoch = fract_dict[key] / xraydb.atomic_mass(atomic_number) / min_val
        
        output_string += f"{xraydb.atomic_symbol(atomic_number)}{stoch:.4f} "
        
    return output_string
                




def calculate_electron_density_from_weight_fractions(fract_dict, mass_density):
    """
    This function calculates the absolute electron density 
    based on a dictionary with the mass fractions for each
    element as well as the mass density.
    
    Input:
        fract_dict: dict
            dictionary with entries "element: weight_fraction".
            For the elements their symbol, their atomic number 
            or their full name can be used.
        
        mass_density: float
            The mass density of the material.
            
            
    Returns:
        rho_elec: float
            The electron density of the material.
    """
    
    # Constant:
    u = 1.660539E-24 # g, not kg!
    
    rho_elec = 0

    for element in fract_dict:
        atomic_number = convert_element_identifier_to_atomic_number(element)
        weight_fraction_i = fract_dict[element]

        A_i = xraydb.atomic_mass(atomic_number)
        Z_i = xraydb.atomic_number(atomic_number)

        rho_elec += weight_fraction_i * Z_i / A_i
        
    return rho_elec*mass_density/u



def calculate_compound_I(material_dict_entry):
    """
    This function computes the ionization potential for 
    a compound material.
    
    The ionization energy of a compound material can be
    calculated using the Bragg additivity rule as follows:

    $$ \ln I_m = \left(\sum \frac{\omega_i\ Z_i}{A_i} \ln I_i \right) \left(\sum \frac{\omega_i\ Z_i}{A_i} \right)^{-1} $$

    with: <br>
    $\omega_i \quad$: weight fraction of element $i$ <br>
    $Z_i \quad$: atomic number of element $i$ <br>
    $A_i \quad$: atomic weight of element $i$ <br>
    $I_i \quad$: ionization energy of element $i$.

    
    Parameters
    ----------
        material_dict_entry: dictionary
            The input is a dictionary that defines a material
            composition and exhibits the following structure
            (example):

                 {'Hydrogen': 0.0743,
                  'Oxygen': 0.2071,
                  'Carbon': 0.5786,
                  'Nitrogen': 0.0196,
                  'Chlorine': 0.0008,
                  'Magnesium': 0.1119,
                  'Silicon': 0.0077}
                 }

                    from skimage.metrics import structural_similarity

    Returns
    -------
        I: float
            The ionization potential for the compound material.
    
    """
    material_composition = material_dict_entry
    numerator   = 0
    denominator = 0
    
    for element in material_composition:
        f = material_composition[element]
        Z = convert_element_identifier_to_atomic_number(element)
        A = xraydb.atomic_mass(Z)
        
        numerator   += f * Z * np.log(get_ionization_energy(Z)) / A
        denominator += f * Z                                    / A
        
        # print(element, Z, A, f)
        
    return np.exp(numerator/denominator)
        
        

def get_ionization_energy(Z):
    """
    This function returns the ionization energy for elements.
    
    The data was taken from Table 2.11 of ICRU report 49:
    Except for H(1), C(6), N(7), O(8), F(9) and Cl(17), 
    the I value for elements in the condensed phase 
    given in Table 2.8 is multiplied by 1.13.
    From this I understand, that the elements for 
    which the value is given in gas phase is not 
    mutliplied by 1.13.
    
    The ICRU report says:
    "This assignment scheme has again been adopted in the
    present work, and is listed in Table 2.11. This scheme 
    is practical but oversimplified; it distinguishes 
    between compounds in the gas and condensed phases, but
    does not take into account the influence of chemical 
    bonds. For constituents of solid materials (except for
    the constituents H, C, N, F and CI), I-values are used 
    which are 13 percent larger than the I-values for 
    elemental substances in the condensed phase. This
    prescription was based on experimental data for 
    aluminum oxide, silicon dioxide and photographic
    emulsion, and its extension to other solids is
    tentative and in need of validation by new 
    experimental data.
    
    Parameters
    ----------
        Z: int
            atomic number of the element
    
    Returns
    -------
        I: float
            The ionization potential of 
            element with atomic number Z.
    
    
    """
    
    result_dict = {
        1:  19.2,         # Table 2.11
        2:  41.8 * 1.13,
        3:  40.0 * 1.13,
        4:  63.7 * 1.13,
        5:  76.0 * 1.13,
        6:  81.0,        # Table 2.11
        7:  82.0,        # Table 2.11
        8:  106,         # Table 2.11
        9:  112,         # Table 2.11
        10: 137,         # Table 2.8, gas
        11: 149 * 1.13,
        12: 156 * 1.13,
        13: 166 * 1.13,
        14: 173 * 1.13,
        15: 173 * 1.13,
        16: 180 * 1.13,
        17: 180,         # Table 2.11
        18: 188,         # Table 2.8, gas
        19: 190 * 1.13,
        20: 191 * 1.13,
        26: 286 * 1.13,
        53: 491 * 1.13 
        }
    
    return result_dict[Z]



def calculate_beta(Ekin):
    """
    Calculates beta based on the given kinetic 
    energy of the proton, with $\beta = v/c$,
    where c is the speed of light and v the 
    velocity of the particle.
    
    Parameters
    ----------
        Ekin: float
            The kinetic energy of the proton.
            
    Returns
    -------
        beta: float
            The beta value of the proton for the 
            given kinetic energy.
    """
    # Rest mass of proton:
    mp = 938.27208816 # MeV/c^2 = 1.67262192369E-27 kg


    # Total energy of protons = kinetic energy plus rest energy:
    Etot = Ekin + mp

    gamma = Etot/mp

    beta = np.sqrt(1-1/gamma**2)
    
    return beta



def calculate_SPR_w(material_dict, Ekin=200):
    """
    The stopping power ratio relative to water, 
    $\mathrm{SPR}_\mathrm{w}$ is calculated according 
    to the Bethe equation:

    $$ \mathrm{SPR}_\mathrm{w} = \rho_e  \frac{\ln \left( \frac{2 m_e c^2 \beta^2}{I (1 - \beta^2)} \right) - \beta^2 }{ \ln \left( \frac{2 m_e c^2 \beta^2}{I_\mathrm{w} (1 - \beta^2)} \right) - \beta^2} $$

    with: 
    $\rho_e \quad$ : the electron density **relative to water**, 
    $m_e \quad $ : electron mass, 
    $I \quad$ : the ionization potential of the given compound,
    $I_\mathrm{w} \quad$: ionization potential of water.


    Parameters
    -----------
        material_dict: dictionary 
            Dictionary with the following entries:
            
            'composition': dict 
                Contains the mass fractions specifying 
                the chemical composition.
                    This dictionary contains the chemical composition. 
                    Example:

                    'composition':   {'Hydrogen': 0.0743,
                                      'Oxygen': 0.2071,
                                      'Carbon': 0.5786,
                                      'Nitrogen': 0.0196,
                                      'Chlorine': 0.0008,
                                      'Magnesium': 0.1119,
                                      'Silicon': 0.0077}
             }
             
             'density': float 
                The mass density of the material in g/cm3.
        
        Ekine: float
            Kinetic energy of protons in units of MeV.
    
    Returns
    -------
        SPR_w: float
            Stopping power ratio of the given material
            relative to water.
            
    """
    
    material_composition_dict = material_dict['composition']
    material_mass_density     = material_dict['density']  
    
    
    ### Constants:
    
    # Speed of light:
    c = float(299792458) # m/s
    
    # Electron mass:
    me = 9.1093837015E-31 # kg
    
    water = {
            'Hydrogen': 0.111902,
            'Oxygen':0.888098
            }
    
    
    ### Values obtained via other functions:
    beta = calculate_beta(Ekin)
    abs_electron_density_compound = calculate_electron_density_from_weight_fractions(material_composition_dict, material_mass_density)
    abs_electron_density_water    = calculate_electron_density_from_weight_fractions(water, 1)
    rel_electron_density = abs_electron_density_compound / abs_electron_density_water
    
    # ionization potential of given compound:
    I = calculate_compound_I(material_composition_dict)
    # ionization potential of water:
    Iw = calculate_compound_I(water)
    
    numerator   = np.log(2 * me * c**2 * beta**2 / ( I  * (1-beta**2) )) - beta**2
    denominator = np.log(2 * me * c**2 * beta**2 / ( Iw * (1-beta**2) )) - beta**2
    
    SPR_w = rel_electron_density * numerator/denominator
    
    return SPR_w



def spr_show(water_true, bone_true, matres2, matres1):

    mass_frac_dict = {
        'Hydrogen'  : 0.034,
        'Carbon'    : 0.155,
        'Nitrogen'  : 0.042,
        'Oxygen'    : 0.435,
        'Sodium'    : 0.001,
        'Magnesium' : 0.002,
        'Phosphor'  : 0.103,
        'Sulfur'    : 0.003,
        'Calcium'   : 0.225
    }

    ICRU_CorticalBoneAdult = {}
    ICRU_CorticalBoneAdult['composition']  = mass_frac_dict
    ICRU_CorticalBoneAdult['stochiometry'] = convert_mass_fractions_to_stochiometry(mass_frac_dict)
    ICRU_CorticalBoneAdult['density']      = 1.920
    mass_frac_dict = {
    'Hydrogen'  : 0.034,
    'Carbon'    : 0.155,
    'Nitrogen'  : 0.042,
    'Oxygen'    : 0.435,
    'Sodium'    : 0.001,
    'Magnesium' : 0.002,
    'Phosphor'  : 0.103,
    'Sulfur'    : 0.003,
    'Calcium'   : 0.225
    }

    ICRU_CorticalBoneAdult = {}
    ICRU_CorticalBoneAdult['composition']  = mass_frac_dict
    ICRU_CorticalBoneAdult['stochiometry'] = convert_mass_fractions_to_stochiometry(mass_frac_dict)
    ICRU_CorticalBoneAdult['density']      = 1.920

    ### Water 

    Water = {}
    Water['composition']  =  {
        'Hydrogen' : 0.111898,
        'Oxygen'   : 0.888102
        }
    Water['stochiometry'] = 'H2O'
    Water['density']      = 1


    print('SPR Water:',calculate_SPR_w(Water, Ekin=200), "SPR Cortical Bone:", calculate_SPR_w(ICRU_CorticalBoneAdult, Ekin=200))

    SPR_gt = water_true * calculate_SPR_w(Water, Ekin=200) + bone_true  * calculate_SPR_w(ICRU_CorticalBoneAdult, Ekin=200)

    fig_SPR_gt = plt.figure()
    plt.imshow(SPR_gt, cmap="gray", vmin=SPR_gt.min(), vmax=SPR_gt.max())
    cbar = plt.colorbar()
    cbar.set_label('SPR')
    plt.title("SPR Ground Truth")
    fig_SPR_gt.savefig(f'/scratch/matildegoncalv/SPR_gt.png')
    plt.close(fig_SPR_gt)

    SPR_rc = matres2 * calculate_SPR_w(Water,Ekin=200) + matres1  * calculate_SPR_w(ICRU_CorticalBoneAdult, Ekin=200)

    fig_SPR_rc = plt.figure()
    plt.imshow(SPR_rc, cmap="gray", vmin=SPR_rc.min(), vmax=SPR_rc.max())
    cbar = plt.colorbar()
    cbar.set_label('SPR')
    plt.title("SPR Reconstruction")
    fig_SPR_rc.savefig(f'/scratch/matildegoncalv/SPR_rc.png')
    plt.close(fig_SPR_rc)

    return SPR_gt, SPR_rc

def compare_to_ground_truth(SPR_rc, SPR_gt):
    
    from skimage.metrics import structural_similarity

    error = SPR_rc - SPR_gt

    mae = np.mean(np.abs(error))
    mse = np.mean(error**2)
    rmse = np.sqrt(mse)

    relative_error = np.divide(error, SPR_gt, out=np.zeros_like(error), where=np.abs(SPR_gt) > 1e-12)
    mean_relative_error_percent = 100 * np.mean(relative_error)
    mean_absolute_relative_error_percent = 100 * np.mean(np.abs(relative_error))
    rms_relative_error_percent = 100 * np.sqrt(np.mean(relative_error**2))

    nrmse_percent = 100 * rmse / np.mean(np.abs(SPR_gt))

    data_range = SPR_gt.max() - SPR_gt.min()
    ssim = structural_similarity(SPR_gt, SPR_rc, data_range=data_range)

    correlation = np.corrcoef(SPR_gt.flatten(), SPR_rc.flatten())[0, 1]

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "NRMSE_percent": nrmse_percent,
        "Mean_relative_error_percent": mean_relative_error_percent,
        "Mean_absolute_relative_error_percent": mean_absolute_relative_error_percent,
        "RMS_relative_error_percent": rms_relative_error_percent,
        "SSIM": ssim,
        "Correlation": correlation,
    }