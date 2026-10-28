import numpy as np
import matplotlib.pyplot as plt

def phantom(size,depth):

    phantom = np.zeros((size, size))
    
    # Define circle parameters
    center = size // 2
    radius_outer = size * 0.4 #12.8
    radius_small_circles = size * 0.04 #1.28

    # Create meshgrid
    Y, X = np.ogrid[:size, :size]
    distance_from_center = np.sqrt((X - center)**2 + (Y - center)**2)
    
    # Create outer circle
    phantom[distance_from_center <= radius_outer] = 1.0 #water density
    
    small_circle_positions = [(size // 7, 0), (- size // 7, 0), (size // 10, size // 10), (-size // 10, size // 10), (size // 10, -size // 10), (-size // 10, -size // 10), (size // 4, size // 10), (-size // 4, size // 10), (size // 4, -size // 10), (-size // 4, -size // 10), (size // 10, size // 4), (-size // 10, size // 4), (size // 10, -size // 4), (-size // 10, -size // 4), (0, size // 7), (0, -size // 7)]
    
    material_densities = [1.92] * len(small_circle_positions)
    for idx in [8, 9, 12, 13]:
        material_densities[idx] = 1.06
        
    for idx in [6, 7, 10, 11]:
        material_densities[idx] = 1.05
        
        
    # Create the small circles with more materials
    for i, (dx, dy) in enumerate(small_circle_positions):
        cx, cy = center + dx, center + dy
        mask = (X - cx)**2 + (Y - cy)**2 <= radius_small_circles**2
        phantom[mask] = material_densities[i] # density
    

    #take a slab of size depth in the center
    center = size // 2
    x_min = center - (depth // 2)
    x_max = x_min + depth   # exclusive upper bound
    phantom_3d = np.zeros((size, size, size))
    phantom_3d[x_min:x_max,:,:] = phantom  # Stack identical slices //mg
    print(x_min,x_max)

    bone_fraction = (phantom_3d == 1.92).astype(float)
    water_fraction = (phantom_3d == 1.0).astype(float)
    blood_fraction = (phantom_3d == 1.06).astype(float) 
    muscle_fraction = (phantom_3d == 1.05).astype(float) 
    
    
    
    water_fraction += muscle_fraction * 1
    bone_fraction  += muscle_fraction * 0.0082

    water_fraction += blood_fraction * 1
    bone_fraction  += blood_fraction * 0.0125

    
    
    bone = bone_fraction * 1.92
    water = water_fraction * 1.0

    x = np.column_stack((bone.flatten(), water.flatten()))

    bone_shape = x[:, 0].reshape((size, size, size))
    plt.imsave("bone_shape.png", bone_shape[size//2, :, :], cmap='gray')
    water_shape = x[:, 1].reshape((size, size, size))
    plt.imsave("water_shape.png", water_shape[size//2, :, :], cmap='gray')
   

    
    # Display the phantom
    #plt.imshow(phantom_3d[depth//2], cmap='gray')
    #plt.title("CT Phantom")
    #plt.axis("off")
    #plt.show(block=False)

    return x

#phantom(64,1)

def phantom_astra(size, depth):
    num_rods = 8
    phantom = np.zeros((size, size))
    
    # Define circle parameters
    center = size // 2
    radius_outer = size * 0.4 #1.28 cm
    radius_small_circles = size * 0.04 #0.128 cm

    # Create meshgrid
    Y, X = np.ogrid[:size, :size]
    distance_from_center = np.sqrt((X - center)**2 + (Y - center)**2)
    
    # Create outer circle
    phantom[distance_from_center <= radius_outer] = 1.0 #water density

    for i in range(num_rods):
        angle = i * 2*np.pi / num_rods
        cx = center + int((radius_outer/2) * np.cos(angle))
        cy = center + int((radius_outer/2) * np.sin(angle))
        rod_mask = (X - cx)**2 + (Y - cy)**2 <= radius_small_circles**2

        phantom[rod_mask] = 1.92 # fill the rods with bone density

    x_min = center - (depth // 2)
    x_max = x_min + depth   # exclusive upper bound
    phantom_3d = np.zeros((size, size, size))
    phantom_3d[x_min:x_max,:,:] = phantom  # Stack identical slices //mg
    
    bone = (phantom_3d == 1.92).astype(float)
    water = (phantom_3d == 1.0).astype(float)

    x = np.column_stack((bone.flatten(), water.flatten()))
    
    return x, bone, water
