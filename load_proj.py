import numpy as np
import matplotlib.pyplot as plt

data = np.load("atten_low_stack.npy")
print(data.shape)      # good
print(data[0])         # show one projection


plt.imshow(data[0], cmap='gray')
plt.colorbar()
plt.title("Attenuation (low energy)")
plt.show()