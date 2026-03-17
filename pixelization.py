import itk

img = itk.imread("pixel_counts_counts.mhd")
arr = itk.array_from_image(img)

with open("pixel_counts.txt", "w") as f:
    for i in range(arr.shape[1]):   # y
        for j in range(arr.shape[2]): # x
            f.write(f"{j} {i} {arr[0,i,j]}\n")

total_particles = 0

# write the txt file and accumulate total
with open("pixel_counts.txt", "w") as f:
    for i in range(arr.shape[1]):   # y
        for j in range(arr.shape[2]): # x
            count = arr[0, i, j]
            f.write(f"{j} {i} {count}\n")
            total_particles += count

print(f"Total number of particles that reached the detector: {int(total_particles)}")