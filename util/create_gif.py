#%% GIF
import os
import glob
from PIL import Image

# Directory containing PNG images
# ex) "C:/Users/krist/Desktop/images" (windowns)
# ex) "/mnt/c/Users/krist/JAX-FEM-vs-PINNs" (linux)
image_dir = '/mnt/c/Users/krist/JAX-FEM-vs-PINNs/fig/sol_contour/Ground_Truth'

# Find images matching the pattern & sort (relative path)
# ex) result*.png = [result1.png, result2.png, ...]
rel_file_paths = sorted(glob.glob(os.path.join(image_dir, "*.png"))) # default: ascending

# Convert to absolute path
abs_file_paths = [os.path.abspath(file_path) for file_path in rel_file_paths]

# Quit if the folder is empty
if not abs_file_paths:
    print("Cannot find any PNG images.")
    exit()

# Open 1st image & create a base image
img_1 = Image.open(abs_file_paths[0])

# Image list from 2nd image
imgs = [Image.open(file_path) for file_path in abs_file_paths[1:]]

# Output directory
output_path = os.path.join(image_dir, "output.gif")

# Save as GIF by appending images (frames)
img_1.save(output_path,
    save_all=True, # 
    append_images=imgs, # 
    duration=100,  # small: fast / large: slow (n: time between frames)
    loop=0) # 0: repeat forever / n: repeat n times

print("GIF created:", output_path)

# %%
