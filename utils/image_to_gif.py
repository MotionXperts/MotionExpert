import os
from PIL import Image

# Usage : Specify the images_directory for the images, the path where the GIF will be saved,
#         and the FPS (frames per second).

# Boxing
images_directory = '/home/weihsin/projects/HybrIK/Boxing/cam3/4_front_1/raw_images'
output_gif_path = '/home/weihsin/projects/HybrIK/Boxing/cam3/4_front_1/output.gif'
FPS = 59

# Figure Skating
images_directory = '/home/weihsin/projects/HybrIK/Output/471706290155159700_0/raw_images'
output_gif_path = '/home/weihsin/projects/HybrIK/Output/471706290155159700_0/output.gif'
FPS = 30
image_files = sorted([
    file for file in os.listdir(images_directory)
    if file.lower().endswith(('.png', '.jpg', '.jpeg'))
])

if not image_files:
    exit()

first_image = Image.open(os.path.join(images_directory, image_files[0]))

frames = [Image.open(os.path.join(images_directory, img)) for img in image_files[1:]]

first_image.save(
    output_gif_path,
    format='GIF',
    save_all=True,
    append_images=frames,
    fps=FPS,
    loop=0         
)