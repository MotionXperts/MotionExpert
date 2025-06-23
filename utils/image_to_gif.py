import os
from PIL import Image

image_folder = 'dir'
output_gif_path = 'output.gif'        

image_files = sorted([
    file for file in os.listdir(image_folder)
    if file.lower().endswith(('.png', '.jpg', '.jpeg'))
])

if not image_files:
    exit()

first_image = Image.open(os.path.join(image_folder, image_files[0]))

frames = [Image.open(os.path.join(image_folder, img)) for img in image_files[1:]]

first_image.save(
    output_gif_path,
    format='GIF',
    save_all=True,
    append_images=frames,
    fps=60,
    loop=0         
)