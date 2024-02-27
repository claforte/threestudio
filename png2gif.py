import glob
import os

import imageio.v2 as imageio

png_dir = "/weka/home-chunhanyao/threestudio/outputs/GSO_drunk_no_sds_hr/BALANCING_CACTUS/save/sober/*.png"
output_file = "/weka/home-chunhanyao/threestudio/outputs/GSO_drunk_no_sds_hr/BALANCING_CACTUS/sober.gif"
images_files = sorted(glob.glob(png_dir))

images = []
for file_path in images_files:
    images.append(imageio.imread(file_path))

imageio.mimsave(output_file, images, fps=10, loop=0)
