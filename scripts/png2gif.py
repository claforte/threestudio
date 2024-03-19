import glob
import os

import imageio.v2 as imageio

png_dir = "/weka/home-chunhanyao/threestudio/outputs/GSO_sv3d-p_static_elev30_smasked_sds_sgs_pbr_refine/*/save/sober"

for obj_dir in sorted(glob.glob(png_dir)):
    obj = obj_dir.split("/")[-3]
    print(obj)
    output_file = os.path.join(obj_dir, "%s_sv3d.mp4" % obj)
    image_files = sorted(glob.glob(os.path.join(obj_dir, "*.png")))
    images = []
    for file_path in image_files:
        images.append(imageio.imread(file_path))

    # imageio.mimsave(output_file, images, fps=10, loop=0) # gif
    imageio.mimsave(output_file, images, fps=10)  # mp4
