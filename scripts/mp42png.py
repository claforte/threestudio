import glob
import os

import cv2

video_file = "/weka/proj-sv3d/EVAL_WildnFree_STUDY/SV3D/sculpture5.mp4"
output_dir = "/weka/home-chunhanyao/threestudio/outputs/EVAL_WildnFree_STUDY/"
file_name = video_file.split("/")[-1].split(".")[0]

vidcap = cv2.VideoCapture(video_file)
success, image = vidcap.read()
count = 0
while success:
    output_file = os.path.join(output_dir, file_name, "rgba_0020", "%d.png" % count)
    cv2.imwrite(output_file, image)  # save frame as JPEG file
    success, image = vidcap.read()
    count += 1
