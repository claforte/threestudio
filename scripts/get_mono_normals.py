import argparse
import glob
import os

from extract_monocular_cues import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    mono_normal_dir = args.output_dir
    dataroot = args.input_dir

    if not os.path.exists(mono_normal_dir):
        os.mkdir(mono_normal_dir)

    run(
        "/weka/home-markboss/omnidata/omnidata_tools/torch/pretrained_models/",
        "/weka/home-markboss/omnidata/omnidata_tools/torch",
        "normal",
        mono_normal_dir,
        dataroot,
    )

    print("Done extracting normals for %s" % dataroot)


if __name__ == "__main__":
    main()
