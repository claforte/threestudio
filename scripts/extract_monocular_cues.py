# adapted from https://github.com/EPFL-VILAB/omnidata
import argparse
import glob
import os.path
import sys
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def run(root_dir, omnidata_path, task, output_path, img_path):
    sys.path.append(omnidata_path)
    # print(sys.path)
    from data.transforms import get_transform
    from modules.midas.dpt_depth import DPTDepthModel
    from modules.unet import UNet

    map_location = (
        (lambda storage, loc: storage.cuda())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get target task and model
    if task == "normal":
        image_size = 384

        ## Version 1 model
        # pretrained_weights_path = root_dir + 'omnidata_unet_normal_v1.pth'
        # model = UNet(in_channels=3, out_channels=3)
        # checkpoint = torch.load(pretrained_weights_path, map_location=map_location)

        # if 'state_dict' in checkpoint:
        #     state_dict = {}
        #     for k, v in checkpoint['state_dict'].items():
        #         state_dict[k.replace('model.', '')] = v
        # else:
        #     state_dict = checkpoint

        pretrained_weights_path = root_dir + "omnidata_dpt_normal_v2.ckpt"
        model = DPTDepthModel(backbone="vitb_rn50_384", num_channels=3)  # DPT Hybrid
        checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
        if "state_dict" in checkpoint:
            state_dict = {}
            for k, v in checkpoint["state_dict"].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.to(device)
        trans_totensor = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                transforms.CenterCrop(image_size),
                get_transform("rgb", image_size=None),
            ]
        )

    elif task == "depth":
        image_size = 384
        pretrained_weights_path = (
            root_dir + "omnidata_dpt_depth_v2.ckpt"
        )  # 'omnidata_dpt_depth_v1.ckpt'
        # model = DPTDepthModel(backbone='vitl16_384') # DPT Large
        model = DPTDepthModel(backbone="vitb_rn50_384")  # DPT Hybrid
        checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
        if "state_dict" in checkpoint:
            state_dict = {}
            for k, v in checkpoint["state_dict"].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        model.to(device)
        trans_totensor = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5, std=0.5),
            ]
        )

    else:
        print("task should be one of the following: normal, depth")
        sys.exit()

    trans_rgb = transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
            transforms.CenterCrop(image_size),
        ]
    )

    def standardize_depth_map(img, mask_valid=None, trunc_value=0.1):
        if mask_valid is not None:
            img[~mask_valid] = torch.nan
        sorted_img = torch.sort(torch.flatten(img))[0]
        # Remove nan, nan at the end of sort
        num_nan = sorted_img.isnan().sum()
        if num_nan > 0:
            sorted_img = sorted_img[:-num_nan]
        # Remove outliers
        trunc_img = sorted_img[
            int(trunc_value * len(sorted_img)) : int(
                (1 - trunc_value) * len(sorted_img)
            )
        ]
        trunc_mean = trunc_img.mean()
        trunc_var = trunc_img.var()
        eps = 1e-6
        # Replace nan by mean
        img = torch.nan_to_num(img, nan=trunc_mean)
        # Standardize
        img = (img - trunc_mean) / torch.sqrt(trunc_var + eps)
        return img

    def save_outputs(img_path, output_file_name):
        if os.path.splitext(img_path)[1].lower() not in [".png", ".jpg", ".jpeg"]:
            return
        with torch.no_grad():
            save_path = os.path.join(output_path, f"{output_file_name}_{task}.png")

            # print(f"Reading input {img_path} ...")
            img = Image.open(img_path)

            img_tensor = trans_totensor(img)[:3].unsqueeze(0).to(device)

            if img_tensor.shape[1] == 1:
                img_tensor = img_tensor.repeat_interleave(3, 1)

            output = model(img_tensor).clamp(min=0, max=1)

            if task == "depth":
                # output = F.interpolate(output.unsqueeze(0), (512, 512), mode='bicubic').squeeze(0)
                output = output.clamp(0, 1)

                np.save(
                    save_path.replace(".png", ".npy"), output.detach().cpu().numpy()[0]
                )

                # output = 1 - output
                #             output = standardize_depth_map(output)
                plt.imsave(save_path, output.detach().cpu().squeeze(), cmap="viridis")

            else:
                # import pdb; pdb.set_trace()
                # np.save(save_path.replace(".png", ".npy"), output.detach().cpu().numpy()[0])
                # print(output.min(), output.max())
                imageio.imsave(
                    save_path.replace(".png", ".hdr"),
                    output.detach()[0].permute(1, 2, 0).cpu().numpy(),
                )
                # trans_topil(output[0]).save(save_path)

            # print(f"Writing output {save_path} ...")

    if os.path.isfile(img_path):
        save_outputs(img_path, os.path.splitext(os.path.basename(img_path))[0])
    elif os.path.isdir(img_path):
        for f in glob.glob(os.path.join(img_path, "*.png")):
            save_outputs(f, os.path.splitext(os.path.basename(f))[0])
    else:
        print("invalid file path!")
        sys.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize output for depth or surface normals"
    )

    parser.add_argument(
        "--omnidata_path", dest="omnidata_path", help="path to omnidata model"
    )
    parser.set_defaults(
        omnidata_path="/weka/home-markboss/omnidata/omnidata_tools/torch"
    )

    parser.add_argument(
        "--pretrained_models",
        dest="pretrained_models",
        help="path to pretrained models",
    )
    parser.set_defaults(
        pretrained_models="/weka/home-markboss/omnidata/omnidata_tools/torch/pretrained_models/"
    )

    parser.add_argument("--task", dest="task", help="normal or depth")
    parser.set_defaults(task="NONE")

    parser.add_argument("--img_path", dest="img_path", help="path to rgb image")
    parser.set_defaults(im_name="NONE")

    parser.add_argument(
        "--output_path",
        dest="output_path",
        help="path to where output image should be stored",
    )
    parser.set_defaults(store_name="NONE")

    args = parser.parse_args()

    os.system(f"mkdir -p {args.output_path}")

    run(args.root_dir, args.omnidata_path, args.task, args.output_path, args.img_path)
