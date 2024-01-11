import os
import shutil
import subprocess

import pytorch_lightning

from threestudio.utils.config import dump_config
from threestudio.utils.misc import parse_version

if parse_version(pytorch_lightning.__version__) > parse_version("1.8"):
    from pytorch_lightning.callbacks import Callback
else:
    from pytorch_lightning.callbacks.base import Callback

import torch
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.utilities.rank_zero import rank_zero_only, rank_zero_warn


class VersionedCallback(Callback):
    def __init__(self, save_root, version=None, use_version=True):
        self.save_root = save_root
        self._version = version
        self.use_version = use_version

    @property
    def version(self) -> int:
        """Get the experiment version.

        Returns:
            The experiment version if specified else the next version.
        """
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    def _get_next_version(self):
        existing_versions = []
        if os.path.isdir(self.save_root):
            for f in os.listdir(self.save_root):
                bn = os.path.basename(f)
                if bn.startswith("version_"):
                    dir_ver = os.path.splitext(bn)[0].split("_")[1].replace("/", "")
                    existing_versions.append(int(dir_ver))
        if len(existing_versions) == 0:
            return 0
        return max(existing_versions) + 1

    @property
    def savedir(self):
        if not self.use_version:
            return self.save_root
        return os.path.join(
            self.save_root,
            self.version
            if isinstance(self.version, str)
            else f"version_{self.version}",
        )


class CodeSnapshotCallback(VersionedCallback):
    def __init__(self, save_root, version=None, use_version=True):
        super().__init__(save_root, version, use_version)

    def get_file_list(self):
        return [
            b.decode()
            for b in set(
                subprocess.check_output(
                    'git ls-files -- ":!:load/*"', shell=True
                ).splitlines()
            )
            | set(  # hard code, TODO: use config to exclude folders or files
                subprocess.check_output(
                    "git ls-files --others --exclude-standard", shell=True
                ).splitlines()
            )
        ]

    @rank_zero_only
    def save_code_snapshot(self):
        os.makedirs(self.savedir, exist_ok=True)
        for f in self.get_file_list():
            if not os.path.exists(f) or os.path.isdir(f):
                continue
            os.makedirs(os.path.join(self.savedir, os.path.dirname(f)), exist_ok=True)
            shutil.copyfile(f, os.path.join(self.savedir, f))

    def on_fit_start(self, trainer, pl_module):
        try:
            self.save_code_snapshot()
        except:
            rank_zero_warn(
                "Code snapshot is not saved. Please make sure you have git installed and are in a git repository."
            )


class ConfigSnapshotCallback(VersionedCallback):
    def __init__(self, config_path, config, save_root, version=None, use_version=True):
        super().__init__(save_root, version, use_version)
        self.config_path = config_path
        self.config = config

    @rank_zero_only
    def save_config_snapshot(self):
        os.makedirs(self.savedir, exist_ok=True)
        dump_config(os.path.join(self.savedir, "parsed.yaml"), self.config)
        shutil.copyfile(self.config_path, os.path.join(self.savedir, "raw.yaml"))

    def on_fit_start(self, trainer, pl_module):
        self.save_config_snapshot()


class CustomProgressBar(TQDMProgressBar):
    def get_metrics(self, *args, **kwargs):
        # don't show the version number
        items = super().get_metrics(*args, **kwargs)
        items.pop("v_num", None)
        return items


class MemoryAnalysisCallback(Callback):
    def __init__(self, save_path, steps):
        super().__init__()
        self.save_path = save_path
        self.steps = set(steps)
        self.validation_count = 0
        # torch.cuda.memory._record_memory_history(enabled=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx - 1 in self.steps:
            torch.cuda.memory._dump_snapshot(
                self.save_path + str(batch_idx) + ".pickle"
            )
            torch.cuda.memory._record_memory_history(enabled=None)
        if batch_idx in self.steps:
            torch.cuda.memory._record_memory_history(
                enabled="all", context="all", stacks="python"
            )

    def on_validation_start(self, trainer, pl_module):
        torch.cuda.memory._record_memory_history(
            enabled="all", context="all", stacks="python"
        )

    def on_validation_end(self, trainer, pl_module):
        torch.cuda.memory._dump_snapshot(
            self.save_path + str(self.validation_count) + ".pickle"
        )
        torch.cuda.memory._record_memory_history(enabled=None)
        self.validation_count += 1


import math

import matplotlib.pyplot as plt


class TensorVizCallback(Callback):
    @staticmethod
    def viz_3d_tensor(t, name, shape, iteration):
        plt.clf()

        data = t.detach().to(torch.float32).reshape(shape)

        C = shape[1]

        scale_max = torch.max(data.max().abs(), data.min().abs())

        width = int(math.ceil(math.sqrt(C)))
        height = int(math.ceil(C / width))

        f, axarr = plt.subplots(width, height, figsize=(30, 30))

        a = axarr.flatten()
        for idx in range(C):
            channel_data = data[:, idx]
            data_min = channel_data.min()
            data_max = channel_data.max()
            data_mean = channel_data.mean()

            im = a[idx].imshow(
                channel_data.squeeze().cpu().numpy(),
                cmap="coolwarm",
                vmin=-scale_max,
                vmax=scale_max,
            )
            a[idx].set_title(
                f"{name} {idx} min={data_min:.2} max={data_max:.2} mean={data_mean:.2}"
            )
            plt.colorbar(im)
        plt.tight_layout()
        plt.savefig(f"{name.replace(' ', '_')}_{iteration}.png")

    def __init__(self, tensors_to_viz, viz_frequency):
        super().__init__()
        self.tensors_to_viz = tensors_to_viz
        self.viz_frequency = viz_frequency

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if batch_idx % self.viz_frequency == 0:
            for name, value in self.tensors_to_viz.items():
                t, shape = value
                TensorVizCallback.viz_3d_tensor(t, name, shape, batch_idx)


class ProgressCallback(Callback):
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path
        self._file_handle = None

    @property
    def file_handle(self):
        if self._file_handle is None:
            self._file_handle = open(self.save_path, "w")
        return self._file_handle

    @rank_zero_only
    def write(self, msg: str) -> None:
        self.file_handle.seek(0)
        self.file_handle.truncate()
        self.file_handle.write(msg)
        self.file_handle.flush()

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        self.write(
            f"Generation progress: {pl_module.true_global_step / trainer.max_steps * 100:.2f}%"
        )

    @rank_zero_only
    def on_validation_start(self, trainer, pl_module):
        self.write(f"Rendering validation image ...")

    @rank_zero_only
    def on_test_start(self, trainer, pl_module):
        self.write(f"Rendering video ...")

    @rank_zero_only
    def on_predict_start(self, trainer, pl_module):
        self.write(f"Exporting mesh assets ...")
