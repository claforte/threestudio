import gc
import os
import re

import psutil
import tinycudann as tcnn
import torch
from packaging import version

from threestudio.utils.config import config_to_primitive
from threestudio.utils.typing import *


def parse_version(ver: str):
    return version.parse(ver)


def get_rank():
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def get_device():
    return torch.device(f"cuda:{get_rank()}")


def load_module_weights(
    path, module_name=None, ignore_modules=None, map_location=None
) -> Tuple[dict, int, int]:
    if module_name is not None and ignore_modules is not None:
        raise ValueError("module_name and ignore_modules cannot be both set")
    if map_location is None:
        map_location = get_device()

    ckpt = torch.load(path, map_location=map_location)
    state_dict = ckpt["state_dict"]
    state_dict_to_load = state_dict

    if ignore_modules is not None:
        state_dict_to_load = {}
        for k, v in state_dict.items():
            ignore = any(
                [k.startswith(ignore_module + ".") for ignore_module in ignore_modules]
            )
            if ignore:
                continue
            state_dict_to_load[k] = v

    if module_name is not None:
        state_dict_to_load = {}
        for k, v in state_dict.items():
            m = re.match(rf"^{module_name}\.(.*)$", k)
            if m is None:
                continue
            state_dict_to_load[m.group(1)] = v

    return state_dict_to_load, ckpt["epoch"], ckpt["global_step"]


def C(value: Any, epoch: int, global_step: int) -> float:
    if isinstance(value, int) or isinstance(value, float):
        pass
    else:
        value = config_to_primitive(value)
        if not isinstance(value, list):
            raise TypeError("Scalar specification only supports list, got", type(value))
        if len(value) == 3:
            value = [0] + value
        if len(value) == 6:
            start_step, start_value, end_value, end_step, end_value2, end_step2 = value
            if global_step < end_step:
                value = [start_step, start_value, end_value, end_step]
            else:
                value = [end_step, end_value, end_value2, end_step2]
        assert len(value) == 4
        start_step, start_value, end_value, end_step = value
        if isinstance(end_step, int):
            current_step = global_step
            value = start_value + (end_value - start_value) * max(
                min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
            )
        elif isinstance(end_step, float):
            current_step = epoch
            value = start_value + (end_value - start_value) * max(
                min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
            )
    return value


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    tcnn.free_temporary_memory()


def finish_with_cleanup(func: Callable):
    def wrapper(*args, **kwargs):
        out = func(*args, **kwargs)
        cleanup()
        return out

    return wrapper


def _distributed_available():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def barrier():
    if not _distributed_available():
        return
    else:
        torch.distributed.barrier()


def broadcast(tensor, src=0):
    if not _distributed_available():
        return tensor
    else:
        torch.distributed.broadcast(tensor, src=src)
        return tensor


def enable_gradient(model, enabled: bool = True) -> None:
    for param in model.parameters():
        param.requires_grad_(enabled)


def get_CPU_mem():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**3


def get_GPU_mem():
    try:
        num = torch.cuda.device_count()
        mem, mems = 0, []
        for i in range(num):
            mem_free, mem_total = torch.cuda.mem_get_info(i)
            mems.append(int(((mem_total - mem_free) / 1024**3) * 1000) / 1000)
            mem += mems[-1]
        return mem, mems
    except:
        try:

            def get_gpu_memory_map():
                import subprocess

                """Get the current gpu usage.

                Returns
                -------
                usage: dict
                    Keys are device ids as integers.
                    Values are memory usage as integers in MB.
                """
                result = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,nounits,noheader",
                    ],
                    encoding="utf-8",
                )
                # Convert lines into a dictionary
                gpu_memory = [int(x) for x in result.strip().split("\n")]
                gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
                return gpu_memory_map

            mems = get_gpu_memory_map()
            mems = [int(mems[k] / 1024 * 1000) / 1000 for k in mems]
            mem = int(sum([m for m in mems]) * 1000) / 1000
            return mem, mems
        except:
            return 0, "CPU"
