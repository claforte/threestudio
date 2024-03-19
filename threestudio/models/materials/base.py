import random
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.utils.base import BaseModule
from threestudio.utils.tonemapping import AbstractToneMapping, NoToneMapping
from threestudio.utils.typing import *


class BaseMaterial(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        tone_mapping_type: str = "no-tonemapping"
        tone_mapping: dict = field(default_factory=dict)

    cfg: Config
    requires_normal: bool = False
    requires_tangent: bool = False

    def configure(self):
        self.tone_mapping = threestudio.find(self.cfg.tone_mapping_type)(
            self.cfg.tone_mapping
        )

    def forward_impl(self, *args, **kwargs) -> Float[Tensor, "*B 3"]:
        raise NotImplementedError

    def forward(self, *args, **kwargs) -> Float[Tensor, "*B 3"]:
        evaled = self.forward_impl(*args, **kwargs)
        if isinstance(evaled, dict):
            evaled["color"] = self.tone_mapping(evaled["color"])
            evaled["illumination"] = self.tone_mapping(evaled["illumination"])
            return evaled
        return self.tone_mapping(evaled)

    def export(self, *args, **kwargs) -> Dict[str, Any]:
        return {}
