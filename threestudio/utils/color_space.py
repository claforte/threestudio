import abc
from dataclasses import dataclass, field

import torch
from jaxtyping import Float

import threestudio
from threestudio.utils.config import parse_structured


def srgb_to_linear(x: Float[torch.Tensor, "*B C"]) -> Float[torch.Tensor, "*B C"]:
    switch_val = 0.04045
    return torch.where(
        torch.greater(x, switch_val),
        ((x.clip(min=switch_val) + 0.055) / 1.055).pow(2.4),
        x / 12.92,
    )


def linear_to_srgb(x: Float[torch.Tensor, "*B C"]) -> Float[torch.Tensor, "*B C"]:
    switch_val = 0.0031308
    return torch.where(
        torch.greater(x, switch_val),
        1.055 * x.clip(min=switch_val).pow(1.0 / 2.4) - 0.055,
        x * 12.92,
    )


def gamma_to_linear(
    x: Float[torch.Tensor, "*B C"], gamma: float = 2.2
) -> Float[torch.Tensor, "*B C"]:
    return x.pow(gamma)


def linear_to_gamma(
    x: Float[torch.Tensor, "*B C"], gamma: float = 2.2
) -> Float[torch.Tensor, "*B C"]:
    return x.pow(1.0 / gamma)


class AbstractColorSpaceConversion(torch.nn.Module, abc.ABC):
    @dataclass
    class Config:
        pass

    cfg: Config

    def __init__(self, cfg):
        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)
        self.configure()

    def configure(self):
        pass

    @abc.abstractmethod
    def forward(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        pass

    @abc.abstractmethod
    def inverse(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        pass


@threestudio.register("no-color-space-conversion")
class NoColorSpaceConversion(AbstractColorSpaceConversion):
    def forward(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        return values

    def inverse(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        return values


@threestudio.register("sRGB-color-space-conversion")
class LinearToSRGBColorSpaceConversion(AbstractColorSpaceConversion):
    def forward(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        return linear_to_srgb(values)

    def inverse(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        return srgb_to_linear(values)


@threestudio.register("gamma-color-space-conversion")
class LinearToGammaColorSpaceConversion(AbstractColorSpaceConversion):
    @dataclass
    class Config(AbstractColorSpaceConversion.Config):
        gamma: float = 2.2

    def forward(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        return linear_to_gamma(values, self.cfg.gamma)

    def inverse(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        return gamma_to_linear(values, self.cfg.gamma)
