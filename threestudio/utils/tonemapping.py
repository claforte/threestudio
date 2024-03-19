import abc
from dataclasses import dataclass, field

import torch
from jaxtyping import Float

import threestudio
from threestudio.utils.config import parse_structured

from .color_space import AbstractColorSpaceConversion, NoColorSpaceConversion


class AbstractToneMapping(torch.nn.Module, abc.ABC):
    @dataclass
    class Config:
        color_space_type: str = "no-color-space-conversion"
        color_space: dict = field(default_factory=dict)

    cfg: Config

    def __init__(self, cfg):
        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)
        self._color_space = threestudio.find(self.cfg.color_space_type)(
            self.cfg.color_space
        )
        self.configure()

    def configure(self):
        pass

    @abc.abstractmethod
    def forward_impl(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        pass

    def forward(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        return self.color_space(self.forward_impl(values))

    @abc.abstractmethod
    def inverse_impl(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        pass

    def inverse(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        return self.inverse_impl(self.color_space.inverse(values))

    @property
    def color_space(self) -> AbstractColorSpaceConversion:
        return self._color_space

    @property
    def transforms_linear_color_space(self) -> bool:
        return not isinstance(self._color_space, NoColorSpaceConversion)


@threestudio.register("no-tonemapping")
class NoToneMapping(AbstractToneMapping):
    def forward_impl(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        return values

    def inverse_impl(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        return values


def inverse_sigmoid(
    values: Float[torch.Tensor, "*B C"], eps: float = 1e-3
) -> Float[torch.Tensor, "*B C"]:
    values = values.clip(min=eps, max=1 - eps)
    values = torch.log(values / (1 - values))
    return values


@threestudio.register("sigmoid-tonemapping")
class SigmoidMapping(AbstractToneMapping):
    @dataclass
    class Config(AbstractToneMapping.Config):
        eps: float = 1e-3

    def forward_impl(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        return torch.sigmoid(values)

    def inverse_impl(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        return inverse_sigmoid(values, self.cfg.eps)


@threestudio.register("log-tonemapping")
class LogMapping(AbstractToneMapping):
    def forward_impl(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        return torch.log(values + 1)

    def inverse_impl(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        return torch.exp(values) - 1


@threestudio.register("softclip-tonemapping")
class SoftClipHDRToneMapping(AbstractToneMapping):
    @dataclass
    class Config(AbstractToneMapping.Config):
        up_threshold: float = 0.9

    def soft_clip(
        self, values: Float[torch.Tensor, "*B C"], up_threshold: float
    ) -> Float[torch.Tensor, "*B C"]:
        return torch.where(
            torch.less_equal(values, up_threshold),
            values,
            (
                1
                - (1 - up_threshold)
                * torch.exp(-((values - up_threshold) / (1 - up_threshold)))
            ),
        )

    def inverse_soft_clip(
        self, values: Float[torch.Tensor, "*B C"], up_threshold: float
    ):
        log_term = torch.log((values - 1) / (up_threshold - 1)).clip(1e-6)
        return torch.where(
            torch.less_equal(values, up_threshold),
            values,
            up_threshold * log_term - log_term + up_threshold,
        )

    def forward_impl(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        return self.soft_clip(values, self.cfg.up_threshold)

    def inverse_impl(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        return self.inverse_soft_clip(values, self.cfg.up_threshold)


@threestudio.register("reinhard-tonemapping")
class ReinhardHDRToneMapping(AbstractToneMapping):
    def forward_impl(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        return values / (1 + values)

    def inverse_impl(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        values = values.clip(0, 1 - 1e-6)
        return -values / (values - 1)


@threestudio.register("extended-reinhard-tonemapping")
class ExtendedReinhardHDRToneMapping(AbstractToneMapping):
    @dataclass
    class Config(AbstractToneMapping.Config):
        normalize_first: bool = False

    def forward_impl(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        if self.cfg.normalize_first:
            values = (values - values.min()) / (values.max() - values.min())
        max_val = values.max()
        numerator = values * (1 + (values / max_val.square()))
        return numerator / (1 + values)


@threestudio.register("cheap-ACES-tonemapping")
class CheapACESFilmicHDRToneMapping(AbstractToneMapping):
    """This is a cheap ACES Filmic approximation. It is close to the actual but rather expensive ACES Filmic curve.
    Also often used in video games.
    https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
    """

    def forward_impl(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        v = values * 0.6
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14
        return ((v * (a * v + b)) / (v * (c * v + d) + e)).clip(min=0, max=1)

    def inverse_impl(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        # Source: wolfram alpha
        values = values.clip(0, 1)
        return -(
            0.833333
            * (
                -3
                + 59 * values
                + torch.sqrt(9 + 13702 * values - 10127 * values.square())
            )
        ) / (-251 + 243 * values)


@threestudio.register("cheap-sRGB-ACES-tonemapping")
class CombinedApproximateACESsRGBToneMapping(AbstractToneMapping):
    """This is a combination of the cheap ACES Filmic approximation and a cheap sRGB approximation.
    This is even more approximate compared to the Cheap ACES filmic approximation
    """

    def configure(self):
        if not isinstance(self._color_space, NoColorSpaceConversion):
            raise ValueError(
                "CombinedApproximateACESsRGBToneMapping only supports NoColorSpaceConversion as it handles the sRGB conversion"
            )

    def forward_impl(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        return values / (values + 0.1667)

    def inverse_impl(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        return values / (6 - 6 * values)

    @property
    def transforms_linear_color_space(self) -> bool:
        return True
