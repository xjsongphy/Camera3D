from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from lab3.reconstruction.base import ReconstructionContext, Reconstructor, ViewerTarget
from lab3.reconstruction.dgs import DGSConfig, DGSReconstructor
from lab3.reconstruction.nerf import NeRFConfig, NeRFReconstructor
from lab3.reconstruction.neus import NeuSConfig, NeuSReconstructor
from lab3.reconstruction.sfm import SfMConfig, SfMReconstructor


@dataclass(frozen=True)
class ReconstructionRegistration:
    config_attr: str
    constructor: Callable[[Any], Reconstructor]
    default_config: Callable[[], Any]


RECONSTRUCTIONS: dict[str, ReconstructionRegistration] = {
    "sfm": ReconstructionRegistration("sfm", SfMReconstructor, SfMConfig),
    "3dgs": ReconstructionRegistration("dgs", DGSReconstructor, DGSConfig),
    "nerf": ReconstructionRegistration("nerf", NeRFReconstructor, NeRFConfig),
    "neus": ReconstructionRegistration("neus", NeuSReconstructor, NeuSConfig),
}
METHOD_ALIASES = {"dgs": "3dgs", "gaussian": "3dgs", "gaussian-splatting": "3dgs"}


def normalize_reconstruction_name(name: str) -> str:
    value = name.strip().lower()
    return METHOD_ALIASES.get(value, value)


def create_reconstructor(name: str, pipeline_config: Any) -> Reconstructor:
    registration = RECONSTRUCTIONS[normalize_reconstruction_name(name)]
    return registration.constructor(getattr(pipeline_config, registration.config_attr))


def create_default_reconstructor(name: str) -> Reconstructor:
    registration = RECONSTRUCTIONS[normalize_reconstruction_name(name)]
    return registration.constructor(registration.default_config())

__all__ = [
    "DGSConfig",
    "DGSReconstructor",
    "NeRFConfig",
    "NeRFReconstructor",
    "NeuSConfig",
    "NeuSReconstructor",
    "ReconstructionContext",
    "Reconstructor",
    "ViewerTarget",
    "SfMConfig",
    "SfMReconstructor",
    "RECONSTRUCTIONS",
    "METHOD_ALIASES",
    "create_default_reconstructor",
    "create_reconstructor",
    "normalize_reconstruction_name",
]
