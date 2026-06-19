"""Lazy registry for reconstruction adapters.

Importing the package does not import every backend. This keeps adapter command
entrypoints independent and makes the registry the only place that knows names.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from importlib import import_module
from typing import Any

from lab3.reconstruction.base import DatasetSplit, ReconstructionContext, Reconstructor, ViewerTarget


@dataclass(frozen=True)
class ReconstructionRegistration:
    config_attr: str
    module: str
    reconstructor_class: str
    config_class: str

    def reconstructor(self):
        return getattr(import_module(self.module), self.reconstructor_class)

    def default_config(self):
        return getattr(import_module(self.module), self.config_class)()

    def parse_config(self, values: dict[str, Any]):
        return getattr(import_module(self.module), "config_from_dict")(values)

    def add_cli_arguments(self, parser: Any) -> None:
        getattr(import_module(self.module), "add_cli_arguments")(parser)

    def cli_overrides(self, arguments: Any) -> dict[str, Any]:
        return getattr(import_module(self.module), "cli_overrides")(arguments)


RECONSTRUCTIONS: dict[str, ReconstructionRegistration] = {
    "sfm": ReconstructionRegistration("sfm", "lab3.reconstruction.sfm", "SfMReconstructor", "SfMConfig"),
    "3dgs": ReconstructionRegistration("dgs", "lab3.reconstruction.dgs", "DGSReconstructor", "DGSConfig"),
    "nerf": ReconstructionRegistration("nerf", "lab3.reconstruction.nerf", "NeRFReconstructor", "NeRFConfig"),
    "neus": ReconstructionRegistration("neus", "lab3.reconstruction.neus", "NeuSReconstructor", "NeuSConfig"),
}
METHOD_ALIASES = {"dgs": "3dgs", "gaussian": "3dgs", "gaussian-splatting": "3dgs"}

_LAZY_EXPORTS = {
    "SfMConfig": ("lab3.reconstruction.sfm", "SfMConfig"),
    "SfMReconstructor": ("lab3.reconstruction.sfm", "SfMReconstructor"),
    "DGSConfig": ("lab3.reconstruction.dgs", "DGSConfig"),
    "DGSReconstructor": ("lab3.reconstruction.dgs", "DGSReconstructor"),
    "NeRFConfig": ("lab3.reconstruction.nerf", "NeRFConfig"),
    "NeRFReconstructor": ("lab3.reconstruction.nerf", "NeRFReconstructor"),
    "NeuSConfig": ("lab3.reconstruction.neus", "NeuSConfig"),
    "NeuSReconstructor": ("lab3.reconstruction.neus", "NeuSReconstructor"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def normalize_reconstruction_name(name: str) -> str:
    value = name.strip().lower()
    return METHOD_ALIASES.get(value, value)


def create_reconstructor(name: str, pipeline_config: Any) -> Reconstructor:
    canonical = normalize_reconstruction_name(name)
    registration = RECONSTRUCTIONS[canonical]
    return registration.reconstructor()(pipeline_config.reconstruction[canonical])


def create_default_reconstructor(name: str) -> Reconstructor:
    registration = RECONSTRUCTIONS[normalize_reconstruction_name(name)]
    return registration.reconstructor()(registration.default_config())


def default_reconstruction_configs() -> dict[str, Any]:
    return {name: registration.default_config() for name, registration in RECONSTRUCTIONS.items()}


def parse_reconstruction_configs(values: dict[str, Any]) -> dict[str, Any]:
    configs: dict[str, Any] = {}
    for name, registration in RECONSTRUCTIONS.items():
        section = values.get(name, values.get(registration.config_attr, {}))
        configs[name] = registration.parse_config(section if isinstance(section, dict) else {})
    return configs


def add_reconstruction_cli_arguments(parser: Any) -> None:
    for registration in RECONSTRUCTIONS.values():
        registration.add_cli_arguments(parser)


def apply_reconstruction_cli_overrides(config: Any, arguments: Any) -> Any:
    configs = dict(config.reconstruction)
    for name, registration in RECONSTRUCTIONS.items():
        values = registration.cli_overrides(arguments)
        if values:
            configs[name] = replace(configs[name], **values)
    return replace(config, reconstruction=configs)


__all__ = [
    *_LAZY_EXPORTS,
    "DatasetSplit", "ReconstructionContext", "Reconstructor", "ViewerTarget",
    "RECONSTRUCTIONS", "METHOD_ALIASES", "create_default_reconstructor",
    "create_reconstructor", "default_reconstruction_configs", "parse_reconstruction_configs",
    "add_reconstruction_cli_arguments", "apply_reconstruction_cli_overrides",
    "normalize_reconstruction_name",
]
