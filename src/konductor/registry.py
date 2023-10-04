"""
The design rationale for a configuration registry which returns models
is that we can partially initialise and gather the relevant variables
throughout the initialisation phase of the program. This allows for 
circular dependencies between classes such as the number of classes defined
by a dataset can be used to configure the model 
"""

import inspect
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from logging import getLogger, warning
from typing import Any, Dict, Set, Type

from .init import ExperimentInitConfig


@dataclass
class BaseConfig(ABC):
    """
    All configuration modules require from_config to initialise and
    get_instance to return an instance
    """

    @classmethod
    @abstractmethod
    def from_config(cls, config: ExperimentInitConfig, *args, **kwargs) -> Any:
        """Run configuration stage of the module"""

    @abstractmethod
    def get_instance(self, *args, **kwargs) -> Any:
        """Get initialised module from configuration"""

    def init_auto_filter(
        self, target, known_unused: Set[str] | None = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Make instance of target class with auto-filtered asdict(self) + kwargs
        known_unused is a set of keyword arguments which may be present and are
        known to be not used by the module itself so we can skip the warning.
        """
        kwargs.update(asdict(self))
        filtered = {
            k: v for k, v in kwargs.items() if k in inspect.signature(target).parameters
        }

        diff = set(kwargs.keys()).difference(set(filtered.keys()))
        if known_unused is not None:
            diff = diff.difference(known_unused)

        if len(diff) > 0:
            warning("Filtered unused arguments from %s: %s", target.__name__, diff)

        return target(**filtered)


class Registry:
    """
    Registry for modules to re-access by a given name.
    Names are case insensitive (all cast to lower).
    """

    def __init__(self, name: str):
        self._name = name.lower()
        self._module_dict: Dict[str, Type[BaseConfig]] = {}
        self._logger = getLogger(name=f"{name}_registry")

    def __len__(self):
        return len(self._module_dict)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__} (name={self._name}, items={self._module_dict})"
        )

    def __getitem__(self, name: str) -> Type[BaseConfig]:
        return self._module_dict[name.lower()]

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._module_dict

    def _register_module(
        self, module: Any, name: str | None = None, force_override: bool = False
    ):
        if not any([inspect.isclass(module), inspect.isfunction(module)]):
            raise TypeError(f"module must be a class or a function, got {type(module)}")

        if name is None:
            name = module.__name__
        name = name.lower()  # enforce lowercase

        if name in self._module_dict:
            if force_override:
                self._logger.warning("Overriding %s", name)
            else:
                raise KeyError(f"{name} is already registered")
        else:
            self._logger.info("adding new module %s", name)

        self._module_dict[name] = module

    def register_module(
        self,
        name: str | None = None,
        module: Any | None = None,
        force_override: bool = False,
    ):
        """Add new module to registry, name is case insensitive (force lower)"""
        if module is not None:
            self._register_module(
                module=module, name=name, force_override=force_override
            )
            return module

        def _register(module):
            self._register_module(
                module=module, name=name, force_override=force_override
            )
            return module

        return _register
