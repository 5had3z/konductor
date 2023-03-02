"""
The design rationale for a configuration registry which returns models
is that we can partially initialise and gather the relevant variables
throughout the initialisation phase of the program. This allows for 
circular dependices between classes such as the numebr of classes defined
by a dataset can be used to configure the model, and 

"""
import abc
from dataclasses import dataclass
import inspect
from logging import getLogger
from typing import Any, Dict, Type

from .init import ExperimentInitConfig


@dataclass
class BaseConfig(metaclass=abc.ABCMeta):
    """
    All configuration modules require from_config to initialise and
    get_instance to return an instance
    """

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config: ExperimentInitConfig, *args, **kwargs) -> Any:
        """Run configuration stage of the module"""
        return cls(*args, **kwargs)

    @abc.abstractmethod
    def get_instance(self, *args, **kwargs) -> Any:
        """Get initialised module from configuration"""


class Registry:
    def __init__(self, name: str):
        self._name = name
        self._module_dict: Dict[str, Type[BaseConfig]] = {}
        self._logger = getLogger(name=name)

    def __len__(self):
        return len(self._module_dict)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__} (name={self._name}, items={self._module_dict})"
        )

    def __getitem__(self, name: str) -> Type[BaseConfig]:
        return self._module_dict[name]

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def __contains__(self, name: str) -> bool:
        return name in self._module_dict

    def _register_module(
        self, module: Any, name: str | None = None, force_override: bool = False
    ):
        if not any([inspect.isclass(module), inspect.isfunction(module)]):
            raise TypeError(f"module must be a class or a function, got {type(module)}")

        name = module.__name__ if name is None else name

        if name in self._module_dict:
            if force_override:
                self._logger.warn(f"Overriding {name} in registry")
            else:
                raise KeyError(f"{name} is already registered in {self.name}")
        else:
            self._logger.info(f"Registering new module {name}")

        self._module_dict[name] = module

    def register_module(
        self,
        name: str | None = None,
        module: Type[BaseConfig] | None = None,
        force_override: bool = False,
    ):
        """"""
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
