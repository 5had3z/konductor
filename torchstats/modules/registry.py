import inspect
from typing import Any, Dict


class Registry:
    def __init__(self, name: str):
        self._name = name
        self._module_dict: Dict[str, Any] = {}

    def __len__(self):
        return len(self._module_dict)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__} (name={self._name}, items={self._module_dict})"
        )

    def __getitem__(self, name: str) -> Any:
        return self._module_dict[name]

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def _register_module(self, module, name=None):
        if not any([inspect.isclass(module), inspect.isfunction(module)]):
            raise TypeError(f"module must be a class or a function, got {type(module)}")

        if name is None:
            name = module.__name__

        if name in self._module_dict:
            raise KeyError(f"{name} is already registered in {self.name}")

        self._module_dict[name] = module

    def register_module(self, name: str | None = None, module=None):
        """"""
        if module is not None:
            self._register_module(module=module, name=name)
            return module

        def _register(module):
            self._register_module(module=module, name=name)
            return module

        return _register
