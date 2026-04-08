from dataclasses import dataclass

try:
    from diffusers.utils import BaseOutput
except ImportError:
    @dataclass
    class BaseOutput:
        def __getitem__(self, k):
            if isinstance(k, str):
                return self.__dict__[k]
            return self.to_tuple()[k]

        def __iter__(self):
            yield from self.to_tuple()

        def to_tuple(self):
            return tuple(self.__dict__.values())
