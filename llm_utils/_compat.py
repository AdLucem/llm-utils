"""Compatibility helpers for older Python runtimes."""

try:
    from dataclasses import dataclass
except ImportError:  # pragma: no cover - Python 3.6 fallback
    def dataclass(cls):
        annotations = getattr(cls, "__annotations__", {})
        fields = list(annotations.keys())
        defaults = {
            name: getattr(cls, name)
            for name in fields
            if hasattr(cls, name)
        }

        def __init__(self, *args, **kwargs):
            if len(args) > len(fields):
                raise TypeError("Too many positional arguments for dataclass fallback.")

            values = {}
            for name, value in zip(fields, args):
                values[name] = value

            for name in fields[len(args):]:
                if name in kwargs:
                    values[name] = kwargs.pop(name)
                elif name in defaults:
                    values[name] = defaults[name]
                else:
                    raise TypeError("Missing required argument: {}".format(name))

            if kwargs:
                unexpected = ", ".join(sorted(kwargs.keys()))
                raise TypeError("Unexpected keyword arguments: {}".format(unexpected))

            for name in fields:
                setattr(self, name, values[name])

        cls.__init__ = __init__
        return cls

try:
    from typing import Literal
except ImportError:  # pragma: no cover - Python 3.6 fallback
    class _LiteralCompat(object):
        def __getitem__(self, _args):
            return object

    Literal = _LiteralCompat()

__all__ = ["Literal", "dataclass"]
