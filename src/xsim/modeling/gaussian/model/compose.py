from typing import Optional
import types
import sys


_composed: dict[tuple, type] = {}


def compose_model(*classes: type, name: Optional[str] = None) -> type:
    # allow a single iterable, e.g. compose_model([A, B, C])
    if len(classes) == 1 and not isinstance(classes[0], type):
        classes = tuple(classes[0])
    if not classes:
        raise ValueError("compose_model() needs at least one class")
    # A single class needs no composition -- return it as-is. (Wrapping it in a
    # same-named subclass would collide with the original when it lives in this
    # module, e.g. the base GaussianModel used for raw features.)
    if len(classes) == 1 and name is None:
        return classes[0]
    name = name or "".join(c.__name__ for c in classes[::-1])

    key = (classes, name)
    cls = _composed.get(key)
    if cls is not None:
        return cls

    # types.new_class() leaves __module__ pointing at whatever module happens
    # to invoke type.__new__ under the hood (the metaclass's module, not this
    # one), and never binds the class to a name anywhere. Both are required
    # for pickle (and torch.save) to be able to look the class back up, so
    # fix them up and register the class here, memoized so repeated identical
    # compositions return the same class object instead of a fresh one.
    cls = types.new_class(name, classes)
    cls.__module__ = __name__
    cls.__qualname__ = name

    module = sys.modules[__name__]
    existing = getattr(module, name, None)
    if existing is not None and existing is not cls:
        raise ValueError(
            f"compose_model: name {name!r} is already bound to a different "
            f"composition ({existing.__bases__} vs {classes})"
        )
    setattr(module, name, cls)
    _composed[key] = cls
    return cls


def compose(base: type, mixins: Optional[list[type]] = None,
            name: Optional[str] = None) -> type:
    # wrapper for convenient type composition from config.
    return compose_model(base, *(mixins or []), name=name)
