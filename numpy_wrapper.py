from wrapper import primitive, notrace_primitive
import numpy as np

notrace_functions = [
    np.ndim, np.shape, np.iscomplexobj, np.result_type
]

def wrap_intdtype(cls):
    class IntdtypeSubclass(cls):
        __new__ = notrace_primitive(cls.__new__)
    return IntdtypeSubclass

def wrap_namespace(old, new):
    unchanged_types = {float, int, type(None), type}
    int_types = {np.int, np.int8, np.int16, np.int32, np.int64, np.integer}
    for name, obj in old.items():
        if obj in notrace_functions:
            new[name] = notrace_primitive(obj)
        elif callable(obj) and type(obj) is not type:
            # wrap all legeable functions with primitive decorator
            new[name] = primitive(obj)
        elif type(obj) is type and obj in int_types:
            new[name] = wrap_intdtype(obj)
        elif type(obj) in unchanged_types:
            new[name] = obj

wrap_namespace(np.__dict__, globals()) # wrap numpy namespace in globals dict
