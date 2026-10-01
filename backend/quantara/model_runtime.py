"""Keep process-global neural RNGs isolated between concurrent local jobs."""

from functools import wraps
from threading import RLock

training_lock = RLock()


def serialized_training(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with training_lock:
            return function(*args, **kwargs)

    return wrapped
