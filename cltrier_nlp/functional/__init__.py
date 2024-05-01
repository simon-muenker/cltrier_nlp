import functools
import logging
import time
import typing

from . import text
from . import neural

__all__ = [text, neural]


def timeit(func: typing.Callable):
    @functools.wraps(func)
    def wrap(*args, **kwargs) -> any:
        start = time.time()
        result = func(*args, **kwargs)
        logging.info(f"> f({func.__name__}) took: {time.time() - start:2.4f} sec")

        return result

    return wrap
