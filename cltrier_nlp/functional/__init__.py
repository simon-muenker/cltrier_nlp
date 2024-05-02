"""
This functional collection aims to improve code usability, readability, and overall
development efficiency. They can be integrated into educational and research projects.
"""
import functools
import logging
import time
import typing

from . import text
from . import neural

__all__ = ["text", "neural"]


def timeit(func: typing.Callable):
    """
    Decorator function to measure the execution time of the function argument.

    Args:
        func (Callable): The function to be measured.

    Returns:
        any: The result of the input function.
    """
    @functools.wraps(func)
    def wrap(*args, **kwargs) -> typing.Any:
        """
        Decorator function that wraps the input function, measures its execution time,
        logs the result, and returns the result.
        """
        start = time.time()
        result = func(*args, **kwargs)
        logging.info(f"> f({func.__name__}) took: {time.time() - start:2.4f} sec")

        return result

    return wrap
