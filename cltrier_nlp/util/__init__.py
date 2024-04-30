import functools
import logging
import time
import typing
import warnings

import langcodes
import langdetect
import torch


def get_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def calculate_model_memory_usage(model: torch.nn.Module) -> str:
    usage_in_byte: int = sum(
        [
            sum(
                [
                    param.nelement() * param.element_size()
                    for param in model.parameters()
                ]
            ),
            sum([buf.nelement() * buf.element_size() for buf in model.buffers()]),
        ]
    )

    return f'{usage_in_byte / (1024.0 * 1024.0):2.4f} MB'


def timeit(func: typing.Callable):
    @functools.wraps(func)
    def wrap(*args, **kwargs) -> any:
        start = time.time()
        result = func(*args, **kwargs)
        logging.info(f'> f({func.__name__}) took: {time.time() - start:2.4f} sec')

        return result

    return wrap


def detect_language(content: str) -> str:
    # Ignore langcodes dependent language data warning
    # DeprecationWarning: pkg_resources is deprecated as an API.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        try:
            return (
                langcodes.Language.get(langdetect.detect(content))
                .display_name()
                .lower()
            )

        except langdetect.lang_detect_exception.LangDetectException:
            return "unknown"
