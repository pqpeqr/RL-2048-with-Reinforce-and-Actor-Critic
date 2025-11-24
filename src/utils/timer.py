# timer.py
from __future__ import annotations

import time
import logging
from functools import wraps
from typing import Callable, TypeVar, ParamSpec

P = ParamSpec("P")
R = TypeVar("R")

_module_logger = logging.getLogger(__name__)
_module_logger.addHandler(logging.NullHandler())


def timer(
    _func: Callable[P, R] | None = None,
    *,
    logger: logging.Logger | None = None,
    level: int = logging.INFO,
):
    """
    timer decorator

    @timer
    @timer(logger=my_logger, level=logging.DEBUG)
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        # Use function's module logger in default case
        log = logger or logging.getLogger(func.__module__)

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()

            log.log(level, "%s RUN TIME: %.6f s", func.__qualname__, end - start)
            return result

        return wrapper

    # @timer
    if _func is not None:
        return decorator(_func)

    # @timer(logger=..., level=...)
    return decorator
