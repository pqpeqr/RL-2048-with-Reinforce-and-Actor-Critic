
import logging

VERBOSE_LEVEL_NUM = 15
VERBOSE_LEVEL_NAME = "VERBOSE"


def _add_verbose_level():
    if not hasattr(logging, "VERBOSE"):
        logging.addLevelName(VERBOSE_LEVEL_NUM, VERBOSE_LEVEL_NAME)
        logging.VERBOSE = VERBOSE_LEVEL_NUM

    if not hasattr(logging.Logger, "verbose"):
        def verbose(self, message, *args, **kwargs):
            if self.isEnabledFor(VERBOSE_LEVEL_NUM):
                self._log(VERBOSE_LEVEL_NUM, message, args, **kwargs)
        logging.Logger.verbose = verbose

_add_verbose_level()