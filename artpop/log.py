# Standard library
import sys
import logging

# Third-party
from astropy.logger import AstropyLogger


class ArtPopLogger(AstropyLogger):

    def _set_defaults(self, level='INFO'):
        """
        Reset logger to its initial state
        """
        # Remove all previous handlers
        for handler in self.handlers[:]:
            self.removeHandler(handler)

        # Set levels
        self.setLevel(level)

        # Set up the stdout handler
        sh = StreamHandler()
        self.addHandler(sh)


class StreamHandler(logging.StreamHandler):
    """
    A specialized StreamHandler that logs INFO and DEBUG messages to
    stdout, and all other messages to stderr.  Also provides coloring
    of the output, if enabled in the parent logger.
    """

    def emit(self, record):
        '''
        The formatter for stderr
        '''
        # Import utils.console only if necessary and at the latest because
        # the import takes a significant time [#4649]
        from astropy.utils.console import color_print

        if record.levelno <= logging.INFO:
            stream = sys.stdout
        else:
            stream = sys.stderr

        if record.levelno < logging.INFO:
            color_print(record.levelname, 'magenta', end='', file=stream)
        elif record.levelno < logging.WARN:
            color_print(record.levelname, 'green', end='', file=stream)
        elif record.levelno < logging.ERROR:
            color_print(record.levelname, 'brown', end='', file=stream)
        else:
            color_print(record.levelname, 'red', end='', file=stream)
        record.message = "{0}".format(record.msg)
        print(": " + record.message, file=stream)


logging.setLoggerClass(ArtPopLogger)
logger = logging.getLogger('ArtPop Logger')
logger._set_defaults()
