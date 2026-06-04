__version__ = '0.0.1'

from windlespy import _caseFiles
from windlespy import _profileAnalysis
from windlespy import _profileCalibration
from windlespy import _aerodynamicForces
from windlespy import _plot
from windlespy import _windTunnel
import os
# from ._specs import specs

# __version__ = specs['__version__']
# __package_name__ = specs['__package_name__']
# __author__ = specs['__author__']
# __author_email__ = specs['__author_email__']
# __url__ = specs['__url__']

"""
Setup the logger
"""
def setup_logging(logging_level, log_file_path=None):
    # Setup a logger
    logger = logging.getLogger("microclimate")

    for logger_filter in logger.filters:
        logger.removeFilter(logger_filter)

    for logger_handler in logger.handlers:
        logger.removeHandler(logger_handler)

    logger.setLevel(logging_level)
    
    if len(logger.filters) == 0:
        if log_file_path is not None:
            logging.basicConfig(
                filename=log_file_path,
                filemode='w',
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                level=logging.DEBUG
                )

    logger.debug("I'm the root logger")
    return logger

import logging
logging_level = logging.DEBUG
logger = setup_logging(logging_level, log_file_path=os.path.join(os.getcwd(),'log.txt'))

