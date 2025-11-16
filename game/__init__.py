import logging

package_logger = logging.getLogger(__name__)
logging.getLogger(__name__).addHandler(logging.NullHandler())
