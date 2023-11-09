from loguru import logger
import sys

logger = logger.bind(corr_id="BASIC  ")

logger.remove()
logger.add(sys.stdout, format="[{level.icon} {level.name[0]}]\t CID: {extra[corr_id]}\t {message}")
