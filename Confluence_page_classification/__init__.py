import logging
from pathlib import Path
from datetime import datetime
from Confluence_page_classification.Config import Config
import urllib3

urllib3.disable_warnings()    # sollte die dämlichen SSL-Warnings in der JIRA-Test-Maschine supressen

# Globale Variable
global_config = Config()

logFilename = Path.cwd().joinpath("logs").joinpath(datetime.now().strftime("%Y%m%d_%H%M%S") + '.log')
print(f"Logfile used: {logFilename}")

# Bit more advanced logging
logger = logging.getLogger("CPC")
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fileHandler = logging.FileHandler(logFilename, encoding="UTF-8")
fileHandler.setLevel(level=logging.DEBUG)
# create console handler with a higher log level
channelHandler = logging.StreamHandler()
channelHandler.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')
channelHandler.setFormatter(formatter)
fileHandler.setFormatter(formatter)
# add the handlers to logger
logger.addHandler(channelHandler)
logger.addHandler(fileHandler)
