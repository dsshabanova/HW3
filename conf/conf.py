#for logging and dynaconf
import logging 

logging.basicConfig(level=logging.INFO)
from dynaconf import Dynaconf

settings = Dynaconf(settings_file="setting.toml")

