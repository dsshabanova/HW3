#code
from model.model import params, split, train_model
from conf.conf import logging
from connector.connector import get_data
from conf.conf import settings
from util.util import load_model


settings.load_file(path="conf/setting.toml")



df = get_data(settings.DATA.data_set)
X_train, X_test, y_train, y_test = split(df)
model = train_model(X_train, y_train, params)
logging.info(f'accuracy: {model.score(X_test, y_test)}')
model = load_model(settings.MODEL.dt_conf)
logging.info(f'prediction: {model.predict(X_test)}')



