import argparse
from model.model import prediction
import logging 


cli = argparse.ArgumentParser(description='Prediction from terminal')
cli.add_argument('--prediction_model',action='store',  dest='prediction_model', type=str, nargs=1)
cli.add_argument('--prediction_params', action='append',  dest='prediction_params', type=float, nargs=2,)
parse = cli.parse_args()
print(prediction(parse.prediction_model[0], parse.prediction_params))
logging.info("Prediction from terminal")

