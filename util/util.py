# here is pickle 
import pickle

import sys
from pathlib import Path

if sys.path[0][-3:] != 'L-HSE-12-2022-MAIN':
    sys.path[0] = str(Path(sys.path[0]).parent)

def save_model(dir: str, model) -> None:
    pickle.dump(model, open(dir, 'wb'))

def load_model(dir: str) -> None:
    return pickle.load(open(dir, 'rb'))
