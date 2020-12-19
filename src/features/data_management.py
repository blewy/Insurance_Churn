import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

from src.config.core import config, DATASET_DIR, TRAINED_MODEL_DIR
from src import __version__ as _version

import logging
import typing as t