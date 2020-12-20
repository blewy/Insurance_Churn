import typing as t

import pandas as pd

from src import __version__ as _version
from src.config.core import config
from src.features.data_management import load_pipeline

from src.features.validation import validate_inputs


pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
_price_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, input_data: t.Union[pd.DataFrame, dict], ) -> dict:
    """Make a prediction using a saved model pipeline."""

    data = pd.DataFrame(input_data)
    errors = False
    validated_data, errors = validate_inputs(input_data=data)
    results = {"predictions": None, "version": _version, "errors": errors}

    if not errors:
        predictions = _price_pipe.predict_proba(
            # x=validated_data[config.model_config.features]
            validated_data[config.model_config.features]
        )[:, 1]
        results = {"predictions": predictions, "version": _version, "errors": errors}

    return results


if __name__ == "__main__":
    inputs = {'feature_0': [0, 3], 'feature_1': [0.5, 1], 'feature_2': [0, 1], 'feature_3': [0, 1], 'feature_4': [0, 1],
              'feature_5': [0, 1], 'feature_7': [0, 1], 'feature_14': [0, 1], 'feature_15': [0, 1]}
    df = pd.DataFrame(data=inputs)
    print(make_prediction(input_data=df))
