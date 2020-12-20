import typing as t

from src.config.core import config

import numpy as np
import pandas as pd
from marshmallow import fields, Schema, ValidationError


class ChurnInputSchema(Schema):
    feature_0 = fields.Float(allow_none=False)
    feature_1 = fields.Float(allow_none=False)
    feature_2 = fields.Float(allow_none=False)
    feature_3 = fields.Float(allow_none=False)
    feature_4 = fields.Float(allow_none=False)
    feature_5 = fields.Float(allow_none=False)
    feature_7 = fields.Float(allow_none=False)
    feature_14 = fields.Float(allow_none=False)
    feature_15 = fields.Float(allow_none=False)


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for na values and filter."""
    validated_data = input_data.copy()
    if config.model_config.numerical_na_not_allowed != ['None']:
        if input_data[config.model_config.numerical_na_not_allowed].isnull().any().any():
            validated_data = validated_data.dropna(
                axis=0, subset=config.model_config.numerical_na_not_allowed
            )

    return validated_data


def validate_inputs(
        *, input_data: pd.DataFrame
) -> t.Tuple[pd.DataFrame, t.Optional[dict]]:
    """Check model inputs for unprocessable values."""

    # convert syntax error field names (beginning with numbers)
    # input_data.rename(columns=config.model_config.variables_to_rename, inplace=True)
    # validated_data = drop_na_inputs(input_data=input_data)
    validated_data = input_data
    # set many=True to allow passing in a list
    schema = ChurnInputSchema(many=True)
    errors = None

    try:
        # replace numpy nans so that Marshmallow can validate
        schema.load(validated_data.replace({np.nan: None}).to_dict(orient="records"))
    except ValidationError as exc:
        errors = exc.messages

    return validated_data, errors
