from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
# Custom Transformer that extracts columns passed as argument to its constructor


class FeatureSelector(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self, feature_names):
        self._feature_names = feature_names

        # Return self nothing else to do here

    def fit(self, x, y=None):
        return self

        # Method that describes what we need this transformer to do

    def transform(self, x, y=None):
        return x[self._feature_names]


# Custom Transformer that store a schema of data inside teh pipeline
class SchemaBuild(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self, feature_names):
        self._feature_names = feature_names
        self._feature_schema = {}

    # Return self with a data schema added
    def fit(self, x, y=None):
        for feature in x.columns:
            feature_schema = {
                "mean": float(x[feature].mean()),
                "std": float(x[feature].std()),
                "min": float(x[feature].min()),
                "max": float(x[feature].max()),
                "values": (x[feature].unique()[0:20]).tolist(),
                "pct_miss": float(np.round(x[feature].isnull().mean(), 3)),
                "type": str(x[feature].dtypes)
                }
            self._feature_schema[feature] = feature_schema
        return self

        # Method that describes what we need this transformer to do

    def transform(self, x, y=None):
        return x
