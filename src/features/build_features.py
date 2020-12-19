from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features import preprocessors as pp
from src.config.core import config

parameters = dict(config.parameters_config)

churn_pipe = Pipeline(
    [
        ('feature_selector',
         pp.FeatureSelector(config.ModelConfig.features)
         ),
        ('feature_schema',
         pp.SchemaBuild(config.ModelConfig.features)
         ),
        ('model_xgb',
         XGBClassifier(**parameters)
         )
    ]
)
