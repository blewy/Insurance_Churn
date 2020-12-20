from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features import preprocessors as pp
from src.config.core import config
import warnings

parameters = dict(config.parameters_config)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
churn_pipe = Pipeline(
    [
        ('feature_selector',
         pp.FeatureSelector(config.model_config.features)
         ),
        ('feature_schema',
         pp.SchemaBuild(config.model_config.features)
         ),
        ('model_xgb',
         XGBClassifier(**parameters, eval_metric=config.model_config.eval_metric,use_label_encoder=False)
         )
    ]
)
