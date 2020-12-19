from pathlib import Path
import typing as t

from pydantic import BaseModel
from strictyaml import load, YAML

import src

# Project Directories
PACKAGE_ROOT = Path(src.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "models"
DATASET_DIR = PACKAGE_ROOT / "data"


class AppConfig(BaseModel):
    """
    Application-level config.
    """

    package_name: str
    pipeline_name: str
    pipeline_save_file: str
    training_data_file: str
    test_data_file: str


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """

    target: str
    features: t.Sequence[str]
    numerical_vars: t.Sequence[str]
    categorical_vars: t.Sequence[str]
    temporal_vars: str
    numerical_vars_with_na: t.Sequence[str]
    numerical_na_not_allowed: t.Sequence[str]
    test_size: float
    random_state: int


class ParametersConfig(BaseModel):
    """
    All configuration relevant to model
    hyper-parameters.
    """

    n_estimators: int
    colsample_bytree: float
    gamma: float
    learning_rate: float
    max_depth: int
    min_child_weight: float
    n_estimators: int
    subsample: float


class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    model_config: ModelConfig
    parameters_config: ParametersConfig


def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: object = None) -> object:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        model_config=ModelConfig(**parsed_config.data),
        parameters_config=ParametersConfig(**parsed_config.data)
    )

    return _config


config = create_and_validate_config()
