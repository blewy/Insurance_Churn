from sklearn.model_selection import train_test_split

from src.features import build_features
from src.features.data_management import (
    load_dataset,
    save_pipeline,
)
from src.config.core import config
from src import __version__ as _version


def run_training() -> None:
    """Train the model."""

    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)

    # divide train and test
    x_train, x_test, y_train, y_test = train_test_split(
        data[config.model_config.features],  # predictors
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config.random_state,
    )

    build_features.churn_pipe.fit(x_train, y_train)

    save_pipeline(pipeline_to_persist=build_features.churn_pipe)


if __name__ == "__main__":
    run_training()
