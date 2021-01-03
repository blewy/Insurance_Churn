import connexion

from api.config import Config


def create_app(*, config_object: Config) -> connexion.App:
    """Create app instance."""

    connexion_app = connexion.App(
        __name__, debug=config_object.DEBUG, specification_dir="spec/"
    )
    flask_app = connexion_app.app
    flask_app.config.from_object(config_object)
    connexion_app.add_api("api.yaml")

    return connexion_app
