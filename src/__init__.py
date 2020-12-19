from src.config.core import config, PACKAGE_ROOT

with open(PACKAGE_ROOT / "VERSION") as version_file:
    __version__ = version_file.read().strip()

