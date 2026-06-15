import importlib
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PACKAGE_PARENT_DIR = PACKAGE_DIR.parent
PROJECT_DIR = PACKAGE_PARENT_DIR.parent
CRPROPA_DATA_DIR = PACKAGE_DIR / "CRPropa3-data"


def ensure_crpropa_data_path() -> Path:
    '''
    Register the local CRPropa3-data checkout for module imports.
    https://github.com/CRPropa/CRPropa3-data
    '''
    if not CRPROPA_DATA_DIR.exists():
        raise FileNotFoundError(
            "CRPropa3-data is missing. Expected official checkout at "
            f"{CRPROPA_DATA_DIR}."
        )
    data_path = str(CRPROPA_DATA_DIR)
    if data_path not in sys.path:
        sys.path.insert(0, data_path)
    return CRPROPA_DATA_DIR


def import_crpropa_data_module(module_name: str):
    '''
    Import a module from the local CRPropa3-data checkout.
    '''
    ensure_crpropa_data_path()
    return importlib.import_module(module_name)
