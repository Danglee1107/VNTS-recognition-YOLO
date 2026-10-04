from os.path import exists
from pathlib import Path


BASE = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_DIR = DATA_DIR / 'processed'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
