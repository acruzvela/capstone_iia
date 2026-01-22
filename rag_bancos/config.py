# rag_bancos/config.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
BASE_DATA_DIR = BASE_DIR / "data" / "bancos"
BASE_EMB_DIR = BASE_DIR / "embeddings"
BASE_FEATURES_DIR = BASE_DIR / "data" / "features"

DB_PATH = Path(r"E:/cruz/informatica/sqlite/ibex35.db")

DATA_DIRS = {
    "BDE": str(BASE_DATA_DIR / "bde"),
    "FED": str(BASE_DATA_DIR / "fed"),
    "BCE": str(BASE_DATA_DIR / "bce"),
    "GENERAL_ES": str(BASE_DATA_DIR / "general_es"),
    "GENERAL_ENG": str(BASE_DATA_DIR / "general_eng"),
}

EMB_DIRS = {

    "BDE": str(BASE_EMB_DIR / "faiss_bde"),
    "BCE": str(BASE_EMB_DIR / "faiss_bce"),
    "FED": str(BASE_EMB_DIR / "faiss_fed"),
    "GENERAL_ENG": str(BASE_EMB_DIR / "faiss_general_eng"),
    "GENERAL_ES": str(BASE_EMB_DIR / "faiss_general_es"),
}

SENTIMENT_FEATURES_DIR = BASE_FEATURES_DIR / "sentiment"

SOURCES = ["BDE", "BCE", "FED", "GENERAL_ES", "GENERAL_ENG"]

SOURCE_LANG = {
    "BDE": "es",
    "BCE": "en",
    "FED": "en",
    "GENERAL_ES": "es",
    "GENERAL_ENG": "en",
}

MACRO_KEYWORDS = {
    "es": [
        "inflación","ipc","pib","crecimiento","recesión","paro","desempleo",
        "tipos","tipo de interés","bce","banco central","banco de españa",
        "euríbor","euribor","deuda","déficit","prima de riesgo",
        "bono","bonos","rentabilidad","yield","mercado","bolsa","ibex",
        "crédito","hipoteca","salarios","consumo","exportaciones","importaciones"
    ],
    "en": [
        "inflation","cpi","gdp","growth","recession","unemployment","jobs",
        "rates","interest rate","central bank","ecb","fed","federal reserve",
        "bond","bonds","yields","treasury","deficit","debt",
        "markets","stocks","equities","credit","mortgage","wages","consumption"
    ],
}

