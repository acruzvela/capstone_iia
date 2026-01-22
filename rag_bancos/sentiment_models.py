import logging
from transformers import pipeline

logger = logging.getLogger(__name__)

def crear_modelo_sentimiento_es():
    """
    Modelo en español para BdE (via transformers.pipeline).
    """
    logger.info("Cargando modelo de sentimiento en ESPAÑOL (pysentimiento/robertuito-sentiment-analysis)...")
    clf = pipeline(
        "sentiment-analysis",
        model="pysentimiento/robertuito-sentiment-analysis",
        device=-1   # CPU para evitar errores CUDA
    )
    return clf

def crear_modelo_sentimiento_en():
    """
    Modelo en inglés (FinBERT).
    """
    logger.info("Cargando modelo de sentimiento en INGLÉS (FinBERT)...")
    clf = pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        device=0  # puedes dejar GPU si funciona bien
    )
    return clf

def safe_max_len(clf, fallback=512) -> int:
    """
    Devuelve un max_length seguro según el modelo/tokenizer.
    Para roberta suele ser max_position_embeddings-2 (~128 si mpe=130).
    """
    mpe = getattr(getattr(clf, "model", None), "config", None)
    mpe = getattr(mpe, "max_position_embeddings", None)

    if isinstance(mpe, int) and mpe > 0:
        # margen típico en RoBERTa
        #return max(8, min(fallback, mpe - 2))
        margin = 2
        return max(8, min(fallback, mpe - margin))


    # fallback: intenta sacar el límite del tokenizer (a veces es enorme)
    tok = getattr(clf, "tokenizer", None)
    tmax = getattr(tok, "model_max_length", None)
    if isinstance(tmax, int) and 0 < tmax < 100000:
        return min(fallback, tmax)

    return fallback
