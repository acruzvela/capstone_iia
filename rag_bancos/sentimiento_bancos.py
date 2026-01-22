
import logging
import pandas as pd

from rag_bancos.sentiment_io import leer_documentos
from rag_bancos.config import DATA_DIRS, SOURCE_LANG, SOURCES, SENTIMENT_FEATURES_DIR
from rag_bancos.sentiment_scoring import score_en, score_es, resumir_por_documento, resumenes_a_filas_doc
from rag_bancos.sentiment_inference import analizar_sentimiento
from rag_bancos.sentiment_models import crear_modelo_sentimiento_es, crear_modelo_sentimiento_en
from rag_bancos.sentiment_aggregation import agregar_diario
from rag_bancos.sentiment_aggregation import agregar_macro_diario, agregar_macro_semanal

# ejecutar python -m rag_bancos.sentimiento_bancos

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def analizar_sentimiento_fuente(source_key: str, clf_es, clf_en, batch_size: int = 16):
    docs = leer_documentos(DATA_DIRS[source_key])
    if not docs:
        return [], []

    lang = SOURCE_LANG.get(source_key, "es")
    if lang == "es":
        clf = clf_es
        score_fn = score_es
    else:
        clf = clf_en
        score_fn = score_en

    return analizar_sentimiento(docs, clf, score_fn, fuente=source_key, batch_size=batch_size)


def main():
    # avoid clearing the screen in pipelines
    logger.info("=== Sentimiento multi-fuente + agregación diaria ===")

    # 1) Cargar modelos una vez
    clf_es = crear_modelo_sentimiento_es()
    clf_en = crear_modelo_sentimiento_en()

    # 2) Ejecutar sentimiento por fuente
    resumenes_all = []
    errors_all = []

    for src in SOURCES:
        logger.info("--- %s ---", src)
        docs = leer_documentos(DATA_DIRS[src])
        if not docs:
            logger.info("⚠️ %s: 0 documentos (ok).", src)
            continue

        lang = SOURCE_LANG.get(src)
        if lang == "es":
            resultados, errores = analizar_sentimiento(docs, clf_es, score_es, fuente=src)
        else:
            resultados, errores = analizar_sentimiento(docs, clf_en, score_en, fuente=src)

        errors_all.extend(errores)

        # resumen por documento
        resumen_src = resumir_por_documento(resultados, fuente=src, top_k=3, snippet_len=220)
        resumenes_all.extend(resumen_src)

    logger.info("🧾 Errores totales: %d", len(errors_all))
    if errors_all:
        logger.warning("Ejemplo error: %s", errors_all[0])

    # 3) Doc-level dataframe
    filas = resumenes_a_filas_doc(resumenes_all)
    df_docs = pd.DataFrame(filas)
    df_docs["date_time"] = df_docs["date"]
    df_docs["date"] = pd.to_datetime(df_docs["date"], utc=True, errors="coerce").dt.date

    SENTIMENT_FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # guarda doc-level
    docs_path = SENTIMENT_FEATURES_DIR / "sentiment_docs.csv"
    df_docs.to_csv(docs_path, index=False, encoding="utf-8")
    logger.info("✅ Guardado sentiment_docs.csv")

    # 4) Agregación diaria
    df_daily = agregar_diario(df_docs, sources=SOURCES)

    # guarda daily
    daily_path = SENTIMENT_FEATURES_DIR / "sentiment_timeseries_daily.csv"
    df_daily.to_csv(daily_path, index=False, encoding="utf-8")
    logger.info("✅ Guardado sentiment_timeseries_daily.csv")

    # guarda macro diario
    df_macro = agregar_macro_diario(df_daily)
    macro_path = SENTIMENT_FEATURES_DIR / "sentiment_macro_daily.csv"
    df_macro.to_csv(macro_path, index=False, encoding="utf-8")
    logger.info("✅ Guardado sentiment_macro_daily.csv")

    # guarda macro semanal
    df_macro_w = agregar_macro_semanal(df_macro)
    df_macro_w.to_csv(
        SENTIMENT_FEATURES_DIR / "sentiment_macro_weekly.csv",
        index=False,
        encoding="utf-8",
    )
    logger.info("✅ Guardado sentiment_macro_weekly.csv")

    # 5) Print mini tabla (últimos días)
    logger.info("\n📈 Vista rápida (últimas 10 filas):\n%s", df_daily.tail(10).to_string(index=False))

    # 5) Print mini tabla (últimos días)
    logger.info("\n📈 Últimos 10 días con docs por fuente:")
    for src in SOURCES:
        sub = df_daily[(df_daily["fuente"] == src) & (df_daily["n_docs"] > 0)].tail(10)
        if len(sub):
            logger.info("--- %s ---\n%s", src, sub.to_string(index=False))

    # 5) Print macro tabla (últimos días)
    logger.info("\n📈 Macro (últimas 10 filas):\n%s", df_macro.tail(10).to_string(index=False))

    # 5) Print macro semanal (últimos 10)
    logger.info("\n📈 Macro semanal (últimos 10):\n%s", df_macro_w.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
