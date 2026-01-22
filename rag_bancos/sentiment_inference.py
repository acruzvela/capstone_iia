import logging
import textwrap
from .sentiment_models import safe_max_len
from .sentiment_preprocess import limpiar_texto, trocear_texto, es_chunk_sospechoso, partir_mas, contiene_keywords
from .config import MACRO_KEYWORDS

logger = logging.getLogger(__name__)

def reintentar_batch_mas_pequeno(clf, batch, max_length=512, small_chars=250):
    batch_small = []
    for t in batch:
        batch_small.extend(textwrap.wrap(t, small_chars))
    batch_small = [x for x in batch_small if x.strip()]
    preds = clf(batch_small, truncation=True, max_length=max_length, padding=True)
    return batch_small, preds

def analizar_sentimiento(
    docs,
    clf,
    score_fn,
    fuente: str,
    max_chars: int = 600,
    batch_size: int = 16,
    clip_chars: int = 2000,
    max_length: int = 512,
    verbose: bool = True,
    lang: str | None = None,
    use_macro_filter: bool = False,
):
    """
    Analiza docs = [(nombre_doc, texto, dt_doc, txt_path), ...]
    Devuelve (resultados, errors)
    """
    resultados = []
    errors = []

    # Límite seguro real del modelo (p.ej. robertuito -> 128)
    max_len_model = safe_max_len(clf, fallback=max_length)

    for nombre_doc, texto, dt_doc, txt_path in docs:
        texto = limpiar_texto(texto)
        trozos = trocear_texto(texto, max_chars=max_chars)
        if not trozos:
            continue

        for i in range(0, len(trozos), batch_size):
            batch = trozos[i:i + batch_size]

            # recorte prudente por caracteres
            if clip_chars is not None:
                batch = [t[:clip_chars] for t in batch]

            # filtro anti-basura (tablas, índices, etc.)
            batch = [t for t in batch if not es_chunk_sospechoso(t)]
            if not batch:
                continue

            # filtro macro opcional
            if use_macro_filter and lang:
                kws = MACRO_KEYWORDS.get(lang, [])
                batch = [t for t in batch if contiene_keywords(t, kws)]
                if not batch:
                    continue

            # -------- 1) intento normal por batch --------
            try:
                preds = clf(
                    batch,
                    truncation=True,
                    max_length=max_len_model,
                    padding=True
                )

            except Exception as e:
                if verbose:
                    logger.warning("Batch %d fallo en %s: %s -> probando 1 a 1", i, nombre_doc, e)

                ok_any = False

                # -------- 2) fallback 1-a-1 --------
                for j, t in enumerate(batch):
                    if es_chunk_sospechoso(t):
                        continue

                    pred = None

                    try:
                        pred = clf(
                            [t],
                            truncation=True,
                            max_length=max_len_model,
                            padding=True
                        )[0]

                    except Exception:
                        # reintento aún más corto (por si acaso)
                        try:
                            pred = clf(
                                [t],
                                truncation=True,
                                max_length=min(256, max_len_model),
                                padding=True
                            )[0]
                        except Exception as e_two:
                            # -------- 3) último intento: partir en mini-trozos --------
                            mini = partir_mas([t], max_chars_small=120)
                            salvado = False

                            for tt in mini:
                                if es_chunk_sospechoso(
                                    tt,
                                    min_len=25,
                                    max_nonalpha_ratio=0.75,
                                    max_digit_ratio=0.40,
                                ):
                                    continue
                                try:
                                    pred = clf(
                                        [tt],
                                        truncation=True,
                                        max_length=max_len_model,
                                        padding=True
                                    )[0]

                                    label = pred.get("label", "")
                                    prob = float(pred.get("score", 0.0))
                                    signed = float(score_fn(label) * prob)
                                    dt_iso = dt_doc.isoformat() if dt_doc else None
                                    date_iso = dt_doc.date().isoformat() if dt_doc else None

                                    resultados.append({
                                        "fuente": fuente,
                                        "documento": nombre_doc,
                                        "dt": dt_iso,
                                        "date": date_iso,
                                        "chunk": i + j,
                                        "label": label,
                                        "score": prob,
                                        "signed": signed,
                                        "text": tt,
                                    })
                                    salvado = True
                                    ok_any = True
                                    break
                                except Exception:
                                    pass

                            if salvado:
                                continue

                            # no se pudo salvar este chunk
                            errors.append({
                                "documento": nombre_doc,
                                "chunk": i + j,
                                "error": repr(e_two),
                                "text_snippet": t[:200].replace("\n", " "),
                            })
                            continue

                    # guardado normal del 1-a-1 si funcionó
                    label = pred.get("label", "")
                    prob = float(pred.get("score", 0.0))
                    signed = float(score_fn(label) * prob)
                    dt_iso = dt_doc.isoformat() if dt_doc else None
                    date_iso = dt_doc.date().isoformat() if dt_doc else None

                    resultados.append({
                        "fuente": fuente,
                        "documento": nombre_doc,
                        "dt": dt_iso,
                        "date": date_iso,
                        "chunk": i + j,
                        "label": label,
                        "score": prob,
                        "signed": signed,
                        "text": t,
                    })
                    ok_any = True

                # si salvamos algo, seguimos con el siguiente batch
                if ok_any:
                    continue

                # si no se salvó nada, pasamos al siguiente batch/doc
                continue

            # -------- guardado normal si batch OK --------
            for idx, pred in enumerate(preds):
                label = pred.get("label", "")
                prob = float(pred.get("score", 0.0))
                signed = float(score_fn(label) * prob)
                dt_iso = dt_doc.isoformat() if dt_doc else None
                date_iso = dt_doc.date().isoformat() if dt_doc else None

                resultados.append({
                    "fuente": fuente,
                    "documento": nombre_doc,
                    "dt": dt_iso,
                    "date": date_iso,
                    "chunk": i + idx,
                    "label": label,
                    "score": prob,
                    "signed": signed,
                    "text": batch[idx],
                })
    return resultados, errors

