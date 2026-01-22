from collections import defaultdict
from .sentiment_io import fecha_from_name

def score_es(label: str) -> float:
    lab = label.upper()
    # posibles variantes: POS / POSITIVE, NEG / NEGATIVE, NEU / NEUTRAL
    if "POS" in lab:
        return 1.0
    if "NEG" in lab:
        return -1.0
    return 0.0

def score_en(label: str) -> float:
    label = label.lower()
    if "positive" in label:
        return 1.0
    if "negative" in label:
        return -1.0
    return 0.0

def _ensure_signed(it: dict) -> float:
    """Devuelve signed fiable. Si no existe, lo calcula desde label+score."""
    if "signed" in it and it["signed"] is not None:
        return float(it["signed"])

    lab = (it.get("label") or "").upper()
    prob = float(it.get("score", 0.0))
    if "POS" in lab:
        return +prob
    if "NEG" in lab:
        return -prob
    return 0.0

def resumir_por_documento(resultados, fuente, top_k: int = 3, snippet_len: int = 220):
    """
    Resume por documento:
      - score medio (signed)
      - nº trozos
      - top_k trozos más negativos y más positivos (con snippet)
    """

    por_doc = defaultdict(list)

    for r in resultados:
        fuente = r.get("fuente", "NA")
        doc = r.get("documento")
        if doc:
            por_doc[(fuente, doc)].append(r)


    resumen = []

    #for doc, items in por_doc.items():
    for (fuente, doc), items in por_doc.items():
        # asegurar signed en todos (una sola vez)
        for it in items:
            it["signed"] = _ensure_signed(it)

        signed_vals = [it["signed"] for it in items]

        # ✅ solo chunks con señal (no neutrales)
        signal = [s for s in signed_vals if abs(s) > 1e-6]

        score_medio = (sum(signal) / len(signal)) if signal else 0.0

        pct_signal = (len(signal) / len(signed_vals)) if signed_vals else 0.0

        # ordenar por signed (más negativo primero)
        items_sorted = sorted(items, key=lambda x: x["signed"])

        top_neg = items_sorted[:top_k]
        top_pos = list(reversed(items_sorted[-top_k:]))

        def pack(it):
            txt = (it.get("text") or "").strip().replace("\n", " ")
            if len(txt) > snippet_len:
                txt = txt[:snippet_len].rstrip() + "..."
            return {
                "chunk": it.get("chunk"),
                "signed": float(it["signed"]),
                "label": it.get("label", ""),
                "prob": float(it.get("score", 0.0)),
                "text": txt
            }

        resumen.append({
            "fuente": fuente,
            "documento": doc,
            "score": float(score_medio),
            "n_trozos": len(items),
            "top_neg": [pack(x) for x in top_neg],
            "top_pos": [pack(x) for x in top_pos],
            "pct_signal": pct_signal,
        })

    return resumen

def resumenes_a_filas_doc(resumenes):
    filas = []
    for r in resumenes:
        doc = r["documento"]
        dt = fecha_from_name(doc)
        filas.append({
            "fuente": r["fuente"],
            "date": dt,
            "documento": doc,
            "score": float(r.get("score", 0.0)),
            "pct_signal": float(r.get("pct_signal", 0.0)),
            "n_trozos": int(r.get("n_trozos", 0)),
        })
    return filas

