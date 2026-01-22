import unicodedata
import re
import textwrap

# Ligaduras típicas de PDFs (muy importantes para tokenización)
_PDF_LIGATURES = {
    "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
    "ﬅ": "ft", "ﬆ": "st",
}

def limpiar_texto(texto: str) -> str:
    """
    Limpieza robusta para textos procedentes de PDF, web o RSS.
    Diseñada para evitar errores en tokenizadores / modelos.

    Pasos:
      1) Normaliza Unicode
      2) Reemplaza ligaduras PDF
      3) Separa palabras pegadas (PortadaÍndices -> Portada Índices)
      4) Elimina caracteres de control invisibles
      5) Colapsa espacios en blanco
    """
    if not texto:
        return ""

    # 1) Normalización Unicode (arregla rarezas de PDFs)
    texto = unicodedata.normalize("NFKC", texto)

    # 2) Reemplazo de ligaduras PDF (fi, fl, etc.)
    for lig, repl in _PDF_LIGATURES.items():
        texto = texto.replace(lig, repl)

    # 3) Separar palabras pegadas por maquetación PDF
    texto = re.sub(
        r"(?<=[a-záéíóúñü])(?=[A-ZÁÉÍÓÚÑÜ])",
        " ",
        texto
    )

    # 4) Eliminar caracteres de control invisibles
    texto = re.sub(r"[\x00-\x08\x0B-\x1F\x7F]", " ", texto)

    # convierte “184 mm de euros” -> “184 millones de euros” (menos “basura”)
    texto = re.sub(r"\b(\d+)\s*mm\b", r"\1 millones", texto, flags=re.IGNORECASE)

    # separa números pegados a letras
    texto = re.sub(r"(?<=\d)(?=[A-Za-zÁÉÍÓÚÑÜáéíóúñü])", " ", texto)
    texto = re.sub(r"(?<=[A-Za-zÁÉÍÓÚÑÜáéíóúñü])(?=\d)", " ", texto)


    # 5) Colapsar espacios y saltos de línea
    texto = " ".join(texto.split())

    return texto

def trocear_texto(texto: str, max_chars: int = 600):
    """
    Corta el texto en trozos de tamaño max_chars aprox.
    No hace falta que sea perfecto, es solo para no pasar al modelo textos enormes.
    """
    # split separa el texto en palabras y join los une usando un espacio en blanco como separador
    texto = " ".join(texto.split())  # compactar espacios / saltos de línea
    trozos = textwrap.wrap(texto, max_chars)
    return [t for t in trozos if t.strip()]

def partir_mas(trozos, max_chars_small: int = 250):
    """
    Recibe una lista de strings (trozos) y los parte aún más pequeño.
    Útil como fallback cuando un batch provoca errores en el modelo/tokenizador.
    """
    mini = []
    for t in trozos:
        mini.extend(textwrap.wrap(t, max_chars_small))
    return [x for x in mini if x.strip()]

def es_chunk_sospechoso(
    t: str,
    min_len: int = 40,
    max_nonalpha_ratio: float = 0.55,
    max_digit_ratio: float = 0.25,
) -> bool:
    t2 = (t or "").strip()
    if len(t2) < min_len:
        return True

    n = len(t2)
    n_alpha = sum(ch.isalpha() for ch in t2)
    n_digit = sum(ch.isdigit() for ch in t2)

    nonalpha_ratio = 1 - (n_alpha / max(n, 1))
    digit_ratio = n_digit / max(n, 1)

    # 1) demasiado “no-letra” (símbolos, números, etc.)
    if nonalpha_ratio > max_nonalpha_ratio:
        return True

    # 2) demasiados dígitos -> típico de tablas/índices del PDF
    if digit_ratio > max_digit_ratio:
        return True

    return False

def contiene_keywords(texto: str, keywords: list[str]) -> bool:
    t = (texto or "").lower()
    return any(k in t for k in keywords)

