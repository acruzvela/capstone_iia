import os
from datetime import datetime, timezone
import feedparser
from bs4 import BeautifulSoup
import re
import json
from pathlib import Path
import hashlib
import glob
from rag_bancos.config import BASE_DATA_DIR, BASE_DIR, BASE_EMB_DIR, DATA_DIRS, EMB_DIRS
from agente.http_utils import get_session

# Feeds
FEEDS_BDE = [
    "https://www.bde.es/wbe/es/inicio/rss/rss-noticias/",
    "https://www.bde.es/wbe/es/inicio/rss/rss-blog/"
]

FEEDS_BCE = [
    "https://www.ecb.europa.eu/rss/press.html",
    "https://www.ecb.europa.eu/rss/blog.html",
    "https://www.ecb.europa.eu/rss/wppub.html",
]

FEEDS_FED = [
    "https://www.federalreserve.gov/feeds/press_all.xml",
]

FEEDS_GENERAL_ENG = [
    "https://www.bankofengland.co.uk/rss/knowledgebank",
    "https://www.bis.org/doclist/rss_all_categories.rss",
    "https://www.bis.org/doclist/all_pressrels.rss",
    "https://www.ft.com/global-economy?format=rss",
    "https://www.ft.com/markets?format=rss",
    "https://www.thestreet.com/.rss/full/",
    "https://dealbreaker.com/.rss/full",
    "https://www.financialsamurai.com/feed/",
    "https://moneyweek.com/feed/all",
    "https://gfmag.com/feed/",
    "https://search.cnbc.com/rs/search/combined/news_view.html?partnerId=wrss01",
    "https://www.investing.com/rss/news.rss",

]

FEEDS_GENERAL_ES = [
    "https://feeds.elpais.com/mrss-s/pages/ep/site/cincodias.elpais.com/section/ultimas-noticias/portada",
    "https://e01-expansion.uecdn.es/rss/portada.xml",
    "https://www.abc.es/rss/atom/economia/",
    "https://cincodias.elpais.com/rss/cincodias/ultimas_noticias.xml",
    "https://www.estrategiasdeinversion.com/rss/rssnoticias.xml",
    "https://www.eleconomista.es/rss/mercados-financieros.xml",
    "https://feeds.elpais.com/mrss-s/list/ep/site/cincodias.elpais.com/section/mercados-financieros",
    "https://cincodias.elpais.com/rss/cincodias/portada.xml",
    "https://e01-expansion.uecdn.es/rss/mercados/cronica-bolsa.xml",
    "https://e01-expansion.uecdn.es/rss/mercados.xml",
    "https://e01-expansion.uecdn.es/rss/mercados/materias-primas.xml",
    "https://e01-expansion.uecdn.es/rss/mercados/dividendos.xml",
    "https://e01-expansion.uecdn.es/rss/mercados/euribor.xml",
    "https://e01-expansion.uecdn.es/rss/ahorro/pensiones.xml",
    "https://e01-expansion.uecdn.es/rss/mercados/criptomonedas.xml",
    "https://e01-expansion.uecdn.es/rss/mercados/divisas.xml",
    "https://e01-expansion.uecdn.es/rss/mercados/fondos.xml",
    "https://e01-expansion.uecdn.es/rss/mercados/renta-fija.xml",
    "https://es.investing.com/rss/news_301.rss",
    "https://es.investing.com/rss/news_1065.rss",
    "https://es.investing.com/rss/news_1064.rss",
    "https://es.investing.com/rss/news_1063.rss",
    "https://es.investing.com/rss/news_1061.rss",
    "https://es.investing.com/rss/news_357.rss",
    "https://es.investing.com/rss/news_356.rss",
    "https://es.investing.com/rss/news_1.rss",
    "https://es.investing.com/rss/news_285.rss",
    "https://es.investing.com/rss/news_11.rss",
    "https://es.investing.com/rss/news_25.rss",
    "https://es.investing.com/rss/news_95.rss",
    "https://es.investing.com/rss/news_12.rss",
    "https://es.investing.com/rss/news_14.rss",
    "https://es.investing.com/rss/news_288.rss",
    "https://es.investing.com/rss/news_287.rss",
]

BDE_PDFS = [
    "https://www.bde.es/f/webbde/INF/MenuHorizontal/Publicaciones/PublicacionesAnuales/InformeInstitucional/2021/INI2021_Capitulo4_Informacionfinanciera.pdf"
]
# ejecutar python -m rag_bancos.rag_bancos

# --------- Utilidades ---------
def limpiar_html(texto: str) -> str:
    return BeautifulSoup(texto or "", "html.parser").get_text(" ", strip=True)

def sanitize_filename(t: str) -> str:
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r'[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑ \-_]', '', t)
    return t[:80].replace(" ", "_")

def entry_datetime(entry):
    """
    Devuelve datetime (UTC si no hay tz) usando published/updated del feed.
    """
    dt_obj = None

    if getattr(entry, "published_parsed", None):
        dt_obj = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
    elif getattr(entry, "updated_parsed", None):
        dt_obj = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)

    return dt_obj

def normalize_title(t):
    return " ".join(t.split()).lower()


def guardar_doc_con_meta(data_dir: str, fname_txt: str, titulo: str, cuerpo: str, meta: dict):
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    txt_path = Path(data_dir) / fname_txt
    txt_path.write_text(f"{titulo.strip()}\n\n{cuerpo.strip()}", encoding="utf-8", errors="ignore")

    meta_path = txt_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

# --------- 1 Leer docs ---------------
def descargar_docs_fuente(nombre_fuente: str, feeds: list[str], prefijo: str, max_por_feed: int = 20):
    data_dir = DATA_DIRS[prefijo]
    os.makedirs(data_dir, exist_ok=True)

    print(f"📰 Descargando documentos de {nombre_fuente} ({prefijo})...")
    nuevos = 0
    ahora = datetime.now(timezone.utc)

    for url in feeds:
        try:
            feed = feedparser.parse(url)

            for entry in feed.entries[:max_por_feed]:
                titulo = limpiar_html(getattr(entry, "title", ""))

                cuerpo = limpiar_html(
                    getattr(entry, "summary", "") or getattr(entry, "description", "")
                )

                # opcional: si el feed trae "content" (Atom suele traerlo)
                if not cuerpo and getattr(entry, "content", None):
                    try:
                        cuerpo = limpiar_html(entry.content[0].value)
                    except Exception:
                        pass

                if not titulo or not cuerpo:
                    continue

                # ✅ fecha real del RSS/Atom
                dt_pub = entry_datetime(entry)  # puede ser None

                if dt_pub is not None:
                    fecha_doc = dt_pub.strftime("%Y%m%d_%H%M%S")
                else:
                    fecha_doc = "UNDATED"   # ✅ no inventar fecha


                #fecha_doc = dt_pub.strftime("%Y%m%d_%H%M%S")

                # url/canonical para dedupe mejor (si existe)
                link = getattr(entry, "link", "") or getattr(entry, "id", "")
                raw_link = (
                    getattr(entry, "id", None)
                    or getattr(entry, "link", None)
                    or f"{prefijo}|{normalize_title(titulo)}"
                )
                link_hash = hashlib.md5(raw_link.encode("utf-8")).hexdigest()[:8]

                
                #fname = f"{fecha_doc}_{prefijo}_{sanitize_filename(titulo)}.txt"
                fname=f"{fecha_doc}_{prefijo}_{link_hash}_{sanitize_filename(titulo)}.txt"
                path = os.path.join(data_dir, fname)

                # si ya existe ese fichero, no reescribimos
                # if os.path.exists(path):
                #     continue

                pattern = os.path.join(data_dir, f"*_{prefijo}_{link_hash}_*.txt")
                if glob.glob(pattern):
                    continue

                meta = {
                    "fuente": prefijo,
                    "source_name": nombre_fuente,
                    "feed_url": url,
                    "title": titulo,
                    "link": link,
                    "link_hash": link_hash,   # 👈 muy útil después
                    "published": dt_pub.isoformat() if dt_pub else None,
                    "retrieved": ahora.isoformat(),
                }


                guardar_doc_con_meta(data_dir, fname, titulo, cuerpo, meta)
                nuevos += 1

        except Exception as e:
            print(f"⚠️ Error con feed {nombre_fuente} ({url}): {e}")

    print(f"✅ Guardados {nuevos} documentos de {nombre_fuente} en {data_dir}")

def descargar_pdfs_bde():
    """
    Ejemplo sencillo: descargar 1 PDF macro del BdE y convertirlo a txt.
    Luego podrás ampliar la lista BDE_PDFS.
    """
    from pathlib import Path
    from pypdf import PdfReader  # asegúrate de tener `pip install pypdf`
    # use shared session with retries and sensible headers/timeouts
    session = get_session(user_agent="CAPSTONE/1.0 (+https://example.com)")

    data_dir = DATA_DIRS["BDE"]
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    print("📄 Descargando/actualizando PDFs del BdE...")
    nuevos = 0

    for url in BDE_PDFS:
        nombre_pdf = url.split("/")[-1]
        ruta_pdf = Path(data_dir) / nombre_pdf
        ruta_txt = ruta_pdf.with_suffix(".txt")

        # descargar PDF si no existe (con retries desde session)
        if not ruta_pdf.exists():
            try:
                resp = session.get(url, timeout=30)
                resp.raise_for_status()
                ruta_pdf.write_bytes(resp.content)
            except Exception as e:
                print(f"⚠️ Error al descargar PDF {url}: {e}")
                continue

        # convertir a txt si no existe
        if not ruta_txt.exists():
            try:
                reader = PdfReader(str(ruta_pdf))
                texto = "\n".join(page.extract_text() or "" for page in reader.pages)
                # write replacing invalid chars instead of silently ignoring
                ruta_txt.write_text(texto, encoding="utf-8", errors="replace")
                nuevos += 1
            except Exception as e:
                print(f"⚠️ Error al convertir PDF a txt {ruta_pdf}: {e}")
                continue

        meta = {
            "fuente": "BDE",
            "source_name": "Banco de España (PDF)",
            "feed_url": None,
            "title": nombre_pdf,
            "link": url,
            "published": None,
            "retrieved": datetime.now(timezone.utc).isoformat(),
            "type": "pdf"
        }
        meta_path = ruta_txt.with_suffix(".meta.json")
        # write metadata reliably (replace errors if any)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8", errors="replace")


    print(f"✅ Convertidos {nuevos} PDF(s) del BdE a .txt en {data_dir}")

# --------- 2) Crear / actualizar FAISS ---------
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# lo creamos una sola vez y lo reutilizamos
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def crear_o_actualizar_faiss_fuente(prefijo: str):
    """
    Construye o actualiza el índice FAISS de una institución (BDE, BCE o FED).
    Usa DATA_DIRS[prefijo] y EMB_DIRS[prefijo].
    """
    data_dir = DATA_DIRS[prefijo]
    emb_dir = EMB_DIRS[prefijo]
    os.makedirs(data_dir, exist_ok=True)

    loader = DirectoryLoader(
        data_dir,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs = loader.load()
    if not docs:
        print(f"⚠️ No hay documentos en {data_dir}. Nada que indexar para {prefijo}.")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    if os.path.exists(emb_dir):
        # try safe load first, fallback to dangerous deserialization if necessary
        try:
            db = FAISS.load_local(emb_dir, embeddings)
        except Exception:
            db = FAISS.load_local(emb_dir, embeddings, allow_dangerous_deserialization=True)
        db.add_documents(chunks)
    else:
        db = FAISS.from_documents(chunks, embeddings)

    db.save_local(emb_dir)
    print(f"✅ Índice FAISS para {prefijo} actualizado en {emb_dir}")
    return db

# --------- 3) Crear cadena RAG ---------
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

def crear_rag_para_db(db, modelo_llm="mistral", temperatura=0.1, k=4):
    retriever = db.as_retriever(search_kwargs={"k": k})
    llm = OllamaLLM(model=modelo_llm, temperature=temperatura)

    prompt = ChatPromptTemplate.from_template(
        "Responde SIEMPRE en español. Usa únicamente la información de los documentos para responder.\n"
        "Si no puedes responder con certeza, dilo claramente.\n\n"
        "Documentos:\n{context}\n\nPregunta: {input}"
    )

    def format_docs(docs_list):
        return "\n\n".join(d.page_content for d in docs_list)

    chain = (
        {
            "context": retriever | format_docs,
            "input": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# --------- 4) Programa principal ---------

def main():
    os.system('cls' if os.name=='nt' else 'clear')
    
    # 1) Actualizar corpus de cada institución
    descargar_docs_fuente("Banco de España", FEEDS_BDE, "BDE", max_por_feed=30)
    descargar_docs_fuente("BCE", FEEDS_BCE, "BCE", max_por_feed=30)
    descargar_docs_fuente("Reserva Federal", FEEDS_FED, "FED", max_por_feed=30)
    descargar_docs_fuente("General en español", FEEDS_GENERAL_ES, "GENERAL_ES", max_por_feed=30)
    descargar_docs_fuente("General en inglés", FEEDS_GENERAL_ENG, "GENERAL_ENG", max_por_feed=30)
    descargar_pdfs_bde()  # opcional, pero muy recomendable

    # 2) Construir / actualizar índices FAISS por institución
    db_bde = crear_o_actualizar_faiss_fuente("BDE")
    db_bce = crear_o_actualizar_faiss_fuente("BCE")
    db_fed = crear_o_actualizar_faiss_fuente("FED")
    db_general_es = crear_o_actualizar_faiss_fuente("GENERAL_ES")
    db_general_eng = crear_o_actualizar_faiss_fuente("GENERAL_ENG")


    # 3) Menú
    while True:
        print("\nElige fuente para el RAG:")
        print("  1) Banco de España (BDE)")
        print("  2) BCE")
        print("  3) Reserva Federal (FED)")
        print("  4) Generalistas en español (GENERAL_ES)")
        print("  5) Generalistas en inglés (GENERAL_ENG)")
        print("  6) Comparar todas las fuentes")
        opcion = input("👉 Opción [1-6]: ").strip()

        if opcion in ("1", "2", "3","4","5"):
            mapa_db = {"1": ("BDE", db_bde), "2": ("BCE", db_bce), "3": ("FED", db_fed), "4": ("GENERAL_ES", db_general_es), "5": ("GENERAL_ENG", db_general_eng)}
            prefijo, db = mapa_db[opcion]
            if db is None:
                print(f"⚠️ No hay índice disponible para {prefijo}.")
            else:
                chain = crear_rag_para_db(db, modelo_llm="llama3")
                print(f"\n💬 Pregunta para {prefijo}:")
                pregunta = input("👉 ")

                respuesta = chain.invoke(pregunta)
                print("\n🧩 Respuesta generada:\n")
                print(respuesta)

                print("\n📚 Documentos más relevantes:")
                docs_scores = db.similarity_search_with_score(pregunta, k=5)
                for i, (doc, score) in enumerate(docs_scores, 1):
                    print(f"{i}. {doc.metadata.get('source', '')}  (score={score:.3f})")

        elif opcion == "6":
            # Comparativa básica: preguntar lo mismo a las tres instituciones
            pregunta = input("\n💬 Escribe una pregunta común para BDE / BCE / FED / GENERAL_ES / GENERAL_ENG:\n👉 ")

            resultados = []
            for prefijo, db in [("BDE", db_bde), ("BCE", db_bce), ("FED", db_fed), ("GENERAL_ES", db_general_es), ("GENERAL_ENG", db_general_eng)]:
                if db is None:
                    resultados.append((prefijo, "⚠️ Sin índice / sin documentos"))
                    continue
                chain = crear_rag_para_db(db, modelo_llm="mistral")
                resp = chain.invoke(pregunta)
                resultados.append((prefijo, resp))

            print("\n🧩 Respuestas comparadas:\n")
            for prefijo, resp in resultados:
                print(f"=== {prefijo} ===")
                print(resp)
                print("\n")

        else:
            print("Opción no válida.")

        salir = input("¿Deseas salir? (y/n): ").strip().lower() 
        if salir == "y": break

if __name__ == "__main__":
    main()
    