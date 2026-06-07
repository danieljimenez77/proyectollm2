"""
rag_memory.py — Componente 3: Retail Vision Intelligence System
Indexación y recuperación semántica de inspecciones históricas.

Características:
- Embeddings multilingües locales (paraphrase-multilingual-MiniLM-L12-v2)
- Vector store persistente con ChromaDB
- Estrategia de chunking híbrida: summary como chunk principal + metadatos para filtrado
- Indexación por issue individual (chunking granular opcional)
- Queries en lenguaje natural sintetizadas por Gemini con contexto recuperado
- Soporte para las 4 queries obligatorias del enunciado
- Prompts cargados desde prompts/
"""

import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("rag_memory")

BASE_DIR       = Path(__file__).parent
PROMPTS_DIR    = BASE_DIR.parent / "prompts"
VECTORSTORE_DIR = BASE_DIR.parent / "vectorstore"
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME       = "gemini-2.5-flash"
EMBEDDING_MODEL  = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K_DEFAULT    = 3

# Nombres de las colecciones en ChromaDB
COLLECTION_SUMMARIES = "inspection_summaries"   # chunk principal por inspección
COLLECTION_ISSUES    = "inspection_issues"       # chunk granular por issue


def load_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if api_key:
        return api_key
    fallback_path = BASE_DIR.parent / "Apikey.txt"
    if fallback_path.exists():
        key = (
            fallback_path
            .read_text(encoding="utf-8-sig")
            .strip()
            .replace('\r', '')
            .replace('\n', '')
        )
        if key:
            return key
    raise EnvironmentError(
        "GEMINI_API_KEY no encontrada. Crea Apikey.txt en la raíz del proyecto."
    )


client = genai.Client(api_key=load_api_key())


# ─────────────────────────────────────────────
# Carga de prompts
# ─────────────────────────────────────────────

def load_prompt_template(filename: str) -> str:
    path = PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt no encontrado: {path}")
    return path.read_text(encoding="utf-8")


# ─────────────────────────────────────────────
# Llamada a Gemini (solo texto)
# ─────────────────────────────────────────────

def _call_gemini(prompt: str) -> str:
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[types.Part.from_text(text=prompt)],
        config=types.GenerateContentConfig(temperature=0),
    )
    return response.text.strip()


# ─────────────────────────────────────────────
# RAG Memory principal
# ─────────────────────────────────────────────

class RAGMemory:
    """
    Memoria semántica del sistema de inspección.
    Indexa inspecciones y permite recuperación por similitud semántica.

    Estrategia de chunking híbrida:
    - COLLECTION_SUMMARIES: un chunk por inspección (summary rico + metadatos)
    - COLLECTION_ISSUES: un chunk por issue individual (granular, para queries específicas)

    Ejemplo de uso:
        rag = RAGMemory()
        rag.index_inspection(inspection_dict)
        response = rag.query("¿Cuándo fue la última vez que Z_S1 tuvo fill rate bajo?")
        print(response)
    """

    def __init__(self):
        logger.info(f"Cargando modelo de embeddings: {EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        self.chroma = chromadb.PersistentClient(
            path=str(VECTORSTORE_DIR),
        )

        # Colección principal: un documento por inspección
        self.col_summaries = self.chroma.get_or_create_collection(
            name=COLLECTION_SUMMARIES,
            metadata={"hnsw:space": "cosine"}
        )

        # Colección granular: un documento por issue
        self.col_issues = self.chroma.get_or_create_collection(
            name=COLLECTION_ISSUES,
            metadata={"hnsw:space": "cosine"}
        )

        logger.info(
            f"RAGMemory iniciado | "
            f"summaries={self.col_summaries.count()} | "
            f"issues={self.col_issues.count()}"
        )

    # ── Generación de summary ────────────────────────────────────────

    def _generate_summary(self, inspection: dict) -> str:
        """
        Usa Gemini para generar un summary semánticamente rico
        de la inspección, optimizado para recuperación futura.
        """
        template = load_prompt_template("rag_summary.txt")
        prompt   = template.replace(
            "{inspection_json}",
            json.dumps(inspection, ensure_ascii=False, indent=2)
        )
        summary = _call_gemini(prompt)
        logger.info(f"Summary generado ({len(summary)} chars)")
        return summary

    # ── Indexación ───────────────────────────────────────────────────

    def index_inspection(
        self,
        inspection: dict,
        index_issues: bool = True
    ) -> str:
        """
        Indexa una inspección en el vector store.

        Args:
            inspection:   dict con el resultado del shelf_inspector
            index_issues: si True, indexa también cada issue individualmente

        Returns:
            inspection_id del registro indexado
        """
        inspection_id = inspection.get("inspection_id", "UNKNOWN")

        # Evitar duplicados
        existing = self.col_summaries.get(ids=[inspection_id])
        if existing["ids"]:
            logger.info(f"[SKIP] {inspection_id} ya está indexado")
            return inspection_id

        # Generar summary rico para embeddings
        summary = self._generate_summary(inspection)

        # Extraer metadatos estructurados para filtrado pre-retrieval
        timestamp = inspection.get("timestamp", "")
        try:
            dt       = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            date_str = dt.strftime("%Y-%m-%d")
            hour     = dt.hour
            weekday  = dt.strftime("%A")  # Monday, Tuesday, etc.
        except Exception:
            date_str = ""
            hour     = -1
            weekday  = ""

        metadata = {
            "inspection_id":  inspection_id,
            "zone_id":        inspection.get("zone_id", ""),
            "overall_status": inspection.get("overall_status", ""),
            "shelf_fill_rate": float(inspection.get("shelf_fill_rate", 0.0)),
            "n_issues":       len(inspection.get("issues", [])),
            "timestamp":      timestamp,
            "date":           date_str,
            "hour":           hour,
            "weekday":        weekday,
            "image_path":     inspection.get("image_path", ""),
        }

        # Calcular embedding del summary
        embedding = self.embedder.encode(summary).tolist()

        # Indexar chunk principal
        self.col_summaries.add(
            ids=[inspection_id],
            embeddings=[embedding],
            documents=[summary],
            metadatas=[metadata]
        )
        logger.info(f"[INDEX] Summary indexado: {inspection_id} | zona={metadata['zone_id']}")

        # Indexar issues individualmente (chunking granular)
        if index_issues:
            self._index_issues(inspection, metadata)

        return inspection_id

    def _index_issues(self, inspection: dict, parent_metadata: dict) -> None:
        """Indexa cada issue de la inspección como chunk individual."""
        issues = inspection.get("issues", [])
        for issue in issues:
            issue_doc_id = f"{parent_metadata['inspection_id']}_{issue.get('issue_id', 'ISS')}"

            # Texto rico para el issue
            issue_text = (
                f"Zona {parent_metadata['zone_id']} — "
                f"Issue tipo '{issue.get('type')}' en '{issue.get('location')}': "
                f"{issue.get('description')} "
                f"Severidad: {issue.get('severity')}. "
                f"Confianza: {issue.get('confidence', 0)*100:.0f}%. "
                f"Área afectada: {issue.get('affected_area_pct', 0):.0f}%. "
                f"Fecha: {parent_metadata.get('date')} {parent_metadata.get('weekday')} "
                f"a las {parent_metadata.get('hour')}h."
            )

            issue_metadata = {
                **parent_metadata,
                "issue_type":       issue.get("type", "other"),
                "issue_severity":   issue.get("severity", "low"),
                "issue_location":   issue.get("location", ""),
                "issue_confidence": float(issue.get("confidence", 0.0)),
            }

            embedding = self.embedder.encode(issue_text).tolist()

            try:
                self.col_issues.add(
                    ids=[issue_doc_id],
                    embeddings=[embedding],
                    documents=[issue_text],
                    metadatas=[issue_metadata]
                )
            except Exception as e:
                logger.warning(f"Issue ya indexado o error: {issue_doc_id} — {e}")

        logger.info(
            f"[INDEX] {len(issues)} issues indexados para {parent_metadata['inspection_id']}"
        )

    # ── Recuperación ─────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        k: int = TOP_K_DEFAULT,
        zone_filter: Optional[str] = None,
        collection: str = "summaries"
    ) -> list[dict]:
        """
        Recupera los k documentos más relevantes para la query.

        Args:
            query:        texto de búsqueda en lenguaje natural
            k:            número de resultados a recuperar
            zone_filter:  si se especifica, filtra por zona
            collection:   "summaries" o "issues"

        Returns:
            Lista de dicts con document, metadata y distance
        """
        col = self.col_summaries if collection == "summaries" else self.col_issues

        if col.count() == 0:
            logger.warning("El vector store está vacío. Indexa inspecciones primero.")
            return []

        query_embedding = self.embedder.encode(query).tolist()

        where = {"zone_id": zone_filter} if zone_filter else None

        results = col.query(
            query_embeddings=[query_embedding],
            n_results=min(k, col.count()),
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            retrieved.append({
                "document": doc,
                "metadata": meta,
                "similarity": round(1 - dist, 4)  # cosine distance → similarity
            })

        logger.info(
            f"Recuperados {len(retrieved)} documentos para query: '{query[:50]}...'"
        )
        return retrieved

    # ── Síntesis con Gemini ──────────────────────────────────────────

    def query(
        self,
        natural_query: str,
        k: int = TOP_K_DEFAULT,
        zone_filter: Optional[str] = None,
        use_issues: bool = False
    ) -> str:
        """
        Responde una query en lenguaje natural usando RAG.
        Recupera documentos relevantes y usa Gemini para sintetizar la respuesta.

        Args:
            natural_query: pregunta del gestor en lenguaje natural
            k:             número de documentos a recuperar
            zone_filter:   filtrar por zona específica
            use_issues:    si True, busca en la colección de issues (más granular)

        Returns:
            Respuesta sintetizada por el LLM con referencias a inspecciones
        """
        collection = "issues" if use_issues else "summaries"
        retrieved  = self.retrieve(natural_query, k=k, zone_filter=zone_filter,
                                   collection=collection)

        if not retrieved:
            return "No se encontraron inspecciones relevantes en el historial."

        # Construir contexto aumentado
        context_parts = []
        for i, r in enumerate(retrieved, 1):
            meta = r["metadata"]
            context_parts.append(
                f"[{i}] inspection_id={meta.get('inspection_id')} | "
                f"zona={meta.get('zone_id')} | "
                f"fecha={meta.get('date')} ({meta.get('weekday')}) {meta.get('hour')}h | "
                f"status={meta.get('overall_status')} | "
                f"fill_rate={meta.get('shelf_fill_rate')}% | "
                f"similitud={r['similarity']}\n"
                f"Resumen: {r['document']}"
            )

        context_str = "\n\n".join(context_parts)

        template = load_prompt_template("rag_query.txt")
        prompt   = (
            template
            .replace("{query}", natural_query)
            .replace("{retrieved_context}", context_str)
        )

        response = _call_gemini(prompt)
        return response

    # ── Utilidades ───────────────────────────────────────────────────

    def index_batch(self, inspections: list[dict]) -> list[str]:
        """Indexa una lista de inspecciones."""
        ids = []
        for insp in inspections:
            try:
                iid = self.index_inspection(insp)
                ids.append(iid)
            except Exception as e:
                logger.error(f"Error indexando {insp.get('inspection_id')}: {e}")
        return ids

    def index_from_file(self, json_path: str | Path) -> str:
        """Indexa una inspección desde un archivo JSON."""
        with open(json_path, "r", encoding="utf-8") as f:
            inspection = json.load(f)
        return self.index_inspection(inspection)

    def index_from_directory(self, dir_path: str | Path) -> list[str]:
        """Indexa todas las inspecciones JSON de un directorio."""
        dir_path = Path(dir_path)
        json_files = sorted(dir_path.glob("*.json"))
        logger.info(f"Indexando {len(json_files)} archivos desde {dir_path}")
        ids = []
        for jf in json_files:
            try:
                iid = self.index_from_file(jf)
                ids.append(iid)
            except Exception as e:
                logger.error(f"Error con {jf.name}: {e}")
        return ids

    def stats(self) -> dict:
        """Retorna estadísticas del vector store."""
        return {
            "summaries_count": self.col_summaries.count(),
            "issues_count":    self.col_issues.count(),
            "vectorstore_path": str(VECTORSTORE_DIR),
            "embedding_model": EMBEDDING_MODEL,
        }

    def reset(self) -> None:
        """Elimina todos los datos del vector store. Usar con precaución."""
        self.chroma.delete_collection(COLLECTION_SUMMARIES)
        self.chroma.delete_collection(COLLECTION_ISSUES)
        self.col_summaries = self.chroma.get_or_create_collection(
            name=COLLECTION_SUMMARIES,
            metadata={"hnsw:space": "cosine"}
        )
        self.col_issues = self.chroma.get_or_create_collection(
            name=COLLECTION_ISSUES,
            metadata={"hnsw:space": "cosine"}
        )
        logger.warning("Vector store reseteado completamente.")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="RAG Memory — Indexación y recuperación de inspecciones"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # index — indexar una inspección
    p_index = subparsers.add_parser("index", help="Indexar una inspección desde JSON")
    p_index.add_argument("json_path", help="Ruta al archivo JSON de inspección")

    # index-dir — indexar directorio
    p_idir = subparsers.add_parser("index-dir", help="Indexar todas las inspecciones de un directorio")
    p_idir.add_argument("dir_path", help="Ruta al directorio con JSONs de inspección")

    # query — consulta en lenguaje natural
    p_query = subparsers.add_parser("query", help="Consultar el historial en lenguaje natural")
    p_query.add_argument("question", help="Pregunta en lenguaje natural")
    p_query.add_argument("--zone", help="Filtrar por zona (opcional)")
    p_query.add_argument("--k", type=int, default=TOP_K_DEFAULT,
                         help=f"Número de resultados a recuperar (default: {TOP_K_DEFAULT})")
    p_query.add_argument("--issues", action="store_true",
                         help="Buscar en colección de issues (más granular)")

    # stats — estadísticas
    subparsers.add_parser("stats", help="Ver estadísticas del vector store")

    # reset — limpiar vector store
    subparsers.add_parser("reset", help="Limpiar el vector store (IRREVERSIBLE)")

    args = parser.parse_args()
    rag  = RAGMemory()

    if args.command == "index":
        iid = rag.index_from_file(args.json_path)
        print(f"✅ Indexado: {iid}")

    elif args.command == "index-dir":
        ids = rag.index_from_directory(args.dir_path)
        print(f"✅ Indexadas {len(ids)} inspecciones")

    elif args.command == "query":
        print(f"\n🔍 Consultando: '{args.question}'\n")
        response = rag.query(
            args.question,
            k=args.k,
            zone_filter=args.zone,
            use_issues=args.issues
        )
        print(response)

    elif args.command == "stats":
        s = rag.stats()
        print(f"\n📊 Vector Store Stats:")
        print(f"   Summaries indexados : {s['summaries_count']}")
        print(f"   Issues indexados    : {s['issues_count']}")
        print(f"   Modelo de embeddings: {s['embedding_model']}")
        print(f"   Ruta                : {s['vectorstore_path']}")

    elif args.command == "reset":
        confirm = input("⚠️  ¿Seguro que quieres borrar todo el vector store? (s/N): ")
        if confirm.lower() == "s":
            rag.reset()
            print("✅ Vector store reseteado.")
        else:
            print("Operación cancelada.")


if __name__ == "__main__":
    main()
