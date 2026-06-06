"""
shelf_inspector.py — Componente 1: Retail Vision Intelligence System
Análisis visual de estantes con Google Gemini 1.5 Flash.

Características:
- API key cargada desde .env o Apikey.txt
- Prompts cargados desde prompts/*.txt
- Tres estrategias: Zero-shot (A), Chain-of-Thought (B), Few-shot (C)
- Cache local por hash MD5
- Rate limiting con backoff exponencial (15 req/min Gemini free tier)
- Fallback gracioso cuando se agota la quota diaria
- Output estructurado en JSON con schema obligatorio del enunciado
"""

import os
import json
import time
import hashlib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum
from typing import Optional

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
logger = logging.getLogger("shelf_inspector")

BASE_DIR    = Path(__file__).parent
PROMPTS_DIR = BASE_DIR.parent / "prompts"
CACHE_DIR   = BASE_DIR.parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "gemini-2.5-flash"

RATE_LIMIT_RPM = 15
BACKOFF_BASE   = 2
MAX_RETRIES    = 5


def load_api_key() -> str:
    """
    Carga la API key en este orden de prioridad:
    1. Variable de entorno GEMINI_API_KEY
    2. Archivo .env en la carpeta src/
    3. Archivo Apikey.txt en la raíz del proyecto
    """
    # 1. Variable de entorno (incluye lo que carga .env)
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if api_key:
        logger.info("API key cargada desde variable de entorno / .env")
        return api_key

    # 2. Archivo Apikey.txt en la raíz del proyecto
    fallback_path = BASE_DIR.parent / "Apikey.txt"
    if fallback_path.exists():
        try:
            key = (
                fallback_path
                .read_text(encoding="utf-8-sig")
                .strip()
                .replace('\r', '')
                .replace('\n', '')
            )
            if key:
                logger.info(f"API key cargada desde {fallback_path}")
                return key
        except Exception as e:
            raise EnvironmentError(
                f"No se pudo leer la API key desde {fallback_path}: {e}"
            ) from e

    raise EnvironmentError(
        "GEMINI_API_KEY no encontrada.\n"
        "Opciones:\n"
        "  1. Crea src/.env con: GEMINI_API_KEY=tu_clave\n"
        "  2. Crea Apikey.txt en la raíz del proyecto con solo la clave\n"
        "Obtén tu clave gratis en: https://aistudio.google.com"
    )


GEMINI_API_KEY = load_api_key()
client = genai.Client(api_key=GEMINI_API_KEY)


class PromptStrategy(str, Enum):
    ZERO_SHOT        = "A"
    CHAIN_OF_THOUGHT = "B"
    FEW_SHOT         = "C"


PROMPT_FILES = {
    PromptStrategy.ZERO_SHOT:        "inspect_A_zero_shot.txt",
    PromptStrategy.CHAIN_OF_THOUGHT: "inspect_B_chain_of_thought.txt",
    PromptStrategy.FEW_SHOT:         "inspect_C_few_shot.txt",
}

FEW_SHOT_EXAMPLES_FILE = "inspect_C_few_shot_examples.txt"


# ─────────────────────────────────────────────
# Carga de prompts desde disco
# ─────────────────────────────────────────────

def load_prompt(strategy: PromptStrategy, image_path: str, zone_id: str) -> str:
    prompt_file = PROMPTS_DIR / PROMPT_FILES[strategy]
    if not prompt_file.exists():
        raise FileNotFoundError(
            f"Archivo de prompt no encontrado: {prompt_file}\n"
            f"Asegúrate de que la carpeta 'prompts/' existe en la raíz del proyecto."
        )

    template = prompt_file.read_text(encoding="utf-8")

    if strategy == PromptStrategy.FEW_SHOT:
        examples_file = PROMPTS_DIR / FEW_SHOT_EXAMPLES_FILE
        if not examples_file.exists():
            raise FileNotFoundError(
                f"Archivo de ejemplos few-shot no encontrado: {examples_file}"
            )
        few_shot_examples = examples_file.read_text(encoding="utf-8")
        template = template.replace("{few_shot_examples}", few_shot_examples)

    template = template.replace("{image_path}", image_path)
    template = template.replace("{zone_id}", zone_id)
    return template


# ─────────────────────────────────────────────
# Cache local por hash MD5
# ─────────────────────────────────────────────

def compute_md5(image_path: Path) -> str:
    h = hashlib.md5()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_key(image_path: Path, strategy: PromptStrategy) -> str:
    return f"{compute_md5(image_path)}_{strategy.value}"


def load_from_cache(image_path: Path, strategy: PromptStrategy) -> Optional[dict]:
    key        = cache_key(image_path, strategy)
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        logger.info(f"[CACHE HIT] {image_path.name} — estrategia {strategy.value}")
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_to_cache(image_path: Path, strategy: PromptStrategy, result: dict) -> None:
    key        = cache_key(image_path, strategy)
    cache_file = CACHE_DIR / f"{key}.json"
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"[CACHE SAVE] {image_path.name} — estrategia {strategy.value}")


# ─────────────────────────────────────────────
# Rate limiting y llamada a la API
# ─────────────────────────────────────────────

_request_timestamps: list = []
_quota_exhausted = False


def _check_and_wait_rate_limit():
    global _request_timestamps
    now    = time.time()
    window = 60.0
    _request_timestamps = [t for t in _request_timestamps if now - t < window]

    if len(_request_timestamps) >= RATE_LIMIT_RPM:
        oldest = _request_timestamps[0]
        wait   = window - (now - oldest) + 0.5
        logger.warning(f"Rate limit: esperando {wait:.1f}s antes de la siguiente petición")
        time.sleep(wait)

    _request_timestamps.append(time.time())


def _call_gemini_with_backoff(prompt: str, image_path: Path) -> str:
    """Llama a Gemini 1.5 Flash con backoff exponencial en caso de error 429."""
    global _quota_exhausted

    if _quota_exhausted:
        raise RuntimeError(
            "⚠️  Quota diaria agotada. Solo se pueden procesar imágenes en caché."
        )

    suffix    = image_path.suffix.lower()
    mime_map  = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                 ".png": "image/png",  ".webp": "image/webp"}
    mime_type   = mime_map.get(suffix, "image/jpeg")
    image_bytes = image_path.read_bytes()

    for attempt in range(MAX_RETRIES):
        try:
            _check_and_wait_rate_limit()
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    types.Part.from_text(text=prompt),
                ],
                config=types.GenerateContentConfig(temperature=0),
            )
            return response.text

        except Exception as e:
            error_str = str(e).lower()

            if "quota" in error_str or "429" in error_str:
                if "daily" in error_str or "per day" in error_str:
                    _quota_exhausted = True
                    logger.error(
                        "⚠️  Quota diaria de Gemini agotada. "
                        "El sistema seguirá funcionando solo con caché."
                    )
                    raise RuntimeError("Quota diaria agotada.")

                wait = BACKOFF_BASE ** attempt
                logger.warning(
                    f"Error 429 (rate limit). "
                    f"Reintento {attempt+1}/{MAX_RETRIES} en {wait}s"
                )
                time.sleep(wait)

            else:
                logger.error(f"Error en llamada a Gemini: {e}")
                raise

    raise RuntimeError(
        f"Gemini no respondió correctamente tras {MAX_RETRIES} intentos."
    )


# ─────────────────────────────────────────────
# Extracción y validación del JSON
# ─────────────────────────────────────────────

def _extract_json(raw_text: str) -> dict:
    """
    Extrae el JSON de la respuesta del modelo.
    La estrategia B devuelve texto + JSON, por eso buscamos el bloque JSON.
    """
    # Intento 1: respuesta completa es JSON válido
    try:
        return json.loads(raw_text.strip())
    except json.JSONDecodeError:
        pass

    # Intento 2: bloque ```json ... ```
    match = re.search(r"```json\s*(.*?)\s*```", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Intento 3: primer { ... } de nivel raíz
    brace_start = raw_text.find("{")
    brace_end   = raw_text.rfind("}")
    if brace_start != -1 and brace_end != -1:
        try:
            return json.loads(raw_text[brace_start:brace_end+1])
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"No se pudo extraer JSON válido de la respuesta del modelo.\n"
        f"Primeros 500 chars:\n{raw_text[:500]}"
    )


def _validate_schema(data: dict, image_path: str, zone_id: str) -> dict:
    """Valida y completa los campos obligatorios del schema."""
    now      = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    defaults = {
        "inspection_id":     f"INS_{now}_001",
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "image_path":        str(image_path),
        "zone_id":           zone_id,
        "overall_status":    "warning",
        "issues":            [],
        "shelf_fill_rate":   0.0,
        "products_detected": [],
        "model_reasoning":   "No se proporcionó razonamiento."
    }

    for key, default in defaults.items():
        if key not in data:
            logger.warning(f"Campo '{key}' ausente. Usando valor por defecto.")
            data[key] = default

    if data["overall_status"] not in ("ok", "warning", "critical"):
        data["overall_status"] = "warning"

    valid_types      = {"empty_shelf", "wrong_product", "damaged",
                        "misaligned", "label_missing", "other"}
    valid_severities = {"low", "medium", "high"}

    for i, issue in enumerate(data.get("issues", [])):
        if "issue_id" not in issue:
            issue["issue_id"] = f"ISS_{i+1:03d}"
        if issue.get("type") not in valid_types:
            issue["type"] = "other"
        if issue.get("severity") not in valid_severities:
            issue["severity"] = "medium"
        issue.setdefault("confidence", 0.5)
        issue.setdefault("affected_area_pct", 0.0)
        issue.setdefault("location", "desconocida")
        issue.setdefault("description", "Sin descripción.")

    return data


# ─────────────────────────────────────────────
# Inspector principal
# ─────────────────────────────────────────────

class ShelfInspector:
    """
    Analiza imágenes de estantes usando Google Gemini 1.5 Flash.
    Los prompts se cargan desde la carpeta prompts/ en tiempo de ejecución.
    """

    def inspect(
        self,
        image_path: str | Path,
        zone_id: str = "Z_UNKNOWN",
        strategy: PromptStrategy = PromptStrategy.CHAIN_OF_THOUGHT,
        force_refresh: bool = False
    ) -> dict:
        """
        Analiza una imagen de estante y retorna el JSON de inspección.

        Args:
            image_path:    Ruta a la imagen.
            zone_id:       ID de la zona (e.g. "Z_S1").
            strategy:      Estrategia de prompting (A, B o C).
            force_refresh: Si True, ignora el caché.

        Returns:
            dict con el resultado según el schema obligatorio del enunciado.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Imagen no encontrada: {image_path}")

        if not force_refresh:
            cached = load_from_cache(image_path, strategy)
            if cached is not None:
                return cached

        logger.info(
            f"Inspeccionando {image_path.name} | "
            f"zona={zone_id} | estrategia={strategy.value}"
        )

        prompt       = load_prompt(strategy, str(image_path), zone_id)
        raw_response = _call_gemini_with_backoff(prompt, image_path)

        result = _extract_json(raw_response)
        result = _validate_schema(result, str(image_path), zone_id)

        save_to_cache(image_path, strategy, result)
        return result

    def inspect_batch(
        self,
        images_dir: str | Path,
        zone_id: str = "Z_UNKNOWN",
        strategy: PromptStrategy = PromptStrategy.CHAIN_OF_THOUGHT,
        extensions: tuple = (".jpg", ".jpeg", ".png", ".webp"),
        force_refresh: bool = False
    ) -> list:
        """Inspecciona todas las imágenes de un directorio."""
        images_dir  = Path(images_dir)
        image_files = [
            f for f in sorted(images_dir.iterdir())
            if f.suffix.lower() in extensions
        ]

        if not image_files:
            logger.warning(f"No se encontraron imágenes en {images_dir}")
            return []

        logger.info(f"Procesando {len(image_files)} imágenes en {images_dir}")
        results = []

        for img_path in image_files:
            try:
                result = self.inspect(
                    img_path, zone_id=zone_id,
                    strategy=strategy, force_refresh=force_refresh
                )
                results.append(result)

            except RuntimeError as e:
                results.append({
                    "inspection_id":  f"ERR_{img_path.stem}",
                    "image_path":     str(img_path),
                    "zone_id":        zone_id,
                    "error":          str(e),
                    "overall_status": "error"
                })
                if "quota" in str(e).lower():
                    logger.warning("Deteniendo procesamiento: quota agotada.")
                    break

            except Exception as e:
                logger.error(f"Error procesando {img_path.name}: {e}")
                results.append({
                    "inspection_id":  f"ERR_{img_path.stem}",
                    "image_path":     str(img_path),
                    "zone_id":        zone_id,
                    "error":          str(e),
                    "overall_status": "error"
                })

        return results

    def compare_strategies(
        self,
        image_path: str | Path,
        zone_id: str = "Z_UNKNOWN"
    ) -> dict:
        """Ejecuta las tres estrategias sobre la misma imagen para comparación."""
        results = {}
        for strategy in PromptStrategy:
            logger.info(f"Ejecutando estrategia {strategy.value}...")
            try:
                results[strategy.value] = self.inspect(
                    image_path, zone_id=zone_id, strategy=strategy
                )
            except Exception as e:
                results[strategy.value] = {"error": str(e)}
        return results


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Shelf Inspector — Análisis visual de estantes"
    )
    parser.add_argument("image", help="Ruta a la imagen del estante")
    parser.add_argument("--zone",     default="Z_S1",
                        help="ID de la zona (default: Z_S1)")
    parser.add_argument(
        "--strategy", choices=["A", "B", "C"], default="B",
        help="A=Zero-shot  B=Chain-of-Thought  C=Few-shot  (default: B)"
    )
    parser.add_argument("--compare",  action="store_true",
                        help="Comparar las tres estrategias sobre la misma imagen")
    parser.add_argument("--force",    action="store_true",
                        help="Ignorar caché y forzar nueva llamada a la API")
    parser.add_argument("--output",   help="Guardar resultado en archivo JSON")
    args = parser.parse_args()

    inspector = ShelfInspector()

    if args.compare:
        result = inspector.compare_strategies(args.image, zone_id=args.zone)
        print("\n=== COMPARACIÓN DE ESTRATEGIAS ===")
        for strat, res in result.items():
            status   = res.get("overall_status", "error")
            fill     = res.get("shelf_fill_rate", "N/A")
            n_issues = len(res.get("issues", []))
            print(
                f"  Estrategia {strat}: "
                f"status={status} | fill_rate={fill}% | issues={n_issues}"
            )
    else:
        strategy = PromptStrategy(args.strategy)
        result   = inspector.inspect(
            args.image, zone_id=args.zone,
            strategy=strategy, force_refresh=args.force
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output:
        out_path = Path(args.output)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nResultado guardado en: {out_path}")


if __name__ == "__main__":
    main()