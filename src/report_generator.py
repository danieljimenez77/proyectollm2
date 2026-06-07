"""
report_generator.py — Componente 4: Retail Vision Intelligence System
Genera informes de inspección en Markdown combinando resultados del
ShelfInspector, RuleEngine y RAGMemory.

Secciones obligatorias del informe:
1. Sumario ejecutivo (máx. 150 palabras)
2. Problemas por zona
3. Reglas activadas
4. Contexto histórico relevante (del RAG)
5. Recomendaciones (máx. 5, ordenadas por urgencia)

Características:
- Informe en Markdown guardado en data/inspections/
- Integra los tres componentes anteriores
- Prompts cargados desde prompts/
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types
from dotenv import load_dotenv

# Importar componentes del sistema
import sys
sys.path.insert(0, str(Path(__file__).parent))
from shelf_inspector import ShelfInspector, PromptStrategy
from rule_engine import RuleEngine
from rag_memory import RAGMemory

# ─────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("report_generator")

BASE_DIR         = Path(__file__).parent
PROMPTS_DIR      = BASE_DIR.parent / "prompts"
INSPECTIONS_DIR  = BASE_DIR.parent / "data" / "inspections"
INSPECTIONS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "gemini-2.5-flash"


def load_api_key() -> str:
    import os
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
    raise EnvironmentError("GEMINI_API_KEY no encontrada.")


client = genai.Client(api_key=load_api_key())


# ─────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────

def load_prompt_template(filename: str) -> str:
    path = PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt no encontrado: {path}")
    return path.read_text(encoding="utf-8")


def _call_gemini(prompt: str) -> str:
    import time
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[types.Part.from_text(text=prompt)],
                config=types.GenerateContentConfig(temperature=0),
            )
            return response.text.strip()
        except Exception as e:
            if "503" in str(e) or "unavailable" in str(e).lower():
                wait = 2 ** attempt
                logger.warning(f"Servidor no disponible. Reintento en {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Gemini no respondió tras 5 intentos.")


def _extract_json(raw: str) -> dict:
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass
    match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    brace_start = raw.find("{")
    brace_end   = raw.rfind("}")
    if brace_start != -1 and brace_end != -1:
        try:
            return json.loads(raw[brace_start:brace_end+1])
        except json.JSONDecodeError:
            pass
    return {}


# ─────────────────────────────────────────────
# Generador de informes
# ─────────────────────────────────────────────

class ReportGenerator:
    """
    Genera informes de inspección en Markdown combinando ShelfInspector,
    RuleEngine y RAGMemory.

    Ejemplo de uso:
        generator = ReportGenerator()
        report_path = generator.generate_session_report(
            image_paths=["foto1.jpg", "foto2.jpg"],
            zone_ids=["Z_S1", "Z_S2"]
        )
        print(f"Informe guardado en: {report_path}")
    """

    def __init__(self):
        self.inspector = ShelfInspector()
        self.engine    = RuleEngine()
        self.rag       = RAGMemory()
        logger.info("ReportGenerator iniciado")

    # ── Sección 1: Sumario ejecutivo ─────────────────────────────────

    def _generate_executive_summary(
        self,
        inspections: list[dict],
        triggered_rules: list[dict]
    ) -> str:
        n_zones    = len(set(i.get("zone_id") for i in inspections))
        n_critical = sum(1 for i in inspections if i.get("overall_status") == "critical")
        n_warning  = sum(1 for i in inspections if i.get("overall_status") == "warning")
        fill_rates = [i.get("shelf_fill_rate", 0) for i in inspections]
        avg_fill   = sum(fill_rates) / len(fill_rates) if fill_rates else 0
        n_issues   = sum(len(i.get("issues", [])) for i in inspections)

        session_data = {
            "zonas_inspeccionadas": n_zones,
            "total_inspecciones":   len(inspections),
            "issues_criticos":      n_critical,
            "warnings":             n_warning,
            "fill_rate_promedio":   f"{avg_fill:.1f}%",
            "total_issues":         n_issues,
            "reglas_activadas":     len(triggered_rules),
            "inspecciones":         [
                {
                    "zone_id":        i.get("zone_id"),
                    "overall_status": i.get("overall_status"),
                    "shelf_fill_rate": i.get("shelf_fill_rate"),
                    "n_issues":       len(i.get("issues", []))
                }
                for i in inspections
            ]
        }

        template = load_prompt_template("report_executive_summary.txt")
        prompt   = template.replace(
            "{session_data}", json.dumps(session_data, ensure_ascii=False, indent=2)
        )
        return _call_gemini(prompt)

    # ── Sección 2: Problemas por zona ────────────────────────────────

    def _generate_zone_section(
        self,
        inspection: dict,
        historical_context: str
    ) -> str:
        zone_id    = inspection.get("zone_id", "Desconocida")
        status     = inspection.get("overall_status", "unknown")
        fill_rate  = inspection.get("shelf_fill_rate", 0)
        issues     = inspection.get("issues", [])

        status_icon = {"ok": "✅", "warning": "⚠️", "critical": "🚨"}.get(status, "❓")

        lines = [
            f"#### Zona {zone_id} {status_icon}",
            f"",
            f"- **Estado:** `{status}`",
            f"- **Fill Rate:** {fill_rate:.1f}%",
            f"- **Issues detectados:** {len(issues)}",
            f""
        ]

        if issues:
            lines.append("**Problemas:**")
            lines.append("")
            for issue in issues:
                sev_icon = {"low": "🟡", "medium": "🟠", "high": "🔴"}.get(
                    issue.get("severity"), "⚪"
                )
                lines.append(
                    f"- {sev_icon} `{issue.get('type')}` — {issue.get('location')}: "
                    f"{issue.get('description')} "
                    f"*(confianza: {issue.get('confidence', 0)*100:.0f}%, "
                    f"área: {issue.get('affected_area_pct', 0):.0f}%)*"
                )
            lines.append("")

        if historical_context:
            lines.append("**Contexto histórico:**")
            lines.append("")
            lines.append(f"> {historical_context}")
            lines.append("")

        return "\n".join(lines)

    # ── Sección 3: Reglas activadas ──────────────────────────────────

    def _generate_rules_section(self, triggered_rules: list[dict]) -> str:
        if not triggered_rules:
            return "_Ninguna regla se activó en esta sesión._\n"

        lines = []
        for t in triggered_rules:
            level_icon = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(
                t.get("alert_level"), "🔔"
            )
            lines.append(
                f"- {level_icon} **{t.get('rule_id')}** "
                f"[`{t.get('alert_level')}`]: {t.get('message')}"
            )
            if t.get("reasons"):
                for r in t["reasons"]:
                    lines.append(f"  - _{r}_")
        return "\n".join(lines) + "\n"

    # ── Sección 5: Recomendaciones ───────────────────────────────────

    def _generate_recommendations(
        self,
        inspections: list[dict],
        triggered_rules: list[dict],
        historical_context: str
    ) -> str:
        template = load_prompt_template("report_recommendations.txt")
        prompt   = (
            template
            .replace(
                "{inspection_data}",
                json.dumps(inspections, ensure_ascii=False, indent=2)
            )
            .replace(
                "{triggered_rules}",
                json.dumps(triggered_rules, ensure_ascii=False, indent=2)
            )
            .replace("{historical_context}", historical_context)
        )

        raw  = _call_gemini(prompt)
        data = _extract_json(raw)

        recommendations = data.get("recommendations", [])
        if not recommendations:
            return "_No se generaron recomendaciones._\n"

        urgency_icon = {
            "immediate": "🚨 Inmediato",
            "today":     "⚠️  Hoy",
            "this_week": "📅 Esta semana"
        }

        lines = []
        for i, rec in enumerate(recommendations[:5], 1):
            urgency = urgency_icon.get(rec.get("urgency", "today"), rec.get("urgency", ""))
            lines.append(f"**{i}. [{urgency}]** {rec.get('action')}")
            lines.append(f"   > _{rec.get('reason')}_")
            lines.append("")

        return "\n".join(lines)

    # ── Informe completo ─────────────────────────────────────────────

    def generate_report(
        self,
        inspections: list[dict],
        session_id: Optional[str] = None
    ) -> str:
        """
        Genera un informe Markdown completo para una lista de inspecciones.

        Args:
            inspections: lista de dicts de inspección del shelf_inspector
            session_id:  identificador de sesión (se genera automáticamente si no se da)

        Returns:
            Ruta al archivo Markdown generado
        """
        if not session_id:
            session_id = datetime.now(timezone.utc).strftime("SESSION_%Y%m%d_%H%M%S")

        logger.info(
            f"Generando informe {session_id} | "
            f"{len(inspections)} inspecciones"
        )

        # Ejecutar reglas contra todas las inspecciones
        all_triggered = []
        for insp in inspections:
            triggered = self.engine.execute_rules(insp)
            all_triggered.extend(triggered)

        # Indexar inspecciones en RAG y recuperar contexto histórico
        historical_contexts = {}
        for insp in inspections:
            try:
                self.rag.index_inspection(insp)
            except Exception as e:
                logger.warning(f"No se pudo indexar {insp.get('inspection_id')}: {e}")

            zone_id = insp.get("zone_id", "")
            try:
                ctx = self.rag.retrieve(
                    f"problemas históricos en zona {zone_id}",
                    k=3,
                    zone_filter=zone_id
                )
                if ctx:
                    historical_contexts[zone_id] = "\n".join(
                        f"- [{r['metadata'].get('inspection_id')}] "
                        f"{r['metadata'].get('date')} — {r['document'][:150]}..."
                        for r in ctx
                    )
                else:
                    historical_contexts[zone_id] = "_Sin historial previo para esta zona._"
            except Exception:
                historical_contexts[zone_id] = "_Sin historial previo para esta zona._"

        # Contexto histórico global para recomendaciones
        global_context = "\n".join(historical_contexts.values())

        # ── Construir el Markdown ────────────────────────────────────

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # Sección 1: Sumario ejecutivo
        logger.info("Generando sumario ejecutivo...")
        executive_summary = self._generate_executive_summary(
            inspections, all_triggered
        )

        # Sección 2: Problemas por zona
        zone_sections = []
        for insp in inspections:
            zone_id = insp.get("zone_id", "")
            hist    = historical_contexts.get(zone_id, "")
            zone_sections.append(
                self._generate_zone_section(insp, hist)
            )

        # Sección 3: Reglas activadas
        rules_section = self._generate_rules_section(all_triggered)

        # Sección 4: Contexto histórico del RAG
        rag_section_lines = []
        for zone_id, ctx in historical_contexts.items():
            rag_section_lines.append(f"**Zona {zone_id}:**\n{ctx}\n")
        rag_section = "\n".join(rag_section_lines) if rag_section_lines else \
            "_Sin contexto histórico disponible._\n"

        # Sección 5: Recomendaciones
        logger.info("Generando recomendaciones...")
        recommendations = self._generate_recommendations(
            inspections, all_triggered, global_context
        )

        # ── Ensamblar informe ────────────────────────────────────────
        report_md = f"""# Informe de Inspección — {session_id}

**Fecha:** {now}
**Zonas inspeccionadas:** {len(set(i.get('zone_id') for i in inspections))}
**Total inspecciones:** {len(inspections)}

---

## 1. Sumario Ejecutivo

{executive_summary}

---

## 2. Problemas por Zona

{"".join(zone_sections)}

---

## 3. Reglas Activadas

{rules_section}

---

## 4. Contexto Histórico Relevante

{rag_section}

---

## 5. Recomendaciones

{recommendations}

---

*Informe generado automáticamente por Retail Vision Intelligence System*
*Modelo: {MODEL_NAME} | Session ID: {session_id}*
"""

        # Guardar en disco
        report_path = INSPECTIONS_DIR / f"{session_id}.md"
        report_path.write_text(report_md, encoding="utf-8")
        logger.info(f"Informe guardado: {report_path}")

        return str(report_path)

    def generate_session_report(
        self,
        image_paths: list[str],
        zone_ids: Optional[list[str]] = None,
        strategy: PromptStrategy = PromptStrategy.CHAIN_OF_THOUGHT,
        session_id: Optional[str] = None
    ) -> str:
        """
        Pipeline completo: inspecciona imágenes y genera el informe.

        Args:
            image_paths: lista de rutas a imágenes
            zone_ids:    lista de zone_ids (uno por imagen, o uno para todas)
            strategy:    estrategia de prompting para el inspector
            session_id:  identificador de sesión

        Returns:
            Ruta al archivo Markdown generado
        """
        if zone_ids is None:
            zone_ids = [f"Z_S{i+1}" for i in range(len(image_paths))]
        elif len(zone_ids) == 1:
            zone_ids = zone_ids * len(image_paths)

        inspections = []
        for img_path, zone_id in zip(image_paths, zone_ids):
            logger.info(f"Inspeccionando {img_path} | zona={zone_id}")
            try:
                insp = self.inspector.inspect(
                    img_path, zone_id=zone_id, strategy=strategy
                )
                inspections.append(insp)
            except Exception as e:
                logger.error(f"Error inspeccionando {img_path}: {e}")

        if not inspections:
            raise RuntimeError("No se pudo completar ninguna inspección.")

        return self.generate_report(inspections, session_id=session_id)

    def generate_from_json(
        self,
        json_paths: list[str],
        session_id: Optional[str] = None
    ) -> str:
        """
        Genera informe a partir de JSONs de inspección ya existentes
        (útil cuando ya se corrió el inspector previamente).
        """
        inspections = []
        for jp in json_paths:
            with open(jp, "r", encoding="utf-8") as f:
                inspections.append(json.load(f))
        return self.generate_report(inspections, session_id=session_id)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Report Generator — Generación de informes de inspección"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # report desde imágenes
    p_img = subparsers.add_parser("inspect", help="Inspeccionar imágenes y generar informe")
    p_img.add_argument("images", nargs="+", help="Rutas a las imágenes")
    p_img.add_argument("--zones", nargs="+", help="IDs de zona (uno por imagen)")
    p_img.add_argument(
        "--strategy", choices=["A", "B", "C"], default="B",
        help="Estrategia de prompting (default: B)"
    )
    p_img.add_argument("--session", help="ID de sesión personalizado")

    # report desde JSONs existentes
    p_json = subparsers.add_parser("from-json", help="Generar informe desde JSONs existentes")
    p_json.add_argument("jsons", nargs="+", help="Rutas a los JSONs de inspección")
    p_json.add_argument("--session", help="ID de sesión personalizado")

    args    = parser.parse_args()
    gen     = ReportGenerator()

    if args.command == "inspect":
        strategy = PromptStrategy(args.strategy)
        path     = gen.generate_session_report(
            image_paths=args.images,
            zone_ids=args.zones,
            strategy=strategy,
            session_id=args.session
        )
        print(f"\n✅ Informe generado: {path}")

    elif args.command == "from-json":
        path = gen.generate_from_json(
            json_paths=args.jsons,
            session_id=args.session
        )
        print(f"\n✅ Informe generado: {path}")


if __name__ == "__main__":
    main()
