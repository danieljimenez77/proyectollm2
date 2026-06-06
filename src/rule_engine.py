"""
rule_engine.py — Componente 2: Retail Vision Intelligence System
Convierte reglas en lenguaje natural a JSON ejecutable y las ejecuta
contra resultados de inspecciones.

Características:
- Conversión de lenguaje natural a schema JSON via Gemini
- Detección de ambigüedades con clarificación al gestor
- Persistencia de reglas en disco (data/rules/)
- Ejecución de reglas contra resultados de inspección
- Logs de ejecución detallados
- Prompts cargados desde prompts/
"""

import os
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
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
logger = logging.getLogger("rule_engine")

BASE_DIR    = Path(__file__).parent
PROMPTS_DIR = BASE_DIR.parent / "prompts"
RULES_DIR   = BASE_DIR.parent / "data" / "rules"
RULES_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "gemini-2.5-flash"

# Zonas disponibles por defecto (el gestor puede añadir más)
DEFAULT_ZONES = ["Z_S1", "Z_S2", "Z_S3", "Z_S4", "Z_S5"]


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
# Llamada a Gemini
# ─────────────────────────────────────────────

def _call_gemini(prompt: str) -> str:
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[types.Part.from_text(text=prompt)],
        config=types.GenerateContentConfig(temperature=0),
    )
    return response.text


def _extract_json(raw_text: str) -> dict:
    try:
        return json.loads(raw_text.strip())
    except json.JSONDecodeError:
        pass
    match = re.search(r"```json\s*(.*?)\s*```", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    brace_start = raw_text.find("{")
    brace_end   = raw_text.rfind("}")
    if brace_start != -1 and brace_end != -1:
        try:
            return json.loads(raw_text[brace_start:brace_end+1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"No se pudo extraer JSON de la respuesta:\n{raw_text[:500]}")


# ─────────────────────────────────────────────
# Persistencia de reglas
# ─────────────────────────────────────────────

def _generate_rule_id() -> str:
    existing = list(RULES_DIR.glob("RULE_*.json"))
    n = len(existing) + 1
    return f"RULE_{n:03d}"


def save_rule(rule: dict) -> Path:
    rule_file = RULES_DIR / f"{rule['rule_id']}.json"
    with open(rule_file, "w", encoding="utf-8") as f:
        json.dump(rule, f, ensure_ascii=False, indent=2)
    logger.info(f"Regla guardada: {rule_file}")
    return rule_file


def load_all_rules() -> list[dict]:
    rules = []
    for rule_file in sorted(RULES_DIR.glob("RULE_*.json")):
        with open(rule_file, "r", encoding="utf-8") as f:
            rules.append(json.load(f))
    return rules


def load_rule(rule_id: str) -> Optional[dict]:
    rule_file = RULES_DIR / f"{rule_id}.json"
    if not rule_file.exists():
        return None
    with open(rule_file, "r", encoding="utf-8") as f:
        return json.load(f)


def delete_rule(rule_id: str) -> bool:
    rule_file = RULES_DIR / f"{rule_id}.json"
    if rule_file.exists():
        rule_file.unlink()
        logger.info(f"Regla eliminada: {rule_id}")
        return True
    return False


# ─────────────────────────────────────────────
# Motor de reglas principal
# ─────────────────────────────────────────────

class RuleEngine:
    """
    Convierte reglas en lenguaje natural a configuraciones JSON ejecutables
    y las ejecuta contra resultados de inspecciones.

    Ejemplo de uso:
        engine = RuleEngine()
        rule, ambiguities = engine.parse_rule("Avísame cuando fill rate < 60%")
        if ambiguities:
            print(engine.format_ambiguity_response(rule))
        else:
            engine.save_rule(rule)
    """

    def __init__(self, available_zones: list[str] = None):
        self.available_zones = available_zones or DEFAULT_ZONES
        logger.info(f"RuleEngine iniciado | zonas: {self.available_zones}")

    def parse_rule(self, natural_language: str) -> tuple[dict, list[str]]:
        """
        Convierte una regla en lenguaje natural al schema JSON.

        Returns:
            (rule_dict, ambiguities)
            - rule_dict: la regla convertida
            - ambiguities: lista de ambigüedades detectadas (vacía si ninguna)
        """
        rule_id    = _generate_rule_id()
        created_at = datetime.now(timezone.utc).isoformat()

        template = load_prompt_template("rule_parse.txt")
        prompt   = (
            template
            .replace("{natural_language}", natural_language)
            .replace("{available_zones}", ", ".join(self.available_zones))
            .replace("{rule_id}", rule_id)
            .replace("{created_at}", created_at)
        )

        logger.info(f"Parseando regla: '{natural_language[:60]}...'")
        raw = _call_gemini(prompt)
        rule = _extract_json(raw)

        # Garantizar campos obligatorios
        rule.setdefault("rule_id", rule_id)
        rule.setdefault("created_at", created_at)
        rule.setdefault("natural_language", natural_language)
        rule.setdefault("validation", {
            "is_valid": True, "ambiguities": [], "assumptions": []
        })

        ambiguities = rule.get("validation", {}).get("ambiguities", [])
        logger.info(
            f"Regla parseada: {rule['rule_id']} | "
            f"ambigüedades: {len(ambiguities)}"
        )
        return rule, ambiguities

    def format_ambiguity_response(self, rule: dict) -> str:
        """
        Genera un mensaje en lenguaje natural explicando las ambigüedades
        al gestor y preguntando cómo resolverlas.
        """
        ambiguities = rule.get("validation", {}).get("ambiguities", [])
        assumptions = rule.get("validation", {}).get("assumptions", [])

        if not ambiguities:
            return "La regla es clara y no tiene ambigüedades."

        template = load_prompt_template("rule_ambiguity_response.txt")
        prompt   = (
            template
            .replace("{natural_language}", rule.get("natural_language", ""))
            .replace("{ambiguities}", "\n".join(f"- {a}" for a in ambiguities))
            .replace("{assumptions}", "\n".join(f"- {a}" for a in assumptions))
        )
        return _call_gemini(prompt)

    def add_rule(
        self,
        natural_language: str,
        auto_save: bool = False
    ) -> tuple[dict, list[str], str]:
        """
        Parsea una regla y, si no tiene ambigüedades o auto_save=True, la guarda.

        Returns:
            (rule, ambiguities, message)
        """
        rule, ambiguities = self.parse_rule(natural_language)

        if not rule.get("validation", {}).get("is_valid", True):
            msg = "❌ La regla no es válida o no se pudo interpretar."
            return rule, ambiguities, msg

        if ambiguities and not auto_save:
            clarification = self.format_ambiguity_response(rule)
            return rule, ambiguities, clarification

        save_rule(rule)
        msg = f"✅ Regla {rule['rule_id']} guardada correctamente."
        if ambiguities:
            msg += f"\n⚠️  Se guardó con {len(ambiguities)} ambigüedad(es) asumidas por defecto."
        return rule, ambiguities, msg

    def execute_rules(self, inspection: dict) -> list[dict]:
        """
        Ejecuta todas las reglas guardadas contra el resultado de una inspección.

        Args:
            inspection: dict con el resultado del shelf_inspector

        Returns:
            Lista de notificaciones para las reglas que se activaron.
        """
        rules       = load_all_rules()
        triggered   = []
        exec_log    = []

        zone_id    = inspection.get("zone_id", "")
        fill_rate  = inspection.get("shelf_fill_rate", 100.0) / 100.0
        issues     = inspection.get("issues", [])
        timestamp  = inspection.get("timestamp", "")

        # Hora actual para filtros de horario
        try:
            dt   = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            hour = dt.hour
        except Exception:
            hour = datetime.now().hour

        for rule in rules:
            rule_id    = rule.get("rule_id", "?")
            conditions = rule.get("conditions", {})
            action     = rule.get("action", {})
            fired      = False
            reasons    = []

            # ── Filtro de zona ──────────────────────────────────────
            zone_filter = conditions.get("zone_filter")
            if zone_filter and zone_id not in zone_filter:
                exec_log.append({
                    "rule_id": rule_id, "fired": False,
                    "reason": f"zona {zone_id} no está en {zone_filter}"
                })
                continue

            # ── Filtro de horario ───────────────────────────────────
            time_filter  = conditions.get("time_filter", {})
            hours_start  = time_filter.get("hours_start")
            hours_end    = time_filter.get("hours_end")
            if hours_start is not None and hours_end is not None:
                if not (hours_start <= hour < hours_end):
                    exec_log.append({
                        "rule_id": rule_id, "fired": False,
                        "reason": f"hora {hour}h fuera del rango {hours_start}-{hours_end}h"
                    })
                    continue

            # ── Filtro de fill rate ─────────────────────────────────
            fill_threshold = conditions.get("fill_rate_threshold")
            if fill_threshold is not None:
                if fill_rate < fill_threshold:
                    fired = True
                    reasons.append(
                        f"fill rate {fill_rate*100:.1f}% < umbral {fill_threshold*100:.1f}%"
                    )

            # ── Filtro de tipo de issue ─────────────────────────────
            issue_types = conditions.get("issue_types", [])
            sev_threshold = conditions.get("severity_threshold", "any")
            location_filter = conditions.get("location_filter", "any")

            sev_order = {"low": 0, "medium": 1, "high": 2, "any": -1}
            min_sev   = sev_order.get(sev_threshold, -1)

            for issue in issues:
                # Tipo de issue
                if issue_types and issue.get("type") not in issue_types:
                    continue

                # Severidad mínima
                issue_sev = sev_order.get(issue.get("severity", "low"), 0)
                if min_sev >= 0 and issue_sev < min_sev:
                    continue

                # Filtro de ubicación (bottom/middle/top)
                if location_filter != "any":
                    loc = issue.get("location", "").lower()
                    loc_keywords = {
                        "bottom": ["inferior", "bajo", "bottom"],
                        "middle": ["medio", "central", "middle"],
                        "top":    ["superior", "alto", "top"]
                    }
                    keywords = loc_keywords.get(location_filter, [])
                    if not any(k in loc for k in keywords):
                        continue

                fired = True
                reasons.append(
                    f"issue '{issue.get('type')}' en '{issue.get('location')}' "
                    f"(severidad: {issue.get('severity')})"
                )

            # ── Registrar resultado ─────────────────────────────────
            exec_log.append({
                "rule_id":    rule_id,
                "fired":      fired,
                "reason":     " | ".join(reasons) if reasons else "ninguna condición activada"
            })

            if fired:
                # Rellenar el template de notificación
                msg_template = action.get("notification_message", "Regla {rule_id} activada.")
                notification_msg = (
                    msg_template
                    .replace("{rule_id}",    rule_id)
                    .replace("{zone_id}",    zone_id)
                    .replace("{fill_rate}",  f"{fill_rate*100:.1f}%")
                    .replace("{issue_type}", issues[0].get("type", "") if issues else "")
                    .replace("{severity}",   issues[0].get("severity", "") if issues else "")
                )

                triggered.append({
                    "rule_id":      rule_id,
                    "alert_level":  action.get("alert_level", "info"),
                    "message":      notification_msg,
                    "reasons":      reasons,
                    "inspection_id": inspection.get("inspection_id", ""),
                    "timestamp":    timestamp
                })
                logger.info(
                    f"[RULE FIRED] {rule_id} | "
                    f"nivel={action.get('alert_level')} | {reasons}"
                )

        # Log completo de ejecución
        logger.info(
            f"Ejecución de reglas: {len(rules)} verificadas, "
            f"{len(triggered)} activadas"
        )
        for entry in exec_log:
            status = "✅ FIRED" if entry["fired"] else "⬜ skip"
            logger.debug(f"  {status} {entry['rule_id']}: {entry['reason']}")

        return triggered

    def test_rule(self, rule_id: str, inspection: dict) -> dict:
        """
        Prueba una regla específica contra una inspección sin necesidad
        de que esté guardada en disco.
        """
        rule = load_rule(rule_id)
        if not rule:
            return {"error": f"Regla {rule_id} no encontrada"}

        # Ejecutar solo esa regla
        original_rules_dir = RULES_DIR
        results = self.execute_rules(inspection)
        fired = [r for r in results if r["rule_id"] == rule_id]

        return {
            "rule_id":  rule_id,
            "fired":    len(fired) > 0,
            "result":   fired[0] if fired else None
        }

    def list_rules(self) -> list[dict]:
        """Retorna resumen de todas las reglas guardadas."""
        rules   = load_all_rules()
        summary = []
        for rule in rules:
            summary.append({
                "rule_id":        rule.get("rule_id"),
                "created_at":     rule.get("created_at"),
                "natural_language": rule.get("natural_language"),
                "alert_level":    rule.get("action", {}).get("alert_level"),
                "is_valid":       rule.get("validation", {}).get("is_valid"),
                "ambiguities":    len(rule.get("validation", {}).get("ambiguities", []))
            })
        return summary


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Rule Engine — Gestión de reglas de inspección"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # add rule
    p_add = subparsers.add_parser("add", help="Añadir una nueva regla")
    p_add.add_argument("rule", help="Regla en lenguaje natural (entre comillas)")
    p_add.add_argument("--auto-save", action="store_true",
                       help="Guardar aunque tenga ambigüedades")

    # list rules
    subparsers.add_parser("list", help="Listar todas las reglas")

    # delete rule
    p_del = subparsers.add_parser("delete", help="Eliminar una regla")
    p_del.add_argument("rule_id", help="ID de la regla (e.g. RULE_001)")

    # test rule
    p_test = subparsers.add_parser("test", help="Probar una regla contra una inspección")
    p_test.add_argument("rule_id", help="ID de la regla")
    p_test.add_argument("--inspection", required=True,
                        help="Ruta al JSON de inspección")

    # execute all rules
    p_exec = subparsers.add_parser("execute", help="Ejecutar todas las reglas contra una inspección")
    p_exec.add_argument("--inspection", required=True,
                        help="Ruta al JSON de inspección")

    args = parser.parse_args()
    engine = RuleEngine()

    if args.command == "add":
        rule, ambiguities, message = engine.add_rule(
            args.rule, auto_save=args.auto_save
        )
        print(f"\n{message}")
        if ambiguities:
            print("\n⚠️  Ambigüedades detectadas:")
            for a in ambiguities:
                print(f"   • {a}")
        print(f"\nJSON de la regla:")
        print(json.dumps(rule, ensure_ascii=False, indent=2))

    elif args.command == "list":
        rules = engine.list_rules()
        if not rules:
            print("No hay reglas guardadas.")
        else:
            print(f"\n{'ID':<12} {'Nivel':<10} {'Válida':<8} {'Regla'}")
            print("-" * 70)
            for r in rules:
                print(
                    f"{r['rule_id']:<12} "
                    f"{r['alert_level']:<10} "
                    f"{'✅' if r['is_valid'] else '❌':<8} "
                    f"{r['natural_language'][:45]}"
                )

    elif args.command == "delete":
        if delete_rule(args.rule_id):
            print(f"✅ Regla {args.rule_id} eliminada.")
        else:
            print(f"❌ Regla {args.rule_id} no encontrada.")

    elif args.command == "test":
        with open(args.inspection, "r", encoding="utf-8") as f:
            inspection = json.load(f)
        result = engine.test_rule(args.rule_id, inspection)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == "execute":
        with open(args.inspection, "r", encoding="utf-8") as f:
            inspection = json.load(f)
        triggered = engine.execute_rules(inspection)
        if not triggered:
            print("Ninguna regla se activó.")
        else:
            print(f"\n🔔 {len(triggered)} regla(s) activada(s):\n")
            for t in triggered:
                level_icon = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(
                    t["alert_level"], "🔔"
                )
                print(f"{level_icon} [{t['alert_level'].upper()}] {t['rule_id']}")
                print(f"   {t['message']}")
                print()


if __name__ == "__main__":
    main()
