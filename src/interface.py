"""
interface.py — Componente 5: Retail Vision Intelligence System
Interfaz conversacional CLI para el gestor de tienda.

Modos de operación:
- inspect: analizar imágenes de estantes
- rule: gestión de reglas (add, list, delete, test)
- history: consultas al historial RAG
- report: generación de informes
- exit: salir

Características:
- Estado de sesión persistente (reglas cargadas, inspecciones de la sesión)
- Errores amigables sin stack traces expuestos al usuario
- Historial de comandos en la sesión
- Ayuda contextual por modo
"""

import json
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path

# Importar componentes
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from shelf_inspector import ShelfInspector, PromptStrategy
from rule_engine import RuleEngine, load_all_rules
from rag_memory import RAGMemory
from report_generator import ReportGenerator

# ─────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.WARNING,  # Solo warnings en la interfaz, no spam de INFO
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("interface")

INSPECTIONS_DIR = BASE_DIR.parent / "data" / "inspections"
CACHE_DIR       = BASE_DIR.parent / "cache"


# ─────────────────────────────────────────────
# Helpers de display
# ─────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════════╗
║       Retail Vision Intelligence System  v1.0            ║
║       LIACD — TP2  |  Gestor de Tienda                   ║
╚══════════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
COMANDOS DISPONIBLES:
─────────────────────────────────────────────────────────
  INSPECCIÓN
    inspect <imagen> [--zone Z_S1] [--strategy A|B|C]
    inspect --dir <carpeta> [--zone Z_S1]

  REGLAS
    add rule "<regla en lenguaje natural>"
    add rule "<regla>" --save          (guardar sin preguntar)
    list rules
    delete rule <RULE_ID>
    test rule <RULE_ID> --last         (probar con última inspección)

  HISTORIAL
    history "<pregunta>"
    history "<pregunta>" --zone <Z_S1>
    history "<pregunta>" --issues      (búsqueda granular por issue)

  INFORMES
    report --last                      (informe de la última inspección)
    report --session <SESSION_ID>      (informe de una sesión)
    report --json <ruta.json>          (informe desde JSON existente)

  SISTEMA
    status                             (estado del sistema)
    help                               (mostrar esta ayuda)
    exit                               (salir)
─────────────────────────────────────────────────────────
"""


def _print_banner():
    print(BANNER)


def _print_help():
    print(HELP_TEXT)


def _ok(msg):
    print(f"✅ {msg}")


def _warn(msg):
    print(f"⚠️  {msg}")


def _err(msg):
    print(f"❌ {msg}")


def _info(msg):
    print(f"ℹ️  {msg}")


def _section(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


def _parse_args(parts: list[str]) -> dict:
    """Parser simple de argumentos tipo --key value o --flag."""
    args   = {"_positional": []}
    i      = 0
    while i < len(parts):
        if parts[i].startswith("--"):
            key = parts[i][2:]
            if i + 1 < len(parts) and not parts[i+1].startswith("--"):
                args[key] = parts[i+1]
                i += 2
            else:
                args[key] = True
                i += 1
        else:
            args["_positional"].append(parts[i])
            i += 1
    return args


# ─────────────────────────────────────────────
# Interfaz principal
# ─────────────────────────────────────────────

class RetailVisionInterface:
    """
    Interfaz conversacional CLI para el gestor de tienda.
    Mantiene estado de sesión entre comandos.
    """

    def __init__(self):
        print("Iniciando sistema...")
        try:
            self.inspector  = ShelfInspector()
            self.engine     = RuleEngine()
            self.rag        = RAGMemory()
            self.generator  = ReportGenerator()
        except Exception as e:
            _err(f"Error al iniciar el sistema: {e}")
            sys.exit(1)

        # Estado de sesión
        self.session_inspections: list[dict] = []
        self.session_id = datetime.now(timezone.utc).strftime("SESSION_%Y%m%d_%H%M%S")
        self.command_history: list[str] = []

        _print_banner()
        self._print_status()
        _print_help()

    def _print_status(self):
        rules = load_all_rules()
        _section("Estado del Sistema")
        print(f"  Sesión activa  : {self.session_id}")
        print(f"  Reglas cargadas: {len(rules)}")
        print(f"  Summaries RAG  : {self.rag.col_summaries.count()}")
        print(f"  Issues RAG     : {self.rag.col_issues.count()}")
        print(f"  Inspecciones   : {len(self.session_inspections)} en esta sesión")
        print()

    # ── Handlers de comandos ─────────────────────────────────────────

    def _handle_inspect(self, parts: list[str]):
        args = _parse_args(parts)

        zone     = args.get("zone", "Z_S1")
        strategy = PromptStrategy(args.get("strategy", "B"))

        # Modo directorio
        if "dir" in args:
            images_dir = Path(args["dir"])
            if not images_dir.exists():
                _err(f"Directorio no encontrado: {images_dir}")
                return
            image_files = [
                f for f in sorted(images_dir.iterdir())
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
            ]
            if not image_files:
                _err(f"No se encontraron imágenes en {images_dir}")
                return
            print(f"Procesando {len(image_files)} imágenes...")
            for img in image_files:
                self._inspect_single(str(img), zone, strategy)
            return

        # Modo imagen individual
        positional = args.get("_positional", [])
        if not positional:
            _err("Indica la ruta de la imagen. Ejemplo: inspect foto.jpg --zone Z_S1")
            return

        self._inspect_single(positional[0], zone, strategy)

    def _inspect_single(self, image_path: str, zone: str, strategy: PromptStrategy):
        img_path = Path(image_path)
        if not img_path.exists():
            _err(f"Imagen no encontrada: {image_path}")
            return

        print(f"Inspeccionando {img_path.name} | zona={zone} | estrategia={strategy.value}...")
        try:
            result = self.inspector.inspect(img_path, zone_id=zone, strategy=strategy)
            self.session_inspections.append(result)

            # Mostrar resultado resumido
            status_icon = {"ok": "✅", "warning": "⚠️", "critical": "🚨"}.get(
                result.get("overall_status"), "❓"
            )
            print(f"\n{status_icon} {result.get('zone_id')} — "
                  f"Status: {result.get('overall_status')} | "
                  f"Fill rate: {result.get('shelf_fill_rate')}%")

            issues = result.get("issues", [])
            if issues:
                print(f"Issues detectados ({len(issues)}):")
                for issue in issues:
                    sev_icon = {"low": "🟡", "medium": "🟠", "high": "🔴"}.get(
                        issue.get("severity"), "⚪"
                    )
                    print(f"  {sev_icon} {issue.get('type')} — {issue.get('location')}")
            else:
                print("Sin issues detectados.")

            # Ejecutar reglas automáticamente
            triggered = self.engine.execute_rules(result)
            if triggered:
                print(f"\n🔔 {len(triggered)} regla(s) activada(s):")
                for t in triggered:
                    level_icon = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(
                        t.get("alert_level"), "🔔"
                    )
                    print(f"  {level_icon} [{t.get('rule_id')}] {t.get('message')}")

            # Indexar en RAG
            try:
                self.rag.index_inspection(result)
            except Exception:
                pass

            print()

        except RuntimeError as e:
            _err(str(e))
        except Exception as e:
            _err(f"Error durante la inspección: {e}")
            logger.exception(e)

    def _handle_rule(self, parts: list[str]):
        if not parts:
            _err("Subcomando requerido: add, list, delete, test")
            return

        subcmd = parts[0].lower()

        if subcmd == "add":
            # Reconstruir la regla desde las partes (puede tener espacios)
            raw = " ".join(parts[1:])
            auto_save = "--save" in raw
            raw = raw.replace("--save", "").strip().strip('"').strip("'")

            if not raw:
                _err('Escribe la regla entre comillas. Ejemplo: add rule "Fill rate < 60%"')
                return

            print(f"Procesando regla: '{raw}'...")
            try:
                rule, ambiguities, message = self.engine.add_rule(raw, auto_save=auto_save)
                print(f"\n{message}")
                if ambiguities and not auto_save:
                    print("\nUsa '--save' para guardar con los valores por defecto.")
            except Exception as e:
                _err(f"Error al procesar la regla: {e}")

        elif subcmd == "list":
            rules = self.engine.list_rules()
            if not rules:
                _info("No hay reglas guardadas.")
                return
            _section(f"Reglas guardadas ({len(rules)})")
            for r in rules:
                valid_icon = "✅" if r.get("is_valid") else "❌"
                ambi = r.get("ambiguities", 0)
                ambi_str = f" ({ambi} ambigüedad(es))" if ambi else ""
                print(f"  {valid_icon} {r['rule_id']} [{r['alert_level']}] "
                      f"{r['natural_language'][:50]}{ambi_str}")

        elif subcmd == "delete":
            if not parts[1:]:
                _err("Indica el ID de la regla. Ejemplo: delete rule RULE_001")
                return
            rule_id = parts[1].upper()
            from rule_engine import delete_rule
            if delete_rule(rule_id):
                _ok(f"Regla {rule_id} eliminada.")
            else:
                _err(f"Regla {rule_id} no encontrada.")

        elif subcmd == "test":
            args = _parse_args(parts[1:])
            positional = args.get("_positional", [])
            if not positional:
                _err("Indica el ID de la regla. Ejemplo: test rule RULE_001 --last")
                return
            rule_id = positional[0].upper()

            if "last" in args:
                if not self.session_inspections:
                    _err("No hay inspecciones en esta sesión. Ejecuta 'inspect' primero.")
                    return
                inspection = self.session_inspections[-1]
            else:
                _err("Usa --last para probar con la última inspección de la sesión.")
                return

            try:
                result = self.engine.test_rule(rule_id, inspection)
                fired = result.get("fired", False)
                icon  = "🔔 ACTIVADA" if fired else "⬜ No activada"
                print(f"\n{icon} — Regla {rule_id}")
                if fired and result.get("result"):
                    print(f"  Mensaje: {result['result'].get('message')}")
            except Exception as e:
                _err(f"Error al probar la regla: {e}")

        else:
            _err(f"Subcomando desconocido: '{subcmd}'. Usa add, list, delete o test.")

    def _handle_history(self, parts: list[str]):
        # Reconstruir query completa
        raw  = " ".join(parts)
        args = _parse_args(raw.split())

        positional = args.get("_positional", [])
        query      = " ".join(positional).strip('"').strip("'")

        if not query:
            _err('Escribe una pregunta. Ejemplo: history "¿Cuándo tuvo problemas Z_S1?"')
            return

        zone       = args.get("zone")
        use_issues = "issues" in args
        k          = int(args.get("k", 3))

        print(f"\n🔍 Consultando historial: '{query}'...\n")
        try:
            response = self.rag.query(
                query,
                k=k,
                zone_filter=zone,
                use_issues=use_issues
            )
            print(response)
            print()
        except Exception as e:
            _err(f"Error al consultar el historial: {e}")
            logger.exception(e)

    def _handle_report(self, parts: list[str]):
        args = _parse_args(parts)

        try:
            if "last" in args:
                if not self.session_inspections:
                    _err("No hay inspecciones en esta sesión. Ejecuta 'inspect' primero.")
                    return
                print("Generando informe de la sesión actual...")
                path = self.generator.generate_report(
                    self.session_inspections,
                    session_id=self.session_id
                )
                _ok(f"Informe guardado: {path}")

            elif "session" in args:
                session_id = args["session"]
                report_path = INSPECTIONS_DIR / f"{session_id}.md"
                if report_path.exists():
                    _ok(f"Informe existente: {report_path}")
                else:
                    _err(f"Sesión '{session_id}' no encontrada en {INSPECTIONS_DIR}")

            elif "json" in args:
                json_path = args["json"]
                print("Generando informe desde JSON...")
                path = self.generator.generate_from_json([json_path])
                _ok(f"Informe guardado: {path}")

            else:
                _err("Indica una opción: --last, --session <ID> o --json <ruta>")

        except Exception as e:
            _err(f"Error al generar el informe: {e}")
            logger.exception(e)

    def _handle_status(self):
        self._print_status()

    # ── Loop principal ───────────────────────────────────────────────

    def run(self):
        """Inicia el loop interactivo de la interfaz."""
        while True:
            try:
                raw = input("retail> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nSaliendo...")
                break

            if not raw:
                continue

            self.command_history.append(raw)
            parts = raw.split()
            cmd   = parts[0].lower()

            try:
                if cmd in ("exit", "quit", "salir"):
                    print("¡Hasta luego!")
                    break

                elif cmd == "help":
                    _print_help()

                elif cmd == "status":
                    self._handle_status()

                elif cmd == "inspect":
                    self._handle_inspect(parts[1:])

                elif cmd == "add" and len(parts) > 1 and parts[1].lower() == "rule":
                    self._handle_rule(["add"] + parts[2:])

                elif cmd == "list" and len(parts) > 1 and parts[1].lower() == "rules":
                    self._handle_rule(["list"])

                elif cmd == "delete" and len(parts) > 1 and parts[1].lower() == "rule":
                    self._handle_rule(["delete"] + parts[2:])

                elif cmd == "test" and len(parts) > 1 and parts[1].lower() == "rule":
                    self._handle_rule(["test"] + parts[2:])

                elif cmd == "history":
                    self._handle_history(parts[1:])

                elif cmd == "report":
                    self._handle_report(parts[1:])

                else:
                    _err(f"Comando desconocido: '{cmd}'. Escribe 'help' para ver los comandos.")

            except Exception as e:
                _err(f"Error inesperado: {e}")
                logger.exception(e)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    interface = RetailVisionInterface()
    interface.run()


if __name__ == "__main__":
    main()
