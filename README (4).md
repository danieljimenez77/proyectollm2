# Retail Vision Intelligence System — TP2

**LIACD — Interação com Modelos de Grande Escala**  
Año lectivo 2025/2026 | Cotación: 6,5 / 20

---

## Índice

1. [Descripción del proyecto](#descripción-del-proyecto)
2. [Estructura del repositorio](#estructura-del-repositorio)
3. [Dataset](#dataset)
4. [Instalación y configuración](#instalación-y-configuración)
5. [Componentes](#componentes)
6. [Uso](#uso)
7. [Evaluación](#evaluación)
8. [Decisiones técnicas y problemas encontrados](#decisiones-técnicas-y-problemas-encontrados)

---

## Descripción del proyecto

Sistema de inspección visual continua de estantes de supermercado con memoria. El sistema recibe imágenes de estantes, las analiza con modelos de lenguaje multimodales (Google Gemini), detecta problemas operacionales, permite definir reglas de detección en lenguaje natural, e indexa el historial de inspecciones en una base de datos vectorial para recuperación semántica.

### Componentes principales

| Componente | Archivo | Descripción |
|---|---|---|
| 1 | `src/shelf_inspector.py` | Análisis visual con LLM multimodal |
| 2 | `src/rule_engine.py` | Generación y ejecución de reglas en lenguaje natural |
| 3 | `src/rag_memory.py` | Indexación y recuperación de inspecciones históricas |
| 4 | `src/report_generator.py` | Generación de informes con contexto histórico |
| 5 | `src/interface.py` | Interfaz conversacional para el gestor de tienda |

---

## Estructura del repositorio

```
tp2 integracion/
├── README.md
├── requirements.txt
├── Apikey.txt                        ← API key de Gemini (NO subir al repositorio)
├── .gitignore
│
├── data/
│   ├── images/                       ← Dataset de imágenes
│   │   ├── normal/                   ← 150 imágenes de estantes normales
│   │   ├── empty/                    ← 100 imágenes de estantes vacíos
│   │   ├── planogram/                ← 100 imágenes de violaciones de planograma
│   │   ├── dirty/                    ← 80 imágenes de estantes sucios/desordenados
│   │   └── ambiguous/                ← 70 imágenes de casos ambiguos
│   ├── inspections/                  ← Inspection records generados
│   └── rules/                        ← Reglas persistidas
│
├── src/
│   ├── shelf_inspector.py
│   ├── rule_engine.py
│   ├── rag_memory.py
│   ├── report_generator.py
│   └── interface.py
│
├── prompts/
│   ├── inspect_A_zero_shot.txt
│   ├── inspect_B_chain_of_thought.txt
│   ├── inspect_C_few_shot.txt
│   └── inspect_C_few_shot_examples.txt
│
├── vectorstore/                      ← ChromaDB persistente (generado en runtime)
├── cache/                            ← Cache de resultados de API (generado en runtime)
└── evaluate.py
```

---

## Dataset

### Fuentes utilizadas

El dataset combina dos fuentes principales:

**1. SKU-110K** (Goldman et al., 2019)
- 11.762 imágenes de supermercado con anotaciones de bounding box
- Fuente principal para imágenes de estantes normales y con productos bien posicionados
- Repositorio: https://github.com/eg4000/SKU110K_CVPR19
- Licencia: uso académico

**2. Grocery Store Dataset** (Hult et al., 2019)
- Imágenes de productos y estantes en condiciones naturales
- Complementa el SKU-110K en categorías de violaciones de planograma y desorden
- HuggingFace: `johnanvik/grocery-store-dataset`
- Licencia: uso académico

### Distribución mínima requerida

| Tipo | Mínimo | Descripción |
|---|---|---|
| Estante normal | 150 | Producto bien posicionado, sin problemas visibles |
| Estante vacío | 100 | Una o más posiciones sin producto |
| Violación de planograma | 100 | Producto en posición incorrecta, etiqueta ausente, producto caído |
| Estante sucio/desordenado | 80 | Producto desalineado, embalajes dañados |
| Caso ambiguo | 70 | Situaciones donde la clasificación no es obvia |

### Justificación de la elección

Se eligió SKU-110K como fuente principal por ser el dataset de referencia en investigación de retail, con anotaciones de alta calidad y suficiente volumen. El Grocery Store Dataset complementa las categorías con menor representación en SKU-110K. Esta combinación permite cubrir todos los tipos de imagen requeridos con diversidad suficiente para generalización.

---

## Instalación y configuración

### Requisitos previos

- Python 3.11 o superior
- Cuenta en [Google AI Studio](https://aistudio.google.com) (gratuita, sin tarjeta de crédito)

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

Contenido de `requirements.txt`:

```
google-genai
pillow
python-dotenv
chromadb
sentence-transformers
```

### 2. Configurar la API key de Gemini

**Opción A — Archivo `.env`** (recomendada):
```
GEMINI_API_KEY=tu_clave_aqui
```

**Opción B — Archivo `Apikey.txt`** en la raíz del proyecto:
```
AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

> ⚠️ **Importante:** Nunca subas la API key al repositorio. Añade `Apikey.txt` y `.env` al `.gitignore`.

### 3. Obtener la API key

1. Ve a https://aistudio.google.com
2. Inicia sesión con una cuenta Google
3. Haz clic en **Get API Key** → **Create API key**
4. Copia la clave (empieza con `AIza`, 39 caracteres)

---

## Componentes

### Componente 1: Shelf Inspector (`src/shelf_inspector.py`)

Analiza imágenes de estantes usando **Google Gemini 2.5 Flash** y produce un JSON estructurado con los problemas detectados.

#### Modelo utilizado

Se utiliza `gemini-2.5-flash`. Durante el desarrollo se probaron otros modelos:

| Modelo | Resultado | Motivo del descarte |
|---|---|---|
| `gemini-2.0-flash` | Error 429 persistente | Límite de tasa muy restrictivo en free tier |
| `gemini-1.5-flash` | Error 404 | No disponible en la versión v1beta de la API |
| `gemini-2.5-flash` | ✅ Funciona | Disponible, multimodal, gratuito con AI Studio |

#### Tres estrategias de prompting

Los prompts están versionados en la carpeta `prompts/` y se cargan desde disco en tiempo de ejecución, nunca hardcodeados en el código.

**Estrategia A — Zero-shot directo** (`inspect_A_zero_shot.txt`):
Instrucción directa pidiendo el análisis y el JSON, sin ejemplos ni estructura de razonamiento guiada.

**Estrategia B — Chain-of-Thought visual** (`inspect_B_chain_of_thought.txt`):
El modelo razona explícitamente en 6 pasos: descripción general → análisis zona a zona → identificación de anomalías → estimación de fill rate → clasificación global → producción del JSON. Es la estrategia por defecto.

**Estrategia C — Few-shot con ejemplos textuales** (`inspect_C_few_shot.txt` + `inspect_C_few_shot_examples.txt`):
Incluye 3 ejemplos textuales de inspecciones correctas antes de la imagen nueva, evitando el consumo extra de quota que implicaría pasar múltiples imágenes.

#### Gestión de quota (API gratuita: 15 req/min, 1500 req/día)

- **Cache por MD5**: cada inspección se guarda en `cache/` identificada por el hash MD5 de la imagen y la estrategia. Si la imagen no cambió, no se consume quota.
- **Rate limiting**: control de ventana deslizante de 60 segundos para no superar 15 req/min.
- **Backoff exponencial**: ante error 429, espera 2, 4, 8, 16, 32 segundos entre reintentos.
- **Fallback gracioso**: si la quota diaria se agota, el sistema sigue funcionando para imágenes en caché y notifica claramente cuando no puede procesar nuevas imágenes.

#### Schema de salida (obligatorio)

```json
{
  "inspection_id": "INS_20250317_143022_001",
  "timestamp": "2025-03-17T14:30:22Z",
  "image_path": "path/to/image.jpg",
  "zone_id": "Z_S3",
  "overall_status": "ok|warning|critical",
  "issues": [
    {
      "issue_id": "ISS_001",
      "type": "empty_shelf|wrong_product|damaged|misaligned|label_missing|other",
      "location": "descripción de la ubicación",
      "severity": "low|medium|high",
      "description": "descripción en lenguaje natural",
      "confidence": 0.0,
      "affected_area_pct": 0.0
    }
  ],
  "shelf_fill_rate": 0.0,
  "products_detected": ["categorías de producto visibles"],
  "model_reasoning": "razonamiento explícito antes de la clasificación"
}
```

---

### Componente 2: Rule Engine (`src/rule_engine.py`)

Convierte reglas en lenguaje natural definidas por el gestor en configuraciones JSON ejecutables, y las ejecuta contra los resultados de inspecciones.

#### Prompts del Rule Engine

| Archivo | Propósito |
|---|---|
| `prompts/rule_parse.txt` | Convierte lenguaje natural a JSON en 4 pasos guiados |
| `prompts/rule_ambiguity_response.txt` | Genera respuesta amigable al gestor cuando hay ambigüedades |

#### Flujo de conversión

El prompt `rule_parse.txt` guía al modelo en 4 pasos:
1. **Comprensión** — qué condición describe la regla y qué acción debe tomarse
2. **Extracción** — zona(s), horario, tipo de issue, umbral de fill rate, severidad, ubicación
3. **Detección de ambigüedades** — lista todos los aspectos no claramente definidos
4. **Producción del JSON** — genera el schema estructurado con el campo `validation`

#### Schema de regla

```json
{
  "rule_id": "RULE_001",
  "created_at": "2026-06-06T15:20:03Z",
  "natural_language": "texto original del gestor",
  "description": "reformulación clara en lenguaje formal",
  "conditions": {
    "zone_filter": ["Z_S1", "Z_S3"],
    "time_filter": {"hours_start": 10, "hours_end": 13},
    "issue_types": ["empty_shelf", "misaligned"],
    "severity_threshold": "low|medium|high|any",
    "fill_rate_threshold": 0.6,
    "location_filter": "bottom|middle|top|any"
  },
  "action": {
    "alert_level": "info|warning|critical",
    "notification_message": "template con {zone_id}, {fill_rate}, {issue_type}, {severity}"
  },
  "validation": {
    "is_valid": true,
    "ambiguities": ["lista de aspectos no claros"],
    "assumptions": ["lista de suposiciones asumidas"]
  }
}
```

#### Ejecución de reglas

Tras cada inspección, el executor recorre todas las reglas guardadas verificando en orden:
- Filtro de zona — ¿la zona inspeccionada está en `zone_filter`?
- Filtro de horario — ¿la hora del timestamp está dentro de `time_filter`?
- Filtro de fill rate — ¿el `shelf_fill_rate` está por debajo del umbral?
- Filtro de issues — ¿algún issue coincide con `issue_types`, `severity_threshold` y `location_filter`?

Cada ejecución produce logs detallados indicando qué reglas se verificaron, cuáles se activaron y por qué.

#### Comportamiento con ambigüedades

Cuando una regla tiene ambigüedades, el sistema **no asume silenciosamente**: genera una respuesta en lenguaje natural explicando cada ambigüedad y preguntando cómo resolverla. La regla solo se guarda si el gestor usa `--auto-save` o confirma explícitamente.

Ejemplo real durante el desarrollo — regla: `"Avísame cuando el fill rate de cualquier zona caiga por debajo del 60%"`

Ambigüedades detectadas automáticamente:
- No se especifica horario → asume cualquier momento
- No se especifica nivel de urgencia → asume `warning`
- Fill rate es una métrica, no un `issue_type` → `issue_types` queda vacío
- No se especifica severidad → asume `any`
- No se especifica ubicación en el estante → asume `any`


---

## Uso

### Shelf Inspector

```bash
cd src/

# Inspección individual con Chain-of-Thought (por defecto)
python shelf_inspector.py "..\data\images\image.jpg" --zone Z_S1 --strategy B

# Comparar las tres estrategias sobre la misma imagen
python shelf_inspector.py "..\data\images\image.jpg" --zone Z_S1 --compare

# Forzar nueva llamada ignorando caché
python shelf_inspector.py "..\data\images\image.jpg" --force
```

### RAG Memory

```bash
cd src/

# Indexar una inspección
python rag_memory.py index "..\cacheesultado.json"

# Indexar todas las inspecciones de un directorio
python rag_memory.py index-dir "..\data\inspections"

# Consultas en lenguaje natural
python rag_memory.py query "¿Cuándo fue la última vez que Z_S1 tuvo problemas?"
python rag_memory.py query "¿Hay productos desalineados?" --zone Z_S1
python rag_memory.py query "¿Dónde hay issues de fill rate bajo?" --issues --k 5

# Ver estadísticas del vector store
python rag_memory.py stats

# Limpiar vector store (irreversible)
python rag_memory.py reset
```

### Rule Engine

```bash
cd src/

# Añadir una regla (pregunta si hay ambigüedades)
python rule_engine.py add "Avísame cuando la prateleira inferior esté más de 40% vacía"

# Añadir guardando aunque haya ambigüedades
python rule_engine.py add "Avisa cuando haya productos desalineados" --auto-save

# Listar todas las reglas guardadas
python rule_engine.py list

# Eliminar una regla
python rule_engine.py delete RULE_001

# Probar una regla contra una inspección
python rule_engine.py test RULE_001 --inspection "..\cache\resultado.json"

# Ejecutar todas las reglas contra una inspección
python rule_engine.py execute --inspection "..\cache\resultado.json"
```

---

## Evaluación

```bash
python evaluate.py --images-dir test_images/ --output evaluation_report.json
```

### Métricas implementadas

**Análisis visual:**
- Issue Detection Rate (recall)
- False Positive Rate
- Severity Accuracy
- JSON Parse Rate
- Hallucination Rate

**RAG:**
- Recall@3
- Faithfulness
- Answer Relevance (LLM-as-judge)

**Rule Engine:**
- Rule Parse Rate
- Rule Correctness
- Ambiguity Detection Rate

---

## Decisiones técnicas y problemas encontrados

### SDK de Google Gemini — Migración obligatoria

El paquete `google-generativeai` fue deprecado y ya no recibe actualizaciones. El nuevo SDK es `google-genai` y tiene una API completamente diferente:

```bash
# Incorrecto (deprecado)
pip install google-generativeai

# Correcto
pip install google-genai
```

Cambios principales en el código:

| Antes (`google-generativeai`) | Ahora (`google-genai`) |
|---|---|
| `import google.generativeai as genai` | `from google import genai` |
| `genai.configure(api_key=...)` | `genai.Client(api_key=...)` |
| `genai.GenerativeModel(MODEL)` | `client.models.generate_content(model=...)` |
| `[prompt, image]` como contenido | `[Part.from_bytes(...), Part.from_text(...)]` |
| `generation_config={"temperature": 0}` | `config=GenerateContentConfig(temperature=0)` |

### Selección del modelo

Durante el desarrollo se probaron tres modelos hasta encontrar el que funciona con la API key gratuita de AI Studio:

1. `gemini-2.0-flash` — error 429 persistente desde la primera llamada, incluso con backoff
2. `gemini-1.5-flash` — error 404, modelo no disponible en la versión `v1beta` de la API
3. `gemini-2.5-flash` — ✅ funciona correctamente con la key gratuita de AI Studio

Para verificar los modelos disponibles en tu cuenta:
```
https://generativelanguage.googleapis.com/v1beta/models?key=TU_API_KEY
```

### Carga de la API key en Windows

En Windows, los archivos `.txt` pueden incluir un BOM (Byte Order Mark) invisible al inicio que corrompe la clave. Para leerla correctamente:

```python
key = Path("Apikey.txt").read_text(encoding="utf-8-sig").strip().replace('\r', '').replace('\n', '')
```

El encoding `utf-8-sig` elimina automáticamente el BOM. El `.strip()` y el reemplazo de `\r\n` eliminan cualquier salto de línea de Windows.

### API key subida accidentalmente a GitHub

Durante el primer commit se incluyó `Apikey.txt` en el repositorio porque el `.gitignore` se creó después de `git add .`. Para eliminarlo del historial de Git sin borrar el archivo local:

```bash
git rm --cached Apikey.txt
git rm --cached -r cache/
git commit -m "fix: eliminar Apikey.txt y cache del repositorio"
```

### Prompts separados del código

Los prompts están en `prompts/*.txt` y se cargan en runtime. Esto permite modificar los prompts sin tocar el código Python y cumple el requisito explícito del enunciado de versionar los prompts separadamente.

Archivos de prompts actuales:

| Archivo | Componente | Propósito |
|---|---|---|
| `inspect_A_zero_shot.txt` | Shelf Inspector | Estrategia A — Zero-shot |
| `inspect_B_chain_of_thought.txt` | Shelf Inspector | Estrategia B — Chain-of-Thought |
| `inspect_C_few_shot.txt` | Shelf Inspector | Estrategia C — Few-shot (plantilla) |
| `inspect_C_few_shot_examples.txt` | Shelf Inspector | Ejemplos para estrategia C |
| `rule_parse.txt` | Rule Engine | Conversión de lenguaje natural a JSON |
| `rule_ambiguity_response.txt` | Rule Engine | Respuesta al gestor sobre ambigüedades |
| `rag_summary.txt` | RAG Memory | Generación de summaries ricos para indexación |
| `rag_query.txt` | RAG Memory | Síntesis de respuesta con contexto recuperado |

---

### Componente 3: RAG Memory (`src/rag_memory.py`)

Indexa el historial de inspecciones en una base de datos vectorial y permite recuperación semántica en lenguaje natural.

#### Stack tecnológico

| Componente | Tecnología | Justificación |
|---|---|---|
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` | Local, gratuito, soporta español y portugués |
| Vector store | ChromaDB (PersistentClient) | Local, persistente en disco, sin servidor |
| Similaridad | Coseno (hnsw:space=cosine) | Estándar para similitud semántica de texto |
| Síntesis | Gemini 2.5 Flash | Genera respuestas en lenguaje natural con contexto recuperado |

#### Estrategia de chunking híbrida

Se implementan dos colecciones en ChromaDB:

**`inspection_summaries`** — un chunk por inspección:
- El texto indexado es un summary generado por Gemini, semánticamente rico
- Los metadatos estructurados (zona, fecha, hora, día de semana, fill rate, status) permiten filtrado pre-retrieval
- Recuperación eficiente para queries generales

**`inspection_issues`** — un chunk por issue individual:
- Cada issue se indexa con texto detallado: tipo, ubicación, descripción, severidad, confianza
- Permite recuperación granular para queries específicas ("¿dónde hay productos misaligned?")
- Aumenta el índice pero mejora el Recall@3 para queries de issue específico

#### Prompts del RAG

| Archivo | Propósito |
|---|---|
| `prompts/rag_summary.txt` | Genera summaries ricos para indexación |
| `prompts/rag_query.txt` | Sintetiza respuesta con contexto recuperado |

El prompt `rag_summary.txt` incluye ejemplos explícitos de buen y mal summary para guiar al modelo:
- **Malo:** "hay problemas en el estante"
- **Bueno:** "zona Z_S3, martes 15h, fill rate 72%, detergente líquido fuera de posición en nivel medio, embalaje dañado en nivel inferior"

#### Queries obligatorias soportadas

```bash
python rag_memory.py query "¿Cuándo fue la última vez que la zona Z_S1 tuvo problemas?"
python rag_memory.py query "¿Qué zonas tuvieron más issues esta semana?"
python rag_memory.py query "¿Existe algún patrón en los problemas detectados los viernes?"
python rag_memory.py query "¿Qué reglas se activaron más este mes?"
python rag_memory.py query "¿Qué problemas se detectaron en Z_S1?" --zone Z_S1
python rag_memory.py query "¿Dónde hay productos desalineados?" --issues
```

#### Problemas encontrados

**Error 503 — Alta demanda del servidor Gemini:**
Durante las pruebas iniciales se obtuvo un `503 UNAVAILABLE` al llamar a Gemini para sintetizar la respuesta RAG. Es un error temporal por alta demanda. La solución es esperar unos minutos y reintentar. El código de producción debería incluir retry con backoff para este caso (igual que el 429 en `shelf_inspector.py`).

La recuperación semántica (embeddings + ChromaDB) funciona correctamente de forma independiente — el error solo afecta a la síntesis final por Gemini.

---

### Componente 5: Interface (`src/interface.py`)

Interfaz conversacional CLI que orquesta todos los componentes anteriores. Mantiene estado de sesión entre comandos y presenta errores de forma amigable sin exponer stack traces.

#### Modos de operación

```
retail> inspect <imagen> [--zone Z_S1] [--strategy A|B|C]
retail> inspect --dir <carpeta> [--zone Z_S1]

retail> add rule "<regla en lenguaje natural>"
retail> add rule "<regla>" --save
retail> list rules
retail> delete rule <RULE_ID>
retail> test rule <RULE_ID> --last

retail> history "<pregunta>"
retail> history "<pregunta>" --zone Z_S1
retail> history "<pregunta>" --issues

retail> report --last
retail> report --session <SESSION_ID>
retail> report --json <ruta.json>

retail> status
retail> help
retail> exit
```

#### Estado de sesión

Al iniciar, la interfaz muestra el estado del sistema:
- Sesión activa (ID generado automáticamente)
- Número de reglas cargadas desde `data/rules/`
- Summaries e issues indexados en el RAG
- Inspecciones realizadas en la sesión actual

Tras cada `inspect`, el sistema automáticamente ejecuta las reglas guardadas y notifica si alguna se activa, sin necesidad de comandos adicionales.

#### Comportamiento verificado en pruebas

```
retail> inspect ..\data\images\image.jpg --zone Z_S1
⚠️ Z_S1 — Status: warning | Fill rate: 92.0%
Issues detectados (4):
  🟠 misaligned — Quinto nivel del estante (packs de Monster)
  🟠 misaligned — Sexto nivel del estante (packs de Monster)
  🟡 empty_shelf — Extremo izquierdo del quinto nivel
  🟡 empty_shelf — Extremo izquierdo del sexto nivel
🔔 1 regla(s) activada(s):
  ⚠️ [RULE_001] Alerta: Productos desalineados detectados en Z_S1.
```

---

## Problema recurrente: Apikey.txt en commits de Git

Durante el desarrollo este problema ocurrió varias veces. La causa es siempre la misma: el archivo `Apikey.txt` se crea después de haber inicializado el repositorio y Git lo detecta como archivo nuevo en el siguiente `git add .`.

**Solución cuando el archivo está en el último commit pero aún no se hizo push:**
```bash
git rm --cached Apikey.txt
git commit --amend --no-edit
git push
```

**Solución cuando ya está en el historial y GitHub bloquea el push:**
```bash
git filter-branch --force --index-filter "git rm --cached --ignore-unmatch Apikey.txt" --prune-empty --tag-name-filter cat -- --all
git push origin main --force
```

**Importante:** Si la key llegó a subirse a un repositorio público, hay que generarla nueva inmediatamente en [aistudio.google.com](https://aistudio.google.com) ya que quedó comprometida.

**Prevención definitiva** — verificar que `.gitignore` contiene:
```
Apikey.txt
.env
cache/
vectorstore/
__pycache__/
*.pyc
data/inspections/
```

---

### Componente 5 (Streamlit): Interface Visual (`src/interface_streamlit.py`)

Interface visual completa construída com Streamlit, substituindo a CLI por uma experiência mais intuitiva com pestañas, botões e visualização de imagens.

#### Iniciar a interface

```bash
cd src/
streamlit run interface_streamlit.py
```

Abre automaticamente em `http://localhost:8501`.

#### Pestañas disponíveis

**📸 Inspeção** — Upload de imagem, seleção de zona e estratégia de prompting, visualização dos issues com código de cores (🔴 alto, 🟠 médio, 🟡 baixo), raciocínio do modelo e regras ativadas automaticamente.

**📋 Regras** — Criação de regras em linguagem natural com deteção de ambiguidades, listagem com botão de eliminação, e visualização do JSON gerado.

**🔍 Histórico** — Consultas RAG com botões de queries frequentes, filtro por zona, modo de pesquisa granular por issue, e documentos recuperados com score de similaridade.

**📄 Relatórios** — Geração de relatórios da sessão atual ou a partir de JSON existente, seletor de relatórios anteriores, visualização inline e botão de download.

---

## Avaliação (`evaluate.py`)

```bash
# Avaliação completa
python evaluate.py --images-dir test_images/ --output evaluation_report.json

# Apenas análise visual
python evaluate.py --images-dir test_images/ --output eval.json --skip-rag --skip-rules

# Com ground truth manual
python evaluate.py --images-dir test_images/ --ground-truth ground_truth.json --output eval.json
```

### Ground Truth

O harness aceita um ficheiro JSON de anotações manuais (`--ground-truth`). Se não for fornecido, o sistema infere o ground truth automaticamente a partir dos nomes dos ficheiros de imagem:

| Nome contém | Status inferido | Issue type |
|---|---|---|
| `empty`, `vaci` | critical | empty_shelf |
| `misalign`, `desalin` | warning | misaligned |
| `damage`, `dan` | warning | damaged |
| `wrong`, `plano` | warning | wrong_product |
| `ok`, `normal`, `good` | ok | — |

### LLM-as-Judge

O avaliador usa o próprio Gemini 2.5 Flash como juiz para métricas qualitativas:

| Prompt | Métrica avaliada |
|---|---|
| `evaluate_hallucination.txt` | Se as afirmações em `description` são verificáveis |
| `evaluate_rag_judge.txt` | Faithfulness e Answer Relevance das respostas RAG |
| `evaluate_report_judge.txt` | Completude, acionabilidade e precisão dos relatórios |

---

## Instalação Completa

### 1. Clonar o repositório

```bash
git clone https://github.com/danieljimenez77/proyectollm2
cd proyectollm2
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Configurar chave de API

Copiar `.env.example` para `.env` e preencher a chave:

```bash
cp .env.example .env
# editar .env e adicionar: GEMINI_API_KEY=a_sua_chave
```

Ou criar `Apikey.txt` na raiz do projeto com apenas a chave na primeira linha.

### 4. Iniciar a interface

```bash
cd src/
streamlit run interface_streamlit.py
```

---

## Estrutura Final do Repositório

```
tp2 integracion/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── evaluate.py
│
├── data/
│   ├── images/
│   │   ├── normal/
│   │   ├── empty/
│   │   ├── planogram/
│   │   ├── dirty/
│   │   └── ambiguous/
│   ├── inspections/
│   └── rules/
│
├── src/
│   ├── shelf_inspector.py
│   ├── rule_engine.py
│   ├── rag_memory.py
│   ├── report_generator.py
│   ├── interface.py
│   └── interface_streamlit.py
│
├── prompts/
│   ├── inspect_A_zero_shot.txt
│   ├── inspect_B_chain_of_thought.txt
│   ├── inspect_C_few_shot.txt
│   ├── inspect_C_few_shot_examples.txt
│   ├── rule_parse.txt
│   ├── rule_ambiguity_response.txt
│   ├── rag_summary.txt
│   ├── rag_query.txt
│   ├── report_executive_summary.txt
│   ├── report_recommendations.txt
│   ├── evaluate_hallucination.txt
│   ├── evaluate_rag_judge.txt
│   └── evaluate_report_judge.txt
│
├── vectorstore/          ← gerado em runtime (ChromaDB)
└── cache/                ← gerado em runtime (cache MD5)
```
