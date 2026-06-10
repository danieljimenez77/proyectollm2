# Retail Vision Intelligence System — TP2

**LIACD — Interação com Modelos de Grande Escala**  
Ano letivo 2025/2026 | Cotação: 6,5 / 20

---

## Índice

1. [Descrição do projeto](#descrição-do-projeto)
2. [Estrutura do repositório](#estrutura-do-repositório)
3. [Dataset](#dataset)
4. [Instalação e configuração](#instalação-e-configuração)
5. [Componentes](#componentes)
6. [Utilização](#utilização)
7. [Avaliação](#avaliação)
8. [Decisões técnicas e problemas encontrados](#decisões-técnicas-e-problemas-encontrados)

---

## Descrição do projeto

Sistema de inspeção visual contínua de prateleiras de supermercado com memória. O sistema recebe imagens de prateleiras, analisa-as com modelos de linguagem multimodais (Google Gemini), deteta problemas operacionais, permite definir regras de deteção em linguagem natural, e indexa o histórico de inspeções numa base de dados vetorial para recuperação semântica.

### Componentes principais

| Componente | Ficheiro | Descrição |
|---|---|---|
| 1 | `src/shelf_inspector.py` | Análise visual com LLM multimodal |
| 2 | `src/rule_engine.py` | Geração e execução de regras em linguagem natural |
| 3 | `src/rag_memory.py` | Indexação e recuperação de inspeções históricas |
| 4 | `src/report_generator.py` | Geração de relatórios com contexto histórico |
| 5 | `src/interface_streamlit.py` | Interface visual para o gestor de loja |

---

## Estrutura do repositório

```
tp2 integracion/
├── README.md
├── requirements.txt
├── .env.example
├── Apikey.txt                        ← Chave de API do Gemini (NÃO fazer commit)
├── .gitignore
├── evaluate.py
│
├── data/
│   ├── images/                       ← Dataset de imagens
│   │   ├── normal/                   ← 150 imagens de prateleiras normais
│   │   ├── empty/                    ← 100 imagens de prateleiras vazias
│   │   ├── planogram/                ← 100 imagens de violações de planograma
│   │   ├── dirty/                    ← 80 imagens de prateleiras sujas/desordenadas
│   │   └── ambiguous/                ← 70 imagens de casos ambíguos
│   ├── inspections/                  ← Registos de inspeção gerados
│   └── rules/                        ← Regras persistidas
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
├── vectorstore/                      ← ChromaDB persistente (gerado em runtime)
└── cache/                            ← Cache de resultados de API (gerado em runtime)
```

---

## Dataset

### Fontes utilizadas

O dataset combina duas fontes principais:

**1. SKU-110K** (Goldman et al., 2019)
- 11.762 imagens de supermercado com anotações de bounding box
- Fonte principal para imagens de prateleiras normais e com produtos bem posicionados
- Repositório: https://github.com/eg4000/SKU110K_CVPR19
- Licença: uso académico

**2. Grocery Store Dataset** (Hult et al., 2019)
- Imagens de produtos e prateleiras em condições naturais
- Complementa o SKU-110K nas categorias de violações de planograma e desordem
- HuggingFace: `johnanvik/grocery-store-dataset`
- Licença: uso académico

### Distribuição mínima requerida

| Tipo | Mínimo | Descrição |
|---|---|---|
| Prateleira normal | 150 | Produto bem posicionado, sem problemas visíveis |
| Prateleira vazia | 100 | Uma ou mais posições sem produto |
| Violação de planograma | 100 | Produto em posição incorreta, etiqueta ausente, produto tombado |
| Prateleira suja/desordenada | 80 | Produto desalinhado, embalagens danificadas |
| Caso ambíguo | 70 | Situações onde a classificação não é óbvia |

### Justificação da escolha

O SKU-110K foi escolhido como fonte principal por ser o dataset de referência em investigação de retalho, com anotações de alta qualidade e volume suficiente. O Grocery Store Dataset complementa as categorias com menor representação no SKU-110K. Esta combinação permite cobrir todos os tipos de imagem requeridos com diversidade suficiente para generalização.

---

## Instalação e configuração

### Pré-requisitos

- Python 3.11 ou superior
- Conta no [Google AI Studio](https://aistudio.google.com) (gratuita, sem cartão de crédito)

### 1. Clonar o repositório

```bash
git clone https://github.com/danieljimenez77/proyectollm2
cd proyectollm2
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Configurar a chave de API do Gemini

**Opção A — Ficheiro `.env`** (recomendada):
```
GEMINI_API_KEY=a_sua_chave_aqui
```

**Opção B — Ficheiro `Apikey.txt`** na raiz do projeto:
```
AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

> ⚠️ **Importante:** Nunca faça commit da chave de API. Adicione `Apikey.txt` e `.env` ao `.gitignore`.

### 4. Obter a chave de API

1. Aceda a https://aistudio.google.com
2. Inicie sessão com uma conta Google
3. Clique em **Get API Key** → **Create API key**
4. Copie a chave (começa com `AIza`, 39 caracteres)

### 5. Iniciar a interface

```bash
cd src/
streamlit run interface_streamlit.py
```

---

## Componentes

### Componente 1: Shelf Inspector (`src/shelf_inspector.py`)

Analisa imagens de prateleiras usando **Google Gemini 2.5 Flash** e produz um JSON estruturado com os problemas detetados.

#### Modelo utilizado

É utilizado o `gemini-2.5-flash`. Durante o desenvolvimento foram testados outros modelos:

| Modelo | Resultado | Motivo do descarte |
|---|---|---|
| `gemini-2.0-flash` | Erro 429 persistente | Limite de taxa muito restritivo no free tier |
| `gemini-1.5-flash` | Erro 404 | Não disponível na versão v1beta da API |
| `gemini-2.5-flash` | ✅ Funciona | Disponível, multimodal, gratuito com AI Studio |

#### Três estratégias de prompting

Os prompts estão versionados na pasta `prompts/` e são carregados do disco em tempo de execução, nunca hardcoded no código.

**Estratégia A — Zero-shot direto** (`inspect_A_zero_shot.txt`):
Instrução direta pedindo a análise e o JSON, sem exemplos nem estrutura de raciocínio guiada.

**Estratégia B — Chain-of-Thought visual** (`inspect_B_chain_of_thought.txt`):
O modelo raciocina explicitamente em 6 passos: descrição geral → análise zona a zona → identificação de anomalias → estimação de fill rate → classificação global → produção do JSON. É a estratégia por defeito.

**Estratégia C — Few-shot com exemplos textuais** (`inspect_C_few_shot.txt` + `inspect_C_few_shot_examples.txt`):
Inclui 3 exemplos textuais de inspeções corretas antes da imagem nova, evitando o consumo extra de quota que implicaria passar múltiplas imagens.

#### Gestão de quota (API gratuita: 15 req/min, 1500 req/dia)

- **Cache por MD5**: cada inspeção é guardada em `cache/` identificada pelo hash MD5 da imagem e da estratégia. Se a imagem não mudou, não é consumida quota.
- **Rate limiting**: controlo de janela deslizante de 60 segundos para não ultrapassar 15 req/min.
- **Backoff exponencial**: perante erro 429, aguarda 2, 4, 8, 16, 32 segundos entre tentativas.
- **Fallback gracioso**: se a quota diária for esgotada, o sistema continua a funcionar para imagens em cache e notifica claramente quando não consegue processar novas imagens.

#### Schema de saída (obrigatório)

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
      "location": "descrição da localização",
      "severity": "low|medium|high",
      "description": "descrição em linguagem natural",
      "confidence": 0.0,
      "affected_area_pct": 0.0
    }
  ],
  "shelf_fill_rate": 0.0,
  "products_detected": ["categorias de produto visíveis"],
  "model_reasoning": "raciocínio explícito antes da classificação"
}
```

---

### Componente 2: Rule Engine (`src/rule_engine.py`)

Converte regras em linguagem natural definidas pelo gestor em configurações JSON executáveis, e executa-as contra os resultados de inspeções.

#### Prompts do Rule Engine

| Ficheiro | Propósito |
|---|---|
| `prompts/rule_parse.txt` | Converte linguagem natural em JSON em 4 passos guiados |
| `prompts/rule_ambiguity_response.txt` | Gera resposta amigável ao gestor quando há ambiguidades |

#### Fluxo de conversão

O prompt `rule_parse.txt` guia o modelo em 4 passos:
1. **Compreensão** — que condição descreve a regra e que ação deve ser tomada
2. **Extração** — zona(s), horário, tipo de issue, limiar de fill rate, severidade, localização
3. **Deteção de ambiguidades** — lista todos os aspetos não claramente definidos
4. **Produção do JSON** — gera o schema estruturado com o campo `validation`

#### Schema de regra

```json
{
  "rule_id": "RULE_001",
  "created_at": "2026-06-06T15:20:03Z",
  "natural_language": "texto original do gestor",
  "description": "reformulação clara em linguagem formal",
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
    "notification_message": "template com {zone_id}, {fill_rate}, {issue_type}, {severity}"
  },
  "validation": {
    "is_valid": true,
    "ambiguities": ["lista de aspetos não claros"],
    "assumptions": ["lista de pressupostos assumidos"]
  }
}
```

#### Execução de regras

Após cada inspeção, o executor percorre todas as regras guardadas verificando por ordem:
- Filtro de zona — a zona inspecionada está em `zone_filter`?
- Filtro de horário — a hora do timestamp está dentro de `time_filter`?
- Filtro de fill rate — o `shelf_fill_rate` está abaixo do limiar?
- Filtro de issues — algum issue coincide com `issue_types`, `severity_threshold` e `location_filter`?

Cada execução produz logs detalhados indicando que regras foram verificadas, quais dispararam e porquê.

#### Comportamento com ambiguidades

Quando uma regra tem ambiguidades, o sistema **não assume silenciosamente**: gera uma resposta em linguagem natural explicando cada ambiguidade e perguntando como resolvê-la. A regra só é guardada se o gestor usar `--auto-save` ou confirmar explicitamente.

Exemplo real durante o desenvolvimento — regra: `"Avisa-me quando o fill rate de qualquer zona cair abaixo de 60%"`

Ambiguidades detetadas automaticamente:
- Não é especificado horário → assume qualquer momento
- Não é especificado nível de urgência → assume `warning`
- Fill rate é uma métrica, não um `issue_type` → `issue_types` fica vazio
- Não é especificada severidade → assume `any`
- Não é especificada localização na prateleira → assume `any`

#### Problema detetado: regras não eram guardadas sem `--auto-save`

Durante os testes observou-se que ao adicionar regras com ambiguidades sem o flag `--auto-save`, o sistema mostrava o JSON e as ambiguidades mas **não guardava o ficheiro em disco**. Por design, o sistema requer confirmação explícita antes de persistir regras ambíguas. A solução é usar `--auto-save` para guardar com os valores por defeito assumidos.

---

### Componente 3: RAG Memory (`src/rag_memory.py`)

Indexa o histórico de inspeções numa base de dados vetorial e permite recuperação semântica em linguagem natural.

#### Stack tecnológico

| Componente | Tecnologia | Justificação |
|---|---|---|
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` | Local, gratuito, suporta português e espanhol |
| Vector store | ChromaDB (PersistentClient) | Local, persistente em disco, sem servidor |
| Similaridade | Cosseno (hnsw:space=cosine) | Padrão para similaridade semântica de texto |
| Síntese | Gemini 2.5 Flash | Gera respostas em linguagem natural com contexto recuperado |

#### Estratégia de chunking híbrida

São implementadas duas coleções no ChromaDB:

**`inspection_summaries`** — um chunk por inspeção:
- O texto indexado é um summary gerado pelo Gemini, semanticamente rico
- Os metadados estruturados (zona, data, hora, dia da semana, fill rate, status) permitem filtragem pré-retrieval
- Recuperação eficiente para queries gerais

**`inspection_issues`** — um chunk por issue individual:
- Cada issue é indexado com texto detalhado: tipo, localização, descrição, severidade, confiança
- Permite recuperação granular para queries específicas
- Aumenta o índice mas melhora o Recall@3 para queries de issue específico

#### Prompts do RAG

| Ficheiro | Propósito |
|---|---|
| `prompts/rag_summary.txt` | Gera summaries ricos para indexação |
| `prompts/rag_query.txt` | Sintetiza resposta com contexto recuperado |

#### Queries obrigatórias suportadas

```bash
python rag_memory.py query "Quando foi a última vez que a zona Z_S1 teve problemas?"
python rag_memory.py query "Que zonas tiveram mais issues esta semana?"
python rag_memory.py query "Existe algum padrão nos problemas detetados às sextas-feiras?"
python rag_memory.py query "Que regras foram mais frequentemente disparadas este mês?"
```

#### Problemas encontrados

**Erro 503 — Alta procura do servidor Gemini:**
Durante os testes iniciais obteve-se um `503 UNAVAILABLE` ao chamar o Gemini para sintetizar a resposta RAG. É um erro temporário por alta procura. O código implementa retry com backoff exponencial para este caso.

---

### Componente 4: Report Generator (`src/report_generator.py`)

Gera relatórios de inspeção em Markdown combinando os resultados do ShelfInspector, RuleEngine e RAGMemory.

#### Secções obrigatórias do relatório

1. **Sumário executivo** (máx. 150 palavras) — estado geral da loja na sessão
2. **Problemas por zona** — lista de problemas, severidade, fill rate e comparação com histórico
3. **Regras disparadas** — regras ativadas com os dados concretos da inspeção
4. **Contexto histórico relevante** — padrões passados recuperados do RAG com referências explícitas
5. **Recomendações** — máximo 5 ações concretas ordenadas por urgência

#### Prompts do Report Generator

| Ficheiro | Propósito |
|---|---|
| `prompts/report_executive_summary.txt` | Gera sumário executivo em máx. 150 palavras |
| `prompts/report_recommendations.txt` | Gera recomendações concretas em JSON |

---

### Componente 5: Interface Streamlit (`src/interface_streamlit.py`)

Interface visual completa construída com Streamlit, com pestañas, botões e visualização de imagens.

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

#### Comportamento verificado nos testes

```
✅ Banner e estado do sistema
✅ Inspeção com cache hit e deteção automática de regras
✅ Listagem de regras com ambiguidades
✅ Consulta RAG com síntese por Gemini e inspection_id
✅ Geração de relatório com retry automático em erros 503
✅ Estado de sessão atualizado entre comandos
```

---

## Utilização

### Interface Streamlit (recomendada)

```bash
cd src/
streamlit run interface_streamlit.py
```

### Shelf Inspector (CLI)

```bash
cd src/

# Inspeção individual com Chain-of-Thought (por defeito)
python shelf_inspector.py "..\data\images\image.jpg" --zone Z_S1 --strategy B

# Comparar as três estratégias sobre a mesma imagem
python shelf_inspector.py "..\data\images\image.jpg" --zone Z_S1 --compare

# Forçar nova chamada ignorando cache
python shelf_inspector.py "..\data\images\image.jpg" --force
```

### RAG Memory (CLI)

```bash
cd src/

# Indexar uma inspeção
python rag_memory.py index "..\cache\resultado.json"

# Indexar todas as inspeções de um diretório
python rag_memory.py index-dir "..\data\inspections\"

# Consultas em linguagem natural
python rag_memory.py query "Quando foi a última vez que Z_S1 teve problemas?"
python rag_memory.py query "Há produtos desalinhados?" --zone Z_S1
python rag_memory.py query "Onde há issues de fill rate baixo?" --issues --k 5

# Ver estatísticas do vector store
python rag_memory.py stats
```

### Rule Engine (CLI)

```bash
cd src/

# Adicionar uma regra (pergunta se há ambiguidades)
python rule_engine.py add "Avisa-me quando a prateleira inferior estiver mais de 40% vazia"

# Adicionar guardando mesmo com ambiguidades
python rule_engine.py add "Avisa quando houver produtos desalinhados" --auto-save

# Listar todas as regras guardadas
python rule_engine.py list

# Eliminar uma regra
python rule_engine.py delete RULE_001

# Executar todas as regras contra uma inspeção
python rule_engine.py execute --inspection "..\cache\resultado.json"
```

### Report Generator (CLI)

```bash
cd src/

# Gerar relatório a partir de JSON existente
python report_generator.py from-json "..\cache\resultado.json"

# Pipeline completo: inspecionar imagem e gerar relatório
python report_generator.py inspect "..\data\images\image.jpg" --zones Z_S1
```

---

## Avaliação

```bash
# Avaliação completa (executar da raiz do projeto)
python evaluate.py --images-dir test_images/ --output evaluation_report.json

# Apenas análise visual
python evaluate.py --images-dir test_images/ --output eval.json --skip-rag --skip-rules

# Com ground truth manual
python evaluate.py --images-dir test_images/ --ground-truth ground_truth.json --output eval.json
```

### Métricas implementadas

**Análise visual:**

| Métrica | Descrição |
|---|---|
| Issue Detection Rate | % de issues do ground truth corretamente identificados (recall) |
| False Positive Rate | % de issues reportados que não existem no ground truth |
| Severity Accuracy | % de issues com severidade corretamente classificada |
| JSON Parse Rate | % de respostas do modelo que são JSON válido parseável |
| Hallucination Rate | % de afirmações no campo description não verificáveis |

**RAG:**

| Métrica | Descrição |
|---|---|
| Recall@3 | % de queries onde o documento relevante está nos top-3 resultados |
| Faithfulness | % de afirmações na resposta RAG suportadas pelos chunks recuperados |
| Answer Relevance | Avaliado por LLM-as-judge: a resposta responde à query? |

**Rule Engine:**

| Métrica | Descrição |
|---|---|
| Rule Parse Rate | % de regras em linguagem natural convertidas em JSON válido |
| Rule Correctness | % de regras convertidas que executam corretamente sobre dados sintéticos |
| Ambiguity Detection | % de regras ambíguas corretamente identificadas como tal |

### LLM-as-Judge

O avaliador usa o próprio Gemini 2.5 Flash como juiz para métricas qualitativas:

| Prompt | Métrica avaliada |
|---|---|
| `evaluate_hallucination.txt` | Se as afirmações em `description` são verificáveis na imagem |
| `evaluate_rag_judge.txt` | Faithfulness e Answer Relevance das respostas RAG |
| `evaluate_report_judge.txt` | Completude, acionabilidade e precisão dos relatórios |

### Ground Truth

O harness aceita um ficheiro JSON de anotações manuais (`--ground-truth`). Se não for fornecido, o sistema infere o ground truth automaticamente a partir dos nomes dos ficheiros:

| Nome contém | Status inferido | Issue type |
|---|---|---|
| `empty`, `vaci` | critical | empty_shelf |
| `misalign`, `desalin` | warning | misaligned |
| `damage`, `dan` | warning | damaged |
| `wrong`, `plano` | warning | wrong_product |
| `ok`, `normal`, `good` | ok | — |

---

## Decisões técnicas e problemas encontrados

### SDK do Google Gemini — Migração obrigatória

O pacote `google-generativeai` foi depreciado e já não recebe atualizações. O novo SDK é `google-genai` com uma API completamente diferente:

```bash
# Incorreto (depreciado)
pip install google-generativeai

# Correto
pip install google-genai
```

Alterações principais no código:

| Antes (`google-generativeai`) | Agora (`google-genai`) |
|---|---|
| `import google.generativeai as genai` | `from google import genai` |
| `genai.configure(api_key=...)` | `genai.Client(api_key=...)` |
| `genai.GenerativeModel(MODEL)` | `client.models.generate_content(model=...)` |
| `[prompt, image]` como conteúdo | `[Part.from_bytes(...), Part.from_text(...)]` |
| `generation_config={"temperature": 0}` | `config=GenerateContentConfig(temperature=0)` |

### Seleção do modelo

Durante o desenvolvimento foram testados três modelos até encontrar o que funciona com a chave de API gratuita do AI Studio:

1. `gemini-2.0-flash` — erro 429 persistente desde a primeira chamada, mesmo com backoff
2. `gemini-1.5-flash` — erro 404, modelo não disponível na versão `v1beta` da API
3. `gemini-2.5-flash` — ✅ funciona corretamente com a chave gratuita do AI Studio

Para verificar os modelos disponíveis na sua conta:
```
https://generativelanguage.googleapis.com/v1beta/models?key=A_SUA_API_KEY
```

### Leitura da chave de API no Windows

No Windows, os ficheiros `.txt` podem incluir um BOM (Byte Order Mark) invisível no início que corrompe a chave. Para lê-la corretamente:

```python
key = Path("Apikey.txt").read_text(encoding="utf-8-sig").strip().replace('\r', '').replace('\n', '')
```

O encoding `utf-8-sig` elimina automaticamente o BOM. O `.strip()` e a substituição de `\r\n` eliminam qualquer quebra de linha do Windows.

### Problema recorrente: Apikey.txt em commits do Git

Durante o desenvolvimento este problema ocorreu várias vezes. A causa é sempre a mesma: o ficheiro `Apikey.txt` é criado depois de o repositório ter sido inicializado e o Git deteta-o como ficheiro novo no `git add .` seguinte.

**Solução quando o ficheiro está no último commit mas ainda não foi feito push:**
```bash
git rm --cached Apikey.txt
git commit --amend --no-edit
git push
```

**Solução quando já está no histórico e o GitHub bloqueia o push:**
```bash
git filter-branch --force --index-filter "git rm --cached --ignore-unmatch Apikey.txt" --prune-empty --tag-name-filter cat -- --all
git push origin main --force
```

**Importante:** Se a chave chegou a ser enviada para um repositório público, é necessário gerar uma nova imediatamente em [aistudio.google.com](https://aistudio.google.com) pois ficou comprometida.

**Prevenção definitiva** — verificar que `.gitignore` contém:
```
Apikey.txt
.env
cache/
vectorstore/
__pycache__/
*.pyc
data/inspections/
```

### Prompts separados do código

Os prompts estão em `prompts/*.txt` e são carregados em runtime. Isto permite modificar os prompts sem tocar no código Python e cumpre o requisito explícito do enunciado de versionar os prompts separadamente.

Ficheiros de prompts atuais:

| Ficheiro | Componente | Propósito |
|---|---|---|
| `inspect_A_zero_shot.txt` | Shelf Inspector | Estratégia A — Zero-shot |
| `inspect_B_chain_of_thought.txt` | Shelf Inspector | Estratégia B — Chain-of-Thought |
| `inspect_C_few_shot.txt` | Shelf Inspector | Estratégia C — Few-shot (template) |
| `inspect_C_few_shot_examples.txt` | Shelf Inspector | Exemplos para estratégia C |
| `rule_parse.txt` | Rule Engine | Conversão de linguagem natural para JSON |
| `rule_ambiguity_response.txt` | Rule Engine | Resposta ao gestor sobre ambiguidades |
| `rag_summary.txt` | RAG Memory | Geração de summaries ricos para indexação |
| `rag_query.txt` | RAG Memory | Síntese de resposta com contexto recuperado |
| `report_executive_summary.txt` | Report Generator | Sumário executivo |
| `report_recommendations.txt` | Report Generator | Recomendações concretas |
| `evaluate_hallucination.txt` | Evaluate | LLM-as-judge para alucinações |
| `evaluate_rag_judge.txt` | Evaluate | LLM-as-judge para RAG |
| `evaluate_report_judge.txt` | Evaluate | LLM-as-judge para relatórios |
