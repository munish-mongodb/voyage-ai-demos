# Voyage AI — Technical Demos

Interactive demos covering the core architectural capabilities of the **voyage-4 model series**. Built for technical architects evaluating Voyage AI for production search and RAG pipelines.

---

## What's in this repo

| File | Type | What it demonstrates |
|------|------|----------------------|
| `voyage_demo.ipynb` | Jupyter notebook | Shared embedding space across model sizes · Reranker instruction following |
| `voyage_multimodal_demo.ipynb` | Jupyter notebook | Standard chunking → contextual chunking → multimodal retrieval → unified RAG pipeline |
| `voyage_mrl_moe.html` | Standalone SPA | MRL dimension/quantization tradeoffs · MoE expert routing · Deployment decision guide |
| `voyage_demo.py` | Python script | Same as `voyage_demo.ipynb` — runnable from the terminal without Jupyter |

> `voyage4_three_levers.html` and `moe_architecture_explainer.html` are embeddable widget fragments (not standalone pages) — see `voyage_mrl_moe.html` for the full standalone version.

---

## Prerequisites

- Python 3.9+
- A [Voyage AI API key](https://dash.voyageai.com) — free tier available
- For the multimodal notebook: ~200 MB disk space (PDF downloaded at runtime)

---

## Setup

```bash
git clone <repo-url>
cd voyageai

# Copy the env template and add your key
cp .env.example .env
# Edit .env and set VOYAGE_API_KEY=your-key-here

# Install Python dependencies
pip install voyageai numpy pymupdf pillow matplotlib seaborn requests python-dotenv
```

---

## Running the demos

### Option A — Google Colab (recommended, no local setup)

1. Upload the `.ipynb` file to [colab.research.google.com](https://colab.research.google.com)
2. In Colab: click the **🔑 key icon** → add a secret named `VOYAGE_API_KEY` → toggle **Notebook access ON**
3. Run all cells top-to-bottom (`Runtime → Run all`)

### Option B — Local Jupyter

```bash
pip install jupyter
jupyter notebook
# Open the .ipynb file from the Jupyter file browser
```

### Option C — Terminal (voyage_demo.py only)

```bash
export VOYAGE_API_KEY=your-key-here
python voyage_demo.py
```

### The SPA (no API key needed)

```bash
open voyage_mrl_moe.html   # macOS
# or just double-click the file in Finder / Explorer
```

The SPA is fully self-contained — no server, no dependencies, no API calls.

---

## Notebook 1 — `voyage_demo.ipynb`

**Runtime:** ~20 seconds · **Models:** `voyage-4`, `voyage-4-lite`, `rerank-2.5`

### Part A — Shared Embedding Space

All models in the voyage-4 series (`voyage-4-large`, `voyage-4`, `voyage-4-lite`, `voyage-4-nano`) share a single embedding space. A vector produced by `voyage-4` and a vector produced by `voyage-4-lite` are directly comparable with cosine similarity.

**What the demo shows:**
- Embeds 3 sentences with both `voyage-4` and `voyage-4-lite`
- Computes cosine similarity within each model and across both models
- The cross-model column tracks the within-model columns — confirming interoperability

**Why this matters for architects:** You can index your corpus once with the highest-quality (and most expensive) model, then serve every query with the lightest model — all against the same index. No re-embedding when you change query model.

### Part B — Reranker Instruction Following

`rerank-2.5` accepts a natural-language instruction that steers which *kind* of result to surface, without changing the query itself.

**What the demo shows:**
- Query: `"how to use layers"` (ambiguous — could be Photoshop, Illustrator, After Effects)
- Run 1: no instruction → generic relevance ranking
- Run 2: instruction prepended → Illustrator advanced tutorials rise to the top
- Side-by-side comparison of both rankings

**Why this matters for architects:** The same retrieval index serves different user intents without building separate indices or modifying queries at the application layer. The reranker absorbs the intent signal.

---

## Notebook 2 — `voyage_multimodal_demo.ipynb`

**Runtime:** ~3–5 minutes · **Models:** `voyage-4-lite`, `voyage-context-3`, `voyage-multimodal-3.5`  
**Document:** "Attention Is All You Need" (arXiv:1706.03762) — downloaded automatically

The notebook tells a sequential story: each section exposes a limitation of the current approach, then introduces the Voyage AI model that fixes it.

### Part A — Baseline: Standard Chunking + `voyage-4-lite`

Extracts raw text from the PDF, splits into 500-character overlapping windows, embeds with `voyage-4-lite`, and runs semantic search.

**Limitation exposed:** Fixed-size chunks cut mid-sentence. Chunks near figures contain only caption text — the visual content of a diagram is invisible. A chunk about "multi-head attention" has no idea it's inside "Section 3.2: Attention."

### Part B — Context-Aware Chunking + `voyage-context-3`

Passes **all chunks together** to `voyage-context-3` via `vo.contextualized_embed(inputs=[all_chunks], ...)`. The model encodes each chunk with full awareness of where it sits in the document structure.

```python
result = vo.contextualized_embed(
    inputs=[all_chunks],        # entire document as one inner list
    model="voyage-context-3",
    input_type="document",
)
contextual_embeddings = result.results[0].embeddings
```

**What improves:** Chunks that describe experimental results now "know" they're in a results section. The benchmark query correctly surfaces the table rows with BLEU scores rather than prose that merely mentions benchmarks.

**Key metric:** Part E quantifies the improvement — `voyage-context-3` produces lower mean off-diagonal cosine similarity (more discriminative embeddings), which translates directly to fewer false positives in retrieval.

> **Note on scores:** Absolute cosine similarity values from `voyage-context-3` will appear numerically lower than `voyage-4-lite` scores. This is expected — different models calibrate their score distributions differently. Lower absolute score ≠ worse retrieval. What matters is ranking, not magnitude.

### Part C — Multimodal: Text Queries → Page Images

Converts every PDF page to a PIL image at 150 DPI, embeds each with `voyage-multimodal-3.5`. Text queries are embedded with the same model — both live in the same vector space.

```python
# Embed a page image
vo.multimodal_embed(inputs=[[page_img]], model="voyage-multimodal-3.5", input_type="document")

# Embed a text query — same model, same space
vo.multimodal_embed(inputs=[[query_text]], model="voyage-multimodal-3.5", input_type="query")
```

**What this unlocks:** The query `"Show me the Transformer model architecture diagram"` retrieves the page containing the encoder-decoder diagram — a page with almost no extractable text. Text-only retrieval scores this page near zero. Cross-modal retrieval finds it at rank 1.

### Part D — Unified Multimodal RAG Pipeline

Re-embeds all text chunks with `voyage-multimodal-3.5` (text-only mode) to put them in the same vector space as the page images. Merges both into a single flat index and searches it with one query.

```
PDF
 ├── Text chunks  ──► voyage-multimodal-3.5 ──► text embeddings  ──┐
 └── Page images  ──► voyage-multimodal-3.5 ──► image embeddings ──┴──► Unified Index
```

A single query now competes text chunks against page images. Conceptual queries surface text; visual queries surface images. The modality selection is implicit — whichever entry has the highest cosine score wins.

### Part E — Similarity Heatmap

Computes pairwise cosine similarity matrices for the first 25 chunks under both approaches. Renders side-by-side seaborn heatmaps and prints mean off-diagonal similarity as a scalar discriminability metric.

**Reading the output:**
- Lower mean off-diagonal = more discriminative embeddings
- Contextual embeddings typically show more block structure — sections of the paper cluster visibly
- Standard embeddings tend toward a higher, flatter similarity surface

---

## The SPA — `voyage_mrl_moe.html`

No API calls. Open in any browser. Three tabs:

### Tab 1 — MRL (Matryoshka Representation Learning)

Three independent levers that control the storage/quality/cost tradeoff of your embedding index:

| Lever | What it controls | Options |
|-------|-----------------|---------|
| Model tier | Query cost · accuracy ceiling | `voyage-4-large` → `voyage-4-nano` |
| MRL dimension | Storage · ANN search speed | 2048 → 64 dims |
| Quantization | Vector storage footprint | float32 → int8 → binary |

**Key insight:** All voyage-4 models are trained with MRL — the first *d* dimensions of a 2048-dim embedding already form a valid *d*-dimensional embedding. Store once at full dims; truncate at serving time with `output_dimension=N`. No re-indexing when you change dimension in production.

The tab shows live metrics (storage GB per 1M vectors, compression ratio, Recall@10 vs baseline) and a full quality reference table for all dim × quantization combinations.

### Tab 2 — MoE (Mixture of Experts)

voyage-4 uses a sparse MoE feed-forward layer: 8 specialist experts, 2 active per token. The router (gating network) selects experts based on token content — 25% of parameters execute per forward pass.

The interactive router simulation shows which experts fire for different input types (legal text, Python code, French query, financial doc, scientific abstract, general text) and the gating scores across the expert pool.

**MoE × MRL are orthogonal:** MoE controls inference compute inside the encoder. MRL controls the output vector size. They are independently tunable — choosing 512-dim MRL doesn't change which experts fire.

### Tab 3 — Deployment Guide

Decision tables for all three axes:
- Which model tier to use at index time vs query time
- Which MRL dimension to target given your storage budget and quality floor
- Which quantization format to use
- Five named reference configurations from "max precision" to "max compression" with storage and Recall@10 numbers

---

## Model reference

| Model | Modality | Max dims | Quantization-aware | MRL | Use case |
|-------|----------|----------|--------------------|-----|----------|
| `voyage-4-large` | Text | 2048 | ✓ | ✓ | Indexing; max recall |
| `voyage-4` | Text | 1024 | ✓ | ✓ | Balanced production |
| `voyage-4-lite` | Text | 512 | ✓ | ✓ | High-QPS queries |
| `voyage-4-nano` | Text | 256 | ✓ | ✓ | On-device / free |
| `voyage-context-3` | Text | 1024 | — | — | Long documents; chunk context |
| `voyage-multimodal-3.5` | Text + Image | 1024 | — | — | PDFs, slides, diagrams |
| `rerank-2.5` | Text | — | — | — | Instruction-following re-rank |

---

## API key security

- `.env` is gitignored — never committed
- Get a key at [dash.voyageai.com](https://dash.voyageai.com)
- Free tier includes enough quota to run all demos several times
- For Colab: use Secrets (🔑 icon) — keeps the key out of notebook cells entirely
