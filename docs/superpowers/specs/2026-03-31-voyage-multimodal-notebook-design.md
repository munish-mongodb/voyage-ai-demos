# Design: Voyage Multimodal + Contextual Chunking Jupyter Notebook

**Date:** 2026-03-31  
**Status:** Approved

---

## Context

The existing `voyage_demo.ipynb` covers shared embedding space (voyage-4 / voyage-4-lite) and reranker instruction following (rerank-2.5). This new notebook is a distinct, standalone companion that showcases two newer and more advanced VoyageAI capabilities:

1. **`voyage-context-3`** — contextual chunk embeddings that understand each chunk's role within the full document
2. **`voyage-multimodal-3.5`** — multimodal embeddings that embed text and images in a shared space

The motivation is to show a complete, realistic PDF intelligence pipeline — from raw document to cross-modal semantic search — using a single real-world document.

---

## Narrative Structure: Problem-First Progression

The notebook tells a story: each section reveals a limitation of the current approach, then introduces the VoyageAI model that solves it.

```
Standard chunking (baseline) 
  → contextual chunking (better text retrieval)
    → multimodal embeddings (unlock visual content)
      → unified RAG pipeline (combine both)
        → heatmap (visual proof of quality difference)
```

---

## PDF Source

**"Attention Is All You Need"** — arXiv:1706.03762  
URL: `https://arxiv.org/pdf/1706.03762`  
~15 pages. Contains: architecture diagrams, performance tables, BLEU score charts. Small enough for fast demo runtime, rich enough to demonstrate multimodal value. Downloaded at notebook runtime.

---

## Sections

### Section 0 — Setup
- `pip install voyageai pymupdf pillow numpy matplotlib seaborn requests`
- API key via Google Colab Secrets (`userdata.get('VOYAGE_API_KEY')`) with `.env` fallback
- Download PDF to `/tmp/attention_is_all_you_need.pdf`
- Helper: `cosine(a, b)` — reuse pattern from existing notebook
- Helper: `render_page_image(pdf_path, page_num)` — renders a PDF page as PIL image via PyMuPDF
- Helper: `standard_chunk(text, size=500, overlap=50)` — fixed-size character chunking

### Part A — Baseline: Standard Chunking + voyage-3-lite
- Extract full text per page using `fitz` (PyMuPDF)
- Apply `standard_chunk()` → produces N chunks of ~500 chars
- Embed all chunks: `vo.embed(chunks, model="voyage-3-lite", input_type="document")`
- Run 3 sample queries:
  1. `"What is the attention mechanism?"` — covered in text
  2. `"How does the model perform on translation tasks?"` — in tables/charts
  3. `"What are the encoder and decoder components?"` — in architecture diagram caption
- Display top-3 retrieved chunks per query with cosine scores
- **Observation callout:** Some chunks are truncated mid-sentence; chunks near figures score poorly because they only capture caption text, not visual content

### Part B — Contextual Chunking with voyage-context-3
- Reuse same chunks from Part A
- Call REST endpoint `POST /v1/contextualizedembeddings` directly (SDK wrapper if available, else `requests`)
  ```python
  payload = {
      "inputs": [chunks],   # all chunks of the document as a single inner list
      "input_type": "document",
      "model": "voyage-context-3"
  }
  ```
- Run the same 3 queries (embed query with `voyage-3-lite` for query side, or `voyage-context-3` with `input_type="query"`)
- Side-by-side table: Part A rank vs Part B rank for each query's top results
- **Key insight callout:** Chunks that were previously buried (e.g., the multi-head attention description) now surface correctly because the model understands document structure

### Part C — Multimodal: Searching What Text Can't See
- Convert each PDF page to a PIL image using PyMuPDF (`page.get_pixmap()`)
- Embed each page image: `vo.multimodal_embed(inputs=[[img]], model="voyage-multimodal-3.5", input_type="document")`
- Embed 3 text queries using the same model: `vo.multimodal_embed(inputs=["query text"], model="voyage-multimodal-3.5", input_type="query")`
- Retrieve top-2 page images per query via cosine similarity
- Render matched pages inline using `matplotlib.pyplot.imshow`
- Demonstrate: query `"Show me the model architecture diagram"` retrieves the page with the transformer diagram — something text extraction completely misses
- **Key insight callout:** Cross-modal retrieval — text queries find relevant images; the embedding space is shared across modalities

### Part D — Unified Multimodal RAG Pipeline
- Build a unified index: combine contextual text chunk embeddings + page image embeddings
- Tag each entry with its source type (`"text_chunk"` or `"page_image"`)
- For each of the 3 queries, search the unified index
- Display results with source type labels — show that some queries are best answered by text chunks, others by page images
- **Pipeline summary:** Markdown diagram showing: PDF → text chunks (voyage-context-3) + page images (voyage-multimodal-3.5) → unified index → query → ranked results

### Part E — Similarity Heatmap
- Compute N×N pairwise cosine similarity matrices for:
  - Standard chunks embedded with `voyage-3-lite`
  - Same chunks embedded with `voyage-context-3`
- Render as side-by-side seaborn heatmaps
- Label axes with short chunk identifiers (e.g., "chunk_01", "chunk_02"…)
- **Observation:** Contextual embeddings produce more structured, block-diagonal patterns — semantically distinct sections of the paper form visible clusters

---

## Models Used

| Model | Purpose | API Method |
|-------|---------|------------|
| `voyage-3-lite` | Baseline text embeddings (Part A) | `vo.embed()` |
| `voyage-context-3` | Contextual chunk embeddings (Part B) | `POST /v1/contextualizedembeddings` |
| `voyage-multimodal-3.5` | Multimodal text+image embeddings (Parts C, D) | `vo.multimodal_embed()` |

---

## Dependencies

```
voyageai       # VoyageAI Python SDK
pymupdf        # PDF text extraction + page-to-image rendering (import as fitz)
pillow         # PIL image handling
numpy          # Vector math
matplotlib     # Image display + heatmap plotting
seaborn        # Styled heatmaps
requests       # PDF download + REST API call for voyage-context-3
```

---

## Colab Compatibility

Follow the same pattern as `voyage_demo.ipynb`:
```python
try:
    from google.colab import userdata
    VOYAGE_API_KEY = userdata.get('VOYAGE_API_KEY')
except ImportError:
    from dotenv import load_dotenv
    load_dotenv()
    VOYAGE_API_KEY = os.environ.get('VOYAGE_API_KEY')
```

---

## Verification

1. Run all cells top-to-bottom in a clean Colab environment
2. Confirm Part A retrieval returns plausible but imperfect results for all 3 queries
3. Confirm Part B retrieval shows measurably better ranking for at least 2 of 3 queries
4. Confirm Part C renders actual page images from the PDF and the architecture diagram page ranks #1 for the diagram query
5. Confirm Part D unified index returns a mix of text chunk and page image results
6. Confirm Part E heatmaps render and show visually distinct patterns between standard vs contextual embeddings
7. Run locally with `.env` API key as a fallback test
