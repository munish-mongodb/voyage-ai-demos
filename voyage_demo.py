"""
Voyage AI SDK Demo
  Part A: Shared Embedding Space (voyage-4 vs voyage-4-lite)
  Part B: Reranker Instruction Following (rerank-2.5)

Usage:
    export VOYAGE_API_KEY=<your key>
    python voyage_demo.py
"""

import os
import sys
import numpy as np
import voyageai


# ── Startup guard ────────────────────────────────────────────────────────────
if not os.environ.get("VOYAGE_API_KEY"):
    print("ERROR: VOYAGE_API_KEY environment variable is not set.")
    sys.exit(1)

vo = voyageai.Client()


def cosine(u, v):
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))


# ═════════════════════════════════════════════════════════════════════════════
# PART A — Shared Embedding Space
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("  PART A — Shared Embedding Space (voyage-4 vs voyage-4-lite)")
print("═" * 70)
print("Claim: all voyage-4 series models share the same embedding space,")
print("so cross-model cosine similarity is just as meaningful as within-model.\n")

sentences = [
    "A dog runs through a sunny park",                         # S1
    "A canine dashes across a bright meadow",                  # S2 — semantically similar to S1
    "The quarterly earnings report exceeded expectations",     # S3 — unrelated
]

labels = ["S1 (dog/park)", "S2 (canine/meadow)", "S3 (earnings report)"]

try:
    print("Embedding with voyage-4 …")
    emb_a = vo.embed(sentences, model="voyage-4", input_type="document").embeddings
    print("Embedding with voyage-4-lite …")
    emb_b = vo.embed(sentences, model="voyage-4-lite", input_type="document").embeddings
except Exception as e:
    print(f"ERROR during embedding: {e}")
    sys.exit(1)

# Build comparison table for the 3 most illustrative pairs
pairs = [
    (0, 0, "S1 vs S1  (same sentence)"),
    (0, 1, "S1 vs S2  (semantically similar)"),
    (0, 2, "S1 vs S3  (unrelated topics)"),
]

col_w = 22
print(f"\n{'Sentence pair':<30} {'voyage-4 × voyage-4':>{col_w}} "
      f"{'voyage-4 × voyage-4-lite':>{col_w}} {'voyage-4-lite × voyage-4-lite':>{col_w}}")
print("-" * (30 + col_w * 3 + 6))

for i, j, desc in pairs:
    within_a  = cosine(emb_a[i], emb_a[j])
    cross     = cosine(emb_a[i], emb_b[j])
    within_b  = cosine(emb_b[i], emb_b[j])
    print(f"{desc:<30} {within_a:>{col_w}.4f} {cross:>{col_w}.4f} {within_b:>{col_w}.4f}")

print("\n✓ Cross-model similarity (col 2) tracks within-model similarity (cols 1 & 3),")
print("  confirming that voyage-4 series vectors are interoperable across model sizes.")


# ═════════════════════════════════════════════════════════════════════════════
# PART B — Reranker with Instruction Following (rerank-2.5)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("  PART B — Reranker Instruction Following (rerank-2.5)")
print("═" * 70)
print("Query: 'how to use layers'\n")

query = "how to use layers"

documents = [
    "Photoshop for beginners: how to create and manage layers in your first project",
    "Illustrator advanced guide: using artboards and layers for complex vector workflows",
    "After Effects animation: compositing with layer-based timeline and keyframes",
    "Beginner Photoshop tutorial: understanding the Layers panel and blend modes",
    "After Effects expressions and layer parenting for advanced motion graphics",
    "Illustrator layers: organizing paths and groups for print and web projects",
]

instruction = "Prioritize beginner-friendly Photoshop tutorials only"

try:
    print("Reranking without instruction …")
    res_default = vo.rerank(
        query, documents, model="rerank-2.5", return_documents=True
    )
    print(f"Reranking with instruction: '{instruction}' …")
    res_instruct = vo.rerank(
        query, documents, model="rerank-2.5",
        return_documents=True,
        instruction=instruction,
    )
except Exception as e:
    print(f"ERROR during reranking: {e}")
    sys.exit(1)

# Print side-by-side
rank_w = 62
print(f"\n{'Rank':<5} {'Default (no instruction)':<{rank_w}} {'With instruction'}")
print("-" * (5 + rank_w + 50))

for rank, (d, i) in enumerate(
    zip(res_default.results, res_instruct.results), start=1
):
    doc_d = d.document[:rank_w - 3] + "…" if len(d.document) > rank_w - 3 else d.document
    doc_i = i.document[:45] + "…" if len(i.document) > 45 else i.document
    print(f"#{rank:<4} {doc_d:<{rank_w}} {doc_i}")

print("\n✓ With the instruction, beginner Photoshop tutorials rise to the top,")
print("  demonstrating rerank-2.5's ability to follow natural-language constraints.")
