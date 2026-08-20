#!/usr/bin/env python3
"""
index.py — hybrid retrieval over the knowledge base.

Two retrievers, fused with Reciprocal Rank Fusion:

  BM25    lexical. Carries most of the weight on regulatory text, which is
          dense with exact terms and control identifiers.
  Dense   semantic. Catches paraphrase, which BM25 cannot.

BM25 is implemented here rather than pulled from a library, so the tokenizer
can be controlled: standard tokenizers destroy identifiers like AC-7,
SA-15(13) and 3.3.5, which are exactly the tokens that matter most in this
corpus.

Embeddings are cached next to the index and keyed to the model name and the
chunk set, so a Colab session that reconnects does not re-embed.

Subcommands
-----------
  build    build BM25 + dense index from processed/chunks.jsonl
  query    retrieve for one question
  stats    corpus statistics
  eval     retrieve for a file of questions (one per line)

  python index.py build
  python index.py query "who must approve the cyber security policy"
  python index.py query "multi-factor authentication" --doc SAU-SAMA-CSF-2017
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

KB = Path(__file__).resolve().parent
CHUNKS = KB / "processed" / "chunks.jsonl"
INDEX = KB / "index"
EMB_NPY = INDEX / "dense.npy"
EMB_META = INDEX / "dense_meta.json"

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
# bge models are trained with an asymmetric prefix on the query side only.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# --------------------------------------------------------------------------
# tokenisation
# --------------------------------------------------------------------------
# Order matters: identifier patterns are tried before the generic word rule so
# that "AC-7" and "3.3.5" survive as single tokens instead of fragmenting.
TOKEN_RE = re.compile(
    r"""
    [A-Za-z]{2}-\d{1,2}\(\d{1,2}\)(?:\(\d{1,2}\))?   # AC-2(3), SA-15(13)
  | [A-Za-z]{2}-\d{1,2}                              # AC-7, SR-11
  | \d+(?:\.\d+){1,3}[a-z]?                          # 3.3.5, 3.3.12-4.a
  | [A-Za-z][A-Za-z0-9']+                            # ordinary words
    """,
    re.VERBOSE,
)

# Words too frequent in this corpus to discriminate. Deliberately short: an
# aggressive stoplist removes terms that matter in regulatory phrasing
# ("should", "must", "may" carry deontic force and are kept).
STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "at", "by",
    "with", "as", "is", "are", "be", "been", "was", "were", "this", "that",
    "these", "those", "it", "its", "which", "from", "such", "any", "all",
}


def tokenize(text: str) -> List[str]:
    out = []
    for tok in TOKEN_RE.findall(text):
        low = tok.lower()
        if low in STOP or len(low) < 2:
            continue
        out.append(low)
    return out


# --------------------------------------------------------------------------
# BM25
# --------------------------------------------------------------------------
class BM25:
    """Okapi BM25. k1 and b at their standard defaults."""

    def __init__(self, corpus_tokens: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n_docs = len(corpus_tokens)
        self.doc_len = np.array([len(d) for d in corpus_tokens], dtype=np.float32)
        self.avg_len = float(self.doc_len.mean()) if self.n_docs else 0.0

        # term -> {doc_index: term_frequency}
        self.postings: Dict[str, Dict[int, int]] = defaultdict(dict)
        for i, toks in enumerate(corpus_tokens):
            for term, tf in Counter(toks).items():
                self.postings[term][i] = tf

        self.idf: Dict[str, float] = {}
        for term, docs in self.postings.items():
            df = len(docs)
            # Robertson/Sparck-Jones idf, floored so common terms cannot go
            # negative and subtract from a document's score.
            self.idf[term] = max(
                math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0), 1e-6
            )

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n_docs, dtype=np.float32)
        for term in tokenize(query):
            docs = self.postings.get(term)
            if not docs:
                continue
            idf = self.idf[term]
            for i, tf in docs.items():
                dl = self.doc_len[i]
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / self.avg_len)
                out[i] += idf * (tf * (self.k1 + 1.0)) / denom
        return out

    def top(self, query: str, k: int) -> List[Tuple[int, float]]:
        s = self.scores(query)
        if not s.any():
            return []
        idx = np.argsort(-s)[: k * 4]
        return [(int(i), float(s[i])) for i in idx if s[i] > 0][:k]


# --------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------
def load_chunks(path: Path = CHUNKS) -> List[Dict[str, Any]]:
    if not path.exists():
        sys.exit(f"ERROR: {path} not found. Run `python fetch.py chunk` first.")
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if not rows:
        sys.exit("ERROR: chunks.jsonl is empty.")
    return rows


def _chunk_fingerprint(chunks: Sequence[Dict[str, Any]]) -> str:
    import hashlib

    h = hashlib.sha256()
    for c in chunks:
        h.update(c["chunk_id"].encode())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------
# dense
# --------------------------------------------------------------------------
def _load_model(name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit(
            "sentence-transformers is not installed:\n"
            "  pip install -q sentence-transformers"
        )
    return SentenceTransformer(name)


def build_dense(chunks, model_name: str = DEFAULT_MODEL, force: bool = False) -> Optional[np.ndarray]:
    INDEX.mkdir(parents=True, exist_ok=True)
    fp = _chunk_fingerprint(chunks)

    if EMB_NPY.exists() and EMB_META.exists() and not force:
        meta = json.loads(EMB_META.read_text())
        if meta.get("model") == model_name and meta.get("fingerprint") == fp:
            emb = np.load(EMB_NPY)
            if emb.shape[0] == len(chunks):
                print(f"  dense: reusing cached embeddings {emb.shape}")
                return emb
        print("  dense: cache is stale (chunks or model changed) — rebuilding")

    model = _load_model(model_name)
    texts = [c["text"] for c in chunks]
    print(f"  dense: encoding {len(texts)} chunks with {model_name}")
    emb = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,   # so dot product == cosine
        convert_to_numpy=True,
    ).astype(np.float32)

    np.save(EMB_NPY, emb)
    EMB_META.write_text(
        json.dumps(
            {"model": model_name, "fingerprint": fp, "n": len(chunks), "dim": int(emb.shape[1])},
            indent=2,
        )
    )
    print(f"  dense: saved {emb.shape} -> {EMB_NPY.relative_to(KB)}")
    return emb


# --------------------------------------------------------------------------
# fusion
# --------------------------------------------------------------------------
def rrf(*ranked_lists: Sequence[Tuple[int, float]], k: int = 60) -> List[Tuple[int, float]]:
    """Reciprocal Rank Fusion.

    Combines rankings by position, not by score, so a lexical score and a
    cosine similarity can be merged without calibrating one against the other.
    """
    fused: Dict[int, float] = defaultdict(float)
    for lst in ranked_lists:
        for rank, (idx, _score) in enumerate(lst, start=1):
            fused[idx] += 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: -kv[1])


# --------------------------------------------------------------------------
# retriever
# --------------------------------------------------------------------------
class Retriever:
    def __init__(self, model_name: str = DEFAULT_MODEL, dense: bool = True):
        self.chunks = load_chunks()
        self.model_name = model_name
        print(f"Indexing {len(self.chunks)} chunks")

        self.tokens = [tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25(self.tokens)
        print(f"  bm25 : {len(self.bm25.postings):,} unique terms, "
              f"avg {self.bm25.avg_len:.0f} tokens/chunk")

        self.emb = build_dense(self.chunks, model_name) if dense else None
        self._model = None

    # -- dense query encoding is lazy: BM25-only searches never load the model
    def _encode_query(self, q: str) -> np.ndarray:
        if self._model is None:
            self._model = _load_model(self.model_name)
        v = self._model.encode(
            [QUERY_PREFIX + q], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        return v.astype(np.float32)

    def search(
        self,
        query: str,
        k: int = 8,
        doc_id: Optional[str] = None,
        status: Optional[str] = None,
        pool: int = 40,
    ) -> List[Dict[str, Any]]:
        """Return k chunks, each carrying its citation anchor and provenance."""
        lex = self.bm25.top(query, pool)

        den: List[Tuple[int, float]] = []
        if self.emb is not None:
            qv = self._encode_query(query)
            sims = self.emb @ qv
            top = np.argsort(-sims)[:pool]
            den = [(int(i), float(sims[i])) for i in top]

        fused = rrf(lex, den) if den else [(i, s) for i, s in lex]

        lex_rank = {i: r for r, (i, _) in enumerate(lex, 1)}
        den_rank = {i: r for r, (i, _) in enumerate(den, 1)}

        out: List[Dict[str, Any]] = []
        for idx, score in fused:
            c = self.chunks[idx]
            if doc_id and c["doc_id"] != doc_id:
                continue
            if status and c.get("status") != status:
                continue
            out.append(
                {
                    **c,
                    "score": round(score, 6),
                    "bm25_rank": lex_rank.get(idx),
                    "dense_rank": den_rank.get(idx),
                }
            )
            if len(out) >= k:
                break
        return out


# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------
def show(hits: List[Dict[str, Any]], width: int = 300) -> None:
    if not hits:
        print("  no results")
        return
    for n, h in enumerate(hits, 1):
        both = "both" if h["bm25_rank"] and h["dense_rank"] else (
            "bm25" if h["bm25_rank"] else "dense"
        )
        print(f"\n[{n}] {h['chunk_id']}   ({both})")
        print(f"    {h.get('title_en','')} — {h.get('issuing_body','')}"
              f" [{h.get('status','')}]  p{h['page']}")
        body = " ".join(h["text"].split())
        print(f"    {body[:width]}{'…' if len(body) > width else ''}")


# --------------------------------------------------------------------------
def cmd_build(args) -> int:
    Retriever(args.model, dense=not args.no_dense)
    print("\nIndex ready.")
    return 0


def cmd_query(args) -> int:
    r = Retriever(args.model, dense=not args.no_dense)
    print(f"\nQ: {args.question}")
    show(r.search(args.question, k=args.k, doc_id=args.doc, status=args.status))
    return 0


def cmd_eval(args) -> int:
    qs = [l.strip() for l in Path(args.file).read_text(encoding="utf-8").splitlines() if l.strip()]
    r = Retriever(args.model, dense=not args.no_dense)
    for q in qs:
        print("\n" + "=" * 70)
        print(f"Q: {q}")
        show(r.search(q, k=args.k), width=180)
    return 0


def cmd_stats(args) -> int:
    chunks = load_chunks()
    print(f"chunks   : {len(chunks)}")
    per_doc = Counter(c["doc_id"] for c in chunks)
    print("\nby document")
    for d, n in per_doc.most_common():
        print(f"  {d:<28} {n:>4}")
    per_status = Counter(c.get("status", "—") for c in chunks)
    print("\nby legal status")
    for s, n in per_status.most_common():
        print(f"  {s:<28} {n:>4}")
    lens = [len(c["text"]) for c in chunks]
    print(f"\nchunk chars  min {min(lens)}  mean {sum(lens)//len(lens)}  max {max(lens)}")
    toks = [len(tokenize(c["text"])) for c in chunks]
    print(f"chunk tokens min {min(toks)}  mean {sum(toks)//len(toks)}  max {max(toks)}")

    ids = re.compile(r"[A-Za-z]{2}-\d{1,2}(?:\(\d{1,2}\))?")
    hits = Counter()
    for c in chunks:
        for m in ids.findall(c["text"]):
            hits[m.upper()] += 1
    if hits:
        print(f"\ncontrol-like identifiers found in text: {len(hits)} distinct")
        print("  " + ", ".join(f"{k}({v})" for k, v in hits.most_common(12)))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--no-dense", action="store_true", help="BM25 only (no model download)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build", help="build the index")

    p = sub.add_parser("query", help="retrieve for one question")
    p.add_argument("question")
    p.add_argument("-k", type=int, default=8)
    p.add_argument("--doc", help="restrict to one doc_id")
    p.add_argument("--status", choices=["binding", "advisory"], help="restrict by legal status")

    p = sub.add_parser("eval", help="retrieve for a file of questions")
    p.add_argument("file")
    p.add_argument("-k", type=int, default=5)

    sub.add_parser("stats", help="corpus statistics")

    args = ap.parse_args()
    return {"build": cmd_build, "query": cmd_query, "eval": cmd_eval, "stats": cmd_stats}[
        args.cmd
    ](args)


if __name__ == "__main__":
    sys.exit(main())
