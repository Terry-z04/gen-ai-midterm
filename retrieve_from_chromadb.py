#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import sys
from typing import Optional, List

# --- deps ---
try:
    import chromadb
    from chromadb.config import Settings
except Exception as e:
    sys.exit("chromadb not installed. Run: pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    sys.exit("sentence-transformers not installed. Run: pip install sentence-transformers")


# --- config aligned with your loader (load_to_chromadb.py) ---
DEFAULT_DB_PATH = "chroma_db"
DEFAULT_COLLECTION = "uchicago_msads_docs"   # same as loader
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # same family as loader


def get_client(db_path: str):
    if not os.path.isdir(db_path):
        raise FileNotFoundError(
            f"Chroma DB folder not found at '{db_path}'. "
            f"Expected a directory created by your loader."
        )
    return chromadb.PersistentClient(path=db_path, settings=Settings(allow_reset=False))


def list_collections(db_path: str):
    client = get_client(db_path)
    cols = client.list_collections()
    if not cols:
        print("No collections found.")
        return
    print("Available collections:")
    for c in cols:
        print(" -", c.name)


def get_collection(client, name: Optional[str]):
    cols = client.list_collections()
    if not cols:
        raise RuntimeError("No collections found in this Chroma DB. Did you run the loader?")
    if name:
        for c in cols:
            if c.name == name:
                return client.get_collection(name=name)
        raise RuntimeError(f"Collection '{name}' not found. Available: {[c.name for c in cols]}")
    # default to your collection if present; otherwise first
    names = [c.name for c in cols]
    if DEFAULT_COLLECTION in names:
        return client.get_collection(name=DEFAULT_COLLECTION)
    return client.get_collection(name=cols[0].name)


def load_st_model(name: str) -> SentenceTransformer:
    try:
        return SentenceTransformer(name)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load embedding model '{name}'. "
            f"Ensure the name is correct and dependencies are installed. Details: {e}"
        )


def embed_query(model: SentenceTransformer, text: str) -> List[float]:
    emb = model.encode([text], normalize_embeddings=True)
    return emb[0].tolist()


'''
def pretty_print(query: str, docs, metas, ids, dists, max_chars: int = 900):
    print("\n=== QUERY ===")
    print(query)
    print("\n=== TOP MATCHES ===")
    if not docs:
        print("(no results)")
        return
    for i, (doc, meta, _id, dist) in enumerate(zip(docs, metas, ids, dists), start=1):
        print(f"\n[{i}] id={_id}  distance={dist:.4f}")
        # Your loader stores: title, url, parent_url, depth, doc_type
        if isinstance(meta, dict):
            show = {k: meta.get(k, "") for k in ("title", "url", "parent_url", "depth", "doc_type")}
            print("metadata:", show)
        snippet = (doc or "")[:max_chars]
        if doc and len(doc) > max_chars:
            snippet += " ..."
        print(snippet)
'''

def pretty_print(query: str, docs, metas, ids, dists, max_chars: int = 900):
    print("\n=== QUERY ===")
    print(query)
    print("\n=== ANSWERS ===")

    if not docs:
        print("(no results)")
        return

    for i, (doc, meta) in enumerate(zip(docs, metas), 1):
        snippet = (doc or "").strip()
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars] + "..."

        # Optional: show clean URL source if available
        source = meta.get("url") if isinstance(meta, dict) and "url" in meta else None
        if source:
            print(f"\n{i}. {snippet}\n(Source: {source})\n")
            
        else:
            print(f"\n{i}. {snippet}\n")    
    


def retrieve(
    question: str,
    db_path: str = DEFAULT_DB_PATH,
    collection: Optional[str] = None,
    model_name: str = DEFAULT_MODEL,
    top_k: int = 3,
    filter_url_contains: Optional[str] = None,
    max_chars: int = 900,
):
    """
    Run a semantic search over the existing Chroma DB and pretty-print results.
    Returns a compact payload useful for an LLM step.
    """
    # Connect + pick collection + embedding model
    client = get_client(db_path)
    col = get_collection(client, collection)
    model = load_st_model(model_name)

    # Embed the query
    q_emb = embed_query(model, question)

    # Over-fetch so we can filter and still return top_k
    n_fetch = max(top_k * 3, top_k)
    out = col.query(
        query_embeddings=[q_emb],
        n_results=n_fetch,
        # NOTE: no "ids" here to avoid version-specific errors
        include=["documents", "metadatas", "distances"],
    )

    docs = out.get("documents", [[]])[0]
    metas = out.get("metadatas", [[]])[0]
    dists = out.get("distances", [[]])[0]

    # Optional filter by URL substring (metadata is expected to contain 'url')
    if filter_url_contains:
        keep_docs, keep_metas, keep_dists = [], [], []
        needle = filter_url_contains.lower()
        for d, m, s in zip(docs, metas, dists):
            url = (m.get("url") if isinstance(m, dict) else "") or ""
            if needle in url.lower():
                keep_docs.append(d)
                keep_metas.append(m)
                keep_dists.append(s)
        if keep_docs:  # only replace if filter keeps something
            docs, metas, dists = keep_docs, keep_metas, keep_dists

    # Trim to top_k after filtering
    docs, metas, dists = docs[:top_k], metas[:top_k], dists[:top_k]

    # Pretty print to console
    pretty_print(question, docs, metas, list(range(1, len(docs) + 1)), dists, max_chars=max_chars)

    # Return a structured payload (handy for downstream LLM use)
    contexts = []
    for d, m in zip(docs, metas):
        src = (m.get("url") if isinstance(m, dict) else "") or ""
        contexts.append({"text": d, "source": src, "metadata": m})

    return {
        "query": question,
        "top_k": top_k,
        "results": contexts,
    }



def main():
    ap = argparse.ArgumentParser(description="Semantic retrieval over local Chroma DB")
    ap.add_argument("--db_path", default=DEFAULT_DB_PATH, help="Path to Chroma DB (default: ./chroma_db)")
    ap.add_argument("--collection", default=None, help=f"Collection name (default: {DEFAULT_COLLECTION})")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Embedding model for the query")
    ap.add_argument("--question", default=None, help="Your query/question")
    ap.add_argument("--top_k", type=int, default=3, help="Number of passages to return")
    ap.add_argument("--filter-url-contains", default=None, help="Only keep hits whose URL contains this substring")
    ap.add_argument("--max_chars", type=int, default=900, help="Max characters to print per passage")
    ap.add_argument("--list", action="store_true", help="List available collections and exit")
    args = ap.parse_args()

    if args.list:
        list_collections(args.db_path)
        return

    if not args.question:
        # Interactive mode
        client = get_client(args.db_path)
        col = get_collection(client, args.collection)
        model = load_st_model(args.model)
        print(f"Interactive mode. Using collection: '{col.name}'. Press Enter on empty line to exit.")
        while True:
            try:
                q = input("\nEnter a question: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break

            payload = retrieve(
                q,
                db_path=args.db_path,
                collection=args.collection,
                model_name=args.model,
                top_k=args.top_k,
                filter_url_contains=args.filter_url_contains,
                max_chars=args.max_chars,
            )
        return

    # One-shot mode
    retrieve(
        args.question,
        db_path=args.db_path,
        collection=args.collection,
        model_name=args.model,
        top_k=args.top_k,
        filter_url_contains=args.filter_url_contains,
        max_chars=args.max_chars,
    )


if __name__ == "__main__":
    main()
