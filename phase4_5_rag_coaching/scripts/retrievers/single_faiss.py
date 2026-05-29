"""
Retriever that queries the single global FAISS index.
"""
import numpy as np
from config import SIMILARITY_THRESHOLD, TOP_K

def retrieve(query: str, embedder, index, policy_map: list) -> str:
    """
    Retrieves top K semantically similar policy rules from the single global index based on a unified query.
    
    Args:
        query (str): The search query composed of predicted label and customer utterances.
        embedder: Loaded SentenceTransformer model.
        index: Loaded FAISS index from disk.
        policy_map (list): Source rules mapped to their index positions.

    Returns:
        str: Joined string of retrieved policies or a fallback message if none.
    """
    vec = embedder.encode([query], normalize_embeddings=True)
    vec = np.array(vec, dtype=np.float32)

    k_actual = min(TOP_K, index.ntotal)
    scores, indices = index.search(vec, k_actual)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        if float(score) < SIMILARITY_THRESHOLD:
            continue
        entry = policy_map[idx]
        results.append({
            "rule":   entry["rule"],
            "source": entry["source"],
            "score":  round(float(score), 4),
        })
        
    if results:
        return "\n".join(f"- {r['source']}" for r in results)
    else:
        return "(no relevant policies retrieved above threshold)"
