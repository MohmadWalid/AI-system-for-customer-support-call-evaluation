import sys, os
BASE = os.path.dirname(os.path.abspath(__file__))
try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    sys.path.insert(0, BASE)
    from config import EMBEDDING_MODEL, FAISS_INDEX_PATH, POLICY_MAP_PATH, SIMILARITY_THRESHOLD, TOP_K
    from data.transcripts import TRANSCRIPTS
    import json

    model = SentenceTransformer(EMBEDDING_MODEL)
    index_path = os.path.join(BASE, FAISS_INDEX_PATH)
    map_path   = os.path.join(BASE, POLICY_MAP_PATH)
    index = faiss.read_index(index_path)
    with open(map_path) as f:
        policy_map = json.load(f)

    lines = [
        f"Model: {EMBEDDING_MODEL}",
        f"Threshold: {SIMILARITY_THRESHOLD}  TOP_K: {TOP_K}",
        f"Policies: {len(policy_map)}",
        "",
    ]

    all_best = []
    for t in TRANSCRIPTS:
        lines.append(f"{'─'*60}")
        lines.append(f"  {t['call_id']} ({t['agent']})")
        lines.append(f"{'─'*60}")
        for u in t["utterances"]:
            if u["speaker"] != "agent":
                continue
            text = u["text"]
            emb = model.encode([text], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
            scores, idxs = index.search(emb, TOP_K)
            scores = scores[0]; idxs = idxs[0]
            best = float(scores[0])
            all_best.append(best)
            verdict = "HIGH" if best >= SIMILARITY_THRESHOLD else "low"
            lines.append(f"\n  Q: {text[:90]}")
            for r, (s, i) in enumerate(zip(scores, idxs), 1):
                if 0 <= i < len(policy_map):
                    lines.append(f"    [{r}] {s:.4f}  [{policy_map[i]['policy_id']}] {policy_map[i]['policy_name']}")
            lines.append(f"    => best={best:.4f} threshold={SIMILARITY_THRESHOLD} -> {verdict}")
        lines.append("")

    arr = np.array(all_best)
    sug = max(0.0, round(float(arr.min()) - 0.05, 2))
    lines += [
        "═"*60,
        f"  Min={arr.min():.4f}  Max={arr.max():.4f}  Mean={arr.mean():.4f}  Median={np.median(arr):.4f}",
        f"  HIGH: {int((arr>=SIMILARITY_THRESHOLD).sum())}/{len(arr)} at threshold={SIMILARITY_THRESHOLD}",
        f"  Suggested: SIMILARITY_THRESHOLD={sug:.2f}",
        "═"*60,
    ]

    out = os.path.join(BASE, "calibrate_results.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

except Exception as e:
    import traceback
    out = os.path.join(BASE, "calibrate_results.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"ERROR: {e}\n\n{traceback.format_exc()}")
