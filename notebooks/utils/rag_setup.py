# General setup for rag experiments

def query_db(question: str, collection, embedder, rag_type, top_k: int = 5) -> list[dict]:
    """Return the top_k most similar classes with their codes and distances."""
    query_embedding = embedder.encode(question).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        if rag_type == "standard":
            hits.append({
                "code":        meta["code"],
                "description": meta["name"] + " " + meta["notes"],
                "score":       round(1 - dist, 4),  # cosine distance → similarity
            })
        else:
            hits.append({
            "nace_21_code":     meta.get("nace_21_code"),
            "company_name":     meta.get("company_name"),
            "company_activity": meta.get("company_activity"),
            "company_purpose":  meta.get("company_purpose","")
        })

    return hits


def format_few_shot_examples(obs_hits: list[dict]) -> str:
    examples = []
    for i, h in enumerate(obs_hits):
        example = (
            f"Example {i+1}:\n"
            f"  Name: {h['company_name']}\n"
            f"  Activity: {h['company_activity']}\n"
            f"  Purpose: {h['company_purpose']}\n"
            f"  Code: {h['nace_21_code']}"
        )
        examples.append(example)
    return "\n\n".join(examples)