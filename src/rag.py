from typing import Tuple

def retrieve_context(collection, question: str) -> Tuple[str, str]:
    """Unified context retrieval routine from Vector Database."""
    results = collection.query(
        query_texts=[question],
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )

    context_strings = []
    source_citations = []

    if results["documents"] and results["documents"][0]:
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for i in range(len(documents)):
            text = documents[i]
            meta = metadatas[i] if metadatas[i] else {}
            dist = distances[i]

            citation_score = f"Cosine Distance: {dist:.4f}"
            source_file = meta.get("source", "Unknown Document")
            page_num = meta.get("page", None)
            page_str = f" (Page {page_num + 1})" if page_num is not None else ""
            chunk_idx = meta.get("chunk_index", "N/A")

            citation = f"- {source_file}{page_str} [Chunk Index: {chunk_idx}] [{citation_score}]"
            source_citations.append(citation)
            context_strings.append(f"[{source_file}{page_str} | Chunk {chunk_idx}]:\n{text}")

    context = "\n\n".join(context_strings)
    unique_citations = list(dict.fromkeys(source_citations))
    sources_output = "\n".join(unique_citations) if unique_citations else "None"

    return context, sources_output