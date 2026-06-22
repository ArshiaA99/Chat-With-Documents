import os
from groq import Groq

groq_client = Groq()

def ask_llm(question: str, context: str) -> str:
    """
    Submits a context-injected payload to Llama 3.3 to answer user questions
    with deterministic constraints to mitigate hallucinations.
    """
    system_prompt = (
        "You are a retrieval-based corporate assistant.\n"
        "Use ONLY the provided user context blocks to construct your answer.\n"
        "If the answer cannot be found inside the context payload, respond exactly with:\n"
        "\"I don't know based on the provided context.\""
    )

    user_payload = f"Context:\n{context}\n\nQuestion: {question}"

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload}
        ],
        temperature=0.0,
        max_tokens=1024
    )
    return response.choices[0].message.content