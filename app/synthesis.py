import os
import requests

API_KEY = os.getenv("GEMINI_API_KEY")

def synthesize_answer(query, context_chunks):
    if not API_KEY:
        return "Error: Gemini API key not set. Run 'export GEMINI_API_KEY=your_key_here' first."

    # Combine all retrieved text chunks into one context
    context_text = "\n\n".join([c["text"] for c in context_chunks])

    prompt = f"""
You are a helpful assistant.
Answer the question using only the context below.

Context:
{context_text}

Question: {query}

Answer:
"""

    # --- Using Gemini 2.5 Flash ---
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={API_KEY}"

    headers = {"Content-Type": "application/json"}
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=body)
        data = response.json()

        if "candidates" in data:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        elif "error" in data:
            return f"Error from Gemini API: {data['error'].get('message', 'unknown error')}"
        else:
            return f"Unexpected response: {data}"
    except Exception as e:
        return f"Error contacting Gemini API: {str(e)}"
