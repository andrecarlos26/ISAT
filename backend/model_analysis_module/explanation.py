import os
import requests
import json

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"

BASE_PROMPT = """Below is information about the decision-making process of a Multi-Layer Perceptron (MLP) model, which attempts to predict how long it will take before system memory is exhausted.

Write a human-readable explanation of how the model arrives at its predictions. The explanation should be suitable for system administrators and include the following:

A high-level overview of the model’s decision-making process (e.g., how input features are processed and how predictions are generated)
Whether the model appears to be making correct predictions, based on available metrics
Any evidence of bias or systematic errors in the model’s behavior
Confidence level or uncertainty in its predictions
Strategic implications for system management or resource planning based on the model’s outputs

The explanation should avoid technical jargon where possible and focus on helping human operators make informed decisions using the model’s outputs."""

def get_openrouter_explanation(summary: str) -> str:
    """
    Sends the summary + base prompt to OpenRouter and returns the explanatory text.
    """
    prompt = f"{BASE_PROMPT}\n\n{summary}"
    resp = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json; charset=utf-8"
        },
        data=json.dumps({
            "model": "deepseek/deepseek-chat-v3-0324:free",
            "messages": [{"role": "user", "content": prompt}]
        })
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]
