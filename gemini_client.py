# gemini_client.py
import os
import google.generativeai as genai

def get_text_model():
    """
    Returns a Gemini model that supports generateContent.
    Automatically handles model availability.
    """
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    preferred_models = [
        "gemini-2.5-flash",
        "gemini-3-flash-preview",
        "gemini-pro",
    ]

    # Try preferred names first
    for name in preferred_models:
        try:
            model = genai.GenerativeModel(name)
            model.generate_content("ping")  # test call
            return model
        except Exception:
            pass

    # Fallback: find any model that supports generateContent
    for m in genai.list_models():
        if "generateContent" in getattr(m, "supported_generation_methods", []):
            return genai.GenerativeModel(m.name)

    raise RuntimeError(
        "No Gemini model available for generateContent. "
        "Check API key, billing, and enabled Gemini API."
    )
