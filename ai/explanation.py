# explanation.py
import os
from xml.parsers.expat import model
from click import prompt
import google.generativeai as genai
from gemini_client import get_text_model
from prompts import EXPLANATION_PROMPT

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def explain(values, who, risk):
    from gemini_client import get_text_model

    model = get_text_model()
    
    return model.generate_content(
        EXPLANATION_PROMPT.format(values=values, who=who, risk=risk)
    ).text
