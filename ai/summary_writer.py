# summary_writer.py
import os
from click import prompt
import google.generativeai as genai
from gemini_client import get_text_model
from prompts import SUMMARY_PROMPT

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def generate_summary(values, who, risk):
    from gemini_client import get_text_model

    model = get_text_model()

    return model.generate_content(
        SUMMARY_PROMPT.format(values=values, who=who, risk=risk)
    ).text
