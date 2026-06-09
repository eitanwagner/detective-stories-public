import os


def get_client():
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return client

def get_anthropic_client():
    import anthropic
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )
    return client

def configure_gemini():
    import google.generativeai as genai
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

def get_google_client():
    from google import genai
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    return client
