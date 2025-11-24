import os
from dotenv import load_dotenv
from openai import OpenAI

# .env yükle
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY .env içinde tanımlı değil!")

client = OpenAI(api_key=OPENAI_API_KEY)