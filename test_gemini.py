import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")

text = "enakku food romba pudichi irundhuchi"

prompt = f"""
Analyze the sentiment and emotion of the following text (which might be in Tanglish - Tamil in Latin script).
Text: "{text}"
Return the result STRICTLY as a JSON object with these keys:
- sentiment: Choose from ['Positive', 'Negative', 'Mixed_feelings', 'unknown_state', 'not-Tamil']
- emotion: Choose from ['joy', 'surprise', 'love', 'disgust', 'neutral', 'anger', 'sadness', 'fear']
JSON:
"""

response = model.generate_content(
    prompt,
    generation_config=genai.types.GenerationConfig(response_mime_type="application/json")
)

print(response.text)
