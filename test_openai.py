import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

text = "enakku food romba pudichi irundhuchi"

prompt = f"""
Analyze the sentiment and emotion of the following text (which might be in Tanglish - Tamil in Latin script).

Text: "{text}"

Return the result STRICTLY as a JSON object with these keys:
- sentiment: Choose from ['Positive', 'Negative', 'Mixed_feelings', 'unknown_state', 'not-Tamil']
- emotion: Choose from ['joy', 'surprise', 'love', 'disgust', 'neutral', 'anger', 'sadness', 'fear']

JSON:
"""

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a specialized linguistic analyst for Tanglish (Tamil-English) content."},
        {"role": "user", "content": prompt}
    ],
    response_format={ "type": "json_object" }
)

print(response.choices[0].message.content)
