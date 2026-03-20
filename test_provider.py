import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

PROVIDERS = [
    {
        "name": "Together AI",
        "url": "https://api.together.xyz/v1/chat/completions",
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    },
    {
        "name": "SiliconFlow",
        "url": "https://api.siliconflow.cn/v1/chat/completions",
        "model": "deepseek-ai/DeepSeek-V3"
    },
    {
        "name": "OpenRouter",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "meta-llama/llama-3.1-8b-instruct"
    }
]

def test_provider(provider):
    print(f"Testing {provider['name']}...")
    try:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": provider["model"],
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10
        }
        
        response = requests.post(provider["url"], headers=headers, json=payload, timeout=5)
        
        if response.status_code == 200:
            print(f"[SUCCESS] {provider['name']} accepted the key!")
            print(f"Response: {response.json()['choices'][0]['message']['content']}")
            return True
        else:
            print(f"[FAILED] {provider['name']} (Code {response.status_code}): {response.text[:100]}")
    except Exception as e:
        print(f"[ERROR] {provider['name']} - {str(e)}")
    return False

found = False
for p in PROVIDERS:
    if test_provider(p):
        found = True
        break

if not found:
    print("\n⚠️ None of the common providers accepted this key as their primary credential.")
