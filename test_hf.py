import requests
import json

# Local Flask server endpoint
URL_PREDICT = "http://127.0.0.1:5000/predict"

def get_prediction(text):
    try:
        payload = {"text": text}
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(URL_PREDICT, json=payload, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Server returned error code {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}

def main():
    print("====================================================")
    print("      Tanglish Sentiment & Emotion Predictor")
    print("           (Powered by Local Flask Server)          ")
    print("====================================================")
    print("Typing messages in Tanglish/Tamil/English to predict.")
    print("Type 'exit' or 'quit' to stop.")
    
    while True:
        user_input = input("\nEnter Text: ").strip()
        
        if user_input.lower() in ['exit', 'quit', '']:
            print("Exiting...")
            break
            
        print("Analyzing...")
        result = get_prediction(user_input)
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            print("\n--- RESULTS ---")
            print(f"Sentiment: {result.get('sentiment')}")
            print(f"Confidence: {result.get('sentiment_confidence')}%")
            print(f"Emotion:   {result.get('emotion')}")
            print(f"Insight:   {result.get('decision')}")
            print("----------------")

if __name__ == "__main__":
    main()
