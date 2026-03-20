import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
import sys
import os
import pickle
import numpy as np
import re
from transformers import AutoTokenizer
from tf_keras.models import load_model

# Constants
MAX_LEN = 64
XLM_MODEL_NAME = "xlm-roberta-base" 
MINILM_MODEL_NAME = "microsoft/Multilingual-MiniLM-L12-H384"
INDIC_MODEL_NAME = "google/muril-base-cased"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models_")

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def test_local():
    # Load label encoders
    with open(os.path.join(MODELS_DIR, "sentiment_label_encoder.pkl"), "rb") as f:
        le_sentiment = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "emotion_label_encoder.pkl"), "rb") as f:
        le_emotion = pickle.load(f)
    
    # Load tokenizers
    xlm_tokenizer = AutoTokenizer.from_pretrained(XLM_MODEL_NAME)
    minilm_tokenizer = AutoTokenizer.from_pretrained(MINILM_MODEL_NAME)
    indic_tokenizer = AutoTokenizer.from_pretrained(INDIC_MODEL_NAME)

    # Load Model
    model_path = os.path.join(MODELS_DIR, "tri_hybrid_sentiment_emotion_model.keras")
    model = load_model(model_path, compile=False)

    text = "enakku food romba pudichi irundhuchi"
    processed_text = preprocess_text(text)
    
    xlm_enc = xlm_tokenizer(processed_text, padding='max_length', max_length=MAX_LEN, truncation=True, return_tensors="tf")
    minilm_enc = minilm_tokenizer(processed_text, padding='max_length', max_length=MAX_LEN, truncation=True, return_tensors="tf")
    indic_enc = indic_tokenizer(processed_text, padding='max_length', max_length=MAX_LEN, truncation=True, return_tensors="tf")
    
    preds = model.predict({
        'xlm_ids': xlm_enc['input_ids'],
        'xlm_mask': xlm_enc['attention_mask'],
        'minilm_ids': minilm_enc['input_ids'],
        'minilm_mask': minilm_enc['attention_mask'],
        'indic_ids': indic_enc['input_ids'],
        'indic_mask': indic_enc['attention_mask']
    }, verbose=0)
    
    sent_idx = np.argmax(preds[0][0])
    emo_idx = np.argmax(preds[1][0])
    
    print(f"Text: {text}")
    print(f"Sentiment: {le_sentiment.inverse_transform([sent_idx])[0]}")
    print(f"Emotion: {le_emotion.inverse_transform([emo_idx])[0]}")

if __name__ == "__main__":
    test_local()
