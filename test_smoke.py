import os
# Force legacy keras for compatibility as per project standards
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
print(f"TensorFlow Version: {tf.__version__}")

try:
    import tf_keras as keras
    from tf_keras.models import Model
    from tf_keras.layers import Input, Dense, LSTM, Conv1D, Dropout, Concatenate, Bidirectional
    print("Using tf_keras")
except ImportError:
    import tensorflow.keras as keras
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, Dropout, Concatenate, Bidirectional
    print("Using tensorflow.keras")

from transformers import AutoTokenizer, TFAutoModel
import pandas as pd
import numpy as np
import pickle

# Paths
MODELS_DIR = r"d:\PROJECT\Final_Project1 - Copy\models_"
DATASET_PATH = r"d:\PROJECT\Final_Project1 - Copy\data\corrected_tanglish_dataset.csv"
MAX_LEN = 64

XLM_MODEL_NAME = "xlm-roberta-base" 
MINILM_MODEL_NAME = "microsoft/Multilingual-MiniLM-L12-H384"
INDIC_MODEL_NAME = "google/muril-base-cased"

def build_model(num_sentiment_classes, num_emotion_classes):
    # Inputs
    xlm_ids = Input(shape=(MAX_LEN,), dtype=tf.int32, name="xlm_ids")
    xlm_mask = Input(shape=(MAX_LEN,), dtype=tf.int32, name="xlm_mask")
    minilm_ids = Input(shape=(MAX_LEN,), dtype=tf.int32, name="minilm_ids")
    minilm_mask = Input(shape=(MAX_LEN,), dtype=tf.int32, name="minilm_mask")
    indic_ids = Input(shape=(MAX_LEN,), dtype=tf.int32, name="indic_ids")
    indic_mask = Input(shape=(MAX_LEN,), dtype=tf.int32, name="indic_mask")

    # Load frozen Transformers (using from_pt=True as per notebook logic)
    # Note: We use a small timeout or just try-except here
    print("Loading Transformers (Architecture Only)...")
    xlm_transformer = TFAutoModel.from_pretrained(XLM_MODEL_NAME, from_pt=True)
    xlm_transformer.trainable = False
    xlm_out = xlm_transformer(xlm_ids, attention_mask=xlm_mask)[0] 

    minilm_transformer = TFAutoModel.from_pretrained(MINILM_MODEL_NAME, from_pt=True)
    minilm_transformer.trainable = False
    minilm_out = minilm_transformer(minilm_ids, attention_mask=minilm_mask)[0]

    indic_transformer = TFAutoModel.from_pretrained(INDIC_MODEL_NAME, from_pt=True)
    indic_transformer.trainable = False
    indic_out = indic_transformer(indic_ids, attention_mask=indic_mask)[0]

    # Combine representations
    combined = Concatenate(axis=-1)([xlm_out, minilm_out, indic_out]) 

    # CNN
    cnn_out = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(combined)
    cnn_out = Dropout(0.3)(cnn_out)

    # LSTM
    lstm_out = Bidirectional(LSTM(128, return_sequences=False))(cnn_out)
    lstm_out = Dropout(0.3)(lstm_out)

    # Dense Heads
    dense_out = Dense(128, activation='relu')(lstm_out)
    dense_out = Dropout(0.2)(dense_out)

    sentiment_output = Dense(num_sentiment_classes, activation='softmax', dtype='float32', name='sentiment')(dense_out)
    emotion_output = Dense(num_emotion_classes, activation='softmax', dtype='float32', name='emotion')(dense_out)

    model = Model(inputs=[xlm_ids, xlm_mask, minilm_ids, minilm_mask, indic_ids, indic_mask], 
                  outputs=[sentiment_output, emotion_output])
    return model

def smoke_test():
    print("\n--- SMOKE TEST START ---")
    
    # 1. Dataset Check
    print(f"Checking dataset at {DATASET_PATH}...")
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        print(f"Dataset found. Rows: {len(df)}")
        if 'category' in df.columns:
            print(f"'category' column found (will be used as sentiment).")
        else:
            print(f"'category' column NOT found!")
    else:
        print(f"Dataset NOT found at {DATASET_PATH}")
        return

    # 2. Tokenizer Check
    print("Checking Tokenizers...")
    try:
        xlm_tok = AutoTokenizer.from_pretrained(XLM_MODEL_NAME)
        print(f"{XLM_MODEL_NAME} tokenizer loaded.")
    except Exception as e:
        print(f"Tokenizer error: {e}")
        return

    # 3. Model Architecture Check
    print("Building Model Architecture (Testing logic)...")
    try:
        # Assuming 5 sentiment classes and 8 emotion classes based on my previous check
        test_model = build_model(5, 8)
        print("Model architecture built successfully.")
    except Exception as e:
        print(f"Model Build failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Weights Check (Optional)
    model_path = os.path.join(MODELS_DIR, "tri_hybrid_sentiment_emotion_model.keras")
    if os.path.exists(model_path):
        print(f"Pre-trained model found at {model_path} ({os.path.getsize(model_path)/1e9:.2f} GB)")
    else:
        print(f"Pre-trained model NOT found at {model_path}. You will need to train first.")

    print("\n--- SMOKE TEST PASSED ---")
    print("The system is ready for training or prediction.")

if __name__ == "__main__":
    smoke_test()
