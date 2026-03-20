import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["USE_TF"] = "1"

import tensorflow as tf
print("Importing tensorflow.keras...")

# Enable Mixed Precision to save memory
try:
    from tensorflow.keras import mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print("Mixed precision enabled.")
except Exception as e:
    print(f"Could not enable mixed precision: {e}")

try:
    import tf_keras as keras
    import tf_keras.backend as K
    print("Using tf_keras")
except ImportError:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
    print("Using tensorflow.keras")

# --------------------------------------------------------
# WORKAROUND for Keras 3 Transformers from_pt=True bug
# --------------------------------------------------------
if not hasattr(K, 'set_value'):
    print("Patching Keras backend set_value...")
    K.set_value = lambda v, val: v.assign(val)

try:
    import keras.backend as native_K
    if not hasattr(native_K, 'set_value'):
        native_K.set_value = lambda v, val: v.assign(val)
except Exception:
    pass
# --------------------------------------------------------

from tf_keras.models import Model
from tf_keras.layers import Input, Dense, LSTM, Conv1D, Dropout, Concatenate, GlobalMaxPooling1D, Bidirectional
print("Importing transformers...")
from transformers import AutoTokenizer, TFAutoModel
print("Importing other libs...")
import pickle
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import gc

# Constants
MAX_LEN = 64 
BATCH_SIZE = 8 # reduced batch size slightly if memory is a constraint with 3 transformers
XLM_MODEL_NAME = "xlm-roberta-base" 
MINILM_MODEL_NAME = "microsoft/Multilingual-MiniLM-L12-H384"
INDIC_MODEL_NAME = "google/muril-base-cased"
MODELS_DIR = "models_"
DATASET_PATH = r"D:\PROJECT\Final_Project1 - Copy\data\corrected_tanglish_dataset.csv"

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def configure_gpu_memory():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU Memory Growth Enabled for {len(gpus)} GPUs")
        except RuntimeError as e:
            print(e)
            
class TextDataGenerator(keras.utils.Sequence):
    def __init__(self, texts, sentiment_labels, emotion_labels, batch_size, xlm_tok, minilm_tok, indic_tok, max_len):
        self.texts = texts
        self.sentiment_labels = sentiment_labels
        self.emotion_labels = emotion_labels
        self.batch_size = batch_size
        self.xlm_tok = xlm_tok
        self.minilm_tok = minilm_tok
        self.indic_tok = indic_tok
        self.max_len = max_len
        self.indices = np.arange(len(self.texts))

    def __len__(self):
        return int(np.ceil(len(self.texts) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_texts = [self.texts[i] for i in batch_indices]
        
        xlm_enc = self.xlm_tok(batch_texts, padding='max_length', truncation=True, max_length=self.max_len, return_tensors="tf")
        minilm_enc = self.minilm_tok(batch_texts, padding='max_length', truncation=True, max_length=self.max_len, return_tensors="tf")
        indic_enc = self.indic_tok(batch_texts, padding='max_length', truncation=True, max_length=self.max_len, return_tensors="tf")
        
        inputs = {
            'xlm_ids': xlm_enc['input_ids'],
            'xlm_mask': xlm_enc['attention_mask'],
            'minilm_ids': minilm_enc['input_ids'],
            'minilm_mask': minilm_enc['attention_mask'],
            'indic_ids': indic_enc['input_ids'],
            'indic_mask': indic_enc['attention_mask']
        }
        
        targets = {
            'sentiment': self.sentiment_labels[batch_indices],
            'emotion': self.emotion_labels[batch_indices]
        }
        
        return inputs, targets
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)

def build_model(num_sentiment_classes, num_emotion_classes):
    # Inputs
    xlm_ids = Input(shape=(MAX_LEN,), dtype=tf.int32, name="xlm_ids")
    xlm_mask = Input(shape=(MAX_LEN,), dtype=tf.int32, name="xlm_mask")
    minilm_ids = Input(shape=(MAX_LEN,), dtype=tf.int32, name="minilm_ids")
    minilm_mask = Input(shape=(MAX_LEN,), dtype=tf.int32, name="minilm_mask")
    indic_ids = Input(shape=(MAX_LEN,), dtype=tf.int32, name="indic_ids")
    indic_mask = Input(shape=(MAX_LEN,), dtype=tf.int32, name="indic_mask")

    # Transformers
    print(f"Loading {XLM_MODEL_NAME}...")
    try:
        xlm_transformer = TFAutoModel.from_pretrained(XLM_MODEL_NAME)
    except:
        xlm_transformer = TFAutoModel.from_pretrained(XLM_MODEL_NAME, from_pt=True)
    # Freeze transformer layers to save memory and speed up
    xlm_transformer.trainable = False
    xlm_out = xlm_transformer(xlm_ids, attention_mask=xlm_mask)[0] 

    print(f"Loading {MINILM_MODEL_NAME}...")
    try:
        minilm_transformer = TFAutoModel.from_pretrained(MINILM_MODEL_NAME)
    except:
        minilm_transformer = TFAutoModel.from_pretrained(MINILM_MODEL_NAME, from_pt=True)
    minilm_transformer.trainable = False
    minilm_out = minilm_transformer(minilm_ids, attention_mask=minilm_mask)[0]

    print(f"Loading {INDIC_MODEL_NAME}...")
    try:
        indic_transformer = TFAutoModel.from_pretrained(INDIC_MODEL_NAME)
    except:
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
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    configure_gpu_memory()
    print("Starting training process with Tri-Hybrid Model (XLM + MiniLM + IndicBERT) -> CNN -> LSTM")
    
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return

    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    if 'category' in df.columns and 'sentiment' not in df.columns:
        df = df.rename(columns={'category': 'sentiment'})
    print(f"Dataset loaded. {len(df)} rows.")
    
    # Optional filtering: remove empty categories or emotions if needed
    df = df.dropna(subset=['text', 'sentiment', 'emotion'])
    
    # Filter out classes with less than 2 instances to allow for stratified splitting
    cat_counts = df['sentiment'].value_counts()
    df = df[df['sentiment'].isin(cat_counts[cat_counts >= 2].index)]

    # Filter out rare emotions too just in case
    emo_counts = df['emotion'].value_counts()
    df = df[df['emotion'].isin(emo_counts[emo_counts >= 2].index)]
    
    df['clean_text'] = df['text'].apply(preprocess_text)

    print("Encoding targets...")
    le_sentiment = LabelEncoder()
    y_sent = keras.utils.to_categorical(le_sentiment.fit_transform(df['sentiment']))
    
    le_emotion = LabelEncoder()
    y_emo = keras.utils.to_categorical(le_emotion.fit_transform(df['emotion']))

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "sentiment_label_encoder.pkl"), "wb") as f:
        pickle.dump(le_sentiment, f)
    with open(os.path.join(MODELS_DIR, "emotion_label_encoder.pkl"), "wb") as f:
        pickle.dump(le_emotion, f)
    
    indices = np.arange(len(df))
    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42, stratify=df['sentiment'])
    
    train_texts = df['clean_text'].iloc[train_idx].tolist()
    val_texts = df['clean_text'].iloc[val_idx].tolist()
    
    train_sent_labels = y_sent[train_idx]
    train_emo_labels = y_emo[train_idx]
    val_sent_labels = y_sent[val_idx]
    val_emo_labels = y_emo[val_idx]

    print("Loading tokenizers...")
    xlm_tok = AutoTokenizer.from_pretrained(XLM_MODEL_NAME)
    minilm_tok = AutoTokenizer.from_pretrained(MINILM_MODEL_NAME)
    indic_tok = AutoTokenizer.from_pretrained(INDIC_MODEL_NAME)

    train_gen = TextDataGenerator(train_texts, train_sent_labels, train_emo_labels, BATCH_SIZE, xlm_tok, minilm_tok, indic_tok, MAX_LEN)
    val_gen = TextDataGenerator(val_texts, val_sent_labels, val_emo_labels, BATCH_SIZE, xlm_tok, minilm_tok, indic_tok, MAX_LEN)

    print("Building model...")
    gc.collect()
    
    hybrid_model = build_model(len(le_sentiment.classes_), len(le_emotion.classes_))
    hybrid_model.summary()
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=2, 
        min_lr=1e-6,
        verbose=1
    )
    
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODELS_DIR, "best_tri_hybrid_model.keras"),
        monitor='val_sentiment_accuracy', 
        save_best_only=True,
        verbose=1
    )

    print("Starting training...")
    try:
        hybrid_model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=20,
            callbacks=[early_stopping, reduce_lr, model_checkpoint]
        )
    except Exception as e:
        print(f"Exception during training: {e}")

    save_path = os.path.join(MODELS_DIR, "tri_hybrid_sentiment_emotion_model.keras")
    print(f"Saving final model to {save_path}...")
    hybrid_model.save(save_path)
    print("Training Complete. Models Saved.")

if __name__ == "__main__":
    main()
