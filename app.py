from fastapi import FastAPI
import tensorflow as tf
import json
import nltk
from nltk.tokenize import word_tokenize

app = FastAPI()

# Load model and vocabulary
model = tf.keras.models.load_model('next_word_model.h5')
with open('vocab.json', 'r') as f:
    vocab = json.load(f)
reverse_vocab = {idx: word for word, idx in vocab.items()}

@app.get("/predict_next_word")
async def predict_next_word(input_text: str):
    nltk.download('punkt_tab')
    tokens = word_tokenize(input_text.lower())
    input_ids = [vocab.get(token, vocab['<unk>']) for token in tokens]
    if not input_ids:
        return {"predicted_word": "<unk>"}
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(
        [input_ids], maxlen=10, padding='pre'
    )
    predictions = model.predict(input_tensor)[0, -1]
    predicted_id = tf.argmax(predictions).numpy()
    predicted_word = reverse_vocab.get(predicted_id, '<unk>')
    return {"predicted_word": predicted_word}