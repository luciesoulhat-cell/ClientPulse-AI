"""Optional Hugging Face inference. Requires transformers, torch and initial download."""
from functools import lru_cache
MODEL_ID = 'cardiffnlp/twitter-roberta-base-sentiment-latest'

@lru_cache(maxsize=1)
def pipeline_model():
    from transformers import pipeline
    return pipeline('text-classification', model=MODEL_ID, tokenizer=MODEL_ID, device=-1)

def predict_sentiment(texts, batch_size=16):
    """Return label, score from pretrained model (English reviews)."""
    model=pipeline_model()
    output=[]
    for i in range(0,len(texts),batch_size):
        for pred in model(list(texts[i:i+batch_size]), truncation=True, max_length=512, batch_size=batch_size):
            output.append((pred['label'].lower(),round(float(pred['score']),3)))
    return output
