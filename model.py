"""Offline, reproducible text classifiers; no external API or model download."""
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
ROOT=Path(__file__).resolve().parent

def train_models(csv_path=ROOT/'training_reviews.csv'):
    df=pd.read_csv(csv_path)
    if not {'review','sentiment','category'}.issubset(df.columns):
        raise ValueError('Training dataset must have review, sentiment, category columns')
    models={}; metrics={}
    for task in ('sentiment','category'):
        x_train,x_test,y_train,y_test=train_test_split(df['review'].fillna(''),df[task],test_size=.25,random_state=42,stratify=df[task])
        model=Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),min_df=1,sublinear_tf=True)),
                        ('clf',LogisticRegression(max_iter=2000,class_weight='balanced',random_state=42))])
        model.fit(x_train,y_train)
        pred=model.predict(x_test)
        metrics[task]={'accuracy':accuracy_score(y_test,pred),'macro_f1':f1_score(y_test,pred,average='macro')}
        model.fit(df['review'].fillna(''),df[task])
        models[task]=model
    return models,metrics

def analyze(df,models):
    df=df.copy()
    if 'review' not in df: raise ValueError("CSV must contain a 'review' column")
    texts=df['review'].fillna('').astype(str)
    if texts.str.strip().eq('').any(): raise ValueError('Remove empty reviews before analyzing')
    for task in ('category','sentiment'):
        model=models[task]
        probs=model.predict_proba(texts)
        idx=probs.argmax(axis=1)
        df['predicted_'+task]=model.classes_[idx]
        df[task+'_confidence']=probs.max(axis=1).round(3)
    # prioritization is a heuristic/business rule, not a clinical or causal AI score
    def priority(r):
        if r.predicted_sentiment=='negative' and r.sentiment_confidence>=.60: return 'High'
        if r.predicted_sentiment=='negative': return 'Review'
        return 'Normal'
    df['priority']=df.apply(priority,axis=1)
    return df
