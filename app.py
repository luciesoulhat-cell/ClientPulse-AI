"""Run with: streamlit run app.py"""
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from model import train_models, analyze
ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='ClientPulse AI',page_icon='💬',layout='wide')
st.title('💬 ClientPulse AI')
st.caption('From customer feedback to clear action priorities | Educational prototype')
with st.sidebar:
    st.header('Data source')
    upload=st.file_uploader('Upload a CSV containing a review column',type='csv')
    st.caption('Or use the included fictional sample reviews.')
    st.divider()
    engine=st.radio('Sentiment engine', ['Offline baseline (fast)', 'Hugging Face RoBERTa (advanced)'], help='Hugging Face mode requires the optional dependencies and downloads the public model the first time.')
    st.info('Analysis runs locally. Hugging Face mode downloads public model weights on first use; reviews are not sent to an API.')
    st.warning('Prototype trained on synthetic examples. Do not rely on it for production decisions without real-world validation.')
@st.cache_resource(show_spinner='Training text classifiers...')
def load_models(): return train_models()
try:
    models,metrics=load_models()
    df=pd.read_csv(upload) if upload is not None else pd.read_csv(ROOT/'demo_reviews.csv')
    if 'review' not in df.columns: st.error("Your CSV needs a column named 'review'.");st.stop()
    if len(df)==0: st.error('Please provide at least one review.');st.stop()
    if len(df)>10000: st.error('Prototype limit: 10,000 reviews per upload.');st.stop()
    result=analyze(df,models)
    if engine.startswith('Hugging Face'):
        from hf_model import predict_sentiment
        with st.spinner('Classifying sentiment with Hugging Face RoBERTa...'):
            labels=predict_sentiment(df['review'].fillna('').astype(str).tolist())
        result['predicted_sentiment']=[x[0] for x in labels]
        result['sentiment_confidence']=[x[1] for x in labels]
        result['priority']=result.apply(lambda r:'High' if r.predicted_sentiment=='negative' and r.sentiment_confidence>=.60 else ('Review' if r.predicted_sentiment=='negative' else 'Normal'),axis=1)
except Exception as e:
    st.error(f'Could not analyze the data: {e}');st.stop()

n=len(result);negative=int((result.predicted_sentiment=='negative').sum());high=int((result.priority=='High').sum())
a,b,c,d=st.columns(4)
a.metric('Reviews analyzed',f'{n:,}')
b.metric('Negative feedback',f'{negative/n:.0%}')
c.metric('High-priority cases',high)
d.metric('Categories identified',result.predicted_category.nunique())
left,right=st.columns(2)
with left:
    st.subheader('Customer sentiment')
    s=result.predicted_sentiment.value_counts().rename_axis('Sentiment').reset_index(name='Reviews')
    st.plotly_chart(px.bar(s,x='Sentiment',y='Reviews',text='Reviews'),use_container_width=True)
with right:
    st.subheader('Issues to investigate')
    issues=result[result.predicted_sentiment=='negative'].predicted_category.value_counts().rename_axis('Category').reset_index(name='Negative reviews')
    if len(issues): st.plotly_chart(px.bar(issues,x='Category',y='Negative reviews',text='Negative reviews'),use_container_width=True)
    else: st.success('No negative reviews detected in the current file.')
st.subheader('Prioritized feedback')
category=st.selectbox('Filter by issue',['All']+sorted(result.predicted_category.unique().tolist()))
view=result if category=='All' else result[result.predicted_category==category]
view=view.sort_values(by='priority',key=lambda s:s.map({'High':0,'Review':1,'Normal':2}))
cols=[x for x in ['review_id','date','channel','review','predicted_category','predicted_sentiment','category_confidence','sentiment_confidence','priority'] if x in view.columns]
st.dataframe(view[cols],use_container_width=True,hide_index=True)
st.download_button('⬇️ Download analyzed CSV',result.to_csv(index=False).encode('utf-8'),file_name='clientpulse_results.csv',mime='text/csv')
with st.expander('How the AI works / limitations'):
    st.write('The baseline uses two supervised classifiers with TF-IDF word and phrase features and logistic regression. The optional Hugging Face engine replaces sentiment with pretrained RoBERTa (issue categories remain from the offline baseline).')
    st.write('Two supervised text classifiers use TF-IDF word and phrase features and logistic regression. A priority rule highlights high-confidence negative reviews. Confidence indicates relative model certainty, not a calibrated probability of correctness.')
    st.write('Models are trained on labeled synthetic examples packaged with the project. Validation metrics below are obtained from a held-out portion of this synthetic dataset and do **not** establish real-world performance. Similar examples across train and test can inflate performance.')
    st.table(pd.DataFrame(metrics).T.rename(columns={'accuracy':'Held-out accuracy','macro_f1':'Held-out macro-F1'}))
    st.write('Human review is needed before following up. Multilingual or unexpected feedback may be misclassified. Uploaded data is not retained by this app.')
