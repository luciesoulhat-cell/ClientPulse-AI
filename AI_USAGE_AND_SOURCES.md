# Tools, data provenance, and AI assistance

## Application
- Python: source code and data-processing workflow.
- Streamlit: https://streamlit.io/ — interface.
- pandas: https://pandas.pydata.org/ — CSV and dataframes.
- scikit-learn: https://scikit-learn.org/ — TF-IDF and logistic regression classifiers for baseline sentiment and categories.
- Plotly: https://plotly.com/python/ — visualizations.
- Hugging Face Transformers: https://huggingface.co/docs/transformers — optional pretrained sentiment model.
- Pretrained model: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest . Check model license and model card before public deployment.
- Kaggle: https://www.kaggle.com/datasets — OPTIONAL source of an external review dataset. Not integrated automatically; no Kaggle data is included.
- GitHub: https://github.com/ — recommended for hosting the submitted source. No repository has been created or published by this package.

## Data
- `training_reviews.csv`: synthetic labeled examples (not real customers).
- `demo_reviews.csv`: synthetic reviews for demonstration.
- No external dataset is bundled. Do not describe these files as sourced from Kaggle.

## AI prompts used in preparing this project (paraphrased)
- 'Create a reproducible Python application that imports customer reviews, classifies sentiment and complaint categories, and shows a dashboard.'
- 'Add an optional Hugging Face pretrained sentiment classifier while preserving a locally runnable baseline.'
- 'Write an ESSEC student user guide, source citations, limitations and demo structure.'

## Reproducibility
Use `requirements.txt` for offline baseline. Use `requirements-huggingface.txt` for optional pretrained inference. Weights download at first use. The local dataset is sufficient for baseline demo.
