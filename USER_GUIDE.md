# ClientPulse AI — User Guide

## Requirements
Python 3.10–3.12. Internet is required only for the first Hugging Face model download or package installation.

## Setup (Mac / Windows / Linux)
1. Unzip the project and open Terminal in its folder.
2. Create an environment: `python -m venv .venv`
3. Activate it: macOS/Linux `source .venv/bin/activate`; Windows PowerShell `.venv\\Scripts\\Activate.ps1`.
4. Install the baseline: `python -m pip install -r requirements.txt`.
5. For pretrained Hugging Face inference, additionally run `python -m pip install -r requirements-huggingface.txt`.
6. Run: `python -m streamlit run app.py`.
7. Open the local URL displayed by Streamlit (usually http://localhost:8501).

## Demo
1. Keep the default included demo dataset or upload your own CSV containing a `review` column.
2. Choose the sentiment engine in the sidebar: Offline baseline or Hugging Face RoBERTa.
3. See sentiment counts and categories, filter feedback, and download analyzed CSV.
4. Optional: the first Hugging Face run downloads pretrained weights, which can take time.

## Kaggle use
Download a customer-review dataset from Kaggle manually, read its license and privacy terms, and adapt its review-text column to `review`. Upload the CSV in the app. The included CSVs are SYNTHETIC and are NOT from Kaggle. No Kaggle authentication or dataset is bundled.

## Limitations
Synthetic training data may not generalize; results are for demonstration, not validated commercial decisions. The Hugging Face model is pretrained on tweets and can also make errors on retail reviews. Review misclassifications manually and check dataset licensing/privacy.
