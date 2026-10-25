# ClientPulse AI — Intelligent Customer Feedback Analyzer

An ESSEC AAA group-project prototype that classifies customer-review sentiment and complaint topics and displays actionable trends. Python + Streamlit + scikit-learn; optionally Hugging Face RoBERTa for sentiment.

**Quick start**
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
For pretrained sentiment, install `requirements-huggingface.txt` and select Hugging Face in the app sidebar. First use downloads the model.

**Data:** bundled demo/training CSVs are synthetic. You can separately download a dataset from Kaggle and rename its text column to `review` for use in the app; check licenses and privacy first.

**GitHub:** publish the directory to your own repository to share a reproducible codebase. The project is not automatically uploaded to GitHub.

See `USER_GUIDE.md` and `AI_USAGE_AND_SOURCES.md` for instructions, citations and disclosures.
