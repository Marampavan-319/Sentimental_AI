# Sentiment AI 2.0

A customer-feedback sentiment analysis application built with Streamlit, PyTorch, and Hugging Face Transformers.

## Current capabilities

- Analyze one customer review and view its predicted sentiment and model confidence.
- Upload a CSV file containing a `Review` column and analyze up to 1,000 reviews.
- View positive/negative review counts and interactive charts.
- Filter analyzed reviews and download the results as a CSV.
- Load the public DistilBERT SST-2 model through a reusable model-loading module.

## Model information

The baseline model is [DistilBERT fine-tuned on SST-2](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english).

It predicts **POSITIVE** or **NEGATIVE** sentiment. It does not predict neutral sentiment, emotions, or product aspects. The displayed confidence is the model score for its selected label; it is not a guarantee of correctness.

The model is public, so a Hugging Face access token is not required. On first launch, Transformers downloads the model and tokenizer and caches them locally.

## Project structure

```text
Sentimental_AI/
├── app.py
├── requirements.txt
├── src/
│   ├── config.py
│   ├── models/
│   │   └── model_loader.py
│   ├── services/
│   │   └── sentiment_analyzer.py
│   └── utils/
│       └── text_processor.py
└── tests/
    └── test_text_processor.py
```

## Setup on Windows PowerShell

1. Create and activate a virtual environment:

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, allow scripts for the current terminal only:

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Run the app:

   ```powershell
   streamlit run app.py
   ```

4. Open the local URL printed by Streamlit, usually http://localhost:8501.

## Run the tests

The text-validation tests use Python's built-in unittest module:

```powershell
python -m unittest discover -s tests -v
```

## CSV format

Your CSV must contain a column named `Review`:

```csv
Review
"The product arrived quickly and works perfectly."
"The quality was disappointing and it stopped working."
```

## Current limitations

- The baseline model is trained on the SST-2 movie-review sentiment dataset and may not understand every product or service domain equally well.
- Only positive and negative labels are supported in this version.
- Bulk processing is limited to 1,000 rows per upload.
- Analysis history and user accounts are not included yet.

## Roadmap

1. Establish a tested sentiment-analysis foundation.
2. Evaluate and integrate suitable emotion and aspect-based models.
3. Add explainable review insights and richer analytics.
4. Add persistent storage and user-facing workflows.
5. Test, document, and deploy the application.

---
Created by [Marampavan-319](https://github.com/Marampavan-319).
