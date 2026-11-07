"""Load the Hugging Face sentiment model."""

import torch
from transformers import pipeline

from src.config import MODEL_ID


def load_sentiment_model():
    """Load the public SST-2 model using the PyTorch backend.

    No Hugging Face token is required because this model is public.
    The Streamlit app caches the returned pipeline for the current session.
    """
    device = 0 if torch.cuda.is_available() else -1

    return pipeline(
        task="sentiment-analysis",
        model=MODEL_ID,
        framework="pt",
        device=device,
    )
