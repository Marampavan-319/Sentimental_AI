"""Business logic for single and batch sentiment predictions."""

from src.config import MAX_BATCH_SIZE
from src.utils.text_processor import ReviewValidationError, validate_review


def analyze_text(text, analyzer):
    """Return the label and confidence score for one review."""
    cleaned = validate_review(text)
    prediction = analyzer(cleaned, truncation=True, max_length=512)[0]

    return {
        "label": prediction["label"].upper(),
        "score": float(prediction["score"]),
    }


def analyze_many(reviews, analyzer):
    """Analyze a list of reviews in batches and preserve their input order."""
    if len(reviews) > MAX_BATCH_SIZE:
        raise ReviewValidationError(
            f"Please analyze no more than {MAX_BATCH_SIZE} reviews at a time."
        )

    cleaned_reviews = []
    for index, review in enumerate(reviews, start=1):
        try:
            cleaned_reviews.append(validate_review(review))
        except ReviewValidationError as exc:
            raise ReviewValidationError(
                f"Review in CSV row {index} is invalid: {exc}"
            ) from exc

    if not cleaned_reviews:
        raise ReviewValidationError("There are no reviews to analyze.")

    predictions = analyzer(
        cleaned_reviews,
        batch_size=16,
        truncation=True,
        max_length=512,
    )

    return [
        {
            "label": prediction["label"].upper(),
            "score": float(prediction["score"]),
        }
        for prediction in predictions
    ]
