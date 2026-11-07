"""Input validation and minimal text cleanup."""

from src.config import MAX_REVIEW_CHARACTERS


class ReviewValidationError(ValueError):
    """Raised when a review cannot be analyzed."""


def clean_review(text):
    """Trim surrounding whitespace and collapse repeated whitespace.

    We deliberately avoid removing punctuation or stop words because they can
    carry sentiment information.
    """
    return " ".join(str(text).split())


def validate_review(text):
    """Validate and return a cleaned review string."""
    if text is None:
        raise ReviewValidationError("Please enter a review before analyzing.")

    cleaned = clean_review(text)

    if not cleaned or cleaned.lower() == "nan":
        raise ReviewValidationError("The review is empty. Please enter some text.")

    if len(cleaned) > MAX_REVIEW_CHARACTERS:
        raise ReviewValidationError(
            f"Reviews must be {MAX_REVIEW_CHARACTERS} characters or fewer."
        )

    return cleaned
