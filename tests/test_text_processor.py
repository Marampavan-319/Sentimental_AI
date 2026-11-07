"""Unit tests for review text validation."""

import unittest

from src.utils.text_processor import (
    ReviewValidationError,
    clean_review,
    validate_review,
)


class TextProcessorTests(unittest.TestCase):
    def test_clean_review_collapses_whitespace(self):
        self.assertEqual(clean_review("  Great   product!  "), "Great product!")

    def test_validate_review_rejects_empty_text(self):
        with self.assertRaises(ReviewValidationError):
            validate_review("   ")

    def test_validate_review_preserves_punctuation(self):
        self.assertEqual(validate_review("I love it!!!"), "I love it!!!")


if __name__ == "__main__":
    unittest.main()
