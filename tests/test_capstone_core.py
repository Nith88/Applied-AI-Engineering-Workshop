import unittest

from capstone_core import (
    content_to_text,
    count_words,
    make_study_plan,
    parse_json_object,
    split_text,
)


class CapstoneCoreTests(unittest.TestCase):
    def test_split_text_respects_overlap_and_keeps_all_content(self):
        chunks = split_text("abcdefghij", chunk_size=6, overlap=2)

        self.assertEqual(chunks, ["abcdef", "efghij"])

    def test_split_text_rejects_invalid_overlap(self):
        with self.assertRaises(ValueError):
            split_text("some text", chunk_size=5, overlap=5)

    def test_json_parser_accepts_fenced_json(self):
        self.assertEqual(
            parse_json_object('```json\n{"urgency":"High"}\n```'),
            {"urgency": "High"},
        )

    def test_tools_validate_input_boundaries(self):
        self.assertEqual(count_words("RAG retrieves evidence."), 3)
        self.assertEqual(len(make_study_plan("AI", 30).splitlines()), 30)
        with self.assertRaises(ValueError):
            count_words("  ")
        with self.assertRaises(ValueError):
            make_study_plan("AI", True)

    def test_content_to_text_handles_model_text_blocks(self):
        self.assertEqual(
            content_to_text([{"type": "text", "text": "first"}, "second"]),
            "first\nsecond",
        )


if __name__ == "__main__":
    unittest.main()
