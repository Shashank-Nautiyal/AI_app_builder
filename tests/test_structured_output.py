import json
import unittest

from structured_output import repair_structured_json


class StructuredOutputTests(unittest.TestCase):
    def test_repair_strips_fences_and_closes_truncated_json(self):
        raw = '```json\n{"filepath":"app.py","task_desc":"wri"\n```'
        repaired = repair_structured_json(raw)
        parsed = json.loads(repaired)

        self.assertEqual(parsed["filepath"], "app.py")
        self.assertEqual(parsed["task_desc"], "wri")

    def test_repair_extracts_json_from_surrounding_text(self):
        raw = 'Here is the result:\n{"status":"done","count":2}\nThanks!'
        repaired = repair_structured_json(raw)
        parsed = json.loads(repaired)

        self.assertEqual(parsed["status"], "done")
        self.assertEqual(parsed["count"], 2)


if __name__ == "__main__":
    unittest.main()
