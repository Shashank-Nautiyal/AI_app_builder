import unittest

from pydantic import ValidationError

from codemind.api.ui_events import build_ui_event


class UIEventTests(unittest.TestCase):
    def test_build_ui_event_validates_done_payload(self):
        event = build_ui_event("done", {"status": "rolled_back", "message": "restored"}, agent="supervisor")
        self.assertEqual(event["type"], "done")
        self.assertEqual(event["data"]["status"], "rolled_back")
        self.assertEqual(event["agent"], "supervisor")

    def test_build_ui_event_rejects_invalid_node_finished_payload(self):
        with self.assertRaises(ValidationError):
            build_ui_event("node_finished", {"missing": "node"})
