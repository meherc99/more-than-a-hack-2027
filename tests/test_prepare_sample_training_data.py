import unittest

from our_tactics.prepare_sample_training_data import build_sparse_labels


class SparseWeakLabelTests(unittest.TestCase):
    def test_samples_only_selected_event_spans_at_requested_stride(self):
        events = [
            {"period": 1, "start": 10, "end": 20, "team": "H", "type": "PASS"},
            {"period": 1, "start": 11, "end": 20, "team": "A", "type": "BALL LOST"},
        ]

        labels = build_sparse_labels(events, stride=5)

        self.assertEqual(labels, {(1, 10): "H", (1, 15): "H", (1, 20): "H"})

    def test_opposing_team_overlap_is_marked_ambiguous(self):
        events = [
            {"period": 2, "start": 10, "end": 20, "team": "H", "type": "PASS"},
            {"period": 2, "start": 15, "end": 25, "team": "A", "type": "RECOVERY"},
        ]

        labels = build_sparse_labels(events, stride=5)

        self.assertEqual(labels[(2, 10)], "H")
        self.assertIsNone(labels[(2, 15)])
        self.assertIsNone(labels[(2, 20)])
        self.assertEqual(labels[(2, 25)], "A")


if __name__ == "__main__":
    unittest.main()