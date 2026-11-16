import unittest

import numpy as np

from our_tactics.possession_solver import AWAY, DEAD, HOME, UNKNOWN
from our_tactics.transition_diagnostics import (
    extract_tracking_features,
    find_transitions,
    transition_kind,
)


class TransitionDiagnosticsTests(unittest.TestCase):
    def test_find_transitions_and_classify_kinds(self):
        labels = np.array([HOME, HOME, AWAY, UNKNOWN, DEAD, HOME], dtype=np.int8)
        transitions = find_transitions(labels)

        self.assertEqual([frame for frame, _, _ in transitions], [2, 3, 4, 5])
        self.assertEqual(transition_kind(transitions[0][1], transitions[0][2]), "team_switch")
        self.assertEqual(transition_kind(transitions[1][1], transitions[1][2]), "to_unknown")
        self.assertEqual(transition_kind(transitions[2][1], transitions[2][2]), "to_dead")
        self.assertEqual(
            transition_kind(transitions[3][1], transitions[3][2]), "from_dead_to_team")

    def test_tracking_features_report_nearest_team_and_bounds(self):
        team_map = np.array([HOME, AWAY], dtype=np.int8)
        inside = extract_tracking_features(
            "0.12,0.5,0;0.8,0.5,0", "0.13,0.5,0", team_map, (105.3, 68.0))
        outside = extract_tracking_features(
            "0.1,0.5,0;0.9,0.5,0", "1.01,0.5,0", team_map, (105.3, 68.0))
        missing = extract_tracking_features(
            "N;N", "NaN,NaN,NaN", team_map, (105.3, 68.0))

        self.assertEqual(inside["nearest_team"], "H")
        self.assertEqual(inside["home_near_2_5m"], 1)
        self.assertFalse(inside["ball_out_of_bounds"])
        self.assertTrue(outside["ball_out_of_bounds"])
        self.assertFalse(missing["ball_visible"])
        self.assertEqual(missing["nearest_team"], "none")


if __name__ == "__main__":
    unittest.main()