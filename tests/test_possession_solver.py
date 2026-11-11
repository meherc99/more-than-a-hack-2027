import os
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from our_tactics.possession_solver import (
    HOME,
    AWAY,
    collapse_short_runs,
    parse_ball,
    parse_player_group,
    smooth_majority,
    team_of_each_track,
)


class PossessionSolverTests(unittest.TestCase):
    def test_parse_ball_and_player_group(self):
        self.assertEqual(parse_ball("0.3,0.6,0.1"), (0.3, 0.6))
        self.assertIsNone(parse_ball("N,0.0,0.0"))
        self.assertEqual(parse_player_group("0.2,0.9"), (0.2, 0.9))
        self.assertIsNone(parse_player_group("N,0.0"))

    def test_smooth_majority_prefers_majority_and_home_tie_break(self):
        codes = np.array([HOME, AWAY, AWAY, HOME, HOME], dtype=np.int8)
        smoothed = smooth_majority(codes, 1)
        self.assertEqual(smoothed[0], HOME)
        self.assertEqual(smoothed[1], AWAY)
        self.assertEqual(smoothed[3], HOME)

        tie = np.array([HOME, AWAY, HOME, AWAY], dtype=np.int8)
        tied = smooth_majority(tie, 1)
        self.assertEqual(tied[0], HOME)
        self.assertEqual(tied[2], AWAY)

    def test_collapse_short_runs(self):
        codes = np.array([HOME, HOME, AWAY, AWAY, AWAY, HOME, HOME], dtype=np.int8)
        collapsed = collapse_short_runs(codes, min_run=3)
        self.assertTrue(np.array_equal(collapsed[:2], np.array([AWAY, AWAY], dtype=np.int8)))
        self.assertTrue(np.array_equal(collapsed[2:5], np.array([AWAY, AWAY, AWAY], dtype=np.int8)))
        self.assertTrue(np.array_equal(collapsed[5:], np.array([AWAY, AWAY], dtype=np.int8)))

    def test_team_of_each_track_from_meta_xml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "meta.xml")
            root = ET.Element("PlayerModel")
            home = ET.SubElement(root, "Team", {"id": "home"})
            ET.SubElement(home, "Name").text = "Barcelona"
            away = ET.SubElement(root, "Team", {"id": "away"})
            ET.SubElement(away, "Name").text = "Athletic"
            for team_id, slot in (("home", "1"), ("away", "2"), ("home", "3")):
                ET.SubElement(root, "Player", {"teamId": team_id, "slot": slot})

            tree = ET.ElementTree(root)
            tree.write(path)

            team_map = team_of_each_track(path)
            self.assertEqual(team_map.tolist(), [HOME, AWAY, HOME])


if __name__ == "__main__":
    unittest.main()
