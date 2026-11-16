"""Summarize tracking evidence around transitions in a generated submission.

Write diagnostic output outside the repository so derived match data is not committed:

    python our_tactics/transition_diagnostics.py \
        --submission /tmp/solver_submission.csv \
        --output /tmp/transition_diagnostics.csv
"""

import argparse
import csv
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluation import possession_metrics as pm  # noqa: E402
from games import GAME, GAMES, require  # noqa: E402
from our_tactics.possession_solver import (  # noqa: E402
    HOME,
    AWAY,
    DEAD,
    UNKNOWN,
    parse_ball,
    parse_player_group,
    team_of_each_track,
)

SAMPLE_OFFSETS = (-12, 0, 12)
NEAR_RADIUS_METRES = 2.5


def load_predictions(path, n_frames):
    table = pm.read_label_csv(path)
    reason = pm.validate_submission(table, n_frames)
    if reason:
        raise ValueError("invalid submission: {}".format(reason))
    return pm.label_codes(table)


def find_transitions(labels):
    indices = (np.flatnonzero(labels[1:] != labels[:-1]) + 1).tolist()
    return [(int(index), int(labels[index - 1]), int(labels[index])) for index in indices]


def transition_kind(before, after):
    teams = (HOME, AWAY)
    if before in teams and after in teams:
        return "team_switch"
    if after == DEAD:
        return "to_dead"
    if before == DEAD and after in teams:
        return "from_dead_to_team"
    if after == UNKNOWN:
        return "to_unknown"
    if before == UNKNOWN:
        return "from_unknown"
    return "other"


def extract_tracking_features(player_data, ball_data, team_map, pitch):
    ball = parse_ball(ball_data)
    if ball is None:
        return {
            "ball_visible": False,
            "ball_out_of_bounds": "",
            "home_nearest_m": "",
            "away_nearest_m": "",
            "nearest_team": "none",
            "home_near_2_5m": 0,
            "away_near_2_5m": 0,
        }

    length, width = pitch
    ball_x, ball_y = ball[0] * length, ball[1] * width
    nearest = [float("inf"), float("inf")]
    nearby = [0, 0]
    for slot, group in enumerate(player_data.split(";")):
        if slot >= len(team_map):
            break
        player = parse_player_group(group)
        if player is None:
            continue
        team = int(team_map[slot])
        dx = player[0] * length - ball_x
        dy = player[1] * width - ball_y
        distance = float(np.hypot(dx, dy))
        nearest[team] = min(nearest[team], distance)
        if distance < NEAR_RADIUS_METRES:
            nearby[team] += 1

    if nearest[HOME] < nearest[AWAY]:
        nearest_team = "H"
    elif nearest[AWAY] < nearest[HOME]:
        nearest_team = "A"
    elif nearest[HOME] != float("inf"):
        nearest_team = "tie"
    else:
        nearest_team = "none"

    return {
        "ball_visible": True,
        "ball_out_of_bounds": not (0.0 <= ball[0] <= 1.0 and 0.0 <= ball[1] <= 1.0),
        "home_nearest_m": nearest[HOME] if nearest[HOME] != float("inf") else "",
        "away_nearest_m": nearest[AWAY] if nearest[AWAY] != float("inf") else "",
        "nearest_team": nearest_team,
        "home_near_2_5m": nearby[HOME],
        "away_near_2_5m": nearby[AWAY],
    }


def collect_tracking_features(raw_path, frames, team_map, pitch):
    wanted = set(frames)
    features = {}
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            try:
                frame = int(parts[0]) - 1
            except ValueError:
                continue
            if frame in wanted:
                features[frame] = extract_tracking_features(
                    parts[1], parts[2], team_map, pitch)
    return features


def load_tag_events(path, fps):
    if path is None:
        return []
    events = []
    for instance in ET.parse(path).iter("instance"):
        tags = {label.findtext("group"): label.findtext("text")
                for label in instance.findall("label")}
        events.append({
            "start": int(round(float(instance.findtext("start")) * fps)),
            "end": int(round(float(instance.findtext("end")) * fps)),
            "code": (instance.findtext("code") or "").strip(),
            "team": (tags.get("Team") or "").strip(),
            "type": (tags.get("Type") or "").strip(),
        })
    return events


def tag_context(events, frame, radius=12):
    context = []
    for event in events:
        if event["start"] <= frame + radius and event["end"] > frame - radius:
            details = [event["code"]]
            if event["team"]:
                details.append("Team=" + event["team"])
            if event["type"]:
                details.append("Type=" + event["type"])
            context.append(";".join(details))
    return " | ".join(context)


def build_report_rows(transitions, labels, features, events, fps):
    rows = []
    label_names = ("H", "A", "D", "X")
    for frame, before, after in transitions:
        target_team = label_names[after] if after in (HOME, AWAY) else ""
        for offset in SAMPLE_OFFSETS:
            sample_frame = frame + offset
            if not 0 <= sample_frame < len(labels):
                continue
            evidence = features.get(sample_frame, {
                "ball_visible": False,
                "ball_out_of_bounds": "",
                "home_nearest_m": "",
                "away_nearest_m": "",
                "nearest_team": "none",
                "home_near_2_5m": 0,
                "away_near_2_5m": 0,
            })
            rows.append({
                "transition_frame": frame,
                "transition_time_s": round(frame / float(fps), 3),
                "kind": transition_kind(before, after),
                "from_label": label_names[before],
                "to_label": label_names[after],
                "sample_offset_frames": offset,
                "sample_frame": sample_frame,
                "sample_label": label_names[int(labels[sample_frame])],
                "ball_visible": evidence["ball_visible"],
                "ball_out_of_bounds": evidence["ball_out_of_bounds"],
                "home_nearest_m": evidence["home_nearest_m"],
                "away_nearest_m": evidence["away_nearest_m"],
                "nearest_team": evidence["nearest_team"],
                "home_near_2_5m": evidence["home_near_2_5m"],
                "away_near_2_5m": evidence["away_near_2_5m"],
                "supports_target_team": bool(
                    target_team and evidence["nearest_team"] == target_team),
                "tag_context": tag_context(events, frame),
            })
    return rows


def write_report(rows, path):
    fields = (
        "transition_frame", "transition_time_s", "kind", "from_label", "to_label",
        "sample_offset_frames", "sample_frame", "sample_label", "ball_visible",
        "ball_out_of_bounds", "home_nearest_m", "away_nearest_m", "nearest_team",
        "home_near_2_5m", "away_near_2_5m", "supports_target_team", "tag_context",
    )
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--submission", required=True, help="generated frame,label CSV")
    parser.add_argument("--output", required=True, help="diagnostic CSV; keep outside the repo")
    parser.add_argument("--data-dir", default=None, help="downloaded match data directory")
    parser.add_argument("--game", default=GAME, help="game identifier")
    args = parser.parse_args()

    n_frames = GAMES[args.game]["n_video_frames"]
    fps = GAMES[args.game]["fps"]
    labels = load_predictions(args.submission, n_frames)
    transitions = find_transitions(labels)
    sample_frames = [frame + offset for frame, _, _ in transitions
                     for offset in SAMPLE_OFFSETS
                     if 0 <= frame + offset < n_frames]

    meta_path, raw_path = require(["atd_meta", "atd_raw"], args.game, args.data_dir)
    team_map = team_of_each_track(meta_path)
    features = collect_tracking_features(raw_path, sample_frames, team_map,
                                         GAMES[args.game]["pitch"])
    try:
        tag_path = require(["smart_tagging"], args.game, args.data_dir)[0]
    except (KeyError, IOError):
        tag_path = None
    events = load_tag_events(tag_path, fps)
    rows = build_report_rows(transitions, labels, features, events, fps)
    write_report(rows, args.output)

    counts = Counter(transition_kind(before, after) for _, before, after in transitions)
    print("transitions: {}".format(len(transitions)))
    for kind, count in sorted(counts.items()):
        print("  {:<20} {}".format(kind, count))
    print("wrote {} evidence rows to {}".format(len(rows), args.output))


if __name__ == "__main__":
    main()