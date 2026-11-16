"""Build sparse, event-derived possession examples from Metrica's sample matches.

The event spans are weak labels: only selected on-ball actions are labelled. Gaps and
opposing-team event overlaps are omitted rather than filled by assumption.

    python our_tactics/prepare_sample_training_data.py \
        --output /tmp/mtah_weak_training.csv
"""

import argparse
import csv
import json
import math
import os
import xml.etree.ElementTree as ET
from collections import Counter
from itertools import zip_longest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SELECTED_EVENT_TYPES = {"PASS", "CARRY", "SHOT", "RECOVERY"}
NEAR_RADIUS_METRES = 2.5


def build_sparse_labels(events, stride=5):
    """Map sampled (period, frame) keys to H/A or None for conflicting event teams."""
    if stride < 1:
        raise ValueError("stride must be positive")

    labels = {}
    for event in events:
        event_type = event["type"].upper()
        team = event["team"]
        if event_type not in SELECTED_EVENT_TYPES or team not in ("H", "A"):
            continue
        start = int(event["start"])
        end = max(start, int(event["end"]))
        for frame in range(start, end + 1, stride):
            key = (int(event["period"]), frame)
            if key not in labels:
                labels[key] = team
            elif labels[key] is not None and labels[key] != team:
                labels[key] = None
    return labels


def _event_interval(period, start, end, team, event_type):
    return {
        "period": int(period),
        "start": int(start),
        "end": int(end),
        "team": team,
        "type": (event_type or "").strip().upper(),
    }


def load_csv_events(path):
    events = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            team_name = row["Team"].strip().lower()
            team = None
            if team_name == "home":
                team = "H"
            elif team_name == "away":
                team = "A"
            events.append(_event_interval(
                row["Period"], row["Start Frame"], row["End Frame"], team, row["Type"]))
    return events


def load_json_events(path, metadata_path):
    metadata = ET.parse(metadata_path).getroot()
    score = metadata.find(".//Score")
    if score is None:
        raise ValueError("{}: missing match team IDs".format(metadata_path))
    home_id = score.get("idLocalTeam")

    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    events = []
    for event in payload["data"]:
        team_id = (event.get("team") or {}).get("id")
        team = None
        if team_id == home_id:
            team = "H"
        elif team_id:
            team = "A"
        events.append(_event_interval(
            event["period"], event["start"]["frame"], event["end"]["frame"],
            team, (event.get("type") or {}).get("name")))
    return events


def _read_csv_tracking_header(path):
    handle = open(path, newline="", encoding="utf-8-sig")
    reader = csv.reader(handle)
    next(reader)
    next(reader)
    names = next(reader)
    player_columns = [index for index, name in enumerate(names)
                      if name.strip().lower().startswith("player")]
    ball_column = next((index for index, name in enumerate(names)
                        if name.strip().lower() == "ball"), None)
    if ball_column is None:
        handle.close()
        raise ValueError("{}: no Ball columns found".format(path))
    return handle, reader, player_columns, ball_column


def _valid_xy(x_value, y_value):
    if not x_value or not y_value or x_value[0].lower() == "n" or y_value[0].lower() == "n":
        return None
    try:
        x, y = float(x_value), float(y_value)
    except ValueError:
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def _tracking_features(player_groups, ball, team_by_slot, pitch):
    if ball is None:
        return {
            "ball_visible": 0,
            "home_nearest_m": -1.0,
            "away_nearest_m": -1.0,
            "home_near_2_5m": 0,
            "away_near_2_5m": 0,
        }

    length, width = pitch
    nearest = [float("inf"), float("inf")]
    nearby = [0, 0]
    for team, player in zip(team_by_slot, player_groups):
        if player is None:
            continue
        dx = (player[0] - ball[0]) * length
        dy = (player[1] - ball[1]) * width
        distance = math.hypot(dx, dy)
        nearest[team] = min(nearest[team], distance)
        nearby[team] += int(distance < NEAR_RADIUS_METRES)

    return {
        "ball_visible": 1,
        "home_nearest_m": nearest[0] if math.isfinite(nearest[0]) else -1.0,
        "away_nearest_m": nearest[1] if math.isfinite(nearest[1]) else -1.0,
        "home_near_2_5m": nearby[0],
        "away_near_2_5m": nearby[1],
    }


def iter_csv_tracking(home_path, away_path):
    home_handle, home_reader, home_columns, home_ball_column = _read_csv_tracking_header(home_path)
    away_handle, away_reader, away_columns, _ = _read_csv_tracking_header(away_path)
    try:
        for home_row, away_row in zip_longest(home_reader, away_reader):
            if home_row is None or away_row is None:
                raise ValueError("home/away tracking files have different row counts")
            home_key = (int(home_row[0]), int(home_row[1]))
            away_key = (int(away_row[0]), int(away_row[1]))
            if home_key != away_key:
                raise ValueError("home/away frame mismatch: {} != {}".format(home_key, away_key))

            player_groups = []
            team_by_slot = []
            for row, columns, team in (
                (home_row, home_columns, 0), (away_row, away_columns, 1)):
                for column in columns:
                    player_groups.append(_valid_xy(row[column], row[column + 1]))
                    team_by_slot.append(team)
            ball = _valid_xy(home_row[home_ball_column], home_row[home_ball_column + 1])
            yield home_key, _tracking_features(
                player_groups, ball, team_by_slot, (105.0, 68.0))
    finally:
        home_handle.close()
        away_handle.close()


def iter_epts_tracking(raw_path, metadata_path):
    metadata = ET.parse(metadata_path).getroot()
    width = float(metadata.findtext(".//FieldSize/Width") or 105)
    height = float(metadata.findtext(".//FieldSize/Height") or 68)
    score = metadata.find(".//Score")
    home_id = score.get("idLocalTeam") if score is not None else None
    team_ids = [player.get("teamId") for player in metadata.iter("Player")]
    team_by_slot = [0 if team_id == home_id else 1 for team_id in team_ids]
    period_ends = {
        parameter.findtext("Name"): parameter.findtext("Value")
        for parameter in metadata.iter("ProviderParameter")
    }
    first_half_end = int(period_ends.get("first_half_end") or 0)
    if first_half_end <= 0:
        raise ValueError("{}: missing first_half_end metadata".format(metadata_path))

    with open(raw_path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0])
            players = [_parse_epts_position(group) for group in parts[1].split(";")]
            ball = _parse_epts_position(parts[2])
            period = 1 if frame <= first_half_end else 2
            yield (period, frame), _tracking_features(
                players, ball, team_by_slot, (width, height))


def _parse_epts_position(raw_position):
    if not raw_position or raw_position[0].lower() == "n":
        return None
    return _valid_xy(*raw_position.split(",")[:2])


def sample_game(folder, stride):
    game = os.path.basename(folder)
    if game in ("Sample_Game_1", "Sample_Game_2"):
        event_path = os.path.join(folder, game + "_RawEventsData.csv")
        home_path = os.path.join(folder, game + "_RawTrackingData_Home_Team.csv")
        away_path = os.path.join(folder, game + "_RawTrackingData_Away_Team.csv")
        events = load_csv_events(event_path)
        tracking = iter_csv_tracking(home_path, away_path)
    elif game == "Sample_Game_3":
        event_path = os.path.join(folder, game + "_events.json")
        metadata_path = os.path.join(folder, game + "_metadata.xml")
        tracking_path = os.path.join(folder, game + "_tracking.txt")
        events = load_json_events(event_path, metadata_path)
        tracking = iter_epts_tracking(tracking_path, metadata_path)
    else:
        raise ValueError("unsupported sample match: {}".format(game))

    labels = build_sparse_labels(events, stride)
    selected, conflicts = [], 0
    for key, features in tracking:
        if key not in labels:
            continue
        label = labels[key]
        if label is None:
            conflicts += 1
            continue
        selected.append({
            "game": game,
            "period": key[0],
            "frame": key[1],
            "label": label,
            **features,
        })
    return selected, conflicts, labels


def write_rows(rows, output_path):
    fields = ("game", "period", "frame", "label", "ball_visible", "home_nearest_m",
              "away_nearest_m", "home_near_2_5m", "away_near_2_5m")
    folder = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(folder, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=os.path.join(ROOT, "sample_data"))
    parser.add_argument("--output", required=True,
                        help="output CSV path; keep derived samples outside the repository")
    parser.add_argument("--stride", type=int, default=5,
                        help="sample every Nth frame inside selected event spans")
    parser.add_argument("--games", nargs="+", default=("Sample_Game_1", "Sample_Game_2", "Sample_Game_3"))
    args = parser.parse_args()

    all_rows = []
    for game in args.games:
        rows, conflicts, _ = sample_game(os.path.join(args.data_dir, game), args.stride)
        all_rows.extend(rows)
        print("{}: {} labeled samples; {} conflicting samples skipped".format(
            game, len(rows), conflicts))
    write_rows(all_rows, args.output)
    print("wrote {} weakly labeled samples to {}".format(len(all_rows), args.output))


if __name__ == "__main__":
    main()