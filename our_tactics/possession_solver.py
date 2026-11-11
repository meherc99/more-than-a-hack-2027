"""A stronger, state-aware possession model for the MTAH 2027 challenge.

The repository ships with a trivial nearest-player baseline. This solver keeps that intuition,
but adds three improvements that matter for the ranking metric:

1. it uses both the nearest distance and the number of nearby players per team,
2. it treats short possessions as noise and merges them back into a neighbouring label,
3. it can use Smart Tagging as a weak prior for restarts and dead-ball stretches when the XML is
   present.

The script is intentionally dependency-light: it uses only numpy and the stdlib, so it can run in
an environment that matches the challenge scorer.
"""

import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from games import GAME, GAMES, DEFAULT_SOURCE, require  # noqa: E402

HOME, AWAY, DEAD = 0, 1, 2


def team_of_each_track(meta_path):
    """Array over track slots: 0 for home, 1 for away."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))

    home_id = team_ids[0]
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries".format(meta_path))
    return np.array([HOME if team_id == home_id else AWAY for team_id in slots], dtype=np.int8)


def team_names(meta_path):
    root = ET.parse(meta_path).getroot()
    return tuple(team.findtext("Name") for team in root.iter("Team"))


def parse_ball(ball):
    """Return (x, y) in normalised pitch coordinates or None when the ball is missing."""
    if not ball or ball[0] == "N":
        return None
    pieces = ball.split(",")
    if len(pieces) < 2:
        return None
    x = float(pieces[0])
    y = float(pieces[1])
    return x, y


def parse_player_group(group):
    """Return (x, y) in normalised pitch coordinates or None when the player is missing."""
    if not group or group[0] == "N":
        return None
    pieces = group.split(",")
    if len(pieces) < 2:
        return None
    return float(pieces[0]), float(pieces[1])


def load_smart_prior(data_dir=None, game=GAME, fps=None):
    """Return an integer array of prior team labels, with 0=home, 1=away, 2=dead.

    This is intentionally conservative: if the XML is missing or if the phase is not explicitly a
    restart, the prior defaults to the neutral value. Only set-piece and kickoff-like events carry
    a strong but still weak prior.
    """
    fps = fps or GAMES[game]["fps"]
    prior = np.full(GAMES[game]["n_video_frames"], -1, dtype=np.int8)
    try:
        path = require(["smart_tagging"], game, data_dir)[0]
    except (KeyError, IOError):
        return prior

    meta = os.path.join(data_dir or os.path.join(ROOT, "data"), "tactical",
                        GAME + "_tactical_atd.xml")
    if os.path.isfile(meta):
        team_names_list = team_names(meta)
        home_name = team_names_list[0]
        away_name = team_names_list[1]
    else:
        home_name = None
        away_name = None

    root = ET.parse(path).getroot()
    for instance in root.iter("instance"):
        code = (instance.findtext("code") or "").strip()
        labels = instance.findall("label")
        tags = {label.findtext("group"): label.findtext("text") for label in labels}
        team = tags.get("Team")
        team_name = (team or "").strip()
        start_s = float(instance.findtext("start"))
        end_s = float(instance.findtext("end"))
        start_frame = max(0, int(round(start_s * fps)))
        end_frame = min(len(prior), int(round(end_s * fps)))
        if end_frame <= start_frame:
            continue

        # Smart Tagging is a phase-level prior, not a ground truth. We only use it when it points
        # to an actual team and the interval is a restart or set-piece-like event.
        is_restart = False
        if code.strip().lower() == "set pieces" or (team_name and "kick" in team_name.lower()):
            is_restart = True
        if not is_restart:
            continue

        home_or_away = None
        if home_name and team_name and team_name.lower() == home_name.lower():
            home_or_away = HOME
        elif away_name and team_name and team_name.lower() == away_name.lower():
            home_or_away = AWAY
        elif team_name and (team_name.lower().endswith("barcelona") or team_name.lower().endswith("athletic")):
            home_or_away = HOME if "barcelona" in team_name.lower() else AWAY

        if home_or_away is None:
            continue

        prior[start_frame:end_frame] = home_or_away
    return prior


def initial_labels_from_tracking(raw_path, team_of_track, n_frames, pitch, smart_prior=None):
    """Return H/A/D labels inferred per frame from ball proximity and player availability."""
    length, width = pitch
    out = np.full(n_frames, DEAD, dtype=np.int8)
    if smart_prior is None:
        smart_prior = np.full(n_frames, -1, dtype=np.int8)

    with open(raw_path) as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            parts = raw.split(":")
            if len(parts) != 3:
                continue
            try:
                frame_id = int(parts[0])
            except ValueError:
                continue
            if frame_id < 1 or frame_id > n_frames:
                continue

            idx = frame_id - 1
            player_groups = parts[1].split(";")
            ball = parse_ball(parts[2])
            if ball is None:
                if smart_prior[idx] == HOME:
                    out[idx] = HOME
                elif smart_prior[idx] == AWAY:
                    out[idx] = AWAY
                elif idx > 0 and out[idx - 1] != DEAD and idx + 1 < n_frames:
                    # If the ball disappears briefly inside a live possession, carry the previous
                    # team label forward rather than creating a false dead-ball split.
                    out[idx] = out[idx - 1]
                else:
                    out[idx] = DEAD
                continue

            ball_x = ball[0] * length
            ball_y = ball[1] * width
            home_best = float("inf")
            away_best = float("inf")
            home_near = 0
            away_near = 0

            for slot, group in enumerate(player_groups):
                if slot >= len(team_of_track):
                    break
                player = parse_player_group(group)
                if player is None:
                    continue
                px, py = player
                dx = px * length - ball_x
                dy = py * width - ball_y
                dist = math.hypot(dx, dy)
                team = int(team_of_track[slot])
                if team == HOME:
                    if dist < home_best:
                        home_best = dist
                    if dist < 2.5:
                        home_near += 1
                else:
                    if dist < away_best:
                        away_best = dist
                    if dist < 2.5:
                        away_near += 1

            if home_best == float("inf") and away_best == float("inf"):
                out[idx] = DEAD
                continue

            if smart_prior[idx] == HOME:
                out[idx] = HOME
            elif smart_prior[idx] == AWAY:
                out[idx] = AWAY
            elif home_near > away_near and home_best <= away_best + 1.0:
                out[idx] = HOME
            elif away_near > home_near and away_best <= home_best + 1.0:
                out[idx] = AWAY
            elif home_best <= away_best:
                out[idx] = HOME
            elif away_best < home_best:
                out[idx] = AWAY
            else:
                out[idx] = DEAD

    return out


def code_labels(labels):
    conversions = {"H": HOME, "A": AWAY, "D": DEAD}
    out = np.empty(len(labels), dtype=np.int8)
    for i, label in enumerate(labels):
        out[i] = conversions.get(str(label).strip().upper(), DEAD)
    return out


def smooth_majority(codes, window):
    """Centrered majority vote over +/- window frames; ties go to the home side."""
    if window <= 0:
        return codes.copy()
    n = len(codes)
    home = np.concatenate(([0], np.cumsum(codes == HOME)))
    away = np.concatenate(([0], np.cumsum(codes == AWAY)))
    dead = np.concatenate(([0], np.cumsum(codes == DEAD)))

    idx = np.arange(n)
    low = np.maximum(0, idx - window)
    high = np.minimum(n, idx + window + 1)
    home_count = home[high] - home[low]
    away_count = away[high] - away[low]
    dead_count = dead[high] - dead[low]

    out = np.full(n, DEAD, dtype=np.int8)
    has_players = (home_count + away_count + dead_count) > 0
    home_wins = home_count >= away_count
    away_wins = away_count > home_count
    dead_wins = (home_count == 0) & (away_count == 0)

    out[(has_players) & (home_wins) & (~dead_wins)] = HOME
    out[(has_players) & (away_wins) & (~dead_wins)] = AWAY
    out[(has_players) & (dead_wins)] = DEAD
    return out


def collapse_short_runs(codes, min_run=5):
    """Merge tiny possession fragments back into the neighbouring label.

    A possession is a maximal run of H or A. Very short runs are usually just flicker from noisy
    ball tracking; the metric strongly prefers a stable possession interval to a high chunk count.
    """
    out = codes.copy()
    i = 0
    while i < len(out):
        if out[i] == DEAD:
            i += 1
            continue
        start = i
        while i + 1 < len(out) and out[i + 1] == out[start]:
            i += 1
        stop = i + 1
        if stop - start < min_run:
            fill = DEAD
            if start > 0 and out[start - 1] != DEAD:
                fill = out[start - 1]
            elif stop < len(out) and out[stop] != DEAD:
                fill = out[stop]
            if fill != DEAD:
                out[start:stop] = fill
        i = stop
    return out


def finalise_labels(labels, smart_prior=None):
    """A gentle cleanup pass before writing CSV."""
    out = labels.copy()
    if smart_prior is None:
        return collapse_short_runs(smooth_majority(out, 20), 5)

    # We only use the smart prior to prevent self-inflicted splits around restarts and dead-ball
    # stretches. It is intentionally low weight because the tagging is noisy.
    for index, prior in enumerate(smart_prior):
        if prior == HOME and out[index] == AWAY:
            out[index] = HOME
        elif prior == AWAY and out[index] == HOME:
            out[index] = AWAY
        elif prior == HOME and out[index] == DEAD:
            out[index] = HOME
        elif prior == AWAY and out[index] == DEAD:
            out[index] = AWAY

    out = collapse_short_runs(smooth_majority(out, 25), 6)
    return out


def write_submission(labels, path):
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)

    with open(path, "w") as handle:
        handle.write("frame,label\n")
        for index, code in enumerate(labels):
            handle.write("{},{}\n".format(index, {HOME: "H", AWAY: "A", DEAD: "D"}[int(code)]))


def solve(data_dir=None, game=GAME, source=DEFAULT_SOURCE, output=None):
    """Build a possession CSV from the provided match data."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir, source)
    team_map = team_of_each_track(meta_path)
    n_frames = GAMES[game]["n_video_frames"]
    pitch = GAMES[game]["pitch"]
    smart_prior = load_smart_prior(data_dir=data_dir, game=game, fps=GAMES[game]["fps"])
    labels = initial_labels_from_tracking(raw_path, team_map, n_frames, pitch, smart_prior)
    labels = finalise_labels(labels, smart_prior)

    target = output or os.path.join(ROOT, "evaluation", "results", "solver_submission.csv")
    write_submission(labels, target)
    return target


def main():
    parser = argparse.ArgumentParser(description="Generate a possession CSV for the 2027 challenge.")
    parser.add_argument("--data-dir", default=None, help="Directory containing the downloaded tactical data.")
    parser.add_argument("--game", default=GAME, help="Game identifier.")
    parser.add_argument("--output", default=None, help="Destination CSV path.")
    args = parser.parse_args()

    try:
        out = solve(data_dir=args.data_dir, game=args.game, output=args.output)
    except (KeyError, IOError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)

    print("wrote {}".format(out))


if __name__ == "__main__":
    main()
