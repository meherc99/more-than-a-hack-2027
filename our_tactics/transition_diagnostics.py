"""transition_diagnostics_v2.py - aggregate tracking evidence around the transitions of a submission.

Uses the same alignment as possession_v2 (IdFrame -> video frame per half), reads only the
tactical tracking and your own submission, and never touches ground truth.

    python solver/transition_diagnostics_v2.py --submission evaluation/results/solver_v2.csv \
        --output /tmp/transition_diagnostics.csv

Use the SUMMARY to understand failure modes and to design general features or rules. Do not use
individual rows to edit labels or to add constants tied to specific frames: that is hand
correction (RULES.md, rule 1). Keep the output CSV outside the repository.
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from possession_solver import FPS, load_tactical  # noqa: E402

OFFSETS = (-12, 0, 12)
NEAR_M = 2.5


def read_submission(path, n_frames):
    labels = np.empty(n_frames, dtype="<U1")
    with open(path) as fh:
        if fh.readline().strip() != "frame,label":
            raise ValueError("header must be 'frame,label'")
        n = 0
        for i, line in enumerate(fh):
            frame, lab = line.strip().split(",")
            if int(frame) != i or i >= n_frames:
                raise ValueError("row %d: unexpected frame %s" % (i, frame))
            labels[i] = lab
            n += 1
    if n != n_frames:
        raise ValueError("expected %d rows, found %d" % (n_frames, n))
    return labels


def kind(before, after):
    if before in "HA" and after in "HA":
        return "team_switch"
    if after == "D" and before in "HA":
        return "to_dead"
    if before == "D" and after in "HA":
        return "from_dead_to_team"
    return "other"


def evidence(store, team, v):
    for s in store.values():
        if s["v0"] <= v <= s["v1"]:
            k = v - s["v0"]
            b = s["ball"][k]
            if np.isnan(b[0]):
                return dict(ball_visible=False, home_m="", away_m="", nearest="none", home_near=0, away_near=0)
            d = np.hypot(s["P"][k, :, 0] - b[0], s["P"][k, :, 1] - b[1])
            d0, d1 = d[team == 0], d[team == 1]
            h = float(np.nanmin(d0)) if np.any(~np.isnan(d0)) else float("inf")
            a = float(np.nanmin(d1)) if np.any(~np.isnan(d1)) else float("inf")
            nearest = "none" if h == a == float("inf") else ("H" if h < a else "A" if a < h else "tie")
            return dict(ball_visible=True, home_m=round(h, 2) if h < 1e9 else "",
                        away_m=round(a, 2) if a < 1e9 else "", nearest=nearest,
                        home_near=int(np.nansum(d0 < NEAR_M)), away_near=int(np.nansum(d1 < NEAR_M)))
    return None


def main():
    from games import GAME, GAMES, DEFAULT_SOURCE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--output", required=True, help="diagnostic CSV; keep it outside the repository")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--game", default=GAME)
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    args = ap.parse_args()

    n_frames = GAMES[args.game]["n_video_frames"]
    labels = read_submission(args.submission, n_frames)
    store, team, _ = load_tactical(args.data_dir, args.game, args.source)

    trans = [i for i in range(1, n_frames)
             if labels[i] != labels[i - 1] and labels[i] != "X" and labels[i - 1] != "X"]
    rows, summ = [], defaultdict(lambda: dict(n=0, vis=0, support=0, with_ball=0))
    for i in trans:
        k, tgt = kind(labels[i - 1], labels[i]), labels[i]
        s = summ[k]
        s["n"] += 1
        for off in OFFSETS:
            ev = evidence(store, team, i + off) if 0 <= i + off < n_frames else None
            if ev is None:
                continue
            rows.append(dict(frame=i, time_s=round(i / FPS, 2), kind=k, frm=labels[i - 1], to=tgt,
                             offset=off, supports_target=(tgt in "HA" and ev["nearest"] == tgt), **ev))
            if off == 0:
                s["vis"] += int(ev["ball_visible"])
                if tgt in "HA" and ev["ball_visible"]:
                    s["with_ball"] += 1
                    s["support"] += int(ev["nearest"] == tgt)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fields = ["frame", "time_s", "kind", "frm", "to", "offset", "ball_visible", "home_m", "away_m",
              "nearest", "home_near", "away_near", "supports_target"]
    with open(args.output, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    graded_min = np.sum(labels != "X") / FPS / 60
    print("transitions: %d (%.1f per graded minute)" % (len(trans), len(trans) / graded_min))
    print("%-20s %6s %14s %26s" % ("kind", "count", "ball visible", "nearest = new team (visible)"))
    for k, s in sorted(summ.items()):
        sup = "%.0f%%" % (100.0 * s["support"] / s["with_ball"]) if s["with_ball"] else "-"
        print("%-20s %6d %13.0f%% %26s" % (k, s["n"], 100.0 * s["vis"] / max(s["n"], 1), sup))
    print("wrote %d evidence rows to %s" % (len(rows), args.output))


if __name__ == "__main__":
    main()
