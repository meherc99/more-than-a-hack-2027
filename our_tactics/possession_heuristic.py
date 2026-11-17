"""possession_heuristic.py - rule-based fallback (no training) for the BAR-ATH possession task.

A cleaned-up version of the original nearest-player solver. Fixes:
  * tracking IdFrames are mapped to video frames per half (games.py offsets), not "id - 1"
  * home/away come from the metadata xml; nothing is hard-coded to a team name
  * X is written only outside the graded halves, never inside them
  * possession is held by the last team that CONTROLLED the ball, so a pass in flight or a
    flickering nearest player no longer flips the label
  * long ball gaps and a ball outside the lines are dead ball (D)
Smart Tagging is not used. Every constant below is a stated assumption, none is tuned on the
public ground truth or on the video.

    python solver/possession_heuristic.py --out evaluation/results/heuristic_v2.csv
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from possession_solver import (DEAD, FPS, HOME, AWAY, L, LETTER, W, fill_ball,  # noqa: E402
                           load_tactical, runs_min_len)

GAP_DEAD_S = 4.0         # ball unseen for longer than this -> dead ball (live gaps are short)
CTRL_RADIUS = 3.0        # metres: a player this close to the observed ball is "on the ball"
CTRL_FRAMES = 10         # 0.4 s of continuous nearest-player control = a real touch/control
MIN_POSSESSION_S = 3.0   # a possession shorter than this is merged into its neighbour


def merge_short(lab, min_len):
    """Absorb H/A spells shorter than min_len into the longer neighbouring spell."""
    lab = lab.copy()
    while True:
        b = np.flatnonzero(np.diff(lab) != 0) + 1
        starts, ends = np.concatenate(([0], b)), np.concatenate((b, [len(lab)]))
        lens = ends - starts
        short = np.flatnonzero(lens < min_len)
        if len(starts) == 1 or len(short) == 0:
            return lab
        i = short[np.argmin(lens[short])]
        nb = [k for k in (i - 1, i + 1) if 0 <= k < len(starts)]
        j = max(nb, key=lambda k: lens[k])
        lab[starts[i]:ends[i]] = lab[starts[j]]


def label_half(ball, P, team):
    T = len(ball)
    est, vis, _, _ = fill_ball(ball, max_interp=int(GAP_DEAD_S * FPS))
    miss = ~vis
    edges = np.diff(np.concatenate(([0], miss.astype(np.int8), [0])))
    dead = np.zeros(T, bool)
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if e - s > GAP_DEAD_S * FPS:
            dead[s:e] = True
    with np.errstate(all="ignore"):
        dead |= vis & ((ball[:, 0] < 0) | (ball[:, 0] > L) | (ball[:, 1] < 0) | (ball[:, 1] > W))
        Dm = np.hypot(P[:, :, 0] - est[:, 0:1], P[:, :, 1] - est[:, 1:2])
    D99 = np.where(np.isnan(Dm), 99.0, Dm)
    d0, d1 = D99[:, team == 0].min(1), D99[:, team == 1].min(1)
    o0, o1 = np.where(vis, d0, 99.0), np.where(vis, d1, 99.0)
    near = np.where(np.minimum(o0, o1) <= CTRL_RADIUS, np.where(o0 <= o1, 0, 1), -1).astype(np.int8)
    ctrl = runs_min_len(near, CTRL_FRAMES)

    lab = np.full(T, DEAD, dtype=np.int8)
    b = np.flatnonzero(np.diff(dead.astype(np.int8)) != 0) + 1
    for s, e in zip(np.concatenate(([0], b)), np.concatenate((b, [T]))):
        if dead[s]:
            continue
        seg = ctrl[s:e]
        idx = np.flatnonzero(seg >= 0)
        if len(idx) == 0:                       # no control seen: majority nearest team
            lab[s:e] = HOME if np.sum(d0[s:e] <= d1[s:e]) >= np.sum(d0[s:e] > d1[s:e]) else AWAY
            continue
        ii = np.maximum.accumulate(np.where(seg >= 0, np.arange(e - s), -1))
        lab[s:e] = merge_short(np.where(ii >= 0, seg[np.clip(ii, 0, None)], seg[idx[0]]).astype(np.int8),
                               int(MIN_POSSESSION_S * FPS))
    return lab


def main():
    from games import GAMES, GAME, DEFAULT_SOURCE, anchor_video_frame
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--game", default=GAME)
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "heuristic_v2.csv"))
    args = ap.parse_args()

    store, team, names = load_tactical(args.data_dir, args.game, args.source)
    g = GAMES[args.game]
    out = np.full(g["n_video_frames"], "X", dtype="<U1")
    kick = {1: g.get("kickoff"), 2: g.get("kickoff_2nd")}
    for half, s in sorted(store.items()):
        lab = label_half(s["ball"], s["P"], team)
        if kick.get(half):
            k = anchor_video_frame(kick[half]) - s["v0"]
            lab[:max(0, min(k, len(lab)))] = DEAD
        out[s["v0"]:s["v1"] + 1] = LETTER[lab]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("frame,label\n")
        fh.write("\n".join("%d,%s" % (i, out[i]) for i in range(len(out))) + "\n")
    graded = out != "X"
    print("wrote %s (%d rows)" % (args.out, len(out)))
    for c in "HAD":
        print("  %s: %.1f%% of graded frames" % (c, 100.0 * np.mean(out[graded] == c)))
    chg = int(np.sum((out[1:] != out[:-1]) & np.isin(out[1:], ["H", "A"]) & np.isin(out[:-1], ["H", "A"])))
    print("  direct H<->A changes: %.1f per graded minute" % (chg / (graded.sum() / FPS / 60)))
    print("next: python evaluation/validate_submission.py %s" % args.out)


if __name__ == "__main__":
    main()
