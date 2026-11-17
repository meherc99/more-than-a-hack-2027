"""possession_v2.py - possession labels (H/A/D/X) for the BAR-ATH tactical video.

Pipeline
    prepare : download the 7 open DFL matches (CC-BY 4.0) with kloppy -> cache/dfl/*.npz
    check   : read the hackathon tracking, verify alignment / teams, write aggregate stats
    train   : degrade DFL tracking so it looks like the hackathon tracking, build
              per-frame features, fit LightGBM (H/A/D), choose the Viterbi penalties by
              leave-one-match-out on DFL
    predict : hackathon tracking -> features -> LightGBM -> Viterbi -> submission CSV

Rules compliance
    * the only input read from the hackathon bundle is the tactical tracking (atd.xml, atd.txt)
    * every learned parameter comes from other games (DFL); nothing is fitted or tuned on the
      public 5-minute ground truth or on the video
    * deterministic: fixed seeds, no randomness at prediction time
"""
import argparse
import glob
import json
import os
import re
import sys
import warnings
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

FPS = 25
L, W = 105.3, 68.0          # pitch size used for every game (metres)
STRIDE = 5                  # model and decoder run at 5 fps
HOME, AWAY, DEAD = 0, 1, 2  # class codes; letters H / A / D
LETTER = np.array(["H", "A", "D"])

DFL_MATCH_IDS = ["J03WPY", "J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPF", "J03WQF"]

# Degrader defaults (from the suggestion repo's documented calibration of the hackathon ATD:
# ball missing on ~7% of live frames and ~80% of dead frames). `check` prints the real overall
# missing-ball fraction so you can confirm it is near 41%.
DEAD_GAP_P = 0.80
LIVE_GAP_FRAC = 0.07


# ----------------------------------------------------------------------------------------
# small numpy helpers
# ----------------------------------------------------------------------------------------
def _last_idx(valid):
    return np.maximum.accumulate(np.where(valid, np.arange(len(valid)), -1))


def _next_idx(valid):
    T = len(valid)
    return np.minimum.accumulate(np.where(valid, np.arange(T), T)[::-1])[::-1]


def win_mean(x, lo, hi):
    """Mean of x over [t+lo, t+hi] (inclusive), ignoring NaN."""
    T = len(x)
    ok = ~np.isnan(x)
    cs = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    cn = np.concatenate(([0], np.cumsum(ok)))
    idx = np.arange(T)
    a = np.clip(idx + lo, 0, T)
    b = np.clip(idx + hi + 1, 0, T)
    n = cn[b] - cn[a]
    s = cs[b] - cs[a]
    out = np.full(T, np.nan)
    m = n > 0
    out[m] = s[m] / n[m]
    return out


def runs_min_len(code, min_len):
    """Keep only runs of identical non-negative codes that last >= min_len frames."""
    T = len(code)
    out = np.full(T, -1, dtype=np.int8)
    b = np.flatnonzero(np.diff(code) != 0) + 1
    starts = np.concatenate(([0], b))
    ends = np.concatenate((b, [T]))
    for s, e in zip(starts, ends):
        if code[s] >= 0 and e - s >= min_len:
            out[s:e] = code[s]
    return out


def fill_ball(ball, max_interp=150):
    """Ball estimate for every frame: observed, linearly interpolated over gaps <= 6 s, else the
    nearest observation in time. Also returns visibility and seconds since / until an observation."""
    T = len(ball)
    t = np.arange(T)
    vis = ~np.isnan(ball[:, 0])
    li, ni = _last_idx(vis), _next_idx(vis)
    has_l, has_n = li >= 0, ni < T
    since = np.where(has_l, (t - li) / FPS, 60.0).clip(0, 60)
    until = np.where(has_n, (ni - t) / FPS, 60.0).clip(0, 60)
    est = ball.copy()
    miss = ~vis
    both = miss & has_l & has_n & ((ni - li) <= max_interp)
    if both.any():
        w = ((t - li) / np.maximum(ni - li, 1))[both][:, None]
        est[both] = ball[li[both]] * (1 - w) + ball[ni[both]] * w
    rest = miss & ~both
    use_l = rest & has_l & (~has_n | ((t - li) <= (ni - t)))
    use_n = rest & has_n & ~use_l
    est[use_l] = ball[li[use_l]]
    est[use_n] = ball[ni[use_n]]
    return est, vis, since, until


# ----------------------------------------------------------------------------------------
# features (no player identities, symmetric in home/away and in pitch direction)
# ----------------------------------------------------------------------------------------
WINDOWS = [(-12, 12), (-50, 50), (-150, 150), (-375, 375), (-125, 0), (-375, 0), (0, 125), (0, 375)]


def _team_shape(P, m):
    X, Y = P[:, m, 0], P[:, m, 1]
    ok = ~np.isnan(X)
    n = ok.sum(1)
    den = np.maximum(n, 1)
    cx = np.where(n > 0, np.nansum(X, 1) / den, np.nan)
    cy = np.where(n > 0, np.nansum(Y, 1) / den, np.nan)
    sx = np.where(n > 1, np.sqrt(np.nansum((X - cx[:, None]) ** 2, 1) / den), np.nan)
    sy = np.where(n > 1, np.sqrt(np.nansum((Y - cy[:, None]) ** 2, 1) / den), np.nan)
    return cx, cy, sx, sy, n.astype(float)


def build_features(ball, P, team, swap=False):
    """ball (T,2) metres with NaN, P (T,N,2) metres with NaN, team (N,) in {0,1,-1}.
    With swap=True the two teams exchange roles (used for augmentation and test-time averaging)."""
    T = len(ball)
    t = np.arange(T)
    tm = np.where(team == 0, 1, np.where(team == 1, 0, -1)) if swap else team
    m0, m1 = tm == 0, tm == 1
    if m0.sum() == 0 or m1.sum() == 0:
        raise ValueError("a team has no tracking slots")
    est, vis, since, until = fill_ball(ball)

    with np.errstate(all="ignore"):
        Dm = np.hypot(P[:, :, 0] - est[:, 0:1], P[:, :, 1] - est[:, 1:2])
        D99 = np.where(np.isnan(Dm), 99.0, Dm)
        d0, d1 = D99[:, m0].min(1), D99[:, m1].min(1)
        c3 = (Dm[:, m0] <= 3).sum(1) - (Dm[:, m1] <= 3).sum(1)
        c6 = (Dm[:, m0] <= 6).sum(1) - (Dm[:, m1] <= 6).sum(1)
        c12 = (Dm[:, m0] <= 12).sum(1) - (Dm[:, m1] <= 12).sum(1)

        # control: nearest player within 3 m of an OBSERVED ball for >= 0.4 s
        o0, o1 = np.where(vis, d0, 99.0), np.where(vis, d1, 99.0)
        near = np.where(np.minimum(o0, o1) <= 3.0, np.where(o0 <= o1, 0, 1), -1).astype(np.int8)
        ctrl = runs_min_len(near, 10)
        cv = ctrl >= 0
        li, ni = _last_idx(cv), _next_idx(cv)
        lt = np.where(li >= 0, ctrl[np.clip(li, 0, T - 1)], -1)
        nt = np.where(ni < T, ctrl[np.clip(ni, 0, T - 1)], -1)
        t_since = np.where(li >= 0, (t - li) / FPS, 60.0).clip(0, 60)
        t_until = np.where(ni < T, (ni - t) / FPS, 60.0).clip(0, 60)
        enc = lambda c: np.where(c == 0, 1.0, np.where(c == 1, -1.0, 0.0))
        ctrl_change = ((lt >= 0) & (nt >= 0) & (lt != nt)).astype(float)

        # ball kinematics from positions (the file's own ball speed is mostly missing)
        bspd = np.full(T, np.nan)
        both = vis[1:] & vis[:-1]
        sp = np.hypot(np.diff(ball[:, 0]), np.diff(ball[:, 1])) * FPS
        bspd[1:][both] = sp[both]
        a, b = np.clip(t - 12, 0, T - 1), np.clip(t + 12, 0, T - 1)
        bmove = np.hypot(est[b, 0] - est[a, 0], est[b, 1] - est[a, 1])
        gl = np.minimum(est[:, 0], L - est[:, 0])
        tl = np.minimum(est[:, 1], W - est[:, 1])
        oob = ((est[:, 0] < 0) | (est[:, 0] > L) | (est[:, 1] < 0) | (est[:, 1] > W)).astype(float)

        # team shape
        c0x, c0y, s0x, s0y, n0 = _team_shape(P, m0)
        c1x, c1y, s1x, s1y, n1 = _team_shape(P, m1)
        cspd = np.full(T, np.nan)
        for cx, cy in ((c0x, c0y), (c1x, c1y)):
            s = np.full(T, np.nan)
            s[FPS:] = np.hypot(cx[FPS:] - cx[:-FPS], cy[FPS:] - cy[:-FPS])
            cspd = s if np.isnan(cspd).all() else (cspd + s) / 2
        att = (est[:, 0] - (c0x + c1x) / 2) * np.sign(c1x - c0x) / L
        bd0 = np.hypot(est[:, 0] - c0x, est[:, 1] - c0y)
        bd1 = np.hypot(est[:, 0] - c1x, est[:, 1] - c1y)
        sep = np.hypot(c0x - c1x, c0y - c1y)

    dmin = np.minimum(d0, d1)
    base = {
        "dd": np.clip(d1 - d0, -30, 30),
        "c3": c3.astype(float), "c12": c12.astype(float), "att": att,
        "vis": vis.astype(float), "ctrl0": (ctrl == 0).astype(float), "ctrl1": (ctrl == 1).astype(float),
        "nvis": n0 + n1, "cspd": cspd, "dmin": np.minimum(dmin, 40), "bspd": bspd, "bmove": bmove,
    }
    cols, names = [], []

    def add(name, x):
        names.append(name)
        cols.append(np.asarray(x, dtype=np.float32))

    for k, v in base.items():
        add(k, v)
    for k, v in (("since_seen", since), ("until_seen", until), ("gap", since + until),
                 ("d0", np.minimum(d0, 40)), ("d1", np.minimum(d1, 40)), ("c6", c6), ("gl", gl),
                 ("tl", tl), ("oob", oob), ("last_ctrl", enc(lt)), ("next_ctrl", enc(nt)),
                 ("t_since_ctrl", t_since), ("t_until_ctrl", t_until), ("ctrl_change", ctrl_change),
                 ("bd0", bd0), ("bd1", bd1), ("sep", sep), ("s0x", s0x), ("s0y", s0y), ("s1x", s1x),
                 ("s1y", s1y), ("n0", n0), ("n1", n1)):
        add(k, v)
    for k, v in base.items():
        for lo, hi in WINDOWS:
            add("%s_w%d_%d" % (k, lo, hi), win_mean(np.asarray(v, dtype=float), lo, hi))
    return np.column_stack(cols), names


# ----------------------------------------------------------------------------------------
# decoder
# ----------------------------------------------------------------------------------------
def viterbi(logp, sw, dead):
    """3-state Viterbi. sw = cost of H<->A, dead = cost of entering or leaving D."""
    T = len(logp)
    pen = np.array([[0, sw, dead], [sw, 0, dead], [dead, dead, 0]], dtype=float)
    score = logp[0].copy()
    back = np.zeros((T, 3), dtype=np.int8)
    for i in range(1, T):
        cand = score[:, None] - pen
        back[i] = cand.argmax(0)
        score = cand.max(0) + logp[i]
    path = np.zeros(T, dtype=np.int8)
    path[-1] = score.argmax()
    for i in range(T - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def decode(proba, sw, dead):
    return viterbi(np.log(np.clip(proba, 1e-6, 1.0)), sw, dead)


# ----------------------------------------------------------------------------------------
# proxy metric (NOT the official scorer; used only to pick penalties on other games)
# ----------------------------------------------------------------------------------------
def _spells(lab):
    b = np.flatnonzero(np.diff(lab) != 0) + 1
    starts = np.concatenate(([0], b))
    ends = np.concatenate((b, [len(lab)]))
    return [(int(lab[s]), int(s), int(e)) for s, e in zip(starts, ends) if lab[s] in (HOME, AWAY)]


def proxy_spell_f1(true, pred, thr=0.5):
    ts, ps = _spells(true), _spells(pred)
    if not ts or not ps:
        return 0.0
    used, hit = set(), 0
    for k, a, b in ts:
        best, bj = 0.0, -1
        for j, (k2, c, d) in enumerate(ps):
            if k2 != k or j in used:
                continue
            inter = min(b, d) - max(a, c)
            if inter <= 0:
                continue
            iou = inter / (max(b, d) - min(a, c))
            if iou > best:
                best, bj = iou, j
        if best >= thr:
            hit += 1
            used.add(bj)
    p, r = hit / len(ps), hit / len(ts)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


# ----------------------------------------------------------------------------------------
# DFL: download / convert
# ----------------------------------------------------------------------------------------
def _call_with_fallback(fn, **kw):
    """kloppy versions differ in accepted keywords; drop the optional ones that are rejected."""
    kw = dict(kw)
    while True:
        try:
            return fn(**kw)
        except TypeError as err:
            for k in ("only_alive", "coordinates"):
                if k in kw and k in str(err):
                    del kw[k]
                    break
            else:
                raise


def _convert_dataset(ds, out_prefix):
    from kloppy.domain import BallState, Ground
    cols, grounds = {}, []
    for fr in ds.frames:
        for p in fr.players_data:
            if p.player_id not in cols and p.team.ground in (Ground.HOME, Ground.AWAY):
                cols[p.player_id] = len(cols)
                grounds.append(0 if p.team.ground == Ground.HOME else 1)
    team = np.array(grounds, dtype=np.int8)
    counts = {}
    for fr in ds.frames:
        counts[fr.period.id] = counts.get(fr.period.id, 0) + 1
    store = {}
    for pid, n in counts.items():
        store[pid] = dict(ball=np.full((n, 2), np.nan, np.float32),
                          P=np.full((n, len(cols), 2), np.nan, np.float32),
                          alive=np.zeros(n, bool), owner=np.full(n, -1, np.int8), k=0)
    for fr in ds.frames:
        s = store[fr.period.id]
        k = s["k"]
        s["k"] += 1
        bc = fr.ball_coordinates
        if bc is not None and bc.x is not None:
            s["ball"][k] = (bc.x * L, bc.y * W)
        for p, pd in fr.players_data.items():
            c = pd.coordinates
            if c is not None and p.player_id in cols:
                s["P"][k, cols[p.player_id]] = (c.x * L, c.y * W)
        s["alive"][k] = fr.ball_state == BallState.ALIVE
        ot = fr.ball_owning_team
        if ot is not None and ot.ground in (Ground.HOME, Ground.AWAY):
            s["owner"][k] = 0 if ot.ground == Ground.HOME else 1
    written = []
    for pid, s in sorted(store.items()):
        path = "%s_p%d.npz" % (out_prefix, pid)
        np.savez_compressed(path, ball=s["ball"], P=s["P"], team=team, alive=s["alive"], owner=s["owner"])
        written.append(path)
        print("  period %d: %d frames, %d players, %.0f%% alive" % (pid, len(s["alive"]), len(team), 100 * s["alive"].mean()))
    return written


def cmd_prepare(args):
    from kloppy import sportec
    os.makedirs(args.out, exist_ok=True)
    if args.local_dir:
        raws = glob.glob(os.path.join(args.local_dir, "**", "*positions_raw*.xml"), recursive=True)
        jobs = []
        for raw in raws:
            token = re.search(r"DFL-MAT-[A-Z0-9]+", os.path.basename(raw))
            if not token:
                continue
            metas = glob.glob(os.path.join(args.local_dir, "**", "*matchinformation*%s*.xml" % token.group(0)), recursive=True)
            if metas:
                jobs.append((token.group(0), metas[0], raw))
        if not jobs:
            sys.exit("no (matchinformation, positions_raw) XML pairs found under " + args.local_dir)
        for name, meta, raw in sorted(jobs):
            print("loading", name)
            ds = _call_with_fallback(sportec.load_tracking, meta_data=meta, raw_data=raw,
                                     only_alive=False, coordinates="kloppy")
            _convert_dataset(ds, os.path.join(args.out, name))
    else:
        for mid in args.match_ids:
            print("downloading + loading", mid)
            ds = _call_with_fallback(sportec.load_open_tracking_data, match_id=mid,
                                     only_alive=False, coordinates="kloppy")
            _convert_dataset(ds, os.path.join(args.out, mid))
    print("done ->", args.out)


# ----------------------------------------------------------------------------------------
# DFL -> hackathon-like training data
# ----------------------------------------------------------------------------------------
def make_labels(ball, P, team, alive, owner):
    """Possession labels on CLEAN tracking. D when the ball is dead; in live play the possession
    changes only when the other team controls the ball (nearest player within 1.5 m for >= 0.4 s),
    dated from the first frame of that control. Before the first control of a live stretch the
    label is that first controller; a stretch with no control takes DFL's owning team."""
    T = len(ball)
    lab = np.full(T, DEAD, dtype=np.int8)
    with np.errstate(all="ignore"):
        d = np.hypot(P[:, :, 0] - ball[:, 0:1], P[:, :, 1] - ball[:, 1:2])
    d99 = np.where(np.isnan(d), 99.0, d)
    d0, d1 = d99[:, team == 0].min(1), d99[:, team == 1].min(1)
    near = np.where(np.minimum(d0, d1) <= 1.5, np.where(d0 <= d1, 0, 1), -1).astype(np.int8)
    near[~alive] = -1
    ctrl = runs_min_len(near, 10)
    b = np.flatnonzero(np.diff(alive.astype(np.int8)) != 0) + 1
    starts, ends = np.concatenate(([0], b)), np.concatenate((b, [T]))
    for s, e in zip(starts, ends):
        if not alive[s]:
            continue
        seg = ctrl[s:e]
        idx = np.flatnonzero(seg >= 0)
        if len(idx) == 0:
            o = owner[s:e]
            o = o[o >= 0]
            lab[s:e] = int(np.bincount(o, minlength=2).argmax()) if len(o) else HOME
            continue
        ii = np.maximum.accumulate(np.where(seg >= 0, np.arange(e - s), -1))
        lab[s:e] = np.where(ii >= 0, seg[np.clip(ii, 0, None)], seg[idx[0]])
    return lab


def degrade(ball, P, alive, rng, stats):
    """Make clean DFL tracking look like the hackathon ATD: noisy, smoothed ball, long ball gaps
    (mostly during dead ball, ending at the restart), player drop-outs."""
    T = len(ball)
    b, _, _, _ = fill_ball(ball, max_interp=10 ** 9)
    b = np.nan_to_num(b, nan=0.0) + rng.normal(0, 0.3, (T, 2))
    kern = np.ones(5) / 5
    for c in range(2):
        b[:, c] = np.convolve(b[:, c], kern, mode="same")
    Q = P + rng.normal(0, 0.15, P.shape)
    hide = np.zeros(T, bool)
    d = np.flatnonzero(np.diff(alive.astype(np.int8)) != 0) + 1
    starts, ends = np.concatenate(([0], d)), np.concatenate((d, [T]))
    for s, e in zip(starts, ends):
        if not alive[s] and rng.random() < stats["dead_gap_p"]:
            hide[s + int(rng.random() * 0.5 * (e - s)):e] = True
    gaps = np.asarray(stats["gap_frames"]) if stats.get("gap_frames") else np.arange(15, 151)
    n_gaps = int(alive.sum() * stats["live_gap_frac"] / max(gaps.mean(), 1))
    live_idx = np.flatnonzero(alive)
    if len(live_idx):
        for s in rng.choice(live_idx, size=n_gaps):
            hide[s:s + int(rng.choice(gaps))] = True
    b[hide] = np.nan
    for n in range(Q.shape[1]):
        for _ in range(rng.poisson(T / FPS / 120.0)):
            s = rng.integers(0, T)
            Q[s:s + int(rng.uniform(2, 15) * FPS), n] = np.nan
    Q[np.isnan(P)] = np.nan
    return b.astype(np.float64), Q.astype(np.float64)


def load_stats(path):
    stats = {"dead_gap_p": DEAD_GAP_P, "live_gap_frac": LIVE_GAP_FRAC, "gap_frames": []}
    if path and os.path.isfile(path):
        with open(path) as fh:
            stats["gap_frames"] = json.load(fh).get("gap_frames", [])
        print("degrader uses gap lengths from", path)
    else:
        print("no atd_stats.json found: degrader uses default gap lengths (run `check` first for better calibration)")
    return stats


# ----------------------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------------------
def match_sequences(files, draws, seed, stats):
    seqs = []
    for f in sorted(files):
        z = np.load(f)
        ball, P, team = z["ball"].astype(np.float64), z["P"].astype(np.float64), z["team"]
        alive, owner = z["alive"], z["owner"]
        lab = make_labels(ball, P, team, alive, owner)
        for dr in range(draws):
            rng = np.random.default_rng(seed + 1000 * dr + sum(map(ord, os.path.basename(f))))
            b, Q = degrade(ball, P, alive, rng, stats)
            for swap in (False, True):
                X, names = build_features(b, Q, team, swap=swap)
                y = lab if not swap else np.where(lab == HOME, AWAY, np.where(lab == AWAY, HOME, DEAD))
                seqs.append(dict(X=X[::STRIDE], y=y[::STRIDE].astype(np.int8), swap=swap, draw=dr, file=f))
    return seqs, names


def fit_lgbm(X, y, n_est, seed=0):
    import lightgbm as lgb
    clf = lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=n_est, learning_rate=0.06,
                             num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.7, reg_lambda=5.0, random_state=seed, n_jobs=4,
                             deterministic=True, force_row_wise=True, verbose=-1)
    clf.fit(X, y)
    return clf


def cmd_train(args):
    stats = load_stats(args.stats)
    stats["dead_gap_p"], stats["live_gap_frac"] = args.dead_gap_p, args.live_gap_frac
    files = sorted(glob.glob(os.path.join(args.cache, "*.npz")))
    if not files:
        sys.exit("no npz files in %s; run `prepare` first" % args.cache)
    by_match = {}
    for f in files:
        by_match.setdefault(os.path.basename(f).rsplit("_p", 1)[0], []).append(f)
    print("matches:", sorted(by_match))
    data, names = {}, None
    for mid, fl in sorted(by_match.items()):
        print("features for", mid)
        data[mid], names = match_sequences(fl, args.draws, args.seed, stats)
    os.makedirs(args.models, exist_ok=True)

    sw_grid, dead_grid = [10, 20, 40, 80], [10, 20, 40]
    best = (20, 20)
    mids = sorted(data)
    if args.folds > 0 and len(mids) > 1:
        folds = mids[:args.folds]
        scores = {(s, d): [] for s in sw_grid for d in dead_grid}
        for held in folds:
            tr = [q for m in mids if m != held for q in data[m]]
            clf = fit_lgbm(np.vstack([q["X"] for q in tr]), np.concatenate([q["y"] for q in tr]), args.n_estimators, args.seed)
            for q in [q for q in data[held] if not q["swap"] and q["draw"] == 0]:
                proba = clf.predict_proba(q["X"])
                for (s, d) in scores:
                    scores[(s, d)].append(proxy_spell_f1(q["y"], decode(proba, s, d)))
            print("fold held out %s done" % held)
        mean = {k: float(np.mean(v)) for k, v in scores.items()}
        for k in sorted(mean):
            print("  switch %3d  dead %3d  proxy spellF1 %.3f" % (k[0], k[1], mean[k]))
        best = max(mean, key=lambda k: (round(mean[k], 3), k[0], k[1]))  # ties -> more smoothing
        print("chosen penalties (switch, dead):", best)
    else:
        print("CV skipped; using default penalties", best)

    allq = [q for m in mids for q in data[m]]
    clf = fit_lgbm(np.vstack([q["X"] for q in allq]), np.concatenate([q["y"] for q in allq]), args.n_estimators, args.seed)
    clf.booster_.save_model(os.path.join(args.models, "lgbm.txt"))
    with open(os.path.join(args.models, "decoder.json"), "w") as fh:
        json.dump({"switch": best[0], "dead": best[1], "features": names, "stride": STRIDE}, fh, indent=1)
    print("saved models to", args.models)


# ----------------------------------------------------------------------------------------
# hackathon tracking
# ----------------------------------------------------------------------------------------
def read_meta(meta_path):
    root = ET.parse(meta_path).getroot()
    teams = [(t.get("id"), (t.findtext("Name") or "").strip(),
              (t.get("role") or t.get("Role") or "").strip().lower()) for t in root.iter("Team")]
    if len(teams) != 2:
        raise ValueError("%s: expected 2 teams, found %d" % (meta_path, len(teams)))
    home = [t for t in teams if t[2] == "home"]
    home = home[0] if home else teams[0]
    away = [t for t in teams if t[0] != home[0]][0]
    slots = [p.get("teamId") for p in root.iter("Player")]
    team = np.array([0 if s == home[0] else 1 if s == away[0] else -1 for s in slots], dtype=np.int8)
    return team, (home[1], away[1]), bool(any(t[2] == "home" for t in teams))


def load_tactical(data_dir, game, source):
    from games import GAMES, require, half_video_blocks, frame_offset
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir, source)
    team, names, by_role = read_meta(meta_path)
    pitch = GAMES[game]["pitch"]
    ranges = {}
    for half, v0, v1 in half_video_blocks(game):
        off = frame_offset(game, half)
        ranges[half] = (v0, v1, off, v0, v1)
    store = {h: dict(v0=r[3], v1=r[4], ball=np.full((r[4] - r[3] + 1, 2), np.nan),
                     P=None) for h, r in ranges.items()}
    seen_min, seen_max, n_lines, nslots = None, None, 0, None
    with open(raw_path) as fh:
        for line in fh:
            c = line.find(":")
            if c < 0:
                continue
            try:
                fid = int(line[:c])
            except ValueError:
                continue
            n_lines += 1
            seen_min = fid if seen_min is None else min(seen_min, fid)
            seen_max = fid if seen_max is None else max(seen_max, fid)
            for half, (i0, i1, off, v0, v1) in ranges.items():
                if i0 <= fid <= i1:
                    parts = line.rstrip().split(":")
                    if len(parts) < 3:
                        break
                    arr = np.array(parts[1].replace(";", ",").split(","), dtype=np.float64).reshape(-1, 3)
                    if nslots is None:
                        nslots = arr.shape[0]
                    s = store[half]
                    if s["P"] is None:
                        s["P"] = np.full((v1 - v0 + 1, nslots, 2), np.nan)
                    k = fid - v0
                    if arr.shape[0] == nslots:
                        s["P"][k] = arr[:, :2] * (pitch[0], pitch[1])
                    bp = parts[2].split(",")
                    try:
                        s["ball"][k] = (float(bp[0]) * pitch[0], float(bp[1]) * pitch[1])
                    except (ValueError, IndexError):
                        pass
                    break
    print("tracking file: %d lines, frame ids %s .. %s, %s slots" % (n_lines, seen_min, seen_max, nslots))
    for half, (i0, i1, off, v0, v1) in ranges.items():
        n_ball = int(np.sum(~np.isnan(store[half]["ball"][:, 0])))
        print("half %d: expected video frames %d..%d (%d frames), ball seen on %.1f%%" %
              (half, i0, i1, v1 - v0 + 1, 100.0 * n_ball / (v1 - v0 + 1)))
        if store[half]["P"] is None:
            sys.exit("no tracking rows fall inside half %d's IdFrame range. The file's frame counter does not "
                     "match games.py (ids %s..%s). Fix the alignment before going further." % (half, seen_min, seen_max))
    if nslots != len(team):
        print("WARNING: %s slots in the tracking but %d Player entries in the xml; check the slot -> team mapping"
              % (nslots, len(team)))
        team = np.concatenate([team, -np.ones(max(0, nslots - len(team)), np.int8)])[:nslots]
    print("teams: H = %s, A = %s (home chosen %s); slots: %d home, %d away, %d other" %
          (names[0], names[1], "from the role attribute" if by_role else "as the first Team in the xml",
           int((team == 0).sum()), int((team == 1).sum()), int((team < 0).sum())))
    return store, team, names


def cmd_check(args):
    from games import GAME
    store, team, names = load_tactical(args.data_dir, args.game, args.source)
    gaps, miss, total, nvis = [], 0, 0, []
    for h, s in store.items():
        m = np.isnan(s["ball"][:, 0])
        miss += int(m.sum())
        total += len(m)
        edges = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
        gaps += (np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)).tolist()
        nvis.append(np.mean((~np.isnan(s["P"][:, :, 0])).sum(1)))
    print("ball missing on %.1f%% of graded frames (the task says ~41%%)" % (100.0 * miss / total))
    print("mean visible player slots per frame: %.1f; %d ball gaps, median length %.0f frames" %
          (float(np.mean(nvis)), len(gaps), float(np.median(gaps)) if gaps else 0))
    os.makedirs(os.path.dirname(args.stats) or ".", exist_ok=True)
    rng = np.random.default_rng(0)
    sample = gaps if len(gaps) <= 2000 else rng.choice(gaps, 2000, replace=False).tolist()
    with open(args.stats, "w") as fh:
        json.dump({"gap_frames": [int(g) for g in sample], "ball_missing_frac": miss / total}, fh)
    print("wrote", args.stats, "(aggregate statistics only)")
    print("\nCheck that H is Barcelona and A is Athletic in the line above. If the names are the wrong way "
          "round, every label is flipped.")


def cmd_predict(args):
    import lightgbm as lgb
    from games import GAMES, anchor_video_frame
    store, team, names = load_tactical(args.data_dir, args.game, args.source)
    model = lgb.Booster(model_file=os.path.join(args.models, "lgbm.txt"))
    with open(os.path.join(args.models, "decoder.json")) as fh:
        dec = json.load(fh)
    g = GAMES[args.game]
    n_frames = g["n_video_frames"]
    out = np.full(n_frames, "X", dtype="<U1")
    kick = {1: g.get("kickoff"), 2: g.get("kickoff_2nd")}
    for half, s in sorted(store.items()):
        X0, _ = build_features(s["ball"], s["P"], team, swap=False)
        X1, _ = build_features(s["ball"], s["P"], team, swap=True)
        p = (model.predict(X0[::STRIDE]) + model.predict(X1[::STRIDE])[:, [1, 0, 2]]) / 2.0
        path = decode(p, dec["switch"], dec["dead"])
        n = s["v1"] - s["v0"] + 1
        full = np.repeat(path, STRIDE)[:n].copy()
        if kick.get(half):
            k = anchor_video_frame(kick[half]) - s["v0"]
            full[:max(0, min(k, n))] = DEAD          # dead until the kick-off
        out[s["v0"]:s["v1"] + 1] = LETTER[full]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("frame,label\n")
        fh.write("\n".join("%d,%s" % (i, out[i]) for i in range(n_frames)))
        fh.write("\n")
    graded = out != "X"
    print("wrote %s  (%d rows)" % (args.out, n_frames))
    for lab in "HAD":
        print("  %s: %.1f%% of graded frames" % (lab, 100.0 * np.mean(out[graded] == lab)))
    chg = int(np.sum((out[1:] != out[:-1]) & np.isin(out[1:], ["H", "A"]) & np.isin(out[:-1], ["H", "A"])))
    print("  direct H<->A changes: %d (%.1f per graded minute)" % (chg, chg / (graded.sum() / FPS / 60)))
    print("next: python evaluation/validate_submission.py %s" % args.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    from games import GAME, DEFAULT_SOURCE

    p = sub.add_parser("prepare", help="download / convert the DFL training matches")
    p.add_argument("--out", default=os.path.join(ROOT, "cache", "dfl"))
    p.add_argument("--match-ids", nargs="+", default=DFL_MATCH_IDS)
    p.add_argument("--local-dir", default=None, help="folder with manually downloaded DFL xml files")
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser("check", help="verify alignment and teams on the hackathon tracking")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--game", default=GAME)
    p.add_argument("--source", default=DEFAULT_SOURCE)
    p.add_argument("--stats", default=os.path.join(ROOT, "cache", "atd_stats.json"))
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("train", help="degrade DFL, fit LightGBM, choose decoder penalties")
    p.add_argument("--cache", default=os.path.join(ROOT, "cache", "dfl"))
    p.add_argument("--models", default=os.path.join(ROOT, "models_v2"))
    p.add_argument("--stats", default=os.path.join(ROOT, "cache", "atd_stats.json"))
    p.add_argument("--draws", type=int, default=1, help="degradation draws per match (2 = more data, slower)")
    p.add_argument("--folds", type=int, default=3, help="leave-one-match-out folds for the penalties (0 = skip)")
    p.add_argument("--n-estimators", type=int, default=250)
    p.add_argument("--dead-gap-p", type=float, default=DEAD_GAP_P)
    p.add_argument("--live-gap-frac", type=float, default=LIVE_GAP_FRAC)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("predict", help="write the submission CSV")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--game", default=GAME)
    p.add_argument("--source", default=DEFAULT_SOURCE)
    p.add_argument("--models", default=os.path.join(ROOT, "models_v2"))
    p.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "solver_v2.csv"))
    p.set_defaults(fn=cmd_predict)

    args = ap.parse_args()
    try:
        args.fn(args)
    except (KeyError, IOError) as err:
        print(str(err), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
