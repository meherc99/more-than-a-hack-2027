import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES  # noqa: E402


def build_labels(n_frames):
    return ["H"] * n_frames


def write_submission(path, labels):
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        for frame, label in enumerate(labels):
            handle.write(f"{frame},{label}\n")
    return path


def main():
    parser = argparse.ArgumentParser(description="Create a submission where every frame is labelled H.")
    parser.add_argument("--game", default=GAME, help="Game name to size the CSV for.")
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "always_home.csv"), help="Output CSV path.")
    args = parser.parse_args()

    n_frames = GAMES[args.game]["n_video_frames"]
    labels = build_labels(n_frames)
    out = write_submission(args.out, labels)
    print(f"Wrote {out} with {n_frames} rows.")


if __name__ == "__main__":
    main()
