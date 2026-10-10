"""Render a grid of frames from one video window, for annotating or checking a gold span.

    python scripts/contact_sheet.py corpus/sintel.mkv 410 440 --step 1 -o sheet.jpg

Frames are read left to right, top to bottom, starting at t_start and spaced
--step seconds apart. The printed legend maps each cell to its timestamp.
"""

import argparse
import math
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video")
    parser.add_argument("t_start", type=float)
    parser.add_argument("t_end", type=float)
    parser.add_argument("--step", type=float, default=1.0, help="seconds between frames")
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--width", type=int, default=240, help="pixel width of each frame")
    parser.add_argument("-o", "--output", default="contact_sheet.jpg")
    args = parser.parse_args()

    n = max(1, int((args.t_end - args.t_start) / args.step))
    rows = math.ceil(n / args.cols)
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y", "-ss", str(args.t_start), "-i", args.video,
            "-t", str(args.t_end - args.t_start),
            "-vf", f"fps=1/{args.step},scale={args.width}:-2,tile={args.cols}x{rows}",
            "-frames:v", "1", args.output,
        ],
        check=True,
    )
    for row in range(rows):
        cells = [args.t_start + (row * args.cols + c) * args.step for c in range(args.cols)]
        print(f"row {row + 1}: " + "  ".join(f"{t:.1f}s" for t in cells if t < args.t_end))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
